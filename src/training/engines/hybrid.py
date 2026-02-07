import logging
import os
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimatorV1
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient, SPSAEstimatorGradient
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

from models.feature_maps import FeatureMapBuilder, resolve_feature_map

from .base import BaseEngine

logger = logging.getLogger(__name__)


class HybridEngine(BaseEngine):
    """
    Optimization Engine using PyTorch + Qiskit.
    """

    def __init__(
        self,
        feature_map: Union[str, FeatureMapBuilder] = "angle",
        epochs: int = 5,
        lr: float = 0.01,
        gradient_method: str = "param_shift",
        device: Optional[str] = None,
        use_v2_primitives: bool = True,
        use_scheduler: bool = True,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        verbose: bool = True,
    ):
        self.feature_map_builder = resolve_feature_map(feature_map)
        self.epochs = epochs
        self.lr = lr
        self.gradient_method = gradient_method.lower()
        self.verbose = verbose
        self.use_v2 = use_v2_primitives
        self.use_scheduler = use_scheduler
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.loss_fn = nn.MSELoss()

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if self.verbose:
            logger.info(f"HybridEngine using device: {self.device}")

        self._setup_estimator()

    def _setup_estimator(self):
        if self.use_v2:
            self.estimator = AerEstimatorV2()
            if self.device.type == "cuda":
                # Attempt to set GPU; using type ignore as Aer V2 options are dynamic
                try:
                    self.estimator.options.update(device="GPU")  # type: ignore
                except Exception:
                    pass # ถ้าไม่มี GPU หรือตั้งค่าไม่ได้ ก็ปล่อยผ่าน (ใช้ CPU ไป)
                if self.verbose:
                    logger.info("HybridEngine: Attempting GPU enable.")
        else:
            options = {}
            if self.device.type == "cuda":
                options["device"] = "GPU"
            self.estimator = AerEstimatorV1(backend_options=options)  # type: ignore

        if self.gradient_method == "spsa":
            self.grad_method = SPSAEstimatorGradient(self.estimator, epsilon=0.01)  # type: ignore
        else:
            self.grad_method = ParamShiftEstimatorGradient(self.estimator)  # type: ignore

        if self.verbose:
            logger.info(f"HybridEngine: Using gradient method: {self.gradient_method}")

    def _create_observable(self, num_qubits: int, last_qubit: int) -> SparsePauliOp:
        # สร้างตัวดำเนินการ Z บน qubit ตัวสุดท้ายที่เหลือรอด (last_qubit)
        return SparsePauliOp.from_sparse_list([("Z", [last_qubit], 1.0)], num_qubits=num_qubits)

    def _save_checkpoint(self, checkpoint_path: str, epoch: int, model: Any, optimizer: Any, history: dict, scheduler: Optional[Any] = None) -> None:
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }
        if scheduler:
            checkpoint_state["scheduler_state_dict"] = scheduler.state_dict()
        
        try:
            torch.save(checkpoint_state, checkpoint_path)
        except Exception as e:
            logger.warning(f"HybridEngine: Failed to save checkpoint: {e}")

    def _load_checkpoint(self, checkpoint_path: str, model: Any, optimizer: Any, scheduler: Optional[Any] = None) -> Tuple[int, dict]:
        if not os.path.exists(checkpoint_path):
            logger.info(f"HybridEngine: No checkpoint found at {checkpoint_path}. Starting fresh.")
            return 0, {}

        logger.info(f"HybridEngine: Found checkpoint at {checkpoint_path}. Resuming...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            start_epoch = checkpoint["epoch"] + 1
            history = checkpoint.get("history", {})
            logger.info(f"HybridEngine: Resuming from epoch {start_epoch}")
            return start_epoch, history
        except Exception as e:
            logger.error(f"HybridEngine: Failed to load checkpoint: {e}. Starting fresh.")
            return 0, {}

    def fit(
        self,
        circuit: QuantumCircuit,
        params: ParameterVector,
        last_qubit: int,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        x_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        initial_state_dict: Optional[dict] = None,
        checkpoint_dir: Optional[str] = None,
        file_id: Optional[str] = None,
    ) -> Tuple[float, dict, Any]:
        num_qubits = circuit.num_qubits
        # สร้าง feature map
        fm, input_params = self.feature_map_builder.build(num_qubits)
        # เอา feature map มาต่อกับ QCNN circuit
        composed = fm.compose(circuit)
        assert isinstance(composed, QuantumCircuit) # ตรวจสอบว่าเป็น QuantumCircuit จริง
        full_circuit: QuantumCircuit = composed # QCNN + Feature Map
        observable = self._create_observable(num_qubits, last_qubit)

        # กรอง parameters ให้เหลือเฉพาะตัวที่มีอยู่ในวงจรจริงๆ
        circuit_params = set(full_circuit.parameters)
        filtered_params = [p for p in params if p in circuit_params]

        if self.verbose:
            logger.info(f"HybridEngine: Optimizing {len(filtered_params)} parameters.")

        qnn = EstimatorQNN( # Estimator คือ ตัวที่คำนวณ output ของวงจร
            circuit=full_circuit,
            input_params=list(input_params),
            weight_params=filtered_params,
            observables=observable,
            estimator=self.estimator,
            gradient=self.grad_method,
            input_gradients=True,
        )

        model = TorchConnector(qnn).to(self.device) # สร้าง PyTorch model จาก QNN

        # Check for initial weights
        if initial_state_dict is not None:
             # Load pre-trained weights if provided (Transfer Learning)
            try:
                model.load_state_dict(initial_state_dict)
                if self.verbose:
                    logger.info("HybridEngine: Loaded initial weights from state dictionary.")
            except Exception as e:
                logger.warning(f"HybridEngine: Failed to load initial weights: {e}. Falling back to random init.")
                # Explicit initialization to avoid Barren Plateaus/Zero-gradient starts
                with torch.no_grad():
                    for p in model.parameters():
                        p.uniform_(-0.5, 0.5)
        else:
            # Explicit initialization to avoid Barren Plateaus/Zero-gradient starts
            # สุ่มค่าเริ่มต้นของ weights ในช่วง [-0.5, 0.5] เพื่อหลีกเลี่ยงปัญหา Barren Plateaus
            with torch.no_grad():
                for p in model.parameters():
                    p.uniform_(-0.5, 0.5)
            if self.verbose:
                logger.info("HybridEngine: Initialized weights with uniform distribution [-0.5, 0.5]")

        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        if self.verbose:
            logger.debug(
                f"HybridEngine: Circuit parameters: {[p.name for p in full_circuit.parameters]}"
            )
            logger.debug(f"HybridEngine: Training params: {[p.name for p in filtered_params]}")
            logger.debug(f"HybridEngine: Data Magnitude: {x_train.abs().mean().item():.3f}")
            if self.device.type == "mps":
                logger.warning(
                    "Using MPS (Metal Performance Shaders). "
                    "If gradients are zero, try 'engine.device=cpu'."
                )

        scheduler = None
        if self.use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            )

        # ย้ายข้อมูลไปยัง device ที่กำหนด (CPU/GPU)
        x_train_dev = x_train.to(self.device)
        y_train_dev = y_train.to(self.device).unsqueeze(1)

        # Prepare validation data if provided
        x_val_dev = None
        y_val_dev = None
        if x_val is not None and y_val is not None:
             x_val_dev = x_val.to(self.device)
             y_val_dev = y_val.to(self.device).unsqueeze(1)

        """
        3. เข้าโหมดการเทรน
        """
        history = {"loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        start_epoch = 0
        checkpoint_path = None

        if checkpoint_dir and file_id:
            checkpoint_path = os.path.join(checkpoint_dir, f"{file_id}_checkpoint.pth.tar")
            loaded_epoch, loaded_history = self._load_checkpoint(checkpoint_path, model, optimizer, scheduler)
            if loaded_epoch > 0:
                start_epoch = loaded_epoch
                history = loaded_history

        for epoch in range(start_epoch, self.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(x_train_dev)
            loss = self.loss_fn(output, y_train_dev)
            loss.backward()

            total_grad = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad += p.grad.norm().item()

            optimizer.step()

            history["loss"].append(float(loss.item()))

            # Calculate Epoch Training Accuracy
            with torch.no_grad():
                preds = torch.sign(output)
                train_acc = (preds == y_train_dev).float().mean().item()
                history["train_acc"].append(float(train_acc))

            # Validation Step
            val_loss_scalar = 0.0
            val_acc = 0.0
            if x_val_dev is not None and y_val_dev is not None:
                model.eval()
                with torch.no_grad():
                    val_out = model(x_val_dev)
                    val_loss = self.loss_fn(val_out, y_val_dev)
                    val_loss_scalar = val_loss.item()

                    val_preds = torch.sign(val_out)
                    val_acc = (val_preds == y_val_dev).float().mean().item()

                history["val_loss"].append(val_loss_scalar)
                history["val_acc"].append(val_acc)

                # Step Scheduler
                if scheduler is not None:
                    scheduler.step(val_loss_scalar)

                # Retrieve current LR
                if scheduler is not None and hasattr(scheduler, "get_last_lr"):
                     current_lr = scheduler.get_last_lr()[0]
                else:
                     current_lr = optimizer.param_groups[0]['lr']

            if self.verbose:
                out_mean = output.mean().item()
                out_std = output.std().item()
                log_msg = (
                    f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f} | "
                    f"Acc: {train_acc:.4f} | "
                    f"Grad: {total_grad:.2e} "
                )
                if x_val_dev is not None:
                    log_msg += f"| Val Loss: {val_loss_scalar:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}"

                logger.info(log_msg)

            # Save Checkpoint
            if checkpoint_path:
                self._save_checkpoint(checkpoint_path, epoch, model, optimizer, history, scheduler)

        # Evaluate on test set if provided
        if x_test is not None and y_test is not None:
            x_test_dev = x_test.to(self.device)
            y_test_dev = y_test.to(self.device).unsqueeze(1)

            model.eval()

            with torch.no_grad():
                preds = torch.sign(model(x_test_dev))
                acc = (preds == y_test_dev).float().mean().item()
            return float(acc), history, model.state_dict()

        return 0.0, history, model.state_dict()

""" สรุปสิ่งที่ทำในไฟล์นี้:
ช่วงที่ 1 : เตรียมเครื่อง
    - เอาโมเดลไปวางบนสนามซ้อม (CPU/GPU)
    - ย้ายข้อมูล (x_train, y_train) ไปวางบนสนามเดียวกัน
ช่วงที่ 2 : เตรียมโมเดล
    - สร้างวงจรควอนตัมที่ประกอบด้วย Feature Map + QCNN + ใส่ตัว Observable ที่คิวบิตสุดท้าย
    - กรองพารามิเตอร์ที่ใช้จริงในวงจรออกมา ว่าสรุปใช้กี่ตัว
    - สร้าง QNN เป็นตัวคำนวณค่า expectation value <Z> ของควอนตัม และเชื่อมต่อกับ PyTorch คือเป็นตัวหา gradient แบบ classic (ใช้ ADAM) ด้วย TorchConnector
ช่วงที่ 3 : เข้าโหมดการเทรน
    - เทรนโมเดลเป็นจำนวน epochs ที่กำหนด
    - ในแต่ละ epoch:
        - คำนวณ output ของโมเดล
        - คำนวณ loss ระหว่าง output กับ y_train
        - คำนวณ gradient ผ่าน backpropagation
            - เช็คขนาดของ gradient เพื่อดูว่าโมเดลยังเรียนรู้ปกติอยู่ไหม
        - อัพเดทพารามิเตอร์ของโมเดลด้วย ADAM optimizer
        - บันทึก loss และ accuracy ใน history
ช่วงที่ 4 : ประเมินผล
    - ถ้ามีชุดทดสอบ (x_test, y_test) ให้ประเมินความแม่นยำหลังเทรนเสร็จ
    - ส่งกลับความแม่นยำสุดท้าย acc (ถ้ามี), ประวัติการเทรน history, และ state_dict = สมองของโมเดลที่เทรนแล้ว
"""
