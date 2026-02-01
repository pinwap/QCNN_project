import logging
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
        verbose: bool = True,
    ):
        self.feature_map_builder = resolve_feature_map(feature_map)
        self.epochs = epochs
        self.lr = lr
        self.gradient_method = gradient_method.lower()
        self.verbose = verbose
        self.use_v2 = use_v2_primitives
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

    def fit(
        self,
        circuit: QuantumCircuit,
        params: ParameterVector,
        last_qubit: int,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        initial_state_dict: Optional[dict] = None,
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

        # ย้ายข้อมูลไปยัง device ที่กำหนด (CPU/GPU)
        x_train_dev = x_train.to(self.device)
        y_train_dev = y_train.to(self.device).unsqueeze(1) # unsqueeze คือการเพิ่มมิติ จาก (N,) เป็น (N,1) y_train เดิมมีรูปร่าง [10] (มี 10 ตัวเลขเรียงกัน) เลยต้องเพิ่มมิติให้เป็น [10,1] เพื่อให้เข้ากับ output ของ model ที่มีรูปร่าง (N, 1)

        """
        3. เข้าโหมดการเทรน
        """
        history = {"loss": [], "train_acc": []}
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = model(x_train_dev)
            loss = self.loss_fn(output, y_train_dev)
            loss.backward() # คำนวณ gradient ผ่าน backpropagation ว่าควรขยับไปทางไหน

            # Diagnostic: Check gradient magnitude
            total_grad = 0.0
            for p in model.parameters(): # ตรวจสอบขนาดของ gradient = p.grad = ทิศทางและค่าความชันที่คำนวณได้ในแต่ละพารามิเตอร์
                if p.grad is not None:
                    total_grad += p.grad.norm().item() # norm() คือ การหาขนาดความยาวรวมของเวกเตอร์ gradient ว่าต้องเดินไปในทิศทางไหนและมากน้อยแค่ไหน
                    # ไว้เช็คว่าโมเดลยังเรียนรู้ปกติอยู่ไหม : 
                    #   - ถ้า total_grad ใกล้ 0 คือไม่มี gradient ให้เรียนรู้ อาจจะเจอปัญหา Barren Plateaus 
                    #   - แต่ถ้าใหญ่มากๆ อาจะเกิด exploding gradients คือ gradient ใหญ่เกินไปทำให้การอัพเดทพารามิเตอร์ไม่เสถียรโดดไปมาจนพัง
                    
            optimizer.step() # เอาค่า gradient ที่คำนวณได้มาคูณกับ learning rate แล้วไปอัพเดท weight พารามิเตอร์ของโมเดล = ก้าวเท้าเดินจริง ๆ

            history["loss"].append(float(loss.item()))

            # Calculate Epoch Training Accuracy
            with torch.no_grad(): # เป็นการบอก PyTorch ว่าในบล็อกนี้ไม่ต้องคำนวณ gradient ช่วยประหยัดหน่วยความจำและเพิ่มความเร็ว
                # ในบล็อกนี้เราแค่จะ "คำนวณความแม่นยำ (Accuracy)" เพื่อดูผลเฉยๆ ไม่ได้จะเอาไปเทรน (Update Weight) ดังนั้นเราจึงไม่จำเป็นต้องคำนวณ Gradient
                preds = torch.sign(output)
                train_acc = (preds == y_train_dev).float().mean().item()
                history["train_acc"].append(float(train_acc))

            if self.verbose:
                out_mean = output.mean().item()
                out_std = output.std().item()
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f} | "
                    f"Acc: {train_acc:.4f} | "
                    f"Grad: {total_grad:.2e} | Mean: {out_mean:.3f}, Std: {out_std:.3f}"
                )
                
        # Evaluate on test set if provided
        if x_test is not None and y_test is not None:
            x_test_dev = x_test.to(self.device)
            y_test_dev = y_test.to(self.device).unsqueeze(1)
            
            # บอกโมเดลให้เข้าสู่โหมดสอบ จะปิดการทำงานบางอย่างที่ใช้เฉพาะตอนเทรน เช่น 
            #   - dropout = ตอนเทรนจะสุ่มปิดเซลล์เพื่อไม่ให้โมเลจำข้อสอบ แต่พอสอบต้องเปิดเพื่อให้ได้ประสิทธิภาพมากที่สุด, 
            #   - batchnorm จะใช้สถิติของทั้ง batch ในการ normalize ตอนเทรน แต่ตอนสอบจะใช้สถิติที่เรียนรู้มาแล้ว
            model.eval() 
            
            with torch.no_grad():
                preds = torch.sign(model(x_test_dev)) # sign() คือการแปลง output เป็น -1 หรือ 1 จาก model(x) ที่จะได้ค่าเป็นทศนิยม 
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