import logging
from typing import Any, Optional, Tuple, Union

import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimatorV1
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_algorithms.gradients import SPSAEstimatorGradient
from qiskit_algorithms.optimizers import ADAM # ใข้ ADAM ของ qiskit แทน torch.optim.Adam เพราะมันออกแบบมาเพื่อใช้กับ quantum circuit โดยเฉพาะ
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN

from models.feature_maps import FeatureMapBuilder, resolve_feature_map

from .base import BaseEngine

logger = logging.getLogger(__name__)


class QiskitEngine(BaseEngine):
    def __init__(
        self,
        feature_map: Union[str, FeatureMapBuilder] = "angle",
        max_iter: int = 50,
        verbose: bool = True,
        use_v2_primitives: bool = True,
    ):
        self.feature_map_builder = resolve_feature_map(feature_map)
        self.max_iter = max_iter
        self.verbose = verbose
        self.use_v2 = use_v2_primitives

        self._setup_estimator()

    def _setup_estimator(self): # เลือกใช้ AerEstimatorV2 เป็นหลัก แต่บางเครื่องเก่าๆ ที่ใช้ V2 ไม่ได้ให้ใช้ V1 แทน กัน error
        if self.use_v2:
            self.estimator = AerEstimatorV2()
            try:
                self.estimator.options.update(device="GPU")  # type: ignore
            except Exception:
                pass
        else:
            self.estimator = AerEstimatorV1()  # type: ignore

        self.grad_method = SPSAEstimatorGradient(self.estimator, epsilon=0.01)  # type: ignore

    def _create_observable(self, num_qubits: int, last_qubit: int) -> SparsePauliOp:
        return SparsePauliOp.from_sparse_list([("Z", [last_qubit], 1.0)], num_qubits=num_qubits)

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
    ) -> Tuple[float, dict, Any]:
        num_qubits = circuit.num_qubits

        # สร้างวงจร = feature map + ต่อกับ quantum circuit + ปิดท้ายด้วย observable ตัวหาค่า <Z> บนคิวบิตสุดท้าย
        fm, input_params = self.feature_map_builder.build(num_qubits)

        composed = fm.compose(circuit)
        assert isinstance(composed, QuantumCircuit)
        full_circuit: QuantumCircuit = composed
        observable = self._create_observable(num_qubits, last_qubit)

        # Filter params ให้เหลือเฉพาะตัวที่มีอยู่ในวงจรจริงๆ defualtมัน1000ตัวไงก็fit ลงมา
        circuit_params = set(full_circuit.parameters)
        filtered_params = [p for p in params if p in circuit_params]

        qnn = EstimatorQNN( # Estimator คือ ตัวที่คำนวณ output ของวงจร
            circuit=full_circuit,
            input_params=list(input_params),
            weight_params=filtered_params,
            observables=observable,
            estimator=self.estimator,
            gradient=self.grad_method,  # type: ignore
            input_gradients=True,
        )

        # Prepare history tracking
        history = {"loss": []}

        def callback(weights, loss):
            history["loss"].append(float(loss))
            if self.verbose:
                # Log every 10 iterations to avoid spamming, or every 1 if max_iter is small
                if len(history["loss"]) % 10 == 0 or len(history["loss"]) == 1:
                    logger.info(
                        f"Qiskit Optimization Iteration {len(history['loss'])}/{self.max_iter} "
                        f"- Loss: {loss:.4f}"
                    )

        # เราไม่ได้เขียน loop เทรนเอง แต่ใช้ classifier.fit() ของ qiksit แทน
        classifier = NeuralNetworkClassifier(
            qnn,
            optimizer=ADAM(maxiter=self.max_iter),
            callback=callback, # เลยต้องใส่ฟังก์ชัน callback บอก qiskit ว่าฝากเพื่อเก็บประวัติ loss ให้หน่อย
        )

        # แปลงข้อมูลจาก torch.Tensor เป็น numpy.ndarray เพื่อให้ qiskit อ่านออก
        X_train_np = x_train.cpu().numpy()
        y_train_np = y_train.cpu().numpy()

        try:
            if self.verbose:
                logger.info(f"Starting Qiskit training for {self.max_iter} iterations...")

            # สั่งเทรนด้วย qiskit มันจะไปวน forward + backward + update weight ให้เอง เริ่ด
            classifier.fit(X_train_np, y_train_np)

            score = 0.0
            if x_test is not None and y_test is not None:
                X_test_np = x_test.cpu().numpy()
                y_test_np = y_test.cpu().numpy()
                # ประเมินผลด้วย test set
                score = classifier.score(X_test_np, y_test_np)

            # Now we return the REAL history captured by the callback!
            return float(score), history, classifier
        except Exception as e:
            logger.error(f"QiskitEngine failed: {e}")
            return 0.0, history, None

""" สรุปสิ่งที่ทำในไฟล์นี้:
ช่วงที่ 1 : เตรียมเครื่อง
    - เลือกใช้ AerEstimatorV2 เป็นหลัก แต่บางเครื่องเก่าๆ ที่ใช้ V2 ไม่ได้ให้ใช้ V1 แทน กัน error
ช่วงที่ 2 : สร้างวงจรควอนตัมที่ประกอบด้วย Feature Map + QCNN + ใส่ตัว Observable ที่คิวบิตสุดท้าย
ช่วงที่ 3 : เทรนโมเดลด้วย Qiskit NeuralNetworkClassifier
"""
