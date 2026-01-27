"""Reusable experiment template to reduce duplication across QCNN variants.

Usage:
    from experiment_template import ExperimentConfig, ExperimentRunner
    cfg = ExperimentConfig(backend="qea-qcnn", dataset="mnist")
    runner = ExperimentRunner(cfg)
    result = runner.run()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
import logging

from data import DataManager, Preprocessor, DataConfig
from QEAQCNN.QCNN_structure import QCNNBuilder
from QEAQCNN.Evaluation import Experiment as QEAExperiment, HybridEvaluator
from qcnn_shared.feature_maps import resolve_feature_map
from QEAQCNN.utils import graph_history, initialize_output_dir, save_model

logger = logging.getLogger(__name__)


BackendType = Literal["qea-qcnn", "qiskit-qcnn"]


@dataclass
class ExperimentConfig: #เก็บการตั้งค่าการทดลอง
    backend: BackendType  # qea-qcnn, qiskit-qcnn
    dataset: str = "mnist" # mnist, fashion-mnist, cifar10
    data_path: str = "../data" # path to data directory
    n_train: int = 400
    n_test: int = 100
    preprocessors: Sequence[Union[str, Preprocessor]] = field(
        default_factory=lambda: ["bilinear_resize_4x4", "flatten"]
    )
    n_qubits: int = 16
    encoding: str = "angle"  # angle, pge, zz

    # QEA QCNN settings
    n_pop: int = 50
    n_gen: int = 20
    n_gates: int = 180
    epochs: int = 5
    lr: float = 0.01

    # Qiskit QCNN settings
    max_iter: int = 200

    # Output control
    save_outputs: bool = True #จะให้เซฟลงไฟล์ไหม?
    script_name: str = "experiment" #ชื่อไฟล์ Log


@dataclass
class ExperimentResult: #เก็บผลลัพธ์ของการทดลอง
    summary: Dict[str, Any] #สรุปผล (เช่น Accuracy, Final Loss)
    # สิ่งที่สร้างขึ้น ( Model Object, Trainer Object) เก็บใส่ Dict
    artifacts: Dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.save_dir: Optional[str] = None
        self.file_id: Optional[str] = None

    # ----------------------
    # Public entry point
    # ----------------------
    def run(self) -> ExperimentResult:
        # 1. ถ้าสั่งให้เซฟ -> สร้าง Folder รอไว้เลย
        if self.config.save_outputs:
            self.save_dir, self.file_id = initialize_output_dir(self.config.script_name)

        data_mgr = self._build_data_manager()

        if self.config.backend == "qea-qcnn":
            return self._run_qea_qcnn(data_mgr)
        if self.config.backend == "qiskit-qcnn":
            return self._run_qiskit_qcnn(data_mgr)
        raise ValueError(f"Unsupported backend: {self.config.backend}")

    # ----------------------
    # Builders (คนเตรียมของ)
    # ----------------------
    def _build_data_manager(self) -> DataManager:
        # แปลง ExperimentConfig ให้เป็น DataConfig (เฉพาะส่วนที่ DataManager ต้องใช้)
        dcfg = DataConfig(
            dataset=self.config.dataset,
            data_path=self.config.data_path,
            n_train=self.config.n_train,
            n_test=self.config.n_test,
            preprocessors=self.config.preprocessors,
            target_dim=self.config.n_qubits,
        )
        return DataManager(dcfg)

    # ----------------------
    # Backend runners
    # ----------------------
    def _run_qea_qcnn(self, data_mgr: DataManager) -> ExperimentResult:
        # 1. สร้างตัวสร้างวงจร (Builder) รอไว้
        builder = QCNNBuilder(self.config.n_qubits)
        # 2. เลือก Feature Map ตามที่ตั้งค่า (Angle, PGE, ZZ)
        feature_map = resolve_feature_map(self.config.encoding)
        
        # 3. ตัวนี้จะเอารหัสพันธุกรรมไปสร้างวงจร -> เทรน 5 Epochs -> ให้คะแนนความแม่น
        evaluator = HybridEvaluator(
            builder,
            epochs=self.config.epochs,
            lr=self.config.lr,
            feature_map=feature_map,
        )
        
        # 4. สร้างการทดลอง QEA QCNN
        experiment = QEAExperiment(
            data_mgr=data_mgr,
            evaluator=evaluator,
            n_pop=self.config.n_pop,
            n_gen=self.config.n_gen,
            n_gates=self.config.n_gates,
        )
        
        # 5. เริ่มวิวัฒนาการ experiment.run()
        # ได้ผลลัพธ์เป็น โมเดลที่ดีที่สุด (best_model) และกราฟประวัติ (history)
        best_model, history = experiment.run()

        # 6. ถ้าสั่งเซฟ -> บันทึกกราฟและโมเดลลงไฟล์
        if self.save_dir and self.file_id and best_model:
            graph_history(best_model, history, experiment=experiment, save_dir=self.save_dir, file_id=self.file_id)
            save_model(best_model, save_dir=self.save_dir, file_id=self.file_id)

        # 7. ห่อผลลัพธ์ลง ExperimentResult ส่งกลับ
        summary = {
            "best_accuracy": best_model.fitness if best_model else None,
            "history": history,
        }
        artifacts = {"best_model": best_model, "history": history}
        return ExperimentResult(summary=summary, artifacts=artifacts)

    def _run_qiskit_qcnn(self, data_mgr: DataManager) -> ExperimentResult:
        # Lazy imports เพื่อที่ว่าถ้าเรารัน QEA เราจะได้ไม่ต้องโหลด library ของส่วนนี้ให้หนักเครื่อง
        from qiskitQCNN.qiskitQCNN_structure import QCNNStructure
        from qiskitQCNN.trainer import QCNNTrainer

        # 1. เตรียมข้อมูล
        x_train, y_train, x_test, y_test = data_mgr.get_data(as_numpy=True)
        # 2. สร้างโครงสร้าง QCNN
        q_struct = QCNNStructure(num_qubits=self.config.n_qubits)
        # 3. เลือก Feature Map (ให้เหมือนกับฝั่ง QEA)
        feature_map = resolve_feature_map(self.config.encoding)
        # 4. ประกอบร่าง
        circuit, input_params, weight_params = q_struct.build_full_circuit(feature_map)
        
        # 5. สร้างเทรนเนอร์ (Trainer) และสั่งเทรน  
        trainer = QCNNTrainer(circuit, input_params, weight_params)
        trainer.train(x_train, y_train, max_iter=self.config.max_iter)

        # 6. วัดผลสอบ (Evaluate) ทั้ง Train และ Test set
        train_score = trainer.evaluate(x_train, y_train)
        test_score = trainer.evaluate(x_test, y_test)

        # 7. ห่อผลลัพธ์ลง ExperimentResult ส่งกลับ
        summary = {"train_score": train_score, "test_score": test_score}
        artifacts = {"trainer": trainer}
        return ExperimentResult(summary=summary, artifacts=artifacts)
