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
    retrain_with_qiskit: bool = False # If True, train best QEA structure using Qiskit Trainer
    structure_code: Optional[List[int]] = None # Use this structure if provided

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

        # 3. Evolution Phase (หาโครงสร้าง)
        logger.info("Starting QEA Evolution Phase...")
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

        # 7. Retrain Phase (ถ้าสั่ง)
        retrain_result: Dict[str, Any] = None
        if self.config.retrain_with_qiskit and best_model:
            logger.info("Retraining best structure using Qiskit Trainer (ADAM)...")
            result_obj = self._run_qiskit_qcnn(
                data_mgr, 
                structure_code=best_model.structure_code,
                retrain_mode=True
            )
            retrain_result = result_obj.summary

        # 8. ห่อผลลัพธ์ลง ExperimentResult ส่งกลับ
        summary = {
            "best_accuracy": best_model.fitness if best_model else None,
            "history": history,
            "retrain_result": retrain_result
        }
        artifacts = {"best_model": best_model, "history": history}
        return ExperimentResult(summary=summary, artifacts=artifacts)

    def _run_qiskit_qcnn(
        self, 
        data_mgr: DataManager, 
        structure_code: List[int] = None,
        retrain_mode: bool = False
    ) -> ExperimentResult:
        # Lazy imports เพื่อที่ว่าถ้าเรารัน QEA เราจะได้ไม่ต้องโหลด library ของส่วนนี้ให้หนักเครื่อง
        from qiskitQCNN.qiskitQCNN_structure import QCNNStructure
        from qiskitQCNN.trainer import QCNNTrainer

        # Override structure code if provided in config
        if structure_code is None and self.config.structure_code is not None:
             structure_code = self.config.structure_code

        # 1. เตรียมข้อมูล
        x_train, y_train, x_test, y_test = data_mgr.get_data(as_numpy=True)
        
        # 2. Build Feature Map Circuit (Shared logic)
        logger.info(f"Building Feature Map ({self.config.encoding})...")
        feature_map_builder = resolve_feature_map(self.config.encoding)
        fm_circuit, fm_params = feature_map_builder.build(self.config.n_qubits)

        if structure_code:
            # Case A: QEA Structure (Reconstruct)
            logger.info("Building QCNN from QEA Structure Code...")
            builder = QCNNBuilder(self.config.n_qubits)
            ansatz, _ = builder.assemble(structure_code)
            
            # Manual composition for QEA
            circuit = fm_circuit.compose(ansatz)
            input_params = fm_params
            weight_params = ansatz.parameters
            
        else:
            # Case B: Standard QCNN Structure
            logger.info("Building Standard QCNN Structure...")
            q_struct = QCNNStructure(num_qubits=self.config.n_qubits)
            
            # Note: QCNNStructure.build_full_circuit creates its own feature map internally via resolve_feature_map.
            # To avoid duplication or mismatch, we let it do its thing by passing the builder/string.
            # OR we modify q_struct to accept a circuit. 
            # Current implementation of build_full_circuit:
            #   fmap_builder = resolve_feature_map(feature_map)
            #   feature_map_circuit, feature_params = fmap_builder.build(self.num_qubits)
            #   ...
            # So passing fm_circuit logic is NOT supported by current QCNNStructure.build_full_circuit.
            # We will use the existing method signature.
            circuit, input_params, weight_params = q_struct.build_full_circuit(self.config.encoding)
        
        # 3. Train
        logger.info(f"Training start (Max Iter: {self.config.max_iter})...")
        # Reuse initial weights if provided (and structure matches) - logic implicit in QCNNTrainer if we passed path
        trainer = QCNNTrainer(circuit, input_params, weight_params, initial_point_path=self.config.initial_weights_path)
        
        trainer.train(x_train, y_train, max_iter=self.config.max_iter)

        # 4. Evaluate
        train_score = trainer.evaluate(x_train, y_train)
        test_score = trainer.evaluate(x_test, y_test)
        logger.info(f"Result: Train={train_score:.4f}, Test={test_score:.4f}")
        
        # 5. Save Weights
        if self.save_dir and trainer.classifier and hasattr(trainer.classifier, 'weights'):
             import os
             import json
             filename = "retrained_weights.json" if retrain_mode else "trained_weights.json"
             weights_path = os.path.join(self.save_dir, "model", filename)
             os.makedirs(os.path.dirname(weights_path), exist_ok=True)
             with open(weights_path, 'w') as f:
                 json.dump(trainer.classifier.weights.tolist(), f)
             logger.info(f"💾 Saved Qiskit weights to {weights_path}")

        # 7. ห่อผลลัพธ์ลง ExperimentResult ส่งกลับ
        summary = {"train_score": train_score, "test_score": test_score}
        artifacts = {"trainer": trainer}
        return ExperimentResult(summary=summary, artifacts=artifacts)
