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
from qcnn_common.feature_maps import resolve_feature_map
from QEAQCNN.utils import graph_history, initialize_output_dir, save_model

logger = logging.getLogger(__name__)


BackendType = Literal["qea-qcnn", "qiskit-qcnn"]


@dataclass
class ExperimentConfig:
    backend: BackendType
    dataset: str = "mnist"
    data_path: str = "../data"
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
    save_outputs: bool = True
    script_name: str = "experiment"


@dataclass
class ExperimentResult:
    summary: Dict[str, Any]
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
        if self.config.save_outputs:
            self.save_dir, self.file_id = initialize_output_dir(self.config.script_name)

        data_mgr = self._build_data_manager()

        if self.config.backend == "qea-qcnn":
            return self._run_qea_qcnn(data_mgr)
        if self.config.backend == "qiskit-qcnn":
            return self._run_qiskit_qcnn(data_mgr)
        raise ValueError(f"Unsupported backend: {self.config.backend}")

    # ----------------------
    # Builders
    # ----------------------
    def _build_data_manager(self) -> DataManager:
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
        builder = QCNNBuilder(self.config.n_qubits)
        feature_map = resolve_feature_map(self.config.encoding)
        evaluator = HybridEvaluator(
            builder,
            epochs=self.config.epochs,
            lr=self.config.lr,
            feature_map=feature_map,
        )
        experiment = QEAExperiment(
            data_mgr=data_mgr,
            evaluator=evaluator,
            n_pop=self.config.n_pop,
            n_gen=self.config.n_gen,
            n_gates=self.config.n_gates,
        )

        best_model, history = experiment.run()

        if self.save_dir and self.file_id and best_model:
            graph_history(best_model, history, experiment=experiment, save_dir=self.save_dir, file_id=self.file_id)
            save_model(best_model, save_dir=self.save_dir, file_id=self.file_id)

        summary = {
            "best_accuracy": best_model.fitness if best_model else None,
            "history": history,
        }
        artifacts = {"best_model": best_model, "history": history}
        return ExperimentResult(summary=summary, artifacts=artifacts)

    def _run_qiskit_qcnn(self, data_mgr: DataManager) -> ExperimentResult:
        # Lazy imports to keep dependencies optional for GA-only runs
        from qiskitQCNN.qiskitQCNN_structure import QCNNStructure
        from qiskitQCNN.trainer import QCNNTrainer

        x_train, y_train, x_test, y_test = data_mgr.get_data(as_numpy=True)

        q_struct = QCNNStructure(num_qubits=self.config.n_qubits)
        feature_map = resolve_feature_map(self.config.encoding)
        circuit, input_params, weight_params = q_struct.build_full_circuit(feature_map)

        trainer = QCNNTrainer(circuit, input_params, weight_params)
        trainer.train(x_train, y_train, max_iter=self.config.max_iter)

        train_score = trainer.evaluate(x_train, y_train)
        test_score = trainer.evaluate(x_test, y_test)

        summary = {"train_score": train_score, "test_score": test_score}
        artifacts = {"trainer": trainer}
        return ExperimentResult(summary=summary, artifacts=artifacts)
