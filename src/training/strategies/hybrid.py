import logging
from typing import List, Optional

import torch

from models.feature_maps import FeatureMapBuilder
from models.qcnn import EvolutionaryQCNN
from training.engines.hybrid import HybridEngine

from .base import EvaluationStrategy

logger = logging.getLogger(__name__)


class HybridStrategy(EvaluationStrategy):
    """
    Evaluation strategy using the HybridEngine (PyTorch + Qiskit).
    """

    def __init__(
        self,
        num_qubits: int,
        epochs: int = 5,
        lr: float = 0.01,
        verbose: bool = True,
        device: Optional[str] = None,
        gradient_method: str = "param_shift",
        feature_map: str | FeatureMapBuilder = "angle",
    ):
        self.num_qubits = num_qubits
        self.engine = HybridEngine(
            feature_map=feature_map,
            epochs=epochs,
            lr=lr,
            device=device,
            gradient_method=gradient_method,
            verbose=verbose,
        )

    def evaluate(
        self,
        structure_code: List[int],
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> float:
        model_builder = EvolutionaryQCNN(self.num_qubits, structure_code)
        qc, params, last_qubit = model_builder.build_with_metadata()

        score, _, _ = self.engine.fit(
            circuit=qc,
            params=params,
            last_qubit=last_qubit,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        return score
