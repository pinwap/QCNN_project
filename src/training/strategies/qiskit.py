import logging
from typing import Any, List, Optional, Tuple

import torch

from models.feature_maps import FeatureMapBuilder
from models.qcnn import EvolutionaryQCNN
from training.engines.qiskit import QiskitEngine

from .base import EvaluationStrategy

logger = logging.getLogger(__name__)


class QiskitStrategy(EvaluationStrategy):
    """
    Evaluation strategy using the QiskitEngine (Pure Qiskit).
    """

    def __init__(
        self,
        num_qubits: int,
        max_iter: int = 50,
        verbose: bool = True,
        feature_map: str | FeatureMapBuilder = "angle",
    ):
        self.num_qubits = num_qubits
        self.engine = QiskitEngine(feature_map=feature_map, max_iter=max_iter, verbose=verbose)

    def evaluate(
        self,
        structure_code: List[int],
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Any]:
        model_builder = EvolutionaryQCNN(self.num_qubits, structure_code)
        qc, params, last_qubit = model_builder.build_with_metadata()

        score, _, model_state = self.engine.fit(
            circuit=qc,
            params=params,
            last_qubit=last_qubit,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
        )
        return score, model_state
