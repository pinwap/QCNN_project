from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class BaseEngine(ABC):
    """
    Abstract interface for QCNN Optimization Engines.
    An Engine knows how to take a quantum circuit and data, and optimize its parameters.
    """

    @abstractmethod
    def fit(
        self,
        circuit: QuantumCircuit,
        params: ParameterVector,
        last_qubit: int,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
    ) -> Tuple[float, dict, Any]:
        """
        Train the model.

        Returns:
            final_score (float): Final accuracy or fitness.
            history (dict): Dictionary mapping metric names to lists of values.
            trained_obj (Any): The trained weights or model wrapper.
        """
        pass
