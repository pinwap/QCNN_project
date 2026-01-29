from abc import ABC, abstractmethod
from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class BaseQCNN(ABC):
    """
    Abstract Base Class for Quantum Convolutional Neural Networks.
    Provides the standard structure for alternating convolution and pooling layers.
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits

    @abstractmethod
    def build(self, *args, **kwargs) -> Tuple[QuantumCircuit, ParameterVector]:
        """
        Build and return the full quantum circuit and its parameters.
        """
        pass

    @abstractmethod
    def build_with_metadata(self, *args, **kwargs) -> Tuple[QuantumCircuit, ParameterVector, int]:
        """
        Build and return (circuit, parameters, last_surviving_qubit).
        """
        pass

    def _get_initial_active_qubits(self) -> List[int]:
        return list(range(self.num_qubits))

    @abstractmethod
    def apply_convolutional_layer(
        self, qc: QuantumCircuit, active_qubits: List[int], params: ParameterVector, param_idx: int
    ) -> int:
        """
        Define how to apply a convolutional layer.
        Returns the updated parameter index.
        """
        pass

    @abstractmethod
    def apply_pooling_layer(
        self, qc: QuantumCircuit, active_qubits: List[int], params: ParameterVector, param_idx: int
    ) -> Tuple[List[int], int]:
        """
        Define how to apply a pooling layer.
        Returns a tuple of (remaining_qubits, updated_parameter_index).
        """
        pass
