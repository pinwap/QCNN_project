from abc import ABC, abstractmethod
from typing import Iterable, Tuple

from qiskit import QuantumCircuit


class FeatureMapBuilder(ABC):
    """Abstract Base Class for building Quantum Feature Maps."""

    @abstractmethod
    def build(self, n_qubits: int) -> Tuple[QuantumCircuit, Iterable]:
        """Return a feature map circuit and its ordered input parameters."""
        pass
