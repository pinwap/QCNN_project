from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap

from .base import FeatureMapBuilder


class ZZFeatureMapBuilder(FeatureMapBuilder):
    """Standard ZZFeatureMap from Qiskit."""

    def __init__(self, reps: int = 1, entanglement: str = "linear"):
        self.reps = reps
        self.entanglement = entanglement

    def build(self, n_qubits: int) -> Tuple[QuantumCircuit, Iterable]:
        fmap = ZZFeatureMap(
            feature_dimension=n_qubits, reps=self.reps, entanglement=self.entanglement
        )
        # Decompose ensures high-level instructions are converted to primitive gates for Aer
        return fmap.decompose(), tuple(fmap.parameters)
