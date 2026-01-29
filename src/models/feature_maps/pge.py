from typing import Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from .base import FeatureMapBuilder


class PGEEncodingMap(FeatureMapBuilder):
    """Simple parameterized gate encoding with local rotations and optional entanglement."""

    def __init__(self, entangle: bool = True, param_prefix: str = "x"):
        self.entangle = entangle
        self.param_prefix = param_prefix

    def build(self, n_qubits: int) -> Tuple[QuantumCircuit, ParameterVector]:
        circuit = QuantumCircuit(n_qubits)
        inputs = ParameterVector(self.param_prefix, n_qubits)
        for i in range(n_qubits):
            circuit.ry(inputs[i], i)
            circuit.rz(inputs[i], i)
        if self.entangle and n_qubits > 1:
            for i in range(n_qubits - 1):
                circuit.cx(i, i + 1)
        return circuit, inputs
