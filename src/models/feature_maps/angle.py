from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from .base import FeatureMapBuilder


class AngleEncodingMap(FeatureMapBuilder):
    """Angle Encoding with RX rotations."""

    def __init__(self, scale: float = np.pi, param_prefix: str = "x"):
        self.scale = scale
        self.param_prefix = param_prefix

    def build(self, n_qubits: int) -> Tuple[QuantumCircuit, ParameterVector]:
        circuit = QuantumCircuit(n_qubits)
        inputs = ParameterVector(self.param_prefix, n_qubits) # สร้างพารามิเตอร์เวกเตอร์ ชื่อ x จำนวน n_qubits
        for i in range(n_qubits):
            circuit.rx(inputs[i] * self.scale, i)
        return circuit, inputs
