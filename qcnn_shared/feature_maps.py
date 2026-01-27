from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap


class FeatureMapBuilder(ABC):
    @abstractmethod
    def build(self, n_qubits: int) -> Tuple[QuantumCircuit, Iterable]:
        """Return a feature map circuit and its ordered input parameters."""


class AngleEncodingMap(FeatureMapBuilder):
    def __init__(self, scale: float = np.pi, param_prefix: str = "x"):
        self.scale = scale
        self.param_prefix = param_prefix

    def build(self, n_qubits: int) -> Tuple[QuantumCircuit, ParameterVector]:
        circuit = QuantumCircuit(n_qubits)
        inputs = ParameterVector(self.param_prefix, n_qubits)
        for i in range(n_qubits):
            circuit.rx(inputs[i] * self.scale, i)
        return circuit, inputs


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


class ZZFeatureMapBuilder(FeatureMapBuilder):
    def __init__(self, reps: int = 1, entanglement: str = "linear"):
        self.reps = reps
        self.entanglement = entanglement

    def build(self, n_qubits: int) -> Tuple[QuantumCircuit, Iterable]:
        fmap = ZZFeatureMap(feature_dimension=n_qubits, reps=self.reps, entanglement=self.entanglement)
        return fmap, tuple(fmap.parameters)


FEATURE_MAP_REGISTRY = {
    "angle": AngleEncodingMap,
    "pge": PGEEncodingMap,
    "zz": ZZFeatureMapBuilder,
}


def resolve_feature_map(feature_map: Union[str, FeatureMapBuilder]) -> FeatureMapBuilder:
    if isinstance(feature_map, FeatureMapBuilder):
        return feature_map
    if isinstance(feature_map, str):
        key = feature_map.lower()
        if key not in FEATURE_MAP_REGISTRY:
            raise ValueError(f"Unknown feature map: {feature_map}")
        return FEATURE_MAP_REGISTRY[key]()
    raise TypeError(f"Unsupported feature map type: {type(feature_map)}")


__all__ = [
    "FeatureMapBuilder",
    "AngleEncodingMap",
    "PGEEncodingMap",
    "ZZFeatureMapBuilder",
    "resolve_feature_map",
]
