from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from .base import BaseQCNN


class StandardQCNN(BaseQCNN):
    """
    Standard QCNN implementation with fixed unitary structures.
    Based on the qiskitQCNN implementation.
    โครงสร้าง QCNN ของ qiskit - Cong Iris et al. 
    """

    def _conv_block(self) -> QuantumCircuit:
        qc = QuantumCircuit(2, name="ConvBlock")
        params = ParameterVector("p", 3)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _pool_block(self) -> QuantumCircuit:
        qc = QuantumCircuit(2, name="PoolBlock")
        params = ParameterVector("p", 3)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def apply_convolutional_layer(
        self, qc: QuantumCircuit, active_qubits: List[int], params: ParameterVector, param_idx: int
    ) -> int:
        n = len(active_qubits)
        # Pairing adjacent qubits in two passes
        pairs = []
        pairs.extend([(i, i + 1) for i in range(0, n - 1, 2)])
        pairs.extend([(i, i + 1) for i in range(1, n - 1, 2)])

        block = self._conv_block()
        for i, j in pairs:
            # Map indices in active_qubits to original qubit indices
            q1, q2 = active_qubits[i], active_qubits[j]
            qc.compose(
                block.assign_parameters(params[param_idx : param_idx + 3]),
                [q1, q2],
                inplace=True,
            )
            param_idx += 3
        qc.barrier()
        return param_idx

    def apply_pooling_layer(
        self, qc: QuantumCircuit, active_qubits: List[int], params: ParameterVector, param_idx: int
    ) -> Tuple[List[int], int]:
        n = len(active_qubits)
        survivors = []
        pairs = []
        for i in range(0, n, 2):
            if i + 1 < n:
                pairs.append((active_qubits[i], active_qubits[i + 1]))
                survivors.append(active_qubits[i + 1])  # Keep Q1 as per paper
            else:
                survivors.append(active_qubits[i])  # Odd leftover passes through

        block = self._pool_block()
        for src, sink in pairs:
            qc.compose(
                block.assign_parameters(params[param_idx : param_idx + 3]),
                [src, sink],
                inplace=True,
            )
            param_idx += 3
        qc.barrier()
        return survivors, param_idx

    def build(self, *args, **kwargs) -> Tuple[QuantumCircuit, ParameterVector]:
        qc, params, _ = self.build_with_metadata(*args, **kwargs)
        return qc, params

    def build_with_metadata(self, *args, **kwargs) -> Tuple[QuantumCircuit, ParameterVector, int]:
        qc = QuantumCircuit(self.num_qubits)
        active_qubits = self._get_initial_active_qubits()

        # We need to create a ParameterVector that spans the total required parameters
        # Placeholder size, will be sliced later
        all_params = ParameterVector("θ", 1000)
        param_idx = 0

        while len(active_qubits) > 1:
            param_idx = self.apply_convolutional_layer(qc, active_qubits, all_params, param_idx)
            active_qubits, param_idx = self.apply_pooling_layer(
                qc, active_qubits, all_params, param_idx
            )

        return qc, all_params[:param_idx], active_qubits[0]
