from typing import Iterator, List, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from .base import BaseQCNN


class EvolutionaryQCNN(BaseQCNN):
    """
    QCNN implementation driven by a chromosome (list of integers).
    Based on the QEAQCNN implementation.
    """

    def __init__(self, num_qubits: int, chromosome: List[int]):
        super().__init__(num_qubits)
        self.chromosome = chromosome
        self._gene_iter: Optional[Iterator[int]] = None

    def _append_single_gate(self, qc: QuantumCircuit, gene: int, qubit: int, param, inverse=False):
        final_param = -1 * param if inverse else param
        if gene == 0:
            qc.rx(final_param, qubit)
        elif gene == 1:
            qc.ry(final_param, qubit)
        elif gene == 2:
            qc.rz(final_param, qubit)

    def _append_two_gate(self, qc: QuantumCircuit, gene: int, q1: int, q2: int, param):
        if gene == 0:
            qc.rxx(param, q1, q2)
        elif gene == 1:
            qc.ryy(param, q1, q2)
        elif gene == 2:
            qc.rzz(param, q1, q2)

    def apply_convolutional_layer(
        self,
        qc: QuantumCircuit,
        active_qubits: List[int],
        params: ParameterVector,
        param_idx: int,
    ) -> int:
        n = len(active_qubits)
        assert self._gene_iter is not None, "Iterator must be initialized before use"
        for i in range(n):
            q_a = active_qubits[i]
            q_b = active_qubits[(i + 1) % n]

            # Conv block uses 5 genes/params
            genes = [next(self._gene_iter) for _ in range(5)]
            p = params[param_idx : param_idx + 5]

            self._append_single_gate(qc, genes[0], q_a, p[0])
            self._append_single_gate(qc, genes[1], q_b, p[1])
            self._append_two_gate(qc, genes[2], q_a, q_b, p[2])
            self._append_single_gate(qc, genes[3], q_a, p[3])
            self._append_single_gate(qc, genes[4], q_b, p[4])

            param_idx += 5
        qc.barrier()
        return param_idx

    def apply_pooling_layer(
        self,
        qc: QuantumCircuit,
        active_qubits: List[int],
        params: ParameterVector,
        param_idx: int,
    ) -> Tuple[List[int], int]:
        n = len(active_qubits)
        next_active = []
        assert self._gene_iter is not None, "Iterator must be initialized before use"
        for i in range(0, n, 2):
            if i + 1 >= n:
                next_active.append(active_qubits[i])
                break

            q_src, q_sink = active_qubits[i], active_qubits[i + 1]

            # Pool block uses 2 genes/params
            genes = [next(self._gene_iter) for _ in range(2)]
            p = params[param_idx : param_idx + 2]

            self._append_single_gate(qc, genes[0], q_src, p[0])
            self._append_single_gate(qc, genes[1], q_sink, p[1])
            qc.cx(q_src, q_sink)
            self._append_single_gate(qc, genes[1], q_sink, p[1], inverse=True)

            next_active.append(q_sink)
            param_idx += 2

        qc.barrier()
        return next_active, param_idx

    def build(self, *args, **kwargs) -> Tuple[QuantumCircuit, ParameterVector]:
        qc, params, _ = self.build_with_metadata(*args, **kwargs)
        return qc, params

    def build_with_metadata(self, *args, **kwargs) -> Tuple[QuantumCircuit, ParameterVector, int]:
        qc = QuantumCircuit(self.num_qubits)
        params = ParameterVector("θ", len(self.chromosome))
        active_qubits = self._get_initial_active_qubits()
        self._gene_iter = iter(self.chromosome)
        param_idx = 0

        while len(active_qubits) > 1:
            param_idx = self.apply_convolutional_layer(qc, active_qubits, params, param_idx)
            active_qubits, param_idx = self.apply_pooling_layer(
                qc, active_qubits, params, param_idx
            )

        return qc, params, active_qubits[0]
