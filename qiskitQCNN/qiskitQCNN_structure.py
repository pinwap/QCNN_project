import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qcnn_shared.feature_maps import FeatureMapBuilder, resolve_feature_map


class QCNNStructure:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    # สร้าง two qubit unitary เกตสำหรับเอามาต่อเป็น convolutional layer
    def _conv_circuit(self, params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    def conv_layer(self, active_qubits, param_prefix):
        """Build a convolution layer over the current active qubits.

        Handles any number of qubits (even/odd) by pairing adjacent qubits in two passes:
        - pass 1: (0,1), (2,3), ...
        - pass 2: (1,2), (3,4), ...
        Unpaired last qubit (odd case) is left untouched.
        """
        n = len(active_qubits)
        qc = QuantumCircuit(n, name=f"{param_prefix}Convolutional Layer")

        pairs = []
        pairs.extend([(i, i + 1) for i in range(0, n - 1, 2)])
        pairs.extend([(i, i + 1) for i in range(1, n - 1, 2)])

        params = ParameterVector(param_prefix, length=len(pairs) * 3)
        p_idx = 0
        for q1, q2 in pairs:
            qc.compose(self._conv_circuit(params[p_idx : p_idx + 3]), [q1, q2], inplace=True)
            qc.barrier()
            p_idx += 3

        return qc

    def _pool_circuit(self, params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def _pool_layer(self, active_qubits, param_prefix):
        """Pooling that halves (ceil) the active set; leftover qubit (odd) passes through."""
        n = len(active_qubits)
        qc = QuantumCircuit(n, name=f"{param_prefix}Pooling Layer")

        pairs = []
        survivors = []
        for i in range(0, n, 2):
            if i + 1 < n:
                pairs.append((i, i + 1))
                survivors.append(active_qubits[i])  # keep sink survivor position
            else:
                survivors.append(active_qubits[i])  # odd leftover carries forward

        params = ParameterVector(param_prefix, length=len(pairs) * 3)
        p_idx = 0
        for src, sink in pairs:
            qc.compose(self._pool_circuit(params[p_idx : p_idx + 3]), [src, sink], inplace=True)
            qc.barrier()
            p_idx += 3

        return qc, survivors

    def create_ansatz(self):
        current_qubits = list(range(self.num_qubits))
        ansatz = QuantumCircuit(self.num_qubits, name="QCNN_Ansatz")

        layer_count = 1
        while len(current_qubits) > 1:
            layer_name = f"L{layer_count}"

            conv = self.conv_layer(current_qubits, f"{layer_name}_c")
            ansatz.compose(conv, current_qubits, inplace=True)

            pool, survivors = self._pool_layer(current_qubits, f"{layer_name}_p")
            ansatz.compose(pool, current_qubits, inplace=True)

            current_qubits = survivors
            layer_count += 1

        return ansatz

    def build_full_circuit(self, feature_map: str | FeatureMapBuilder | None = None):
        fmap_builder = resolve_feature_map(feature_map)
        feature_map_circuit, feature_params = fmap_builder.build(self.num_qubits)
        ansatz = self.create_ansatz()

        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(feature_map_circuit, range(self.num_qubits), inplace=True)
        circuit.compose(ansatz, range(self.num_qubits), inplace=True)

        return circuit, feature_params, ansatz.parameters
