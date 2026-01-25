import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import z_feature_map
from qiskit.quantum_info import SparsePauliOp

class QCNNStructure:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        
    # สร้าง two qubit unitary เกตสำหรับเอามาต่อเป็น convolutional layer
    def _conv_circuit(self, params):
        target = QuantumCircuit(2)
        target.rz(-np.pi/2, 1)
        target.cx(1,0)
        target.rz(params[0],0)
        target.ry(params[1],1)
        target.cx(0,1)
        target.ry(params[2],1)
        target.cx(1,0)
        target.rz(np.pi/2,0)
        return target

    # เอามาต่อเป็น convolutional layer
    def conv_layer(self, num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name=f"{param_prefix}Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        
        # คู่่ (0,1), (2,3),...
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(self._conv_circuit(params[param_index : (param_index + 3)]), [q1, q2], inplace=True)
            qc.barrier()
            param_index += 3
        # คู่ (1,2), (3,4),...
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(self._conv_circuit(params[param_index : (param_index + 3)]), [q1, q2], inplace=True)
            qc.barrier()
            param_index += 3

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

    def _pool_layer(self, sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name=f"{param_prefix}Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(self._pool_circuit(params[param_index : (param_index + 3)]), [source, sink], inplace = True)
            qc.barrier()
            param_index += 3
        return qc
    
    def create_ansatz(self):
        current_qubits = list(range(self.num_qubits))
        ansatz = QuantumCircuit(self.num_qubits, name="QCNN_Ansatz")
        
        layer_count = 1
        while len(current_qubits) > 1:
            # 1. Convolutional Layer
            num_active = len(current_qubits)
            layer_name = f"L{layer_count}"
            conv = self._conv_layer(num_active, f"{layer_name}_c")
            ansatz.compose(conv, current_qubits, inplace=True)
        
            # 2. Pooling Layer (ลดจำนวนลงครึ่งหนึ่ง)
            # แบ่งเป็นครึ่งซ้าย (Sources) และครึ่งขวา (Sinks) หรือคู่เว้นคู่
            sources = current_qubits[0::2] # เก็บไว้
            sinks = current_qubits[1::2]   # ทิ้ง (Trace out / Measure)
            
            mid = len(current_qubits) // 2
            sources = current_qubits[:mid]
            sinks = current_qubits[mid:]
            
            pool = self._pool_layer(sources, sinks, f"{layer_name}_p")
            ansatz.compose(pool, current_qubits, inplace=True)
            
            # อัปเดต Qubit ที่เหลือรอด (เฉพาะ Sources)
            current_qubits = sources
            layer_count += 1
            
        return ansatz

    def build_full_circuit(self):
        feature_map = z_feature_map(self.num_qubits)
        ansatz = self.create_ansatz()
        
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(feature_map, range(self.num_qubits), inplace=True)
        circuit.compose(ansatz, range(self.num_qubits), inplace=True)
        
        return circuit, feature_map.parameters, ansatz.parameters