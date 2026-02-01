from typing import Iterable, Iterator, List, Optional, Tuple, Any

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression

from .base import BaseQCNN


class EvolutionaryQCNN(BaseQCNN):
    """
    QCNN implementation driven by a chromosome (list of integers).
    Based on the QEAQCNN implementation.
    
    สร้างวงจร QCNN จากรหัสโครโมโซม Quantum Evolutionary Algorithm (QEA)
    """

    def __init__(self, num_qubits: int, chromosome: List[int]):
        super().__init__(num_qubits)
        self.chromosome = chromosome
        self._gene_iter: Optional[Iterator[int]] = None
        self.layer_count = 0
        
#-------------------------------
# สร้างเกตต่างๆ ตาม gene ในโครโมโซม
#-------------------------------

    def _append_single_gate(self, 
                            qc: QuantumCircuit, 
                            gene: int, 
                            qubit: int, 
                            param: Any, # มุมหมุน
                            inverse=False):
        """
        เติม Single Gate ลงในวงจร
        Gene: 0->Rx, 1->Ry, 2->Rz, 3->Identity
        บอกมาว่าจะสร้างเกตอะไร มุมเท่าไหร่ บน qubit ไหน
        """
        # ถ้าเป็น Inverse ให้คูณมุมด้วย -1 (สำหรับ เกต G2^t (Inverse of G2) ใน Pooling)
        final_param = -1 * param if inverse else param
        
        if gene == 0:
            qc.rx(final_param, qubit)
        elif gene == 1:
            qc.ry(final_param, qubit)
        elif gene == 2:
            qc.rz(final_param, qubit)
        # Gene 3 = Identity (Do nothing)

    def _append_two_gate(self,
                         qc: QuantumCircuit,
                         gene: int,
                         q1: int,
                         q2: int,
                         param):
        """
        เติม Two Qubit Gate ลงในวงจร
        Gene: 0->RXX, 1->RYY, 2->RZZ, 3->Identity
        """
        if gene == 0:
            qc.rxx(param, q1, q2)
        elif gene == 1:
            qc.ryy(param, q1, q2)
        elif gene == 2:
            qc.rzz(param, q1, q2)
        # Gene 3 = Identity (Do nothing)

# -------------------------------
# สร้างบล็อค Conv, Pool
# Conv Block (5 gates) และ Pool Block (2 gates)
# -------------------------------        
    def add_convolutional_block(self,
                                qc: QuantumCircuit,
                                q_pair: Tuple[int, int],
                                genes: List[int],
                                params: List[Any],
                                ) -> None:
        q_a, q_b = q_pair
        p = list(params)
        # Structure: G1(a) -> G2(b) -> GG(a,b) -> G3(a) -> G4(b)
        genes = [next(self._gene_iter) for _ in range(5)]
        self._append_single_gate(qc, genes[0], q_a, p[0]) 
        self._append_single_gate(qc, genes[1], q_b, p[1])
        self._append_two_gate(qc, genes[2], q_a, q_b, p[2])
        self._append_single_gate(qc, genes[3], q_a, p[3])
        self._append_single_gate(qc, genes[4], q_b, p[4])
    
    def apply_convolutional_layer( #ใส่บล็อคลงทุกคิวบิตเป็นเลเยอร์
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
            q_b = active_qubits[(i + 1) % n] # Circular

            # เบิก 5 Genes->gate และ 5 Parameters->มุมหมุนเกต
            genes = [next(self._gene_iter) for _ in range(5)]
            p = params[param_idx : param_idx + 5]
            param_idx += 5
            self.add_convolutional_block(qc, (q_a, q_b), genes, p,)
        qc.barrier()
        return param_idx # ส่งคืน index parameter ปัจจุบัน หลังใช้ไปแล้ว

    def add_pooling_block(self,
        qc: QuantumCircuit,
        q_pair: Tuple[int, int],
        genes: List[int],
        params: Iterable[Any],
    ) -> None:
        """
        สร้าง Pooling Block 1 ชิ้น (ใช้ 2 Genes) ลงในวงจร
        q_pair: (source, sink)
        """
        q_src, q_sink = q_pair
        p = list(params)
        # Structure: G1(src) -> G2(sink) -> CNOT -> G2_inverse(sink)
        self._append_single_gate(qc, genes[0], q_src, p[0])
        self._append_single_gate(qc, genes[1], q_sink, p[1])
        qc.cx(q_src, q_sink)
        self._append_single_gate(qc, genes[1], q_sink, p[1], inverse=True)

    def apply_pooling_layer(
        self,
        qc: QuantumCircuit,
        active_qubits: List[int],
        params: ParameterVector,
        param_idx: int,
    ) -> Tuple[List[int], int]: # ส่งคืน active qubits ตัวที่เหลือ กับ param index ปัจจุบัน
        n = len(active_qubits)
        next_active = []
        assert self._gene_iter is not None, "Iterator must be initialized before use"
        for i in range(0, n, 2):
            # กรณีเหลือเศษ (Odd Logic)
            if i + 1 >= n:
                next_active.append(active_qubits[i]) # ตัวสุดท้ายได้ผ่านไปเฉยๆ
                break
            
            q_src, q_sink = active_qubits[i], active_qubits[i + 1]
            genes = [next(self._gene_iter) for _ in range(2)]
            p = params[param_idx : param_idx + 2]

            self.add_pooling_block(qc, (q_src, q_sink), genes, p)
            # เก็บตัวรอด
            next_active.append(q_sink)
            param_idx += 2

        qc.barrier()
        return next_active, param_idx

    def build(self, *args, **kwargs) -> Tuple[QuantumCircuit, ParameterVector]:
        qc, params, _ = self.build_with_metadata(*args, **kwargs)
        return qc, params # คืนแค่ วงจร กับ มุมหมุน

    def build_with_metadata(self, *args, **kwargs) -> Tuple[QuantumCircuit, ParameterVector, int]:
        qc = QuantumCircuit(self.num_qubits)
        params = ParameterVector("θ", len(self.chromosome))
        active_qubits = self._get_initial_active_qubits() # [0,1,2,...,num_qubits-1]
        self._gene_iter = iter(self.chromosome) # เตรียม iterator คือการเบิก gene ทีละตัวจากโครโมโซม 
                                                # พอเป็น iterator poiter จะเดินหน้าไปเรื่อยๆ ชี้ที่ gene ปัจจุบัน
        param_idx = 0
        
        while len(active_qubits) > 1:
            param_idx = self.apply_convolutional_layer(qc, active_qubits, params, param_idx)
            active_qubits, param_idx = self.apply_pooling_layer(
                qc, active_qubits, params, param_idx
            )
            self.layer_count += 1

        return qc, params, active_qubits[0] # คืนวงจร, มุมหมุน, และ qubit สุดท้ายที่เหลือ