from typing import Iterable, List, Tuple, Union

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterExpression, ParameterVector


class GateFactory:
    # สร้างเกตต่างๆ ที่ใช้ใน QCNN
    @staticmethod
    def append_single_gate(
        qc: QuantumCircuit,
        gene: int,
        qubit: int,
        param: Union[float, ParameterExpression],
        inverse: bool = False,
    ) -> None:
        """
        เติม Single Gate ลงในวงจร
        Gene: 0->Rx, 1->Ry, 2->Rz, 3->Identity
        """
        # ถ้าเป็น Inverse (สำหรับ Pooling) ให้คูณมุมด้วย -1
        final_param = -1 * param if inverse else param

        if gene == 0:
            qc.rx(final_param, qubit)
        elif gene == 1:
            qc.ry(final_param, qubit)
        elif gene == 2:
            qc.rz(final_param, qubit)
        # Gene 3 = Identity (Do nothing)

    @staticmethod
    def append_two_gate(
        qc: QuantumCircuit,
        gene: int,
        q1: int,
        q2: int,
        param: Union[float, ParameterExpression],
    ) -> None:
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
        # Gene 3 = Identity


class BlockFactory:
    # Conv Block (5 gates) และ Pool Block (2 gates)
    def __init__(self) -> None:
        self.factory = GateFactory()

    def add_conv_block(
        self,
        qc: QuantumCircuit,
        q_pair: Tuple[int, int],
        genes: List[int],
        params: Iterable[ParameterExpression],
    ) -> None:
        """
        สร้าง Convolution Block 1 ชิ้น (ใช้ 5 Genes) ลงในวงจร
        qc: QuantumCircuit หลัก
        q_pair: (index_q_a, index_q_b)
        genes: list ของรหัส gene 5 ตัว
        params: list ของ Parameter 5 ตัว
        """
        q_a, q_b = q_pair
        params_list = list(params)
        # Structure: G1(a) -> G2(b) -> GG(a,b) -> G3(a) -> G4(b)
        self.factory.append_single_gate(qc, genes[0], q_a, params_list[0])
        self.factory.append_single_gate(qc, genes[1], q_b, params_list[1])
        self.factory.append_two_gate(qc, genes[2], q_a, q_b, params_list[2])
        self.factory.append_single_gate(qc, genes[3], q_a, params_list[3])
        self.factory.append_single_gate(qc, genes[4], q_b, params_list[4])

    def add_pool_block(
        self,
        qc: QuantumCircuit,
        q_pair: Tuple[int, int],
        genes: List[int],
        params: Iterable[ParameterExpression],
    ) -> None:
        """
        สร้าง Pooling Block 1 ชิ้น (ใช้ 2 Genes) ลงในวงจร
        q_pair: (source, sink)
        """
        q_src, q_sink = q_pair
        params_list = list(params)
        # Structure: G1(src) -> G2(sink) -> CNOT -> G2_inverse(sink)
        self.factory.append_single_gate(qc, genes[0], q_src, params_list[0])
        self.factory.append_single_gate(qc, genes[1], q_sink, params_list[1])
        qc.cx(q_src, q_sink)
        self.factory.append_single_gate(qc, genes[1], q_sink, params_list[1], inverse=True)


class QCNNBuilder:
    # รับรหัสโครโมโซมมาสร้างเป็น QCNN < block
    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.blocks = BlockFactory()

        self.total_genes: int = 0  # เพื่อเรียกดูจำนวน gene ที่ใช้ไป

    def assemble(self, chromosome: List[int]) -> Tuple[QuantumCircuit, int]:
        """
        สร้างวงจร Qiskit จากรหัสโครโมโซม
        """
        # 1. เตรียมวงจรและ ParameterVector
        qr = QuantumRegister(self.n_qubits, "q")  # กล่อง array เก็บ qubit ขนาด self.n_qubits ชื่อ 'q'
        qc = QuantumCircuit(qr)

        # 2. สร้าง ParameterVector สำหรับเก็้บมุมพารามิเตอร์ ขนาดเท่าจำนวน gene
        theta = ParameterVector(
            "theta", len(chromosome)
        )  # จะสร้างตัวแปรชื่อ theta[0], theta[1], ... theta[len(chromosome)-1] ให้เราอัตโนมัติ

        gene_iter = iter(chromosome)
        param_idx = 0  # ตัวนับตำแหน่งพารามิเตอร์

        active_qubits = list(range(self.n_qubits))  # เก็บ Index ของ Qubit ที่ยังรอดอยู่

        # วนลูปสร้าง Layer จนกว่าจะเหลือ Qubit น้อยกว่า 2
        layer_count = 1

        while len(active_qubits) > 1:
            n = len(active_qubits)

            # --- สร้าง Barrier เพื่อความสวยงามในภาพวาด ---
            qc.barrier()

            # ==========================
            # 1. Convolution Layer
            # ==========================
            for i in range(n):
                q_a = active_qubits[i]
                q_b = active_qubits[(i + 1) % n]  # Circular

                # เบิก 5 Genes และ 5 Parameters
                block_genes = [next(gene_iter) for _ in range(5)]
                block_params = theta[param_idx : param_idx + 5]
                param_idx += 5

                # ต่อ Block
                self.blocks.add_conv_block(qc, (q_a, q_b), block_genes, block_params)
            # จบ ได้convolution 1 layer

            # ==========================
            # 2. Pooling Layer
            # ==========================
            next_active_qubits = []  # เก็บ Qubit ที่รอดหลัง Pooling
            qc.barrier()

            for i in range(0, n, 2):
                # กรณีเหลือเศษ (Odd Logic)
                if i + 1 >= n:
                    next_active_qubits.append(active_qubits[i])  # ตัวสุดท้ายได้ผ่านไปเฉยๆ
                    break

                q_src = active_qubits[i]
                q_sink = active_qubits[i + 1]

                # เบิก 2 Genes และ 2 Parameters
                block_genes = [next(gene_iter) for _ in range(2)]
                block_params = theta[param_idx : param_idx + 2]
                param_idx += 2

                # ต่อ Block
                self.blocks.add_pool_block(qc, (q_src, q_sink), block_genes, block_params)
                # จบ ได้convolution 1 layer

                # เก็บตัวรอด
                next_active_qubits.append(q_sink)

            active_qubits = next_active_qubits
            layer_count += 1

        self.total_genes = param_idx
        return qc, active_qubits[0]  # คืน Qubit สุดท้ายที่รอด
