import copy
from typing import Any, List, Optional

from .gene import QuantumGene


class QuantumChromosome:
    """
    2gene แทน 1 gate ，num_gates = จำนวน gate
    genes = หลาย gene = คือโครโมโซมที่แทน QCNN 1 วงจร = 1 คน
    """

    def __init__(self, num_gates: int) -> None:
        self.num_gates = num_gates
        
        # สร้าง gene จำนวน num_gates*2 เพื่อ 2 gene = 1 gate
        self.genes: List[QuantumGene] = [QuantumGene() for _ in range(num_gates * 2)]

        self.binary_code: List[int] = []
        self.structure_code: List[int] = [] # โค้ดโครงสร้าง QCNN (0-3) ไว้ส่งให้ build circuit
        self.fitness: float = 0.0 # ความแม่นของวงจร
        self.best_model_state: Optional[Any] = None # เก็บ weights ของโมเดลที่ดีที่สุด

    def collapse(self) -> List[int]:
        """
        Measure all genes to get a classical binary string and convert to gate codes.
        วัดโครโมโซมทั้งหมดในประชากร แล้วแปลงเป็นเกต
        """
        # 1. วัด gene แต่ละตัว binary
        self.binary_code = [gene.observe() for gene in self.genes]

        # 2. แปลง Binary คู่ เป็น รหัสเกต (0-3)
        self.structure_code = [
            (self.binary_code[i] << 1) | self.binary_code[i + 1]
            for i in range(0, len(self.binary_code), 2)
        ]
        return self.structure_code

    def update_genes(self, global_best_chromosome: "QuantumChromosome") -> None:
        """
        Update each gene's rotation towards the global best chromosome.
        หมุนมุมโดยเทียบ คน(วงจร)นี้ กับ คน(วงจร)ที่ Best
        """
        best_binary = global_best_chromosome.binary_code
        best_fitness = global_best_chromosome.fitness

        # เทียบทีละบิต
        for i, gene in enumerate(self.genes):
            gene.rotate(
                current_bit=self.binary_code[i],
                best_bit=best_binary[i],
                fitness_current=self.fitness,
                fitness_best=best_fitness,
            )

    def copy(self) -> "QuantumChromosome":
        """
        Create a deep copy of the chromosome.
        ฟังก์ชันสำหรับ copy ตัวเอง (ใช้ตอนเก็บ Global Best)
        """
        new_instance = QuantumChromosome(self.num_gates)
        new_instance.genes = copy.deepcopy(self.genes)
        new_instance.binary_code = list(self.binary_code)
        new_instance.structure_code = list(self.structure_code)
        try:
             # Try deepcopy for safe standalone copy (especially for dicts/lists)
            new_instance.best_model_state = copy.deepcopy(self.best_model_state)
        except Exception:
             # Fallback if the model state is not pickle-able or deepcopy-able (e.g. some Qiskit objects)
            new_instance.best_model_state = self.best_model_state
        new_instance.fitness = self.fitness
        return new_instance
