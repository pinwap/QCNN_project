import copy
from typing import List

from .gene import QuantumGene


class QuantumChromosome:
    """
    A collection of QuantumGenes that represents a solution candidate.
    """

    def __init__(self, num_gates: int) -> None:
        self.num_gates = num_gates
        # Each gate is represented by 2 bits/genes
        self.genes: List[QuantumGene] = [QuantumGene() for _ in range(num_gates * 2)]

        self.binary_code: List[int] = []
        self.structure_code: List[int] = []
        self.fitness: float = 0.0

    def collapse(self) -> List[int]:
        """Measure all genes to get a classical binary string and convert to gate codes."""
        # 1. Observe each gene
        self.binary_code = [gene.observe() for gene in self.genes]

        # 2. Convert binary pairs to integer gate codes (0-3)
        self.structure_code = [
            (self.binary_code[i] << 1) | self.binary_code[i + 1]
            for i in range(0, len(self.binary_code), 2)
        ]
        return self.structure_code

    def update_genes(self, global_best_chromosome: "QuantumChromosome") -> None:
        """Update each gene's rotation towards the global best chromosome."""
        best_binary = global_best_chromosome.binary_code
        best_fitness = global_best_chromosome.fitness

        for i, gene in enumerate(self.genes):
            gene.rotate(
                current_bit=self.binary_code[i],
                best_bit=best_binary[i],
                fitness_current=self.fitness,
                fitness_best=best_fitness,
            )

    def copy(self) -> "QuantumChromosome":
        """Create a deep copy of the chromosome."""
        new_instance = QuantumChromosome(self.num_gates)
        new_instance.genes = copy.deepcopy(self.genes)
        new_instance.binary_code = list(self.binary_code)
        new_instance.structure_code = list(self.structure_code)
        new_instance.fitness = self.fitness
        return new_instance
