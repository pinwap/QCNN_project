import copy
import logging
from typing import List, Optional, Tuple

from optimization.qea.chromosome import QuantumChromosome

from .strategies.base import EvaluationStrategy

logger = logging.getLogger(__name__)


class EvolutionarySearch:
    """
    ทำ QEA เพื่อหาโครงสร้าง QCNN ที่ดีที่สุด
    Orchestrates the Evolutionary Algorithm for Quantum Architecture Search.
    Manages the population, evolution steps, and evaluation strategies.
    """

    def __init__(
        self,
        data_manager,
        strategy: EvaluationStrategy,
        n_pop: int = 10,
        n_gen: int = 5,
        n_gates: int = 180,
    ):
        self.data_manager = data_manager
        self.strategy = strategy
        self.n_gen = n_gen

        # Initial population
        self.population = [QuantumChromosome(n_gates) for _ in range(n_pop)]
        self.global_best: Optional[QuantumChromosome] = None
        self.history: List[float] = []

    def _apply_crossover(self) -> None:
        """
        Apply the crossover operator based on the research paper table.
        """
        logger.info("Triggering Population Crossover...")
        n_pop = len(self.population)
        n_genes = len(self.population[0].genes)

        new_population = [p.copy() for p in self.population]

        for i in range(n_pop):
            for j in range(n_genes):
                source_idx = (i + j) % n_pop
                new_population[i].genes[j] = copy.deepcopy(self.population[source_idx].genes[j])

        self.population = new_population

    def run(self) -> Tuple[Optional[QuantumChromosome], List[float]]:
        """
        Execute the evolutionary optimization loop.
        """
        # 1. Prepare Data
        x_train, y_train, x_test, y_test = self.data_manager.get_data()

        if x_train is None:
            logger.error("Data loading failed. Experiment cannot proceed.")
            return None, []

        logger.info(
            f"Starting {self.n_gen} generations of evolution (Pop size: {len(self.population)})"
        )

        stagnation_counter = 0

        for gen in range(self.n_gen):
            logger.info(f"--- Generation {gen + 1}/{self.n_gen} ---")

            # 2. Evaluate Population
            for i, chromo in enumerate(self.population):
                logger.info(f"  Evaluating candidate {i + 1}/{len(self.population)}...")

                # A. Measure quantum chromosome to get classical structure code
                struct_code = chromo.collapse()

                # B. Call the Evaluation Strategy
                acc, model_state = self.strategy.evaluate(struct_code, x_train, y_train, x_test, y_test)
                chromo.fitness = acc
                chromo.best_model_state = model_state

                logger.info(f"  Candidate {i + 1} Result: Accuracy = {acc:.4f}")

            # 3. Update Global Best
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.global_best is None or current_best.fitness > self.global_best.fitness:
                self.global_best = current_best.copy()
                stagnation_counter = 0
                logger.info(f"  >>> New Global Best achieved: {self.global_best.fitness:.4f}")
            else:
                stagnation_counter += 1
                logger.info(f"  Stagnation level: {stagnation_counter}/10")

            # 4. Evolution Step
            if stagnation_counter >= 10:
                self._apply_crossover()
                stagnation_counter = 0
            else:
                for chromo in self.population:
                    chromo.update_genes(self.global_best)

            self.history.append(self.global_best.fitness)

        return self.global_best, self.history
