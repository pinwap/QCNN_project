import logging
from typing import Any, List

from models.qcnn.evolutionary_qcnn import EvolutionaryQCNN
from training.evolution import EvolutionarySearch
from training.pipeline import ProductionPipeline

logger = logging.getLogger(__name__)


class AutoEvolutionPipeline:
    """
    เรียกใช้ Evolutionary Search -> โครงสร้าง QCNN ที่ดีที่สุด แล้วเอาไป retrain ต่อเลย
    Automated workflow that runs Evolutionary Search to find the best QCNN architecture,
    and then immediately launches a Production Pipeline to retrain that best architecture
    for a longer period (higher number of epochs).
    """

    def __init__(
        self,
        evolution_search: EvolutionarySearch,
        production_engine: Any,
        num_qubits: int,
    ):
        self.evolution_search = evolution_search
        self.production_engine = production_engine
        self.num_qubits = num_qubits

    def run(self, x_train, y_train, x_test=None, y_test=None):
        """
        Executes Phase 1 (Evolution) and Phase 2 (Retraining).
        """
        logger.info("Starting Auto-Evolution Workflow...")

        # 1. Phase 1: Evolutionary Search
        logger.info("PHASE 1: Starting Evolutionary Architecture Search...")
        best_chromo, evolution_history = self.evolution_search.run()

        if best_chromo is None:
            logger.error("Evolutionary search failed to find a valid chromosome.")
            return 0.0, [], None, None, []

        # Get the final verified structure code
        if hasattr(best_chromo, "collapse"):
            struct_code = best_chromo.collapse()
        elif hasattr(best_chromo, "genes"):
            struct_code = best_chromo.genes
        else:
            struct_code = best_chromo

        logger.info(f"PHASE 1 COMPLETE. Best Accuracy: {best_chromo.fitness:.4f}")
        logger.info(f"Best Structure Code: {struct_code}")

        # 2. Phase 2: Retraining
        logger.info("PHASE 2: Initializing Production Model with discovered architecture...")
        # Ensure it's a list for the constructor
        final_struct: List[int] = list(struct_code)  # type: ignore
        production_model = EvolutionaryQCNN(num_qubits=self.num_qubits, chromosome=final_struct)

        logger.info(
            f"PHASE 2: Retraining discovered model for {self.production_engine.epochs} epochs..."
        )
        production_pipeline = ProductionPipeline(
            model=production_model, engine=self.production_engine
        )

        final_score, training_history, trained_obj = production_pipeline.run(
            x_train, y_train, x_test, y_test
        )

        logger.info(f"Auto-Evolution Workflow Complete. Final Retrained Score: {final_score:.4f}")

        return final_score, training_history, trained_obj, best_chromo, evolution_history
