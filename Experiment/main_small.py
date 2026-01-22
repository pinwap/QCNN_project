import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from QCNN.DataManager import MNISTDataManager
from QCNN.Evaluation import Experiment, HybridEvaluator
from QCNN.QCNN_structure import QCNNBuilder
from QCNN.Evaluation import HybridEvaluator, Experiment
from QCNN.utils import graph_history, initialize_output_dir, save_model

logger = logging.getLogger(__name__)


def main():
    # 0. Initialize Output Directory & Logging
    save_dir, file_id = initialize_output_dir(script_name="main_small")

    logger.info("Starting QCNN Experiment with QEA...")
    # 1. Create Components
    data_manager = MNISTDataManager(
        data_path="../data", n_train=200, n_test=50
    )  # Small subset for demo
    builder = QCNNBuilder(n_qubits=16)
    evaluator = HybridEvaluator(
        builder, epochs=1, lr=0.01, verbose=True
    )  # 5 Epochs according to paper

    # 2. Inject into Experiment
    experiment = Experiment(
        data_mgr=data_manager,
        evaluator=evaluator,
        n_pop=1,  # ลองรัน population เล็กๆ ก่อน
        n_gen=2,  # ลองรัน 3 รุ่น
        n_gates=180,  # 4 Layers approx
    )

    # 3. Run
    best_model, history = experiment.run()

    logger.info("\n🏁 Experiment Finished!")
    if best_model:
        logger.info(f"Final Best Accuracy: {best_model.fitness:.4f}")
        logger.info(f"History: {history}")

        # Save Results
        graph_history(
            best_model, history, experiment=experiment, save_dir=save_dir, file_id=file_id
        )
        save_model(best_model, save_dir=save_dir, file_id=file_id)

    else:
        logger.info("\nExperiment Failed (Data or Execution Error)")


if __name__ == "__main__":
    main()
