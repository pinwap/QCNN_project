import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from QCNN.DataManager import MNISTDataManager
from QCNN.Evaluation import Experiment, HybridEvaluator
from QCNN.QCNN_structure import QCNNBuilder
from QCNN.Evaluation import HybridEvaluator, Experiment
from QCNN.utils import graph_history, setup_logging

logger = logging.getLogger(__name__)

# Configure logging
setup_logging()


def main():
    logger.info("Starting QCNN Experiment with QEA...")
    # 1. Create Components
    data_manager = MNISTDataManager(
        data_path="../data", n_train=200, n_test=50
    )  # Small subset for demo
    builder = QCNNBuilder(n_qubits=16)
    evaluator = HybridEvaluator(
        builder, epochs=5, lr=0.01, verbose=True
    )  # 5 Epochs according to paper

    # 2. Inject into Experiment
    experiment = Experiment(
        data_mgr=data_manager,
        evaluator=evaluator,
        n_pop=5,  # ลองรัน population เล็กๆ ก่อน
        n_gen=3,  # ลองรัน 3 รุ่น
        n_gates=180,  # 4 Layers approx
    )

    # 3. Run
    best_model, history = experiment.run()

    print("\n🏁 Experiment Finished!")
    if best_model:
        print(f"Final Best Accuracy: {best_model.fitness:.4f}")
        print(f"History: {history}")
        graph_history(best_model, history)

    else:
        print("\nExperiment Failed (Data or Execution Error)")


if __name__ == "__main__":
    main()
