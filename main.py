from QCNN.DataManager import MNISTDataManager
from QCNN.Evaluation import Experiment, HybridEvaluator
from QCNN.QCNN_structure import QCNNBuilder
from QCNN.Evaluation import HybridEvaluator, Experiment
import matplotlib.pyplot as plt

def main():
    # 1. Create Components
    data_manager = MNISTDataManager(
        data_path="../data", n_train=200, n_test=50
    )  # Small subset for demo
    builder = QCNNBuilder(n_qubits=16)
    evaluator = HybridEvaluator(builder, epochs=5, lr=0.01)  # 5 Epochs according to paper

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
    
        # plot history
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(history) + 1), history, marker='o', linestyle='-', color='b')
        plt.title('QEA-QCNN Optimization History')
        plt.xlabel('Generation')
        plt.ylabel('Best Accuracy')
        plt.grid(True)
        plt.show()
    else:
        print("\nExperiment Failed (Data or Execution Error)")
    
if __name__ == "__main__":
    main()
