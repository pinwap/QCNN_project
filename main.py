from QCNN.DataManager import MNISTDataManager
from QCNN.QCNN_structure import QCNNBuilder
from QCNN.Evaluation import HybridEvaluator, Experiment

def main():
    print("Hello from qcnn-project!")

if __name__ == "__main__":
    # 1. Create Components (Dependencies)
    data = MNISTDataManager(n_train=200, n_test=50) # Small subset for demo
    builder = QCNNBuilder(n_qubits=16)
    evaluator = HybridEvaluator(builder, epochs=5) # 5 Epochs according to paper
    
    # 2. Inject into Experiment
    experiment = Experiment(
        data_mgr=data,
        builder=builder,
        evaluator=evaluator,
        n_pop=5,   # ลองรันเล็กๆ ก่อน
        n_gen=3,
        n_gates=180 # 4 Layers approx
    )
    
    # 3. Run
    best_model, hist = experiment.run()
    
    print("\n🏁 Experiment Finished!")
    print(f"Final Best Accuracy: {best_model.fitness:.4f}")
    print(f"History: {hist}")