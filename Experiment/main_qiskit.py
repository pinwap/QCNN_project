import matplotlib.pyplot as plt
import numpy as np
import logging
from data import DataManager, DataConfig
from qiskitQCNN.qiskitQCNN_structure import QCNNStructure
from qiskitQCNN.Evaluation import QCNNTrainer
from QCNN.utils import initialize_output_dir, graph_history

def main():
    # 0. Initialize Output Directory & Logging
    save_dir, file_id = initialize_output_dir(script_name="main_qiskit")
    logger = logging.getLogger(__name__)

    # 1. Prepare Data
    logger.info("--- Step 1: Generating Data ---")
    dm = DataManager(DataConfig(dataset="synthetic", n_train=35, n_test=15))
    x_train, y_train, x_test, y_test = dm.get_data(as_numpy=True)
    dm.train_images, dm.train_labels, dm.test_images, dm.test_labels = x_train, y_train, x_test, y_test

    # 2. Build Model (Flexible Qubits!)
    logger.info("\n--- Step 2: Building QCNN Model ---")
    q_struct = QCNNStructure(num_qubits=8) # เปลี่ยนเป็น 4 หรือ 16 ก็ได้ถ้าแก้ data gen ให้รองรับ
    circuit, input_params, weight_params = q_struct.build_full_circuit()
    
    # 3. Setup Trainer & Train
    logger.info("\n--- Step 3: Training ---")
    trainer = QCNNTrainer(circuit, input_params, weight_params)
    trainer.train(dm.train_images, dm.train_labels, max_iter=40)

    # 4. Evaluation
    logger.info("\n--- Step 4: Evaluation ---")
    logger.info("Train Score:")
    train_score = trainer.evaluate(dm.train_images, dm.train_labels)
    logger.info("Test Score:")
    test_score = trainer.evaluate(dm.test_images, dm.test_labels)

    # Save Result
    graph_history(
        best_model=test_score,
        history=trainer.objective_func_vals,
        experiment=trainer,
        save_dir=save_dir,
        file_id=file_id
    )

    # 5. Visual Check
    y_predict = trainer.predict(dm.test_images)
    
    # Plotting result (Optional visualizer function)
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"xticks": [], "yticks": []})
    for i in range(min(4, len(dm.test_images))):
        ax[i // 2, i % 2].imshow(dm.test_images[i].reshape(2, 4), aspect="equal")
        pred_text = "Vertical" if y_predict[i] == 1 else "Horizontal"
        ax[i // 2, i % 2].set_title(f"Pred: {pred_text}")
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.show()

if __name__ == "__main__":
    main()
    