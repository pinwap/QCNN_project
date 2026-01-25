import matplotlib.pyplot as plt
import numpy as np
from DataManager import DataManager
from qiskitQCNN_structure import QCNNStructure
from Evaluation import QCNNTrainer

def main():
    # 1. Prepare Data
    print("--- Step 1: Generating Data ---")
    dm = DataManager(num_images=50)
    dm.generate_synthetic_dataset()

    # 2. Build Model (Flexible Qubits!)
    print("\n--- Step 2: Building QCNN Model ---")
    q_struct = QCNNStructure(num_qubits=8) # เปลี่ยนเป็น 4 หรือ 16 ก็ได้ถ้าแก้ data gen ให้รองรับ
    circuit, input_params, weight_params = q_struct.build_full_circuit()
    
    # 3. Setup Trainer & Train
    print("\n--- Step 3: Training ---")
    trainer = QCNNTrainer(circuit, input_params, weight_params)
    trainer.train(dm.train_images, dm.train_labels, max_iter=200)

    # 4. Evaluation
    print("\n--- Step 4: Evaluation ---")
    print("Train Score:")
    trainer.evaluate(dm.train_images, dm.train_labels)
    print("Test Score:")
    trainer.evaluate(dm.test_images, dm.test_labels)

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
    