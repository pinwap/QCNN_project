import os

# Fix for Colab: Unset incompatible backend variable before importing matplotlib
if "MPLBACKEND" in os.environ:
    del os.environ["MPLBACKEND"]

import matplotlib

matplotlib.use("Agg")
import datetime
import logging

import matplotlib.pyplot as plt

from QCNN.DataManager import MNISTDataManager
from QCNN.Evaluation import Experiment, HybridEvaluator
from QCNN.QCNN_structure import QCNNBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("qcnn_experiment.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting QCNN Application")
    # 1. Create Components
    logger.info("Initializing components...")
    data_manager = MNISTDataManager(
        data_path="../data", n_train=200, n_test=50
    )  # Small subset for demo
    builder = QCNNBuilder(n_qubits=16)
    evaluator = HybridEvaluator(
        builder, epochs=5, lr=0.01
    )  # 5 Epochs according to paper

    # 2. Inject into Experiment
    logger.info("Setting up experiment...")
    experiment = Experiment(
        data_mgr=data_manager,
        evaluator=evaluator,
        n_pop=5,  # ลองรัน population เล็กๆ ก่อน
        n_gen=3,  # ลองรัน 3 รุ่น
        n_gates=180,  # 4 Layers approx
    )

    # 3. Run
    logger.info("Running experiment...")
    best_model, history = experiment.run()

    logger.info("Experiment Finished!")
    if best_model:
        logger.info(f"Final Best Accuracy: {best_model.fitness:.4f}")
        logger.info(f"History: {history}")

        # --- 💾 ส่วนที่เพิ่ม: Save ผลลัพธ์ลง Drive ---
        # สร้างชื่อไฟล์ตามเวลา (จะได้ไม่ทับของเก่าเวลารันหลายรอบ)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Saving results with timestamp: {timestamp}")

        # 1. วาดกราฟและบันทึกเป็นรูปภาพ (.png)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(history) + 1), history, marker="o", linestyle="-", color="b")
        plt.title(f"QEA-QCNN History (Best Acc: {best_model.fitness:.4f})")
        plt.xlabel("Generation")
        plt.ylabel("Accuracy")
        plt.grid(True)

        # 🔥 สำคัญ: ใช้ savefig แทน show
        # ตรวจสอบว่าอยู่บน Colab หรือไม่ ถ้าไม่ใช่ให้บันทึกลงโฟลเดอร์ plots ปกติ
        save_path = "/content/drive/My Drive/QCNN_Results"
        if not os.path.exists("/content/drive/My Drive"):
            save_path = "plots"
            os.makedirs(save_path, exist_ok=True)

        graph_filename = os.path.join(save_path, f"qcnn_graph_{timestamp}.png")
        plt.savefig(graph_filename)
        print(f"✅ Graph saved to: {graph_filename}")
        plt.close()  # ปิดกราฟเพื่อคืน Ram

        # 2. บันทึกประวัติคะแนนดิบ (.txt หรือ .npy) เก็บไว้พล็อตกราฟใหม่ทีหลัง
        history_filename = os.path.join(save_path, f"history_{timestamp}.txt")
        with open(history_filename, "w") as f:
            f.write(str(history))
        print(f"✅ History data saved to: {history_filename}")

        # 3. (Optional) บันทึกโมเดลที่ดีที่สุด เก็บไว้เผื่อเอาไปใช้ต่อ
        # torch.save(best_model, f"{save_path}/best_model_{timestamp}.pth")

    else:
        print("\nExperiment Failed (Data or Execution Error)")


if __name__ == "__main__":
    main()
