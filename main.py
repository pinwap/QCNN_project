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

        # --- 💾 ส่วนที่เพิ่ม: Save ผลลัพธ์ลง Drive ---
        import datetime
        
        # สร้างชื่อไฟล์ตามเวลา (จะได้ไม่ทับของเก่าเวลารันหลายรอบ)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. วาดกราฟและบันทึกเป็นรูปภาพ (.png)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(history) + 1), history, marker='o', linestyle='-', color='b')
        plt.title(f'QEA-QCNN History (Best Acc: {best_model.fitness:.4f})')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        # 🔥 สำคัญ: ใช้ savefig แทน show
        save_path = '/content/drive/My Drive/QCNN_Results'
        graph_filename = f"{save_path}/qcnn_graph_{timestamp}.png"
        plt.savefig(graph_filename) 
        print(f"✅ Graph saved to: {graph_filename}")
        plt.close() # ปิดกราฟเพื่อคืน Ram

        # 2. บันทึกประวัติคะแนนดิบ (.txt หรือ .npy) เก็บไว้พล็อตกราฟใหม่ทีหลัง
        history_filename = f"{save_path}/history_{timestamp}.txt"
        with open(history_filename, "w") as f:
            f.write(str(history))
        print(f"✅ History data saved to: {history_filename}")
        
        # 3. (Optional) บันทึกโมเดลที่ดีที่สุด เก็บไว้เผื่อเอาไปใช้ต่อ
        # torch.save(best_model, f"{save_path}/best_model_{timestamp}.pth")
    
    else:
        print("\nExperiment Failed (Data or Execution Error)")
    
if __name__ == "__main__":
    main()
