from matplotlib import pyplot as plt
import datetime

def graph_history(best_model, history: dict):
    # Save ผลลัพธ์ลง Drive
    # สร้างชื่อไฟล์ตามเวลา (จะได้ไม่ทับของเก่าเวลารันหลายรอบ)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. วาดกราฟและบันทึกเป็นรูปภาพ (.png)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, marker='o', linestyle='-', color='b')
    plt.title(f'QEA-QCNN History (Best Acc: {best_model.fitness:.4f})')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # ใช้ savefig แทน show
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
