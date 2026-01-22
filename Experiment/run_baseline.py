import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from QCNN.DataManager import MNISTDataManager
from QCNN.QCNN_structure import QCNNBuilder
from QCNN.Evaluation import HybridEvaluator
from QCNN.utils import setup_logging
import logging

logger = logging.getLogger(__name__)
setup_logging()


def run_baseline():
    print("\n📉 Running Standard QCNN (Baseline)...")

    # 1. Setup
    # ใช้ข้อมูลน้อยหน่อยเพื่อความเร็วในการ Demo (Train 100รูป)
    data_mgr = MNISTDataManager(n_train=100, n_test=50)
    builder = QCNNBuilder(n_qubits=16)

    # ⚠️ Key Point: Standard QCNN ต้องเทรนนานๆ (20 Epochs) เพราะไม่มี QEA ช่วย
    evaluator = HybridEvaluator(builder, epochs=20, lr=0.01, verbose=True)

    # 2. Prepare Data
    x_train, y_train, x_test, y_test = data_mgr.get_data()

    # 3. Create Random Structure (สุ่มโครงสร้างมา 1 แบบแล้วใช้ยาว)
    # สมมติใช้ 180 เกต
    # สุ่มเลข 0-3 (Rx, Ry, Rz, I)
    fixed_structure = np.random.randint(0, 4, 180).tolist()

    print(f"🔒 Fixed Structure: {fixed_structure[:10]}...")

    # 4. Train
    # หมายเหตุ: evaluator.evaluate จะคืนค่า Acc สุดท้ายมา
    # ถ้าอยากได้กราฟ Loss ของ Baseline จริงๆ ต้องไปแก้ Evaluation ให้ return loss history
    # แต่ตอนนี้เอาแค่ Acc สุดท้ายไปบอกอาจารย์ก่อนได้ครับ
    final_acc = evaluator.evaluate(fixed_structure, x_train, y_train, x_test, y_test)

    print(f"✅ Baseline Final Accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    run_baseline()
