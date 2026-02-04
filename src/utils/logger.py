import datetime
import json
import logging
import os
from typing import Any, List, Optional

import torch

logger = logging.getLogger(__name__) #__name__ เป็นชื่อไฟล์นี้ เวลารันจะได้รู้ว่า log มาจากไฟล์ไหน

# ตั้งค่าการ log ว่าจะเขียนยังไง บันทึกลงที่ไหน
def initialize_output_dir(
    script_name: str,
    base_output_dir: str = "outputs",
    preprocessor_name: str = "",
    feature_map_name: str = "",
    override_output_dir: Optional[str] = None,
):
    """
    Creates the standardized output directory structure.
    Returns the save_dir and a unique file_id.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ทำความสะอาดชื่อ preprocessor และ feature map
    p_name = preprocessor_name.replace("_", "") if preprocessor_name else "raw"
    fm_name = feature_map_name if feature_map_name else "na"

    # สร้าง file_id เอาชื่อต่าง ๆ มาต่อกัน
    file_id = f"{script_name}_{fm_name}_{p_name}_{timestamp}"

    # กำหนด path ของโฟลเดอร์ที่จะบันทึกผลลัพธ์
    base_dir = override_output_dir if override_output_dir else base_output_dir
    save_dir = os.path.join(base_dir, file_id)
    plots_dir = os.path.join(save_dir, "plots") #อยู่ใน ^
    model_dir = os.path.join(save_dir, "model")

    # สร้างโฟลเดอร์ ถ้า exist แล้วจะไม่ error
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Setup file logging
    # กำหนด path ของ log file
    log_file = os.path.join(save_dir, f"{file_id}_logs.log")

    root_logger = logging.getLogger() #ดึง logger หลักมาใช้
    file_handler = logging.FileHandler(log_file) #สร้าง handler สำหรับเขียนข้อความลงไฟล์ log_file
    # กำหนดรูปแบบของ log message
        # %(asctime)s - เวลาที่ log ถูกสร้าง
        # %(name)s - ชื่อ logger
        # %(levelname)s - ระดับของ log (INFO, DEBUG, ERROR, etc.)
        # %(message)s - ข้อความ log
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    # เพิ่ม handler ที่สร้างขึ้นไปยัง root logger เพื่อให้ log ข้อความลงไฟล์
    root_logger.addHandler(file_handler)

    print(f"✅ Output initialized at: {save_dir}")
    print(f"✅ Log file: {log_file}")

    # คืนค่า path ของโฟลเดอร์บันทึกผลลัพธ์ และ file_id
    return save_dir, file_id


def save_experiment_data(
    final_score: float,
    history: Any,
    save_dir: str,
    file_id: str,
    best_structure_code: Optional[List[int]] = None,
    config: Optional[Any] = None, #ค่า config ที่ใช้ตั้งค่าการทดลอง
):
    """
    Saves raw history and configuration data to JSON and TXT.
    """
    # 1. Save JSON Data
    # เตรียมไฟล์ JSON สำหรับบันทึกผลลัพธ์
    data_filename = os.path.join(save_dir, f"{file_id}_results.json")
    # สร้าง dictionary สำหรับเก็บข้อมูลทั้งหมดที่จะบันทึก
    output_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "file_id": file_id,
        "results": {
            "final_score": final_score,
            "best_structure_code": best_structure_code,
            "history": history,
        },
        "configuration": config,
    }
    # เปิดไฟล์เพื่อเขียนข้อมูล JSON
    with open(data_filename, "w") as f:
        json.dump(output_data, f, indent=4) # indent คือจัดย่อหน้าสวยๆ

    # 2. Save raw history text
    plots_dir = os.path.join(save_dir, "plots")
    history_txt = os.path.join(plots_dir, f"{file_id}_history.txt")
    with open(history_txt, "w") as f:
        f.write(str(history))

    logger.info(f"Experiment data saved to {save_dir}")


def save_model(model: Any, save_dir: str, file_id: str, name: str = "model"):
    """
    Saves a model (or any object) to the model directory.
    """
    model_dir = os.path.join(save_dir, "model")
    os.makedirs(model_dir, exist_ok=True) #สร้างโฟลเดอร์เผื่อยังไม่มี
    save_path = os.path.join(model_dir, f"{file_id}_{name}.pth")
    # บันทึกโมเดลโดยใช้ torch.save
    torch.save(model, save_path)
    logger.info(f"Model saved to: {save_path}")
