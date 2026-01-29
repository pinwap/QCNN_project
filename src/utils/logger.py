import datetime
import json
import logging
import os
from typing import Any, List, Optional

import torch

logger = logging.getLogger(__name__)


def initialize_output_dir(
    script_name: str,
    base_output_dir: str = "outputs",
    preprocessor_name: str = "",
    feature_map_name: str = "",
):
    """
    Creates the standardized output directory structure.
    Returns the save_dir and a unique file_id.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clean up names for filename safety
    p_name = preprocessor_name.replace("_", "") if preprocessor_name else "raw"
    fm_name = feature_map_name if feature_map_name else "na"

    file_id = f"{script_name}_{fm_name}_{p_name}_{timestamp}"

    save_dir = os.path.join(base_output_dir, file_id)
    plots_dir = os.path.join(save_dir, "plots")
    model_dir = os.path.join(save_dir, "model")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Setup file logging
    log_file = os.path.join(save_dir, f"{file_id}_logs.log")

    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)

    print(f"✅ Output initialized at: {save_dir}")
    print(f"✅ Log file: {log_file}")

    return save_dir, file_id


def save_experiment_data(
    final_score: float,
    history: Any,
    save_dir: str,
    file_id: str,
    best_structure_code: Optional[List[int]] = None,
    config: Optional[Any] = None,
):
    """
    Saves raw history and configuration data to JSON and TXT.
    """
    # 1. Save JSON Data
    data_filename = os.path.join(save_dir, f"{file_id}_results.json")
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

    with open(data_filename, "w") as f:
        json.dump(output_data, f, indent=4)

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
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"{file_id}_{name}.pth")
    torch.save(model, save_path)
    logger.info(f"Model saved to: {save_path}")
