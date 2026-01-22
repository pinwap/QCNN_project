import datetime
import logging
import os
from matplotlib import pyplot as plt
import torch


def initialize_output_dir(script_name: str):
    """
    Creates the standardized output directory structure:
    /output/{script_name}_{timestamp}/
        ├── {script_name}_{timestamp}_logs.log
        ├── plots/
        └── model/

    Returns:
        save_dir (str): The absolute path to the run directory.
        file_id (str): The identifier string "{script_name}_{timestamp}".
    """
    # 1. Generate Timestamp & ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_id = f"{script_name}_{timestamp}"

    # 2. Determine Output Root (Assumes utils.py is in QCNN/utils.py, so project root is ../)
    # Get absolute path of this file
    current_file_path = os.path.abspath(__file__)
    # Go up 2 levels: QCNN/utils.py -> QCNN -> ProjectRoot
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    output_root = os.path.join(project_root, "output")

    # 3. Create Run Directory
    save_dir = os.path.join(output_root, file_id)
    plots_dir = os.path.join(save_dir, "plots")
    model_dir = os.path.join(save_dir, "model")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 4. Setup Logging
    log_file = os.path.join(save_dir, f"{file_id}_logs.log")

    # Clear any existing handlers to prevent duplicate logs if re-run
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers = []

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # Force print to ensure user sees this even if logging has issues
    print(f"✅ Output initialized at: {save_dir}")
    print(f"✅ Log file: {log_file}")

    return save_dir, file_id


def setup_logging(level=logging.INFO, log_dir="logs", filename="main"):
    """
    Legacy support wrapper using the new system if possible,
    or falling back to basic logging.
    Ideally, use initialize_output_dir instead.
    """
    # Just forward to the new system, ignoring legacy custom dir for consistency
    # or keep it simple for backward compatibility if needed.
    # For now, let's just use the new system to enforce the rule.
    initialize_output_dir(filename)


def graph_history(best_model, history: dict, experiment=None, save_dir=None, file_id=None):
    """
    Plots training history and saves raw data.
    Args:
        best_model: The best chromosome/model from training.
        history (dict/list): Training history data.
        experiment: Experiment object (optional) for title info.
        save_dir (str): Directory to save plots.
        file_id (str): Timestamped identifier for filenames.
    """
    if save_dir is None or file_id is None:
        print(
            "⚠️ Warning: save_dir or file_id not provided to graph_history. Saving to default local 'logs'."
        )
        save_dir = "logs"
        file_id = f"legacy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)

    plots_dir = os.path.join(save_dir, "plots")
    # Ensure plots subdir exists (redundant safety)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Create and save plot as PNG
    plt.figure(figsize=(10, 6))

    if experiment:
        plt.suptitle(
            f"Population: {len(experiment.population)}, Generations: {experiment.n_gen}, "
            f"Gates: {best_model.num_gates}"
        )
    else:
        plt.suptitle("QEA-QCNN Training History")

    plt.plot(range(1, len(history) + 1), history, marker="o", linestyle="-", color="b")
    plt.title(f"QEA-QCNN History (Best Acc: {best_model.fitness:.4f})")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.grid(True)

    graph_filename = os.path.join(plots_dir, f"{file_id}_graph.png")
    plt.savefig(graph_filename)
    print(f"✅ Graph saved to: {graph_filename}")
    plt.close()

    # 2. Save raw history data
    history_filename = os.path.join(plots_dir, f"{file_id}_history.txt")
    with open(history_filename, "w") as f:
        f.write(str(history))
    print(f"✅ History data saved to: {history_filename}")


def graph_loss(history: dict, save_dir=None, file_id=None):
    """Plot and save the loss history graph."""
    if save_dir is None or file_id is None:
        save_dir = "logs"
        file_id = f"legacy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)

    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    filename = os.path.join(plots_dir, f"{file_id}_loss_history.png")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, marker="o", linestyle="-", color="r")
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(filename)
    print(f"✅ Loss graph saved to: {filename}")
    plt.close()


def save_model(model, save_dir, file_id):
    """
    Saves the PyTorch model/object.
    """
    model_dir = os.path.join(save_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(model_dir, f"{file_id}.pth")
    torch.save(model, save_path)
    print(f"✅ Model saved to: {save_path}")
