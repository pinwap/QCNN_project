from matplotlib import pyplot as plt
import datetime
import logging
import os


def setup_logging(level=logging.INFO, log_dir="logs", filename='main'):
    """Configures global logging settings to file and console."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"qcnn_experiment_{filename}_{timestamp}.log")

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    # Print primarily so user knows where log is even if stdout is cluttered
    print(f"✅ Logging initialized. Log file: {log_file}")


def graph_history(best_model, history: dict, experiment=None, save_dir="logs/results"):
    # Save accuracy results
    # Create filename with timestamp (to avoid overwriting when running multiple times)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Create and save plot as PNG
    plt.figure(figsize=(10, 6))
    
    # Display experiment parameters
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

    # Save figure
    graph_filename = os.path.join(save_dir, f"qcnn_graph_{timestamp}.png")
    plt.savefig(graph_filename)
    print(f"✅ Graph saved to: {graph_filename}")
    plt.close()  # Close plot to free memory

    # 2. Save raw history data for future re-plotting
    history_filename = os.path.join(save_dir, f"history_{timestamp}.txt")
    with open(history_filename, "w") as f:
        f.write(str(history))
    print(f"✅ History data saved to: {history_filename}")

    # 3. (Optional) บันทึกโมเดลที่ดีที่สุด เก็บไว้เผื่อเอาไปใช้ต่อ
    # torch.save(best_model, f"{save_path}/best_model_{timestamp}.pth")

def graph_loss(history: dict, filename: str = "loss_history.png"):
    """Plot and save the loss history graph."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, marker="o", linestyle="-", color="r")
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(filename)
    print(f"✅ Loss graph saved to: {filename}")
    plt.close()  # ปิดกราฟเพื่อคืน Ram