from typing import List

from matplotlib import pyplot as plt


def plot_loss_history(history: List[float], save_path: str, title: str = "Training Loss History"):
    """
    Plots the loss over epochs (lower is better).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, marker="o", linestyle="-", color="r")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_fitness_history(
    history: List[float], save_path: str, title: str = "Evolution Fitness History"
):
    """
    Plots the accuracy/fitness over generations (higher is better).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, marker="o", linestyle="-", color="b")
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
def plot_training_metrics(
    history: dict,
    save_dir: str,
    file_id: str,
    final_score: float = 0.0,
):
    """
    Plots Loss and Accuracy on a single graph using dual Y-axes.
    """
    epochs = range(1, len(history.get("loss", [])) + 1)
    if not epochs:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Plot Loss (Left Axis) ---
    color_loss = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color_loss)
    ax1.plot(epochs, history["loss"], marker="o", color=color_loss, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(True, alpha=0.3)

    # --- Plot Accuracy (Right Axis) ---
    if "train_acc" in history:
        ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
        color_acc = "tab:green"
        ax2.set_ylabel("Training Accuracy", color=color_acc)
        ax2.plot(epochs, history["train_acc"], marker="s", color=color_acc, label="Accuracy")
        ax2.tick_params(axis="y", labelcolor=color_acc)
        ax2.set_ylim(0, 1.05)  # Accuracy is always 0-1

    plt.title(f"Training Metrics (Final Score: {final_score:.4f})")
    fig.tight_layout()  # Ensure the right y-label is not clipped

    # Save combined plot
    plt.savefig(f"{save_dir}/{file_id}_metrics.png")
    plt.close()
