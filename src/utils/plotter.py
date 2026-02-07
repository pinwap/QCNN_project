from typing import List, Sequence

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
    title: str = "Training Metrics"
):
    """
    Plots Loss (Train/Val) and Accuracy (Train/Val) on a single graph using dual Y-axes.
    """
    epochs = range(1, len(history.get("loss", [])) + 1)
    if not epochs:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Plot Loss (Left Axis) ---
    color_loss_train = "tab:red"
    color_loss_val = "lightcoral"

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color_loss_train)

    lines = []
    l1, = ax1.plot(epochs, history["loss"], marker="o", linestyle="-", color=color_loss_train, label="Train Loss")
    lines.append(l1)

    if "val_loss" in history and history["val_loss"]:
        # Ensure val_loss has same length or handle potential mismatch if needed,
        # usually they match if logged every epoch.
        l2, = ax1.plot(epochs, history["val_loss"], marker="o", linestyle="--", color=color_loss_val, label="Val Loss")
        lines.append(l2)

    ax1.tick_params(axis="y", labelcolor=color_loss_train)
    ax1.grid(True, alpha=0.3)

    # --- Plot Accuracy (Right Axis) ---
    if "train_acc" in history:
        ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
        color_acc_train = "tab:green"
        color_acc_val = "lightgreen"

        ax2.set_ylabel("Accuracy", color=color_acc_train)
        l3, = ax2.plot(epochs, history["train_acc"], marker="s", linestyle="-", color=color_acc_train, label="Train Acc")
        lines.append(l3)

        if "val_acc" in history and history["val_acc"]:
            l4, = ax2.plot(epochs, history["val_acc"], marker="s", linestyle="--", color=color_acc_val, label="Val Acc")
            lines.append(l4)

        ax2.tick_params(axis="y", labelcolor=color_acc_train)
        ax2.set_ylim(0, 1.05)  # Accuracy is always 0-1

    # Combined Legend
    labels = [l.get_label() for l in lines]
    # Place legend in a relatively empty spot, e.g., center right
    ax1.legend(lines, labels, loc="center right")

    plt.title(f"{title} (Final Score: {final_score:.4f})")
    fig.tight_layout()  # Ensure the right y-label is not clipped

    # Save combined plot
    plt.savefig(f"{save_dir}/{file_id}_metrics.png")
    plt.close()


def plot_train_val_loss(
    train_history: Sequence[float],
    val_history: Sequence[float] | None,
    save_path: str,
    title: str = "Training vs Validation Loss",
):
    if not train_history:
        return

    epochs = range(1, len(train_history) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_history, marker="o", color="tab:blue", label="Train Loss")

    if val_history:
        plt.plot(
            range(1, len(val_history) + 1),
            val_history,
            marker="s",
            linestyle="--",
            color="tab:orange",
            label="Val Loss",
        )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
