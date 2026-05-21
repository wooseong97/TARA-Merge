import os

import matplotlib.pyplot as plt


def plot_training_curves(output_dir, train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(output_dir, "training_curves.png"),
        dpi=150,
        bbox_inches="tight",
    )
