import matplotlib.pyplot as plt
import pandas as pd

def save_loss_curve(
    csv_path,
    output_path,
):

    df = pd.read_csv(
        csv_path
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        df["epoch"],
        df["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        df["epoch"],
        df["val_loss"],
        label="Val Loss",
    )

    plt.legend()

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Training Loss Curve"
    )

    plt.tight_layout()

    plt.savefig(
        output_path
    )

    plt.close()


def save_miou_curve(
    csv_path,
    output_path,
):

    df = pd.read_csv(
        csv_path
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        df["epoch"],
        df["val_miou"],
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "mIoU"
    )

    plt.title(
        "Validation mIoU"
    )

    plt.tight_layout()

    plt.savefig(
        output_path
    )

    plt.close()