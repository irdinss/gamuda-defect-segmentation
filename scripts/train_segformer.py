from pathlib import Path

import torch
import yaml
import shutil
import time

from backend.utils.reproducibility import set_seed

from backend.dataloaders.segmentation_dataloader import (
    create_dataloaders,
)

from backend.models.segformer_model import (
    build_segformer,
)

from backend.training.losses import (
    CombinedSegmentationLoss,
)

from backend.training.trainer import (
    Trainer,
)

from backend.training.checkpoint import (
    save_checkpoint,
)

from backend.training.history import (
    TrainingHistory,
)

from backend.training.plots import (
    save_loss_curve,
    save_miou_curve,
)

from backend.training.experiment import (
    create_experiment_folder,
)

CONFIG_PATH = "configs/segformer.yaml"

DATASET_ROOT = (
    "/content/drive/MyDrive/"
    "gamuda-segmentation/dataset/"
    "CONCRETE-25FEB.v3i.coco-segmentation"
)

CHECKPOINT_DIR = (
    "/content/drive/MyDrive/"
    "gamuda-segmentation/checkpoints"
)


CLASS_WEIGHTS = [
    1.07,
    51.03,
    296.58,
    172.99,
    29.35,
]


def load_config():

    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def main():

    config = load_config()

    experiment_name = "exp001_baseline"

    experiment_dir = (
        Path(
            "/content/drive/MyDrive/"
            "gamuda-segmentation/experiments"
        )
        / experiment_name
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(
        config["training"]["seed"]
    )

    Path(
        CHECKPOINT_DIR
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing device: {device}\n"
    )

    train_loader, valid_loader = (
        create_dataloaders(
            dataset_root=DATASET_ROOT,
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            train_subset_size=config["dataset"]["train_subset_size"],
        )
    )

    model = build_segformer(
        num_classes=config["model"]["num_classes"]
    ).to(device)

    model = model.to(device)

    criterion = (
        CombinedSegmentationLoss(
            class_weights=CLASS_WEIGHTS
        )
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
    )

    best_miou = 0.0

    epochs = (
        config["training"]["epochs"]
    )

    print(
        f"Starting training "
        f"for {epochs} epochs\n"
    )

    history = TrainingHistory()

    start_time = time.time()

    for epoch in range(epochs):

        print(
            "=" * 60
        )

        print(
            f"Epoch "
            f"{epoch + 1}/{epochs}"
        )

        train_loss = (
            trainer.train_epoch()
        )

        val_loss, val_miou = (
            trainer.validate_epoch()
        )

        history.update(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            val_miou=val_miou,
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val mIoU: "
            f"{val_miou:.4f}"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            miou=val_miou,
            path=(
                f"{CHECKPOINT_DIR}/"
                "last_segformer.pth"
            ),
        )

        if val_miou > best_miou:

            best_miou = val_miou

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                miou=val_miou,
                path=(
                    f"{CHECKPOINT_DIR}/"
                    "best_segformer.pth"
                ),
            )

            print(
                f"New Best mIoU: "
                f"{best_miou:.4f}"
            )

        print()

    training_minutes = (
        time.time() - start_time
    ) / 60
    
    print("=" * 60)

    print(
        f"Training Complete"
    )

    print(
        f"Best Validation mIoU: "
        f"{best_miou:.4f}"
    )

    csv_path = (
        experiment_dir
        / "training_metrics.csv"
    )

    history.save_csv(
        csv_path
    )

    history.save_summary(
        output_path=
            experiment_dir
            / "training_summary.json",

        experiment_id=
            experiment_name,

        config=config,

        training_minutes=training_minutes
    )

    save_loss_curve(
        csv_path,
        experiment_dir
        / "loss_curve.png",
    )

    save_miou_curve(
        csv_path,
        experiment_dir
        / "miou_curve.png",
    )

    shutil.copy(
        CONFIG_PATH,
        experiment_dir
        / "config_snapshot.yaml",
    )

    with open(
        experiment_dir
        / "notes.md",
        "w",
    ) as f:

        f.write(
            "# Experiment Notes\n\n"
            "## Goal\n"
            "Baseline SegFormer-B0 training.\n\n"
            "## Observations\n"
            "- \n\n"
            "## Next Actions\n"
            "- \n"
        )


if __name__ == "__main__":
    main()