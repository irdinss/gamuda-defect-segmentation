from pathlib import Path
import json

import torch

from backend.config import (
    DATASET_ROOT,
    EXPERIMENTS_DIR,
    CHECKPOINT_DIR,
    SEGFORMER_CONFIG_PATH,
    load_yaml,
)

from backend.models.segformer_model import (
    build_segformer,
)

from backend.dataloaders.segmentation_dataloader import (
    create_dataloaders,
)

from backend.training.metrics import (
    compute_miou,
)

config = load_yaml(
    SEGFORMER_CONFIG_PATH
)

experiment_name = (
    config["experiment"]["name"]
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def main():

    evaluation_dir = (
        EXPERIMENTS_DIR
        / experiment_name
        / "evaluation"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _, valid_loader = create_dataloaders(
        dataset_root=DATASET_ROOT,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        train_subset_size=None,
    )

    model = build_segformer(
        num_classes=config["model"]["num_classes"]
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{experiment_name}_best.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    total_miou = 0.0

    with torch.no_grad():

        for images, masks in valid_loader:

            images = images.to(device)

            masks = masks.to(device)

            outputs = model(
                pixel_values=images
            )

            logits = outputs.logits

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            miou = compute_miou(
                predictions.cpu(),
                masks.cpu(),
                num_classes=config["model"]["num_classes"],
            )

            total_miou += miou

    mean_miou = (
        total_miou
        / len(valid_loader)
    )

    metrics = {
        "experiment": experiment_name,
        "miou": float(mean_miou),
    }

    with open(
        evaluation_dir / "metrics.json",
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    print(metrics)

if __name__ == "__main__":
    main()