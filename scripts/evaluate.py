from pathlib import Path
import json

import torch
import torch.nn.functional as F

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


def compute_per_class_iou(
    predictions,
    targets,
    num_classes,
):
    ious = []

    for cls in range(num_classes):

        pred_mask = predictions == cls
        target_mask = targets == cls

        intersection = (
            pred_mask & target_mask
        ).sum().item()

        union = (
            pred_mask | target_mask
        ).sum().item()

        if union == 0:
            ious.append(None)
        else:
            ious.append(
                intersection / union
            )

    return ious


def compute_pixel_accuracy(
    predictions,
    targets,
):
    correct = (
        predictions == targets
    ).sum().item()

    total = targets.numel()

    return correct / total


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
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    num_classes = (
        config["model"]["num_classes"]
    )

    class_names = [
        "background",
        "crack",
        "spalling",
        "efflorescence",
        "rebar"
    ]

    total_pixel_accuracy = 0.0
    total_miou = 0.0

    per_class_iou_sum = [
        0.0
    ] * num_classes

    per_class_iou_count = [
        0
    ] * num_classes

    class_frequency = [
        0
    ] * num_classes

    with torch.no_grad():

        for images, masks in valid_loader:

            images = images.to(device)

            masks = masks.to(device)

            outputs = model(
                pixel_values=images
            )

            logits = F.interpolate(
                outputs.logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            pixel_acc = compute_pixel_accuracy(
                predictions,
                masks,
            )

            total_pixel_accuracy += pixel_acc

            batch_ious = compute_per_class_iou(
                predictions,
                masks,
                num_classes,
            )

            valid_ious = []

            for cls_idx, iou in enumerate(batch_ious):

                if iou is not None:

                    per_class_iou_sum[
                        cls_idx
                    ] += iou

                    per_class_iou_count[
                        cls_idx
                    ] += 1

                    valid_ious.append(iou)

            if len(valid_ious) > 0:

                total_miou += (
                    sum(valid_ious)
                    / len(valid_ious)
                )

            for cls in range(num_classes):

                class_frequency[
                    cls
                ] += (
                    masks == cls
                ).sum().item()

    mean_miou = (
        total_miou
        / len(valid_loader)
    )

    mean_pixel_accuracy = (
        total_pixel_accuracy
        / len(valid_loader)
    )

    per_class_iou = {}

    for idx in range(num_classes):

        if per_class_iou_count[idx] > 0:

            per_class_iou[
                class_names[idx]
            ] = (
                per_class_iou_sum[idx]
                / per_class_iou_count[idx]
            )

        else:

            per_class_iou[
                class_names[idx]
            ] = None

    class_frequency_dict = {
        class_names[idx]: int(freq)
        for idx, freq in enumerate(
            class_frequency
        )
    }

    metrics = {
        "experiment": experiment_name,
        "miou": float(mean_miou),
        "pixel_accuracy": float(
            mean_pixel_accuracy
        ),
        "per_class_iou": per_class_iou,
        "class_frequency": class_frequency_dict,
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

    with open(
        evaluation_dir / "per_class_iou.json",
        "w",
    ) as f:

        json.dump(
            per_class_iou,
            f,
            indent=4,
        )

    with open(
        evaluation_dir / "class_frequency.json",
        "w",
    ) as f:

        json.dump(
            class_frequency_dict,
            f,
            indent=4,
        )

    print(
        json.dumps(
            metrics,
            indent=4,
        )
    )


if __name__ == "__main__":
    main()