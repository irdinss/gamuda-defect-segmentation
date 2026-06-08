import yaml
from pathlib import Path
from backend.dataloaders.segmentation_dataloader import (
    create_dataloaders
)

with open("configs/dataset.yaml") as f:
    dataset_cfg = yaml.safe_load(f)

DATASET_ROOT = Path(dataset_cfg["dataset"]["root"])

def main():

    train_loader, valid_loader = (
        create_dataloaders(
            dataset_root=DATASET_ROOT,
            batch_size=4
        )
    )

    images, masks = next(
        iter(train_loader)
    )

    print(
        "Images:",
        images.shape
    )

    print(
        "Masks:",
        masks.shape
    )


if __name__ == "__main__":
    main()