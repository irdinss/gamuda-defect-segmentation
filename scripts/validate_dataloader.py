from backend.dataloaders.segmentation_dataloader import (
    create_dataloaders
)
from backend.config import DATASET_ROOT


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