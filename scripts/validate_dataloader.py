from backend.dataloaders.segmentation_dataloader import (
    create_dataloaders
)


DATASET_ROOT = (
    r"C:\GAMUDA\Dataset"
    r"\CONCRETE-25FEB.v3i.coco-segmentation"
)


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