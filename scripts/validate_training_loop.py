import torch

from backend.models.unet_model import (
    create_unet
)

from backend.dataloaders.segmentation_dataloader import (
    create_dataloaders
)

from backend.training.train_one_epoch import (
    train_one_epoch
)


DATASET_ROOT = (
    r"C:\GAMUDA\Dataset"
    r"\CONCRETE-25FEB.v3i.coco-segmentation"
)


def main():

    device = "cpu"

    model = create_unet()

    model.to(device)

    train_loader, _ = (
        create_dataloaders(
            dataset_root=DATASET_ROOT,
            batch_size=2
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    criterion = (
        torch.nn.CrossEntropyLoss()
    )

    loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    print(
        f"Loss: {loss:.4f}"
    )


if __name__ == "__main__":
    main()