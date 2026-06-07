from torch.utils.data import DataLoader

from backend.datasets.torch_dataset import (
    TorchSegmentationDataset
)

from backend.transforms.segmentation_transforms import (
    get_train_transforms,
    get_valid_transforms
)


def create_dataloaders(
    dataset_root,
    batch_size=8,
    num_workers=2
):

    train_dataset = (
        TorchSegmentationDataset(
            dataset_root=dataset_root,
            split="train",
            transform=get_train_transforms()
        )
    )

    valid_dataset = (
        TorchSegmentationDataset(
            dataset_root=dataset_root,
            split="valid",
            transform=get_valid_transforms()
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return (
        train_loader,
        valid_loader
    )