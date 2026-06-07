import yaml
from torch.utils.data import DataLoader

from backend.datasets.torch_dataset import (
    TorchSegmentationDataset
)

from backend.transforms.segmentation_transforms import (
    get_train_transforms,
    get_valid_transforms
)

from backend.utils.dataset_subset import (
    create_benchmark_subset,
)

CONFIG_PATH = "configs/segformer.yaml"

def load_config():

    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
    

config = load_config()

def create_dataloaders(
    dataset_root,
    batch_size=8,
    num_workers=2,
    train_subset_size=None,
):

    train_dataset = (
        TorchSegmentationDataset(
            dataset_root=dataset_root,
            split="train",
            transform=get_train_transforms()
        )
    )

    if train_subset_size is not None:
        train_dataset = create_benchmark_subset(
            train_dataset,
            subset_size=train_subset_size,
            seed=42,
        )

        print(
            f"Using subset of "
            f"{len(train_dataset)} "
            f"training samples"
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
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return (
        train_loader,
        valid_loader
    )