import torch
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader

from backend.datasets.torch_dataset import (
    TorchSegmentationDataset
)

from backend.utils.dataset_subset import (
    create_benchmark_subset
)

from backend.transforms.segmentation_transforms import (
    get_train_transforms
)


def create_benchmark_loader(
    dataset_root
):

    dataset = TorchSegmentationDataset(
        dataset_root=dataset_root,
        split="train",
        transform=get_train_transforms()
    )

    subset = create_benchmark_subset(
        dataset,
        subset_size=512
    )

    return DataLoader(
        subset,
        batch_size=8,
        shuffle=True,
        num_workers=
    )


def create_model():

    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=5
    )