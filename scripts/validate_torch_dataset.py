import yaml
from pathlib import Path
from backend.datasets.torch_dataset import (
    TorchSegmentationDataset
)

from backend.transforms.segmentation_transforms import (
    get_train_transforms
)

with open("configs/dataset.yaml") as f:
    dataset_cfg = yaml.safe_load(f)

DATASET_ROOT = Path(dataset_cfg["dataset"]["root"])

def main():

    dataset = (
        TorchSegmentationDataset(
            dataset_root=DATASET_ROOT,
            split="train",
            transform=get_train_transforms()
        )
    )

    image, mask = dataset[0]

    print(
        "Image Shape:",
        image.shape
    )

    print(
        "Mask Shape:",
        mask.shape
    )

    print(
        "Mask Classes:",
        mask.unique()
    )


if __name__ == "__main__":
    main()