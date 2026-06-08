import yaml
from pathlib import Path
from backend.datasets.torch_dataset import (
    TorchSegmentationDataset
)

from backend.utils.dataset_subset import (
    create_benchmark_subset
)

with open("configs/dataset.yaml") as f:
    dataset_cfg = yaml.safe_load(f)

DATASET_ROOT = Path(dataset_cfg["dataset"]["root"])

def main():

    dataset = (
        TorchSegmentationDataset(
            dataset_root=DATASET_ROOT,
            split="train"
        )
    )

    subset = (
        create_benchmark_subset(
            dataset,
            subset_size=512
        )
    )

    print(
        f"Full Dataset: "
        f"{len(dataset)}"
    )

    print(
        f"Benchmark Subset: "
        f"{len(subset)}"
    )


if __name__ == "__main__":
    main()