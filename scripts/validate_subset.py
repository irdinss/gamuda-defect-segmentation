from backend.datasets.torch_dataset import (
    TorchSegmentationDataset
)

from backend.utils.dataset_subset import (
    create_benchmark_subset
)

from backend.config import DATASET_ROOT

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
            subset_size=1024
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