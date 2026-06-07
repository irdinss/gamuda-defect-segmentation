from backend.datasets.torch_dataset import (
    TorchSegmentationDataset
)

from backend.utils.dataset_subset import (
    create_benchmark_subset
)


DATASET_ROOT = (
    r"C:\GAMUDA\Dataset"
    r"\CONCRETE-25FEB.v3i.coco-segmentation"
)


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