from backend.datasets.coco_dataset import (
    CocoSegmentationDataset
)


DATASET_ROOT = (
    r"C:\GAMUDA\Dataset"
    r"\CONCRETE-25FEB.v3i.coco-segmentation"
)


def main():

    dataset = CocoSegmentationDataset(
        dataset_root=DATASET_ROOT,
        split="train"
    )

    print(
        f"Dataset Size: "
        f"{len(dataset)}"
    )


if __name__ == "__main__":
    main()