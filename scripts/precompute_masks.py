import cv2
import yaml
from tqdm import tqdm
from pathlib import Path

from backend.datasets.coco_dataset import (
    CocoSegmentationDataset,
)

with open("configs/dataset.yaml") as f:
    dataset_cfg = yaml.safe_load(f)

DATASET_ROOT = Path(dataset_cfg["dataset"]["root"])


def process_split(split):

    dataset = CocoSegmentationDataset(
        dataset_root=DATASET_ROOT,
        split=split,
    )

    mask_dir = (
        Path(DATASET_ROOT)
        / f"{split}_masks"
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for image_info in tqdm(dataset.images):

        image_id = image_info["id"]

        filename = (
            Path(
                image_info["file_name"]
            ).stem
        )

        mask = dataset.create_mask(
            image_id
        )

        output_path = (
            mask_dir
            / f"{filename}.png"
        )

        cv2.imwrite(
            str(output_path),
            mask,
        )

    print(
        f"{split} complete"
    )


def main():

    process_split("train")

    process_split("valid")

    process_split("test")


if __name__ == "__main__":
    main()