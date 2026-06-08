from collections import Counter

import numpy as np
import yaml
from pathlib import Path

from backend.datasets.coco_dataset import (
    CocoSegmentationDataset
)

with open("configs/dataset.yaml") as f:
    dataset_cfg = yaml.safe_load(f)

DATASET_ROOT = Path(dataset_cfg["dataset"]["root"])

CLASS_NAMES = {
    0: "Background",
    1: "Crack",
    2: "Efflorescence",
    3: "Exposed Rebar",
    4: "Spalling",
}


def main():

    dataset = CocoSegmentationDataset(
        dataset_root=DATASET_ROOT,
        split="train"
    )

    pixel_counts = Counter()

    total_images = len(dataset)

    for idx, image_info in enumerate(dataset.images):

        image_id = image_info["id"]

        mask = dataset.create_mask(
            image_id=image_id
        )

        unique, counts = np.unique(
            mask,
            return_counts=True
        )

        for cls, count in zip(
            unique,
            counts
        ):
            pixel_counts[int(cls)] += int(count)

        if (idx + 1) % 500 == 0:
            print(
                f"Processed "
                f"{idx + 1}/{total_images}"
            )

    print("\nPIXEL DISTRIBUTION\n")

    total_pixels = sum(
        pixel_counts.values()
    )

    for cls_id in sorted(
        pixel_counts.keys()
    ):

        count = pixel_counts[cls_id]

        pct = (
            count / total_pixels
        ) * 100

        print(
            f"{CLASS_NAMES[cls_id]:15s}"
            f"{count:15,d}"
            f"{pct:10.2f}%"
        )


if __name__ == "__main__":
    main()