import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from backend.datasets.coco_dataset import (
    CocoSegmentationDataset
)
from backend.config import DATASET_ROOT


def main():

    dataset = CocoSegmentationDataset(
        dataset_root=DATASET_ROOT,
        split="train"
    )

    output_dir = Path("debug_masks")
    output_dir.mkdir(exist_ok=True)

    for idx in range(
        400,
        len(dataset.images),
        400
    ):

        image_info = dataset.images[idx]

        image_id = image_info["id"]

        file_name = Path(
            image_info["file_name"]
        ).stem

        mask = dataset.create_mask(
            image_id=image_id
        )

        print(
            f"\nIndex: {idx}"
        )

        print(
            "Unique Classes:",
            np.unique(mask)
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(mask)
        plt.axis("off")

        plt.savefig(
            output_dir / f"mask_{file_name}.png",
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()

        print(
            f"Saved: mask_{file_name}.png"
        )


if __name__ == "__main__":
    main()