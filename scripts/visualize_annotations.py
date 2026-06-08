import json
from pathlib import Path
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

with open("configs/dataset.yaml") as f:
    dataset_cfg = yaml.safe_load(f)

DATASET_ROOT = Path(dataset_cfg["dataset"]["root"])

SPLIT = "train"


def main():

    annotation_file = (
        DATASET_ROOT
        / SPLIT
        / "_annotations.coco.json"
    )

    with open(annotation_file, "r") as f:
        coco = json.load(f)

    image_info = coco["images"][0]

    image_path = (
        DATASET_ROOT
        / SPLIT
        / image_info["file_name"]
    )

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    overlay = image.copy()

    image_id = image_info["id"]

    annotations = [
        ann
        for ann in coco["annotations"]
        if ann["image_id"] == image_id
    ]

    for ann in annotations:

        if not ann["segmentation"]:
            continue

        polygon = np.array(
            ann["segmentation"][0]
        ).reshape(-1, 2)

        polygon = polygon.astype(np.int32)

        cv2.polylines(
            overlay,
            [polygon],
            True,
            (255, 0, 0),
            3
        )

    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()