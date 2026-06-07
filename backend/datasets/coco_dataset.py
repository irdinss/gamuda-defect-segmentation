import json
from pathlib import Path

import cv2
import numpy as np


class CocoSegmentationDataset:

    def __init__(
        self,
        dataset_root,
        split="train"
    ):
        self.dataset_root = Path(dataset_root)

        self.split = split

        self.annotation_file = (
            self.dataset_root
            / split
            / "_annotations.coco.json"
        )

        with open(
            self.annotation_file,
            "r",
            encoding="utf-8"
        ) as f:
            self.coco = json.load(f)

        self.images = self.coco["images"]

        self.annotations = self.coco["annotations"]
        self.annotations_by_image_id = {}
        for annotation in self.annotations:
            image_id = annotation["image_id"]
            if image_id not in self.annotations_by_image_id:
                self.annotations_by_image_id[image_id] = []
            self.annotations_by_image_id[
                image_id
            ].append(annotation)

        self.categories = self.coco["categories"]

    def __len__(self):
        return len(self.images)

    def create_mask(self, image_id):

        image_info = next(
            img
            for img in self.images
            if img["id"] == image_id
        )

        height = image_info["height"]
        width = image_info["width"]

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        image_annotations = (
            self.annotations_by_image_id.get(
                image_id,
                []
            )
        )

        for ann in image_annotations:

            category_id = ann["category_id"]

            if not ann["segmentation"]:
                continue

            polygon = np.array(
                ann["segmentation"][0]
            ).reshape(-1, 2)

            polygon = polygon.astype(np.int32)

            cv2.fillPoly(
                mask,
                [polygon],
                color=category_id
            )

        return mask