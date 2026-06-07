import json
from pathlib import Path


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

        self.categories = self.coco["categories"]

    def __len__(self):
        return len(self.images)