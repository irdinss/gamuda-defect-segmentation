from pathlib import Path

import cv2
import torch

from backend.datasets.coco_dataset import (
    CocoSegmentationDataset
)


class TorchSegmentationDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        dataset_root,
        split="train",
        transform=None
    ):
        self.dataset = (
            CocoSegmentationDataset(
                dataset_root=dataset_root,
                split=split
            )
        )

        self.dataset_root = Path(
            dataset_root
        )

        self.split = split

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image_info = (
            self.dataset.images[idx]
        )

        image_path = (
            self.dataset_root
            / self.split
            / image_info["file_name"]
        )

        image = cv2.imread(
            str(image_path)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = (
            self.dataset.create_mask(
                image_info["id"]
            )
        )

        if self.transform:

            transformed = (
                self.transform(
                    image=image,
                    mask=mask
                )
            )

            image = transformed["image"]

            mask = transformed["mask"]

        image = (
            torch.tensor(
                image,
                dtype=torch.float32
            )
            .permute(2, 0, 1)
        )

        mask = torch.tensor(
            mask,
            dtype=torch.long
        )

        return image, mask