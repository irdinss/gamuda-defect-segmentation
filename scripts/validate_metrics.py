import torch

from backend.training.metrics import (
    compute_miou,
    compute_per_class_iou,
)

preds = torch.randint(
    0,
    5,
    (2, 512, 512)
)

targets = torch.randint(
    0,
    5,
    (2, 512, 512)
)

per_class = compute_per_class_iou(
    preds,
    targets,
    5,
)

miou = compute_miou(
    preds,
    targets,
    5,
)

print(
    "Per Class IoU:",
    per_class,
)

print(
    "mIoU:",
    miou,
)