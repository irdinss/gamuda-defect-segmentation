import torch

from backend.training.losses import (
    CombinedSegmentationLoss
)

weights = [
    1.07,
    51.03,
    296.58,
    172.99,
    29.35
]

criterion = CombinedSegmentationLoss(
    class_weights=weights
)

logits = torch.randn(
    2,
    5,
    128,
    128
)

targets = torch.randint(
    0,
    5,
    (2, 128, 128)
)

loss = criterion(
    logits,
    targets
)

print("Loss:", loss.item())