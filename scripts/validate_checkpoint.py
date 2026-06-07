import torch

from backend.models.segformer_model import (
    build_segformer,
)

from backend.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)

model = build_segformer(
    num_classes=5
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)

save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=1,
    miou=0.35,
    path="temp_checkpoint.pth",
)

checkpoint = load_checkpoint(
    path="temp_checkpoint.pth",
    model=model,
    optimizer=optimizer,
)

print(
    checkpoint["epoch"]
)

print(
    checkpoint["miou"]
)