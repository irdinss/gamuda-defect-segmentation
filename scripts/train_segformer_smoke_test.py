import torch
import torch.nn.functional as F

from backend.dataloaders.segmentation_dataloader import (
    create_dataloaders,
)

from backend.models.segformer_model import (
    build_segformer,
)

from backend.training.losses import (
    CombinedSegmentationLoss,
)
from backend.config import DATASET_ROOT


CLASS_WEIGHTS = [
    1.07,
    51.03,
    296.58,
    172.99,
    29.35,
]


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing device: {device}\n")

    train_loader, _ = create_dataloaders(
        dataset_root=DATASET_ROOT,
        batch_size=2,
        num_workers=0,
    )

    model = build_segformer(
        num_classes=5
    ).to(device)

    criterion = (
        CombinedSegmentationLoss(
            class_weights=CLASS_WEIGHTS
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    model.train()

    images, masks = next(
        iter(train_loader)
    )

    print(
        f"Image shape: {images.shape}"
    )

    print(
        f"Mask shape: {masks.shape}"
    )

    images = images.to(device)

    masks = masks.to(device)

    outputs = model(
        pixel_values=images
    )

    logits = outputs.logits

    print(
        f"Raw logits shape: {logits.shape}"
    )

    logits = F.interpolate(
        logits,
        size=masks.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    print(
        f"Upsampled logits shape: {logits.shape}"
    )

    loss = criterion(
        logits,
        masks,
    )

    print(
        f"Loss: {loss.item():.4f}"
    )

    optimizer.zero_grad()

    loss.backward()

    print(
        "Backward pass successful"
    )

    optimizer.step()

    print(
        "Optimizer step successful"
    )

    print(
        "\nSmoke test PASSED"
    )


if __name__ == "__main__":
    main()