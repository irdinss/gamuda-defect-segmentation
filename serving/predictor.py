from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A

from backend.config import (
    CHECKPOINT_DIR,
    SEGFORMER_CONFIG_PATH,
    load_yaml,
)

from backend.models.segformer_model import (
    build_segformer,
)

config = load_yaml(
    SEGFORMER_CONFIG_PATH
)

experiment_name = (
    config["experiment"]["name"]
)

IMAGE_SIZE = 512

CLASS_COLORS = {
    0: [0, 0, 0],          # Background
    1: [255, 0, 0],        # Crack
    2: [0, 255, 0],        # Efflorescence
    3: [0, 0, 255],        # Exposed Rebar
    4: [255, 255, 0],      # Spalling
}

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = build_segformer(
    num_classes=config["model"]["num_classes"]
)

checkpoint = torch.load(
    CHECKPOINT_DIR / f"{experiment_name}_best.pth",
    map_location=device,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


def create_color_mask(mask):

    h, w = mask.shape

    output = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    for class_id, color in CLASS_COLORS.items():

        output[
            mask == class_id
        ] = color

    return output


def create_presentation_overlay(
    image,
    prediction,
):

    display_mask = prediction.copy()

    kernel = np.ones(
        (5, 5),
        np.uint8,
    )

    display_mask = cv2.dilate(
        display_mask.astype(np.uint8),
        kernel,
        iterations=2,
    )

    colored_mask = create_color_mask(
        display_mask
    )

    overlay = cv2.addWeighted(
        image,
        0.6,
        colored_mask,
        0.4,
        0,
    )

    return colored_mask, overlay


def run_prediction(
    image_path: Path
):

    image = cv2.imread(
        str(image_path)
    )

    original_image = image.copy()

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    transform = A.Compose([
        A.Resize(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    transformed = transform(
        image=image_rgb
    )

    image_tensor = (
        torch.tensor(
            transformed["image"]
        )
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        outputs = model(
            pixel_values=image_tensor
        )

        original_h, original_w = original_image.shape[:2]
        logits = F.interpolate(
            outputs.logits,
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=False,
        )

        prediction = (
            logits
            .argmax(dim=1)
            .squeeze()
            .cpu()
            .numpy()
        )

    prediction_mask = (
        create_color_mask(
            prediction
        )
    )

    _, overlay = (
        create_presentation_overlay(
            original_image,
            prediction,
        )
    )

    total_pixels = prediction.size

    crack = round(
        100 * np.sum(prediction == 1)
        / total_pixels,
        2,
    )

    efflorescence = round(
        100 * np.sum(prediction == 2)
        / total_pixels,
        2,
    )

    corrosion = round(
        100 * np.sum(prediction == 3)
        / total_pixels,
        2,
    )

    spall = round(
        100 * np.sum(prediction == 4)
        / total_pixels,
        2,
    )

    return {
        "overlay": overlay,
        "prediction_mask": prediction_mask,
        "crack": crack,
        "efflorescence": efflorescence,
        "corrosion": corrosion,
        "spall": spall,
    }