import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.config import (
    CHECKPOINT_DIR,
    EXPERIMENTS_DIR,
    SEGFORMER_CONFIG_PATH,
    load_yaml,
)

from backend.models.segformer_model import (
    build_segformer,
)

config = load_yaml(
    SEGFORMER_CONFIG_PATH
)

CLASS_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
    4: [255, 255, 0],
}

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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
    )

    args = parser.parse_args()

    experiment_name = (
        config["experiment"]["name"]
    )

    image_path = Path(
        args.image
    )

    image_stem = (
        image_path.stem
    )

    prediction_dir = (
        EXPERIMENTS_DIR
        / experiment_name
        / "predictions"
        / image_stem
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = build_segformer(
        num_classes=config["model"]["num_classes"]
    )

    checkpoint = torch.load(
        CHECKPOINT_DIR
        / f"{experiment_name}_best.pth",
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    image = cv2.imread(
        str(image_path)
    )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image_tensor = (
        torch.tensor(
            image_rgb
        )
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        outputs = model(
            pixel_values=image_tensor
        )

        prediction = (
            outputs.logits
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

    overlay = cv2.addWeighted(
        image,
        0.7,
        prediction_mask,
        0.3,
        0,
    )

    cv2.imwrite(
        str(
            prediction_dir
            / "image.png"
        ),
        image,
    )

    cv2.imwrite(
        str(
            prediction_dir
            / "prediction.png"
        ),
        prediction_mask,
    )

    cv2.imwrite(
        str(
            prediction_dir
            / "overlay.png"
        ),
        overlay,
    )

    print(
        f"Saved results to "
        f"{prediction_dir}"
    )

if __name__ == "__main__":
    main()