import torch

from backend.models.unet_model import (
    create_unet
)


def main():

    model = create_unet()

    x = torch.randn(
        2,
        3,
        1024,
        1024
    )

    y = model(x)

    print(
        "Output Shape:",
        y.shape
    )


if __name__ == "__main__":
    main()