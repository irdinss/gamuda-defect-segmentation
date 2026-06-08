import time

import torch

from backend.models.unet_model import (
    create_unet
)


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def main():

    model = create_unet()

    params = count_parameters(model)

    print(
        f"Parameters: "
        f"{params:,}"
    )

    x = torch.randn(
        1,
        3,
        1024,
        1024
    )

    for _ in range(5):
        _ = model(x)

    start = time.time()

    for _ in range(20):
        _ = model(x)

    end = time.time()

    latency = (
        (end - start) / 20
    ) * 1000

    print(
        f"Latency: "
        f"{latency:.2f} ms"
    )


if __name__ == "__main__":
    main()