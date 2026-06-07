import random

from torch.utils.data import Subset


def create_benchmark_subset(
    dataset,
    subset_size=512,
    seed=42
):

    random.seed(seed)

    indices = random.sample(
        range(len(dataset)),
        subset_size
    )

    return Subset(
        dataset,
        indices
    )