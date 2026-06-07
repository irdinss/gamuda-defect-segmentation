import yaml
from pathlib import Path


CONFIG_FILE = Path(
    "configs/benchmark.yaml"
)


def main():
    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    print("\nBENCHMARK CONFIG\n")

    if "benchmark_subset_fraction" in config:
        print(
            f"Subset Fraction: "
            f"{config['benchmark_subset_fraction']}"
        )
    elif "benchmark_subset_size" in config:
        print(
            f"Subset Size: "
            f"{config['benchmark_subset_size']}"
        )

    print("\nModels:")

    for model in config["models"]:
        print(f" - {model}")

    print("\nMetrics:")

    for metric in config["metrics"]:
        print(f" - {metric}")

    print(
        f"\nMeasure Latency: "
        f"{config.get('measure_latency', False)}"
    )


if __name__ == "__main__":
    main()