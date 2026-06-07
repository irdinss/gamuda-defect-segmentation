import json
from collections import Counter
from pathlib import Path


DATASET_ROOT = Path(
    r"C:\GAMUDA\Dataset\CONCRETE-25FEB.v3i.coco-segmentation"
)


def load_split(split_name):
    annotation_file = (
        DATASET_ROOT
        / split_name
        / "_annotations.coco.json"
    )

    with open(annotation_file, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    train_coco = load_split("train")

    image_counter = Counter()

    category_counter = Counter()

    for ann in train_coco["annotations"]:

        image_counter[
            ann["image_id"]
        ] += 1

        category_counter[
            ann["category_id"]
        ] += 1

    categories = {
        c["id"]: c["name"]
        for c in train_coco["categories"]
    }

    print("\nDATASET QUALITY REPORT")
    print("=" * 50)

    print(
        f"\nImages with annotations: "
        f"{len(image_counter)}"
    )

    print(
        f"Average annotations/image: "
        f"{sum(image_counter.values()) / len(image_counter):.2f}"
    )

    print("\nClass Counts")

    for category_id, count in sorted(
        category_counter.items()
    ):
        print(
            f"{categories[category_id]:20s}"
            f"{count:10d}"
        )


if __name__ == "__main__":
    main()