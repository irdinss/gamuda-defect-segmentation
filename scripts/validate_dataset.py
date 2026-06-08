import json
from backend.config import DATASET_ROOT



def load_coco(annotation_file):
    with open(annotation_file, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_split(split_name):
    annotation_file = (
        DATASET_ROOT
        / split_name
        / "_annotations.coco.json"
    )

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Missing annotation file: {annotation_file}"
        )

    coco = load_coco(annotation_file)

    images = coco["images"]
    annotations = coco["annotations"]

    print(f"\n[{split_name.upper()}]")
    print(f"Images      : {len(images)}")
    print(f"Annotations : {len(annotations)}")

    return coco


def main():
    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    train_coco = summarize_split("train")
    summarize_split("valid")
    summarize_split("test")

    print("\nClasses")

    for category in train_coco["categories"]:
        print(
            f"  ID={category['id']} "
            f"Name={category['name']}"
        )


if __name__ == "__main__":
    main()