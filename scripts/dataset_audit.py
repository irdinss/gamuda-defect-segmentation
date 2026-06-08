import json
import yaml
from collections import Counter
from pathlib import Path

with open("configs/dataset.yaml") as f:
    dataset_cfg = yaml.safe_load(f)

DATASET_ROOT = Path(dataset_cfg["dataset"]["root"])

REPORT_FILE = Path(
    "reports/dataset_audit.txt"
)


def load_annotations(split_name):
    annotation_file = (
        DATASET_ROOT
        / split_name
        / "_annotations.coco.json"
    )

    with open(annotation_file, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    train_coco = load_annotations("train")

    categories = {
        c["id"]: c["name"]
        for c in train_coco["categories"]
    }

    counts = Counter()

    for annotation in train_coco["annotations"]:
        counts[annotation["category_id"]] += 1

    lines = []

    lines.append("DATASET AUDIT")
    lines.append("=" * 50)
    lines.append("")

    for category_id, count in sorted(counts.items()):
        line = (
            f"{categories[category_id]:20s}"
            f"{count:10d}"
        )

        lines.append(line)

    report = "\n".join(lines)

    print(report)

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_FILE.write_text(
        report,
        encoding="utf-8"
    )

    print(
        f"\nSaved report to: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()