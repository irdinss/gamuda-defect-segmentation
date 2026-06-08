from pathlib import Path
import yaml

with open("configs/dataset.yaml") as f:
    cfg = yaml.safe_load(f)

root = Path(cfg["dataset"]["root"])

print(root)
print(root.exists())

print((root / "train" / "_annotations.coco.json").exists())
print((root / "valid" / "_annotations.coco.json").exists())
print((root / "test" / "_annotations.coco.json").exists())