from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Config files
DATASET_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "project.yaml"
)

SEGFORMER_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "segformer.yaml"
)

DEEPLAB_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "deeplab.yaml"
)

UNET_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "unet.yaml"
)

# Project config
with open(DATASET_CONFIG_PATH, "r") as f:
    project_cfg = yaml.safe_load(f)

DATASET_ROOT = (
    PROJECT_ROOT /
    project_cfg["dataset"]["root"]
)

CHECKPOINT_DIR = (
    PROJECT_ROOT /
    project_cfg["artifacts"]["checkpoints"]
)

EXPERIMENTS_DIR = (
    PROJECT_ROOT /
    project_cfg["artifacts"]["experiments"]
)

NUM_CLASSES = (
    project_cfg["dataset"]["num_classes"]
)

CLASS_NAMES = (
    project_cfg["classes"]
)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)