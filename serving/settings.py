from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_DIR = (
    PROJECT_ROOT /
    "artifacts" /
    "uploads"
)

PREDICTION_DIR = (
    PROJECT_ROOT /
    "artifacts" /
    "predictions"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)