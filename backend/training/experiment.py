from pathlib import Path
import shutil

def create_experiment_folder(
    experiment_name,
):

    root = (
        Path("reports")
        / "experiments"
        / experiment_name
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return root

