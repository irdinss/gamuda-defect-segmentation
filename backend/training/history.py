from pathlib import Path
import json

import pandas as pd


class TrainingHistory:

    def __init__(self):

        self.records = []

    def update(
        self,
        epoch,
        train_loss,
        val_loss,
        val_miou,
    ):

        self.records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_miou": val_miou,
            }
        )

    def save_csv(
        self,
        output_path,
    ):

        df = pd.DataFrame(
            self.records
        )

        df.to_csv(
            output_path,
            index=False,
        )

    def save_summary(
        self,
        output_path,
        experiment_id,
        config,
        training_minutes,
    ):

        best_record = max(
            self.records,
            key=lambda x: x["val_miou"],
        )

        summary = {
            "experiment_id":
                experiment_id,

            "epochs":
                len(self.records),

            "best_epoch":
                best_record["epoch"],

            "best_val_miou":
                best_record["val_miou"],

            "final_val_miou":
                self.records[-1]["val_miou"],

            "final_train_loss":
                self.records[-1][
                    "train_loss"
                ],

            "final_val_loss":
                self.records[-1][
                    "val_loss"
                ],

            "config":
                config,

            "training_minutes":
                training_minutes,
        }

        with open(
            output_path,
            "w",
        ) as f:

            json.dump(
                summary,
                f,
                indent=4,
            )