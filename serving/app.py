from pathlib import Path

import cv2

from fastapi import (
    FastAPI,
    UploadFile,
    File,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.staticfiles import (
    StaticFiles,
)

from serving.predictor import (
    run_prediction,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(
    "artifacts/uploads"
)

PREDICTION_DIR = Path(
    "artifacts/predictions"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

app.mount(
    "/predictions",
    StaticFiles(
        directory=PREDICTION_DIR
    ),
    name="predictions",
)


@app.get("/health")
def health():

    return {
        "status": "online"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    upload_path = (
        UPLOAD_DIR /
        file.filename
    )

    with open(
        upload_path,
        "wb"
    ) as f:

        f.write(
            await file.read()
        )

    prediction = run_prediction(
        upload_path
    )

    overlay = prediction["overlay"]

    prediction_mask = prediction[
        "prediction_mask"
    ]

    overlay_name = (
        f"{upload_path.stem}_overlay.png"
    )

    mask_name = (
        f"{upload_path.stem}_mask.png"
    )

    overlay_path = (
        PREDICTION_DIR /
        overlay_name
    )

    mask_path = (
        PREDICTION_DIR /
        mask_name
    )

    cv2.imwrite(
        str(overlay_path),
        overlay,
    )

    cv2.imwrite(
        str(mask_path),
        prediction_mask,
    )

    return {

        "overlay_image":
        (
            f"http://localhost:8000/"
            f"predictions/"
            f"{overlay_name}"
        ),

        "prediction_mask":
        (
            f"http://localhost:8000/"
            f"predictions/"
            f"{mask_name}"
        ),

        "crack":
            prediction["crack"],

        "spall":
            prediction["spall"],

        "corrosion":
            prediction["corrosion"],

        "efflorescence":
            prediction["efflorescence"],
    }