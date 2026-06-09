# Infrastructure Defect Segmentation System

Senior AI Engineer (Computer Vision) Take-Home Assessment

---

## Overview

This project implements an end-to-end semantic segmentation system for infrastructure defect detection.

Given an inspection image, the system predicts:

* Pixel-level defect segmentation masks
* Defect class labels
* Overlay visualizations
* Quantitative evaluation metrics

Target defect classes:

* Crack
* Spall
* Efflorescence
* Corrosion
* Background

The solution demonstrates:

* Deep learning model development
* Transfer learning and fine-tuning
* Evaluation and experiment tracking
* Deployment architecture design
* End-to-end software engineering practices

---


## Setup

### Clone Repository

```bash
git clone https://github.com/irdinss/gamuda-defect-segmentation.git
```

### Create Environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

Download the dataset from:

```text
https://drive.google.com/drive/folders/1ChyEiAxrwC5YZ1Sr_j_uB6dkvLjnPEdj?usp=sharing
```

The dataset may be extracted to any location.
```text
D:\datasets\CONCRETE-25FEB.v3i.coco-segmentation
                    or
~/datasets/CONCRETE-25FEB.v3i.coco-segmentation
```
The dataset path will be used at runtime through command-line arguments.

---

## Training

Train the segmentation model:

```bash
python -m scripts.train_segformer --data_dir "/path/to/CONCRETE-25FEB.v3i.coco-segmentation"
```

---

## Evaluation

Evaluate the trained model:

```bash
python -m scripts.evaluate --data_dir "/path/to/CONCRETE-25FEB.v3i.coco-segmentation"
```

Outputs:

* mIoU
* Per-class IoU
* Dice Score
* Confusion Matrix

---

## Inference

Run inference on a single image:

```bash
python -m scripts.predict --image path/to/image.jpg
```

Outputs:

* Segmentation mask
* Overlay visualization
* Class statistics

---

## Results Summary

| Metric               | Value  |
| -------------------- | ------ |
| Best Validation mIoU | 0.5611 |
| Best Epoch           | 27     |
| Final Epoch          | 30     |

Best checkpoint:

```text
artifacts/checkpoints/exp005_e30_b32_lr001_d5_c5_best.pth
```

---

## Deployment

#### Platform: Hugging Face Spaces

#### Frontend: React

#### Backend: Python + PyTorch

Deployment URL:

```text
https://irdinabisri-gamuda-defect-segmentation.hf.space/
```

Demo Video:

```text
[PLACEHOLDER_VIDEO_LINK]
```

---

## Documentation

Additional technical documentation:
* ARCHITECTURE.md
* DECISIONS.md


