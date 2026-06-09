
# Architecture Document

## Infrastructure Defect Segmentation System

Candidate Name: Nurfarah Irdina Binti Mohamad Bisri

Role Applied: Senior AI Engineer (Computer Vision)

---

# 1. Problem Overview

This project builds an end-to-end system that identifies infrastructure defects at the pixel level from inspection images.

Given an image of a concrete structure, the system generates a segmentation mask that highlights both the location and type of defect. The target defect categories are:

* Crack
* Spall
* Efflorescence
* Corrosion
* Background

Unlike image classification, which only predicts whether a defect exists, segmentation identifies the exact area affected by the defect. This level of detail supports inspection workflows, maintenance planning, reporting, and future measurement tasks.

The system targets a practical inspection workflow where engineers upload an image and receive a visual overlay showing detected defects.

---

# 2. High-Level System Architecture

```text
Image Upload
      │
      ▼
Preprocessing
      │
      ▼
SegFormer-B0
      │
      ▼
Segmentation Mask
      │
      ▼
Overlay Generation
      │
      ▼
Result Visualization
```

The system consists of four main stages:

1. Data preparation
2. Model training
3. Inference
4. Visualization

Each stage operates as an independent module. This structure keeps the codebase easier to maintain and allows future model replacements without major changes to the rest of the system.

---

# 3. Data Pipeline Design

## Dataset Selection

The project uses a publicly available infrastructure defect segmentation dataset stored in COCO segmentation format.

The dataset contains pixel-level annotations for several defect categories commonly found in concrete structures.

COCO format was selected for several practical reasons:

* Strong support across computer vision tools and libraries
* Straightforward validation and inspection
* Easy extension when adding new classes
* Compatibility with common annotation platforms

Using a widely adopted format also reduces friction when integrating future datasets.

---

## Dataset Validation

Before training, the dataset underwent a validation process to identify potential annotation issues.

Validation scripts checked for:

* Missing annotations
* Invalid polygons
* Empty masks
* Class distribution
* Dataset consistency

Poor annotations often create training instability and unreliable evaluation results. Early validation helps prevent these issues from affecting model performance.

---

## Class Imbalance

Infrastructure inspection datasets often contain a large imbalance between background pixels and defect pixels.

Most images contain large background regions, while some defect categories occupy only a small portion of the image. During dataset analysis, classes such as Spall and Efflorescence appeared much less frequently than Background.

To reduce this imbalance, class weighting was added to the loss function. This adjustment increases the contribution of underrepresented classes during training and helps prevent the model from favoring dominant classes.

---

## Data Augmentation

Data augmentation helps the model handle variations commonly found in real inspection images.

The augmentation pipeline includes:

* Horizontal flipping
* Vertical flipping
* Rotation
* Geometric transformations

The purpose is not simply to increase the number of training samples. Instead, augmentation exposes the model to different viewpoints, orientations, and image compositions that may appear during field inspections.

Since inspection images are often captured from different angles and distances, augmentation improves the model's ability to generalize beyond the training set.

---

# 4. Model Selection

## Candidate Models Considered

Several segmentation architectures were evaluated before implementation.

### U-Net

Advantages:

* Simple architecture
* Easy to train
* Low computational requirements

Disadvantages:

* Limited ability to capture broader image context
* Performance may degrade in complex backgrounds

---

### DeepLabV3+

Advantages:

* Strong segmentation performance
* Proven track record in production environments

Disadvantages:

* Higher computational cost
* More demanding deployment requirements

---

### SegFormer-B0 (Selected)

Advantages:

* Strong balance between accuracy and efficiency
* Lightweight architecture suitable for limited GPU resources
* Effective performance on fine-grained segmentation tasks
* Ability to capture both local and global image information

Disadvantages:

* Higher computational requirements than U-Net
* Relies on pretrained weights for best performance

---

## Why SegFormer-B0 Was Selected

The assessment environment relied on limited GPU resources, which required a model that balanced performance and practicality.

The selected architecture needed to provide:

* Competitive segmentation quality
* Reasonable training time
* Manageable memory usage
* Straightforward deployment

SegFormer-B0 offered the strongest balance across these requirements.

Compared to larger transformer-based models, SegFormer-B0 trains efficiently on a Tesla T4 GPU while still benefiting from transformer-based feature extraction. Compared to lighter convolutional models, it captures broader image context that can help distinguish defects from surrounding concrete textures.

For the scope of this project, SegFormer-B0 provided the most practical trade-off between accuracy, speed, and deployment complexity.

---

# 5. Training Pipeline

## Transfer Learning Strategy

Training a transformer-based segmentation model from scratch requires a large amount of labeled data.

The available dataset is relatively small compared to the datasets used to train modern vision models. Instead of starting from random initialization, the project uses pretrained SegFormer weights and fine-tunes them on the defect segmentation dataset.

This approach reduces training time, improves convergence, and allows the model to benefit from visual features learned from larger datasets.

---

## Loss Function

The training objective combines:

* Cross Entropy Loss
* Dice Loss

Cross Entropy Loss focuses on correct pixel classification.

Dice Loss focuses on overlap quality between predicted masks and ground-truth masks.

Each loss addresses a different aspect of segmentation performance. Combining both encourages accurate pixel predictions while also improving mask quality, particularly for smaller defect regions.

---

## Training Workflow

```text
Dataset
   │
   ▼
DataLoader
   │
   ▼
Augmentation
   │
   ▼
SegFormer
   │
   ▼
Loss Calculation
   │
   ▼
Backpropagation
   │
   ▼
Checkpoint Saving
```

The training process follows these steps:

1. Load a batch of images and masks
2. Apply augmentation
3. Generate predictions
4. Calculate loss
5. Update model weights
6. Run validation
7. Save checkpoints

The system tracks validation performance throughout training and stores the best-performing checkpoint based on validation mIoU.

---

# 6. Inference Pipeline

During inference, the system accepts an uploaded image and prepares it for prediction.

The preprocessing stage includes:

1. Image resizing
2. Tensor conversion
3. Normalization

The model then generates a pixel-level prediction map.

The prediction map is converted into a colorized segmentation mask, and the system blends the mask with the original image to create an overlay.

This output allows users to quickly identify defect locations without inspecting raw prediction values.

---

# 7. Visualization Layer

The visualization layer generates three outputs:

* Original image
* Segmentation mask
* Overlay image

Maintenance and inspection teams typically rely on visual evidence when reviewing defects. Presenting only numerical outputs or raw masks would make interpretation more difficult.

The overlay image provides a direct comparison between the original structure and the predicted defect regions, making results easier to review and communicate.

---

# 8. Deployment Architecture

The deployment architecture remains intentionally lightweight.

```text
User
 │
 ▼
Web Interface
 │
 ▼
Inference API
 │
 ▼
SegFormer Model
 │
 ▼
Prediction Result
```

The system separates the user interface from the inference logic.

The web interface handles image uploads and result presentation, while the inference API manages preprocessing, model execution, and prediction generation.

This separation keeps deployment simple while allowing future upgrades to either component independently.

The target deployment environment uses free-tier cloud resources to demonstrate a complete working solution without introducing unnecessary infrastructure complexity.

---

# 9. Scalability Considerations

The current implementation focuses on the assessment scope. A production deployment would require additional considerations as usage grows.

## Larger Dataset

A significant increase in dataset size would introduce new training challenges.

Potential improvements include:

* Distributed training across multiple GPUs
* Dataset versioning and tracking
* Automated validation pipelines
* More structured experiment management

These additions would improve reproducibility and reduce operational overhead as the dataset expands.

---

## Higher Inference Volume

A larger number of prediction requests would require infrastructure changes.

Potential improvements include:

* Dedicated inference servers
* Batch inference processing
* Model caching
* Load balancing

These changes would improve throughput and reduce response times under heavier workloads.

---

## Edge Deployment

Deploying the model on mobile devices or embedded hardware would require additional optimization.

Potential approaches include:

* Model quantization
* ONNX export
* TensorRT optimization

These techniques reduce memory usage and improve inference speed on resource-constrained devices.

---

# 10. Future Improvements

Several enhancements remain outside the scope of this assessment but would provide value in a production setting.

Potential future work includes:

* Additional defect categories
* Active learning workflows
* Semi-supervised training
* Temporal inspection analysis
* Video-based defect segmentation
* Multi-model ensemble approaches

The current architecture supports these extensions without requiring major structural changes.

---

# Conclusion

This project focuses on building a complete and maintainable segmentation system rather than pursuing benchmark performance alone.

The architecture emphasizes:

* Reproducibility
* Modularity
* Ease of experimentation
* Practical deployment

SegFormer-B0 provides a strong balance between segmentation quality, computational efficiency, and deployment feasibility. The modular design also leaves room for future improvements as dataset size, deployment requirements, and business needs evolve.






# Training Configuration

| Parameter          | Value                                     |
| ------------------ | ----------------------------------------- |
| Backbone           | SegFormer-B0                              |
| Pretrained Weights | nvidia/segformer-b0-finetuned-ade-512-512 |
| Epochs             | 30                                        |
| Batch Size         | 32                                        |
| Learning Rate      | 0.001                                     |
| Input Resolution   | 512 × 512                                 |
| Optimizer          | AdamW                                     |
| Scheduler          | Cosine Annealing                          |
| Loss Function      | Weighted Cross Entropy + Dice Loss        |

Best experiment:

```text
exp005_e30_b32_lr001_d5_c5
```

Best validation epoch:

```text
27
```

Best validation mIoU:

```text
0.5611
```

---

# Deployment Architecture

```text
User
 │
 ▼
React Frontend
 │
 ▼
Python Backend API
 │
 ▼
Inference Service
 │
 ▼
SegFormer-B0 Model
 │
 ▼
Segmentation Prediction
 │
 ▼
Overlay Visualization
 │
 ▼
Response
```

Responsibilities:

Frontend:

* Image upload
* Visualization
* User interaction

Backend:

* Input validation
* Image preprocessing
* Model inference
* Result generation

Model Layer:

* SegFormer-B0 execution
* Mask generation
* Class prediction
