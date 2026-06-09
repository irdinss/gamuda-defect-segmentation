# Technical Decisions Documentation

## Infrastructure Defect Segmentation System

This document records the major technical decisions made during the implementation of the project, including alternatives that were considered, trade-offs that were accepted, and situations that may justify revisiting each decision in the future.

---

# Decision 1: Select SegFormer-B0 as the Primary Segmentation Backbone

## Context

Initial inspection of the dataset revealed two challenges.

First, several defect classes occupy only a small fraction of each image.
Cracks in particular appear as thin structures that can easily disappear
during aggressive downsampling.

Second, defect appearance depends heavily on surrounding context.
Concrete texture, lighting variation, stains, and shadows can look visually
similar to actual defects when viewed locally.

The selected model therefore needed to capture both local detail and
large-scale context while remaining trainable on limited GPU resources.

## Alternatives Considered

### U-Net

U-Net was considered because it remains a strong baseline for many
segmentation tasks and can be trained efficiently.

However, preliminary review suggested that its convolution-only encoder may
struggle to distinguish true defects from visually similar background
patterns when larger image context is required.

### DeepLabV3+

DeepLabV3+ was considered because of its strong segmentation performance and
effective multi-scale feature extraction.

The main concern was training and inference cost. The assessment requires an
end-to-end deliverable including deployment, making model efficiency an
important consideration alongside accuracy.

### SegFormer-B0 (Selected)

SegFormer-B0 combines a transformer-based encoder with a lightweight decoder,
allowing the model to capture wider image context without introducing a large
computational footprint.

The architecture provides a favourable balance between segmentation quality,
training speed, memory usage, and deployment feasibility.

## Why This Decision Was Made

The project prioritised end-to-end delivery rather than maximising model size.

SegFormer-B0 offered enough capacity to learn contextual defect patterns
while still supporting rapid experimentation on free-tier GPU resources.

This balance enabled more time to be spent on evaluation, deployment, and
failure analysis instead of waiting for longer training cycles.

## Trade-Off Accepted

The solution accepts a lower theoretical accuracy ceiling compared to larger
SegFormer variants in exchange for faster iteration and easier deployment.

## Revisit Conditions

This decision should be revisited if:

- Additional GPU resources become available
- Dataset size increases significantly
- Production requirements prioritise accuracy over inference cost

In those situations, SegFormer-B2 or SegFormer-B3 would become strong
candidates for evaluation.


---

# Decision 2: Use Weighted Cross Entropy and Dice Loss

## Context

Dataset analysis revealed significant class imbalance. Large background regions dominate most images while some defect categories occupy only small areas.

#### Class Frequency
Pixel distribution analysis showed that the background class accounts for approximately 94% of all labelled pixels.

```text
Background      116,207,621
Crack             2,215,716
Spalling            347,546
Efflorescence       630,069
Rebar Exposure    4,593,160
```

A loss function optimising only overall pixel accuracy would therefore be
heavily rewarded for predicting background correctly while missing smaller
defect regions.

Weighted Cross Entropy was introduced to increase the contribution of
underrepresented classes, while Dice Loss was added to improve overlap quality
for small and irregular defect masks.

The combined objective produced more balanced optimisation behaviour than
either loss function alone.


---

# Decision 3: Use Configuration-Driven Experiment Management

## Context

Model development required repeated experimentation with learning rates, batch sizes, augmentation settings, checkpoint paths, and dataset locations.

During early development, several training runs produced different results despite using the same codebase. The primary cause was configuration values being modified directly inside training scripts and not consistently tracked between experiments.

A more structured approach was required to ensure experiments remained reproducible and easier to compare.


### Configuration Files

Configuration files centralize experiment settings in a single location and separate runtime parameters from implementation logic.

This approach makes it easier to:

* Reproduce previous experiments
* Compare training runs
* Switch datasets and checkpoints
* Adjust hyperparameters without modifying source code



## Trade-Off Accepted

The implementation accepted a small increase in project complexity in exchange for better maintainability and experiment tracking.

---

# Decision 4: Prioritize End-to-End Completion Over Model Benchmark Chasing

## Context

The available dataset contains sufficient examples to fine-tune an existing segmentation model but is relatively small compared to the scale typically required to train a transformer-based architecture from random initialization.

The project also needed to deliver a working end-to-end system within the assessment timeline.

---

## Alternatives Considered

### Train From Scratch

Training from scratch provides complete control over feature learning and removes dependency on external pretrained weights.

However, it typically requires significantly more training data, longer training schedules, and greater compute resources before achieving competitive performance.

There was a substantial risk that available training time would be spent on model convergence rather than evaluation, deployment, and system integration.

### Fine-Tune a Pretrained Model (Selected)

Fine-tuning leverages features learned from large-scale image datasets and adapts them to infrastructure defect segmentation.

Benefits include:
* Faster convergence
* Improved performance on limited datasets
* More stable training behaviour

## Why This Decision Was Made

The objective of the project was not to develop a new segmentation architecture but to build a complete infrastructure defect segmentation system.

Fine-tuning enabled the model to reach useful performance levels within a practical training budget while still allowing task-specific adaptation to the defect classes.

---

# Decision 5: Use Semantic Segmentation Instead of Object Detection

## Context

The target defects do not follow regular geometric shapes.

Cracks appear as thin branching structures, spalling can form irregular damaged regions, and efflorescence often spreads across surfaces without clear rectangular boundaries.

The intended output requires identifying the exact defect location rather than simply confirming its presence.

---

## Alternatives Considered

### Object Detection

Object detection models such as YOLO and Faster R-CNN provide efficient localisation using bounding boxes.

This approach would simplify inference and reduce computational requirements.

However, bounding boxes often include large areas of non-defective background and do not accurately represent the true shape of surface defects.

As a result, defect size estimation and detailed visual inspection become less reliable.

---

### Semantic Segmentation (Selected)

Semantic segmentation produces a class prediction for every pixel in the image.

This provides:

* Precise defect localisation
* More accurate defect boundaries
* Better visualisation for inspection workflows
* Potential support for defect area estimation

---
## Overall Conderation

The primary value of the system comes from identifying where defects exist and how extensively they affect a structure.

Pixel-level predictions provide substantially more information than rectangular bounding boxes and align more closely with how inspection engineers review infrastructure conditions.

Since segmentation masks were already available in the dataset, the additional annotation cost normally associated with segmentation was not a limiting factor

---

## Trade-Off Accepted

The implementation accepts additional training and inference complexity in exchange for significantly more detailed defect localization.

---

## Revisit Conditions

Object detection may become appropriate if:

* Real-time processing becomes critical
* The objective shifts from defect localisation to defect presence detection

---

# Decision 6: Train at 512×512 Resolution Instead of Native Image Resolution

## Context

The original dataset contains images with varying dimensions and resolutions.

Several defect categories, particularly cracks, occupy only a small portion of the image and can become difficult to detect if images are aggressively downsampled. At the same time, training at native resolution significantly increases GPU memory usage and reduces the number of experiments that can be completed within a fixed time budget.

A resolution needed to be selected that preserved important defect features while remaining practical for training and deployment.

---

## Alternatives Considered

### Lower Resolution (256×256)

Lower resolutions reduce memory consumption and accelerate both training and inference.

However, preliminary inspection suggested that fine crack structures could lose important visual detail after resizing. Thin defects may become fragmented or disappear entirely, making learning more difficult.

### Native Resolution (1024x1024)

Training at native image resolution preserves the maximum amount of visual information and may improve detection of very small defects.

However, larger images increase GPU memory requirements, reduce achievable batch sizes, and significantly slow experimentation cycles. This limits the number of training runs that can be completed and evaluated within the available project timeline.

### 512×512 Resolution (Selected)

A resolution of 512×512 provided a practical balance between image detail and computational efficiency.

Benefits using this resolution:

* Sufficient detail for small defect structures
* Stable memory consumption during training
* Faster experimentation compared to native resolution


### Why This Decision Was Made

The project required multiple training iterations, hyperparameter adjustments, and evaluation runs within limited compute resources.

A 512×512 input size retained enough visual information for crack detection while keeping training time and GPU memory usage manageable.

The selected resolution enabled faster experimentation without introducing the substantial computational cost associated with native-resolution training.



## Trade-Off Accepted

The implementation accepts some loss of fine-grained image detail in exchange for improved training efficiency, faster iteration cycles, and more predictable deployment behaviour.


## Observed Limitations

Some failure cases occur when defects occupy only a very small number of pixels in the original image.

After resizing, these features may become less distinguishable from background texture, contributing to missed detections or incomplete segmentation masks.

## Revisit Conditions

Higher input resolutions should be evaluated if:

* Additional GPU resources become available
* Requirements prioritise accuracy over inference speed
