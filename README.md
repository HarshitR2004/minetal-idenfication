# Explainable Mineral Classification using EfficientNet-B3

A deep learning based computer vision model for mineral specimen classification using transfer learning and Grad-CAM explainability.

---

## Project Overview

This project performs mineral image classification using a fine-tuned EfficientNet-B3 model trained on RGB mineral specimen images.

The system combines:

* transfer learning
* data augmentation
* class imbalance handling
* explainable AI through Grad-CAM

to build an end-to-end explainable mineral classification pipeline.

The model learns visually discriminative mineral features such as:

* crystal geometry
* metallic reflectance
* texture distribution
* surface morphology

---

## Features

* EfficientNet-B3 transfer learning
* Advanced image augmentation pipeline
* Class-weighted CrossEntropyLoss
* Grad-CAM explainability visualizations
* Confusion matrix and classification metrics
* PyTorch training and evaluation pipeline
* Inference-ready architecture

---

## Model Pipeline

```text
Input Image
   ↓
Preprocessing + Augmentation
   ↓
EfficientNet-B3 Backbone
   ↓
Classification Head
   ↓
Softmax Probabilities
   ↓
Grad-CAM Explainability
```

---

## Dataset

* RGB mineral specimen image dataset
* Multiple mineral classes including:

  * biotite
  * bornite
  * chrysocolla
  * malachite
  * muscovite
  * pyrite
  * quartz

### Dataset Split

* 70% Training
* 15% Validation
* 15% Testing

---

## Training Configuration

| Component     | Value             |
| ------------- | ----------------- |
| Backbone      | EfficientNet-B3   |
| Framework     | PyTorch           |
| Optimizer     | AdamW             |
| Loss Function | CrossEntropyLoss  |
| Scheduler     | ReduceLROnPlateau |
| Image Size    | 224×224           |
| Batch Size    | 32                |

### Augmentations

* Random Resized Crop
* Horizontal Flip
* Rotation
* Color Jitter

---

# Results

## Test Accuracy

```text
Test Accuracy: 90.9%
```

---

## Training Curves

The model demonstrates stable convergence with consistent validation performance and limited overfitting.

![Training Curves](assets\confusion_matrix.png)

---

## Confusion Matrix

The confusion matrix shows strong classification performance across most mineral classes, with minor confusion between visually similar minerals.

![Confusion Matrix](assets/confusion_matrix.png)

---

# Grad-CAM Explainability

Grad-CAM visualizations indicate that the model focuses on mineral-specific visual regions such as:

* reflective crystal structures
* texture distributions
* surface morphology
* geometric mineral patterns

rather than relying on background artifacts.

---

## Pyrite Grad-CAM

The model strongly attends to reflective crystal facets and metallic geometric structures while classifying pyrite.

![Pyrite GradCAM](assets\gradcam_pyrite.png)

---

## Chrysocolla Grad-CAM

The model focuses on diffuse texture regions and color concentration patterns for chrysocolla classification.

![Chrysocolla GradCAM](assets\gradcam_chrysocolla.png)

---

# Installation

Clone the repository:

```bash
git clone https://github.com/HarshitR2004/minetal-idenfication
cd mineral-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Notebook

Open and run:

```text
notebooks/mineral_classification.ipynb
```

---

# Inference

The notebook includes:

* single image prediction
* confidence scores
* Grad-CAM visualization

Streamlit app (`app.py`) — displays predictions and Grad-CAM overlays directly in the browser. The app shows results inline and does not save output images or figures to disk by default (no files are written unless you explicitly modify the code to do so).

Future standalone inference scripts and deployment support can be added easily.

---

