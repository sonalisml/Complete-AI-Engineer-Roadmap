# Lunar Crater Detection using YOLOv8 + CBAM

## Overview

This project detects **lunar craters** using YOLOv8 and explores how adding a **CBAM (Convolutional Block Attention Module)** to the YOLOv8 backbone affects object detection.

## Objective

* Train a standard **YOLOv8n baseline**
* Add **CBAM attention** to the YOLOv8 backbone
* Train the modified model
* Compare the baseline and attention-enhanced models

## Dataset

Annotated lunar crater dataset in YOLO format.

* **Class:** `crater`
* **Format:** YOLO bounding-box annotations
* **Splits:** Train, Validation, Test

## Model Architecture

### Baseline

```text
Input → YOLOv8n Backbone → Neck → Detection Head
```

### Modified

```text
Input → YOLOv8n Backbone → CBAM → Neck → Detection Head
```

CBAM combines:

* Channel Attention
* Spatial Attention

The attention module was implemented as a custom PyTorch module and integrated into the YOLOv8 architecture.

## Technologies

* Python
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* NumPy

## Project Workflow

```text
Dataset
   ↓
Dataset Analysis
   ↓
YOLOv8n Baseline
   ↓
Evaluation
   ↓
CBAM Integration
   ↓
YOLOv8n + CBAM
   ↓
Training & Evaluation
   ↓
Model Comparison
```

## Key Learning

This project demonstrates how to:

* Prepare and analyze a YOLO dataset
* Train and evaluate YOLOv8
* Understand the YOLO backbone
* Create a custom attention module
* Register a custom module with Ultralytics
* Modify a pretrained YOLO architecture
* Transfer pretrained weights
* Compare baseline and modified models

## Future Scope

* Experiment with other attention mechanisms
* Improve small-crater detection
* Optimize inference speed
* Deploy the trained detector for real-time applications.
