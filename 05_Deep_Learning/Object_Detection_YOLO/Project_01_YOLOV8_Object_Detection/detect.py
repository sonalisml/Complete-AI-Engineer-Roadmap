#Warnings
#=======================
import warnings
warnings.filterwarnings("ignore")
#=======================
#IMPORT
#=======================
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#=======================
#IMPORT YOLO MODEL
#=======================
from ultralytics import YOLO
print("YOLOV8 Environment Ready")
# ==========================================================
# Load YOLOv8 Model
# ==========================================================
model = YOLO("yolov8n.pt")
print("YOLOv8 Model Loaded Successfully")
# ==========================================================
# Dataset Path
# ==========================================================
dataset_path = "dataset_yolo"
print("Dataset Path :", dataset_path)
# ==========================================================
# Read data.yaml
# ==========================================================
results = model.train(

    data=dataset_path,

    epochs=50,

    imgsz=640,

    batch=16,

    patience=10,

    device="cpu",

    project="runs",

    name="bolt_washer_detection"

)
print("\nTraining completed successfully")