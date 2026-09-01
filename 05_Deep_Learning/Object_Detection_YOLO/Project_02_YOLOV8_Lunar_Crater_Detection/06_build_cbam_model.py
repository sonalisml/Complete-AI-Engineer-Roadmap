from ultralytics import YOLO
from ultralytics.nn import tasks

from attention.cbam import CBAM

# Register CBAM
tasks.CBAM = CBAM

print("CBAM registered successfully")


# Build modified YOLOv8n
model = YOLO("yolov8n_cbam.yaml")

print("\nModified YOLOv8n created successfully")


# Load pretrained YOLOv8n weights
model.load("yolov8n.pt")

print("Pretrained YOLOv8n weights loaded")


# Display model
print(model.model)