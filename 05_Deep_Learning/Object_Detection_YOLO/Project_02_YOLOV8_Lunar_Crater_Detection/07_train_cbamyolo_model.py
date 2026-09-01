from ultralytics import YOLO
from ultralytics.nn import tasks

from attention.cbam import CBAM


# ============================================
# Register CBAM
# ============================================

tasks.CBAM = CBAM
print("CBAM registered")


# ============================================
# Create modified YOLOv8n
# ============================================

model = YOLO("yolov8n_cbam.yaml")

print("YOLOv8n + CBAM architecture created")


# ============================================
# Load pretrained YOLOv8n weights
# ============================================

model.load("yolov8n.pt")

print("Pretrained weights loaded")


# ============================================
# Train
# ============================================

results = model.train(

    data= "dataset/data.yaml",

    epochs=1,

    imgsz=640,

    batch=4,

    patience=10,

    device="cpu",

    project="runs",

    name="cbam"

)

print("\nYOLOv8n + CBAM training completed")