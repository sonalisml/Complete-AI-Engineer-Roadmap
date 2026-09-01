from ultralytics import YOLO

# Load best trained model
model = YOLO("runs/baseline/weights/best.pt")

print("Best baseline model loaded successfully")

# Evaluate on TEST set
metrics = model.val(
    data="Lunar Crater Detection 2.v1i.yolov8/data.yaml",
    split="test"
)

print("\nBaseline test evaluation completed")

