#=====================
#=====================

from ultralytics import YOLO


model = YOLO("runs/detect/runs/bolt_washer_detection/weights/best.pt")
print("Best model loaded successfully")


metrics = model.val(
    
    data ="dataset_yolo/data.yaml",
    split= "test"
)

print("test completed")