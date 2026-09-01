#YOLO model loaded
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
print("yolo model loaded succsessfully")

results = model.train(
    data = "dataset/data.yaml",
    epochs = 2,
    imgsz=640,
    batch =16,
    patience = 10,
    device = "cpu",
    project ="runs",
    name = "baseline",
)
print("model craeted succesfully")
