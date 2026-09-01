from ultralytics import YOLO
model = YOLO("yolov8n.pt")

print("\n========YOLO backbone =======")
for layer in model.model.yaml["backbone"]:
    print(layer)

print("\n========== YOLOv8 HEAD ==========\n")
for layer in model.model.yaml["head"]:
    print(layer)
print(model.model)