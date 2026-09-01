from ultralytics import YOLO

model = YOLO("runs/detect/runs/bolt_washer_detection/weights/best.pt")
print("Best model loaded succsessfully")

results = model.predict(
    source = "tt.jpg",
    conf = 0.25,
    save = True
)
print("tested succesfully")

for result in results:
    boxes = result.boxes
    for box in boxes:
        #Bounding box cordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        #Confidence
        confidence = box.conf[0].item()
        #Class_ids
        class_id = int(box.cls[0].item())
        #Class name
        class_name = model.names[class_id]

        print("\n-----------------------------")
        print("Object      :", class_name)
        print("Confidence  :", confidence)
        print("Coordinates :")
        print("x1 =", x1)
        print("y1 =", y1)
        print("x2 =", x2)
        print("y2 =", y2)
        print("\nPrediction completed successfully")

# Calculate center of bounding box
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2

print("Center:")
print("cx =", center_x)
print("cy =", center_y)
