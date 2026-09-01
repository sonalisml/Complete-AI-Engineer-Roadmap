from ultralytics import YOLO

baseline = YOLO("yolov8.pt")
baseline_results = baseline.val(
    data = "dataset/data.yaml",
    imgsz = 640,
    device = "cpu"
)

cbam_model = YOLO("runs/cbam-7/weights/best.pt")
cbam_results = cbam_model.val(
    data = "dataset/data.yaml",
    imgsz = 640,
    device = "cpu"
)
print("\n===== BASELINE YOLOv8n =====")
print("Precision :", baseline_results.box.mp)
print("Recall    :", baseline_results.box.mr)
print("mAP50     :", baseline_results.box.map50)
print("mAP50-95  :", baseline_results.box.map)


print("\n===== YOLOv8n + CBAM =====")
print("Precision :", cbam_results.box.mp)
print("Recall    :", cbam_results.box.mr)
print("mAP50     :", cbam_results.box.map50)
print("mAP50-95  :", cbam_results.box.map)