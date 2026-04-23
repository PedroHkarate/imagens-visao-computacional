from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/dataset.yaml",
    epochs=1,
    imgsz=640
)