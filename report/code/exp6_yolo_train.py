from ultralytics import YOLO


model = YOLO("yolo11n.pt")
model.train(
    data="data.yaml",
    epochs=120,
    imgsz=960,
    batch=16,
    device=0,
    seed=0,
    patience=40,
    hsv_h=0.0,       # color is the class; do not shift hue
    hsv_s=0.5,
    hsv_v=0.4,
    degrees=180.0,
    fliplr=0.5,
    flipud=0.5,
    translate=0.1,
    scale=0.5,
    mosaic=1.0,
    project="runs",
    name="blocks",
    exist_ok=True,
)
