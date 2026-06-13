from ultralytics import YOLO
import cv2


class YoloBlockDetector:
    def __init__(self, weights, conf=0.25, iou=0.5, device="cpu"):
        self.model = YOLO(weights)
        self.names = self.model.names
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect_blocks(self, image):
        result = self.model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        detected = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            color = str(self.names[int(box.cls[0])])
            conf = float(box.conf[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detected.append(
                {"type": color, "center": (cx, cy), "angle": 0.0, "conf": conf}
            )
        return detected
