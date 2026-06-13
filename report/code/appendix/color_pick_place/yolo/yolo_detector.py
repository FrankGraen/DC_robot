#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO-based block detector, drop-in compatible with ColorBlockDetector.

YOLO does the robust "where + which color" job (replacing the brittle full-frame
HSV thresholding). The in-plane angle and the precise center are then recovered
*exactly the way the old pipeline did it* -- by running the existing color mask
inside the detected ROI and taking cv2.minAreaRect. That keeps the ``angle`` /
``center`` convention identical, so convert.py (angle_gain=-1, square wrap) and
everything downstream need no retuning.

Interface contract consumed by main.py / auto_calib.py:
    recognize() -> [{"type": str, "center": (u, v), "angle": float}, ...]
    release()
Plus the same interactive window: press 'a' to accept, 's' to save the raw frame.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Make the parent package (color_pick_place) importable for the shared color logic.
_THIS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _THIS_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from detect import ColorBlockDetector, DRAW_COLORS  # noqa: E402
from convert import camera_observation_to_robot_pose  # noqa: E402


DEFAULT_WEIGHTS = str(_THIS_DIR / "runs" / "blocks" / "weights" / "best.pt")


def refine_center_angle(
    image: np.ndarray,
    color: str,
    box: tuple[int, int, int, int],
    min_area: float = 200.0,
    roi_pad: int = 12,
) -> tuple[tuple[int, int], float]:
    """Recover (center, angle) inside a YOLO box using the legacy color mask.

    Returns the minAreaRect center+angle in full-image pixel coordinates,
    matching exactly what ColorBlockDetector.detect_blocks would have produced.
    Falls back to the box center / angle 0 if the mask yields no usable contour.
    """
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    x1p, y1p = max(0, x1 - roi_pad), max(0, y1 - roi_pad)
    x2p, y2p = min(w, x2 + roi_pad), min(h, y2 + roi_pad)
    box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    roi = image[y1p:y2p, x1p:x2p]
    if roi.size == 0 or color not in DRAW_COLORS:
        return box_center, 0.0

    mask = ColorBlockDetector.build_color_mask(roi, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not contours:
        return box_center, 0.0

    (cx, cy), (rw, rh), angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    if rw <= 0 or rh <= 0:
        return box_center, 0.0
    return (int(round(cx)) + x1p, int(round(cy)) + y1p), float(angle)


class YoloBlockDetector:
    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        camera_index: int = 0,
        calib_path: str | Path = "camera_calibration_result.json",
        width: int | None = 1920,
        height: int | None = 1080,
        min_area: float = 200.0,
        conf: float = 0.25,
        iou: float = 0.5,
        device: str = "cpu",
        roi_pad: int = 12,
        window_name: str = "YOLO Detection Results",
        display_width: int = 960,
    ) -> None:
        from ultralytics import YOLO

        weights = str(weights)
        if not Path(weights).is_file():
            raise FileNotFoundError(f"YOLO weights not found: {weights} (train first with train.py)")
        self.model = YOLO(weights)
        self.names = self.model.names
        self.calib_path = Path(calib_path)
        self.min_area = min_area
        self.conf = conf
        self.iou = iou
        self.device = device
        self.roi_pad = roi_pad
        self.window_name = window_name
        self.display_width = display_width

        self.cap = cv2.VideoCapture(camera_index)
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"摄像头打开失败：{camera_index}")

    # ---- detection -------------------------------------------------------

    def detect_blocks(self, image: np.ndarray) -> tuple[list[dict], np.ndarray]:
        display = image.copy()
        results = self.model.predict(image, conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
        detected: list[dict] = []

        for b in results.boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in b.xyxy[0].tolist()]
            color = str(self.names[int(b.cls[0])])
            conf = float(b.conf[0])
            center, angle = refine_center_angle(
                image, color, (x1, y1, x2, y2), min_area=self.min_area, roi_pad=self.roi_pad
            )
            detected.append({"type": color, "center": center, "angle": angle})

            draw_color = DRAW_COLORS.get(color, (0, 255, 0))
            cv2.rectangle(display, (x1, y1), (x2, y2), draw_color, 2)
            cv2.circle(display, center, 4, draw_color, -1)
            cv2.putText(
                display,
                f"{color} {conf:.2f} {angle:.0f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                draw_color,
                2,
            )
        return detected, display

    def add_robot_coordinates(self, detections: list[dict], display: np.ndarray) -> None:
        for index, detection in enumerate(detections):
            draw_color = DRAW_COLORS.get(str(detection.get("type")), (0, 255, 0))
            center = detection.get("center", (20, 40 + index * 28))
            try:
                x_robot, y_robot, z_robot, pitch, yaw, roll = camera_observation_to_robot_pose(
                    detection,
                    camera_calib_path=self.calib_path,
                )
            except Exception as exc:
                detection["robot_error"] = str(exc)
                text = "robot: convert failed"
            else:
                detection["robot_xyz"] = (x_robot, y_robot, z_robot)
                detection["robot_pose"] = (x_robot, y_robot, z_robot, pitch, yaw, roll)
                text = f"R:({x_robot:.1f},{y_robot:.1f},{z_robot:.1f})"
            cv2.putText(
                display,
                text,
                (int(center[0]) + 10, int(center[1]) + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                draw_color,
                2,
            )

    def show(self, display: np.ndarray) -> None:
        shown = display
        if self.display_width > 0 and display.shape[1] > self.display_width:
            scale = self.display_width / float(display.shape[1])
            shown = cv2.resize(
                display,
                (self.display_width, int(round(display.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imshow(self.window_name, shown)

    # ---- interactive loop (same UX as ColorBlockDetector) ----------------

    def recognize(self) -> list[dict]:
        last_results: list[dict] = []
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    continue
                raw_frame = frame.copy()
                last_results, display = self.detect_blocks(raw_frame)
                self.add_robot_coordinates(last_results, display)
                self.show(display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    filename = f"raw_image_{datetime.now():%Y%m%d_%H%M%S}.png"
                    cv2.imwrite(filename, raw_frame)
                    print(f"已保存原始图像：{filename}")
                if key == ord("a"):
                    return last_results
        finally:
            self.close_window()

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
        self.close_window()

    def close_window(self) -> None:
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    def __enter__(self) -> "YoloBlockDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


if __name__ == "__main__":
    with YoloBlockDetector() as detector:
        print(detector.recognize())
