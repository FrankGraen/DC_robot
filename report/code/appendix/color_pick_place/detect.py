#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
颜色方块检测类。

初始化时打开摄像头并读取与 blue_square_pose.py 一致的相机标定文件。
调用 recognize() 后实时显示识别结果，按 s 保存原始图像，按 a 关闭窗口并返回结果列表。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from convert import camera_observation_to_robot_pose


COLOR_RANGES = {
    "red": [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [179, 255, 255])],
    "green": [([35, 50, 50], [85, 255, 255])],
    "blue": [([78, 70, 35], [135, 255, 255])],
    "yellow": [([20, 100, 100], [30, 255, 255])],
}

DRAW_COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}


class ColorBlockDetector:
    def __init__(
        self,
        camera_index: int = 0,
        calib_path: str | Path = "camera_calibration_result.json",
        width: int | None = 1920,
        height: int | None = 1080,
        min_area: float = 500.0,
        window_name: str = "Detection Results",
        display_width: int = 960,
    ) -> None:
        self.camera_index = camera_index
        self.calib_path = Path(calib_path)
        self.min_area = min_area
        self.window_name = window_name
        self.display_width = display_width
        self.camera_matrix, self.dist_coeffs = self.load_calibration(self.calib_path)

        self.cap = cv2.VideoCapture(camera_index)
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"摄像头打开失败：{camera_index}")

    @staticmethod
    def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
        return camera_matrix, dist_coeffs

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    @staticmethod
    def enhance_blue_visibility(image: np.ndarray) -> np.ndarray:
        balanced = image.astype(np.float32)
        channel_mean = balanced.reshape(-1, 3).mean(axis=0)
        gray_mean = float(channel_mean.mean())
        balanced *= gray_mean / np.maximum(channel_mean, 1.0)
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s.astype(np.float32) * 1.45, 0, 255).astype(np.uint8)
        v = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(v)
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    @staticmethod
    def build_color_mask(image: np.ndarray, color_name: str) -> np.ndarray:
        detect_image = ColorBlockDetector.enhance_blue_visibility(image) if color_name == "blue" else image
        hsv = cv2.cvtColor(detect_image, cv2.COLOR_BGR2HSV)
        mask = np.zeros_like(hsv[:, :, 0])
        for lower, upper in COLOR_RANGES[color_name]:
            lower_array = np.array(lower, dtype=np.uint8)
            upper_array = np.array(upper, dtype=np.uint8)
            color_mask = cv2.inRange(hsv, lower_array, upper_array)
            mask = cv2.bitwise_or(mask, color_mask)

        if color_name == "blue":
            b, g, r = cv2.split(image)
            h, s, v = cv2.split(hsv)
            channel_mask = (
                (h >= 75)
                & (h <= 140)
                & (s >= 65)
                & (v >= 35)
                & (b >= 45)
                & (b.astype(np.int16) > r.astype(np.int16) + 45)
                & (b.astype(np.int16) > g.astype(np.int16) + 8)
            ).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, channel_mask)
        elif color_name == "green":
            b, g, r = cv2.split(image)
            channel_mask = (
                (g >= 45)
                & (g.astype(np.int16) > r.astype(np.int16) + 25)
                & (g.astype(np.int16) > b.astype(np.int16) + 5)
            ).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, channel_mask)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def detect_blocks(self, image: np.ndarray) -> tuple[list[dict], np.ndarray]:
        display = image.copy()
        detected_blocks = []

        for color_name in COLOR_RANGES:
            mask = self.build_color_mask(image, color_name)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= self.min_area:
                    continue

                rect = cv2.minAreaRect(contour)
                (cx, cy), (rw, rh), angle = rect
                if rw <= 0 or rh <= 0:
                    continue

                aspect_ratio = max(rw, rh) / min(rw, rh)
                if not 0.65 <= aspect_ratio <= 1.55:
                    continue

                center = (int(round(cx)), int(round(cy)))
                detected_blocks.append(
                    {
                        "type": color_name,
                        "center": center,
                        "angle": float(angle),
                    }
                )

                draw_color = DRAW_COLORS.get(color_name, (0, 255, 0))
                box = cv2.boxPoints(rect).astype(np.int32)
                x, y, _, _ = cv2.boundingRect(contour)
                cv2.drawContours(display, [box], 0, draw_color, 2)
                cv2.circle(display, center, 4, draw_color, -1)
                cv2.putText(
                    display,
                    f"{color_name} {float(angle):.1f}",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    draw_color,
                    2,
                )

        return detected_blocks, display

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
                robot_xyz = (x_robot, y_robot, z_robot)
                robot_pose = (x_robot, y_robot, z_robot, pitch, yaw, roll)
                detection["robot_xyz"] = robot_xyz
                detection["robot_pose"] = robot_pose
                text = f"R:({x_robot:.1f},{y_robot:.1f},{z_robot:.1f})"

            x_text = int(center[0]) + 10
            y_text = int(center[1]) + 26
            cv2.putText(
                display,
                text,
                (x_text, y_text),
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

    def __enter__(self) -> "ColorBlockDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


if __name__ == "__main__":
    with ColorBlockDetector() as detector:
        results = detector.recognize()
    print(results)
