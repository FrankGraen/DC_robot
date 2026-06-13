import cv2
import numpy as np

COLOR_RANGES = {
    "red": [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [179, 255, 255])],
    "green": [([35, 50, 50], [85, 255, 255])],
    "blue": [([78, 70, 35], [135, 255, 255])],
    "yellow": [([20, 100, 100], [30, 255, 255])],
}


def build_color_mask(image, color_name):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros_like(hsv[:, :, 0])
    for lower, upper in COLOR_RANGES[color_name]:
        mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_blocks(image, color_name, min_area=500.0):
    mask = build_color_mask(image, color_name)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= min_area:
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), angle = rect
        if rw <= 0 or rh <= 0:
            continue
        detections.append({"type": color_name, "center": (cx, cy), "angle": angle})
    return detections
