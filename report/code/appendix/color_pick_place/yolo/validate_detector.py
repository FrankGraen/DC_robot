#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline check that the YOLO pipeline preserves the legacy pixel centers.

For each image it runs:
  * the trained YOLO + ROI-minAreaRect refinement (the new pipeline), and
  * the legacy full-frame color detector (detect.py logic),
then matches detections by nearest center and reports the pixel delta. The center
feeds convert.py -> robot xy, so small deltas mean grasp accuracy is preserved.
No camera is opened; runs straight on saved frames.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _THIS_DIR.parent
for p in (str(_THIS_DIR), str(_PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from detect import ColorBlockDetector, DRAW_COLORS  # noqa: E402
from yolo_detector import refine_center_angle  # noqa: E402

CLASS_ORDER = ["red", "green", "blue"]


def color_detect(image: np.ndarray, min_area: float = 500.0) -> list[dict]:
    """Legacy detect.py centers/angles (full-frame color mask + minAreaRect)."""
    out: list[dict] = []
    for color in CLASS_ORDER:
        mask = ColorBlockDetector.build_color_mask(image, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) <= min_area:
                continue
            (cx, cy), (rw, rh), angle = cv2.minAreaRect(c)
            if rw <= 0 or rh <= 0 or not 0.65 <= max(rw, rh) / min(rw, rh) <= 1.55:
                continue
            out.append({"type": color, "center": (int(round(cx)), int(round(cy))), "angle": float(angle)})
    return out


def yolo_detect(model, image: np.ndarray, conf: float, iou: float, min_area: float) -> list[dict]:
    res = model.predict(image, conf=conf, iou=iou, device="cpu", verbose=False)[0]
    out: list[dict] = []
    for b in res.boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b.xyxy[0].tolist()]
        color = str(model.names[int(b.cls[0])])
        center, angle = refine_center_angle(image, color, (x1, y1, x2, y2), min_area=min_area)
        out.append({"type": color, "center": center, "angle": angle, "box": (x1, y1, x2, y2)})
    return out


def match(yolo: list[dict], color: list[dict], max_dist: float) -> tuple[list[float], int, int]:
    """Greedy nearest-center match within same class. Returns deltas, yolo_only, color_only."""
    used = [False] * len(color)
    deltas: list[float] = []
    yolo_only = 0
    for y in yolo:
        best, best_d = -1, 1e9
        for i, c in enumerate(color):
            if used[i] or c["type"] != y["type"]:
                continue
            d = float(np.hypot(y["center"][0] - c["center"][0], y["center"][1] - c["center"][1]))
            if d < best_d:
                best, best_d = i, d
        if best >= 0 and best_d <= max_dist:
            used[best] = True
            deltas.append(best_d)
        else:
            yolo_only += 1
    color_only = used.count(False)
    return deltas, yolo_only, color_only


def run(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(args.weights)
    images = sorted(Path(args.images).glob("*.png"))
    if args.n:
        images = images[: args.n]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_deltas: list[float] = []
    tot_yolo_only = tot_color_only = tot_matched = 0
    per_class = {c: {"yolo": 0, "color": 0} for c in CLASS_ORDER}

    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        yd = yolo_detect(model, img, args.conf, args.iou, args.min_area)
        cd = color_detect(img, args.min_area_color)
        for d in yd:
            per_class[d["type"]]["yolo"] += 1
        for d in cd:
            per_class[d["type"]]["color"] += 1

        deltas, yo, co = match(yd, cd, args.max_dist)
        all_deltas += deltas
        tot_matched += len(deltas)
        tot_yolo_only += yo
        tot_color_only += co

        disp = img.copy()
        for d in yd:
            x1, y1, x2, y2 = d["box"]
            col = DRAW_COLORS.get(d["type"], (0, 255, 0))
            cv2.rectangle(disp, (x1, y1), (x2, y2), col, 2)
            cv2.drawMarker(disp, d["center"], col, cv2.MARKER_CROSS, 18, 2)
        for d in cd:
            cv2.drawMarker(disp, d["center"], (255, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 1)
        cv2.imwrite(str(out_dir / f"{path.stem}.jpg"), disp, [cv2.IMWRITE_JPEG_QUALITY, 85])

    arr = np.array(all_deltas) if all_deltas else np.array([0.0])
    print("\n===== detector parity (YOLO vs legacy color) =====")
    print(f"images               : {len(images)}")
    print(f"matched detections   : {tot_matched}")
    print(f"center delta (px)    : median={np.median(arr):.2f}  p95={np.percentile(arr, 95):.2f}  max={arr.max():.2f}")
    print(f"YOLO-only (color miss/new): {tot_yolo_only}")
    print(f"color-only (YOLO miss/color-FP): {tot_color_only}")
    print(f"per-class detections : {per_class}")
    print(f"overlays -> {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare YOLO vs legacy color detector centers")
    parser.add_argument("--weights", default=str(_THIS_DIR / "runs" / "blocks" / "weights" / "best.pt"))
    parser.add_argument("--images", default=str(_THIS_DIR / "dataset" / "images" / "val"))
    parser.add_argument("--out", default=str(_THIS_DIR / "parity_overlays"))
    parser.add_argument("--n", type=int, default=0, help="limit number of images (0 = all)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--min-area", type=float, default=200.0, help="ROI contour floor for YOLO angle refine")
    parser.add_argument("--min-area-color", type=float, default=500.0, help="legacy color detector area floor")
    parser.add_argument("--max-dist", type=float, default=25.0, help="px to count two centers as the same block")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
