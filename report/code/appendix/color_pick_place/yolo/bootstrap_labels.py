#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bootstrap YOLO labels for the raw block images using the existing color detector.

This reuses ``ColorBlockDetector.build_color_mask`` (a staticmethod, so no camera
is opened) plus the same contour/area/aspect-ratio filtering as
``detect.detect_blocks``. For every raw frame it writes:

    labels/<stem>.txt        one YOLO line per block: ``cls cx cy w h`` (normalized)
    review_overlays/<stem>.jpg   the same frame with labeled boxes drawn

Classes are colors (matches the chosen "color == class" scheme):
    red=0, green=1, blue=2     (yellow is intentionally excluded)

The labels are *pseudo* labels meant to be human-reviewed afterwards. Images that
get zero detections are reported separately because those most likely need a box
added by hand.

Usage:
    python bootstrap_labels.py
    python bootstrap_labels.py --min-area 800 --src ../../yolo_raw --out .
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Make the parent package (color_pick_place) importable so we can reuse detect.py.
_THIS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _THIS_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from detect import ColorBlockDetector, DRAW_COLORS  # noqa: E402


# Only red/green/blue are sorted downstream; keep the id order stable and explicit.
CLASS_ORDER = ["red", "green", "blue"]
CLASS_ID = {name: index for index, name in enumerate(CLASS_ORDER)}


def _interior_median_v(hsv: np.ndarray, contour: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Median HSV value (brightness) of the pixels inside one contour."""
    x, y, w, h = box
    roi = hsv[y:y + h, x:x + w]
    cmask = np.zeros((h, w), np.uint8)
    cv2.drawContours(cmask, [contour - [x, y]], -1, 255, -1)
    pixels = roi[cmask > 0]
    if pixels.size == 0:
        return 255.0
    return float(np.median(pixels[:, 2]))


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)


def _nms_per_class(boxes: list[dict], iou_thr: float) -> list[dict]:
    """Drop the smaller of two same-class boxes whose IoU exceeds iou_thr."""
    kept: list[dict] = []
    for color in CLASS_ORDER:
        group = sorted([b for b in boxes if b["type"] == color], key=lambda b: -b["box"][2] * b["box"][3])
        survivors: list[dict] = []
        for cand in group:
            if all(_iou(cand["box"], s["box"]) <= iou_thr for s in survivors):
                survivors.append(cand)
        kept.extend(survivors)
    return kept


def detect_label_boxes(
    image: np.ndarray,
    min_area: float,
    aspect_min: float,
    aspect_max: float,
    blue_min_v: float = 140.0,
    nms_iou: float = 0.5,
) -> tuple[list[dict], int]:
    """Return axis-aligned boxes for every accepted block, mirroring detect_blocks.

    Each entry: {"type", "box": (x, y, w, h), "angle"} in pixel units. Blue
    candidates whose interior is darker than ``blue_min_v`` are rejected (these
    are checkerboard black squares, not real cyan blocks); the count of such
    rejections is returned so callers can flag cluttered/checkerboard frames.
    Same-class duplicates are then suppressed by IoU NMS.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    boxes: list[dict] = []
    removed_dark_blue = 0
    for color_name in CLASS_ORDER:
        mask = ColorBlockDetector.build_color_mask(image, color_name)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= min_area:
                continue

            (_, _), (rw, rh), angle = cv2.minAreaRect(contour)
            if rw <= 0 or rh <= 0:
                continue
            aspect_ratio = max(rw, rh) / min(rw, rh)
            if not aspect_min <= aspect_ratio <= aspect_max:
                continue

            box = cv2.boundingRect(contour)
            if color_name == "blue" and _interior_median_v(hsv, contour, box) < blue_min_v:
                removed_dark_blue += 1
                continue
            boxes.append({"type": color_name, "box": box, "angle": float(angle)})

    boxes = _nms_per_class(boxes, nms_iou)
    return boxes, removed_dark_blue


def to_yolo_line(box: dict, img_w: int, img_h: int) -> str:
    x, y, w, h = box["box"]
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    cid = CLASS_ID[box["type"]]
    return f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def draw_overlay(image: np.ndarray, boxes: list[dict]) -> np.ndarray:
    display = image.copy()
    for box in boxes:
        x, y, w, h = box["box"]
        color = DRAW_COLORS.get(box["type"], (0, 255, 0))
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            display,
            f"{box['type']} {box['angle']:.0f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    return display


def run(args: argparse.Namespace) -> None:
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    images_dir = out / "images"
    labels_dir = out / "labels"
    overlays_dir = out / "review_overlays"
    labels_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the symlinked images dir if present, else read straight from src.
    search_dir = images_dir if images_dir.is_dir() and any(images_dir.glob("*.png")) else src
    image_paths = sorted(search_dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"no .png images found in {search_dir}")

    class_counts = {name: 0 for name in CLASS_ORDER}
    empty_images: list[str] = []
    flagged_images: list[str] = []
    per_image: list[dict] = []
    total_boxes = 0

    for index, path in enumerate(image_paths, start=1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"[skip] unreadable: {path.name}", flush=True)
            continue
        img_h, img_w = image.shape[:2]

        boxes, removed_dark_blue = detect_label_boxes(
            image, args.min_area, args.aspect_min, args.aspect_max,
            blue_min_v=args.blue_min_v, nms_iou=args.nms_iou,
        )
        lines = [to_yolo_line(box, img_w, img_h) for box in boxes]
        (labels_dir / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        overlay = draw_overlay(image, boxes)
        if removed_dark_blue > 0:
            cv2.putText(overlay, "REVIEW: checkerboard/clutter", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        cv2.imwrite(str(overlays_dir / f"{path.stem}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])

        for box in boxes:
            class_counts[box["type"]] += 1
        total_boxes += len(boxes)
        if not boxes:
            empty_images.append(path.name)
        if removed_dark_blue > 0:
            flagged_images.append(path.name)
        per_image.append({"image": path.name, "n_boxes": len(boxes),
                          "removed_dark_blue": removed_dark_blue,
                          "by_color": {c: sum(1 for b in boxes if b["type"] == c) for c in CLASS_ORDER}})

        if index % 25 == 0 or index == len(image_paths):
            print(f"[bootstrap] {index}/{len(image_paths)} processed", flush=True)

    summary = {
        "images_total": len(image_paths),
        "boxes_total": total_boxes,
        "class_counts": class_counts,
        "images_with_zero_boxes": empty_images,
        "flagged_for_review": flagged_images,
        "blue_min_v": args.blue_min_v,
        "nms_iou": args.nms_iou,
        "min_area": args.min_area,
        "aspect_range": [args.aspect_min, args.aspect_max],
    }
    (out / "bootstrap_summary.json").write_text(
        json.dumps({**summary, "per_image": per_image}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "review_priority.txt").write_text(
        "# Images flagged for CLOSE review (had checkerboard/clutter false positives removed):\n"
        + "\n".join(flagged_images) + "\n",
        encoding="utf-8",
    )

    print("\n===== bootstrap summary =====")
    print(f"images              : {summary['images_total']}")
    print(f"total boxes         : {summary['boxes_total']}")
    print(f"per-class boxes     : {class_counts}")
    print(f"zero-box images     : {len(empty_images)}  (most likely need a manual box)")
    print(f"flagged for review  : {len(flagged_images)}  (checkerboard/clutter -> review closely)")
    print(f"labels  -> {labels_dir}")
    print(f"overlays-> {overlays_dir}  (flagged frames carry an orange banner)")
    print(f"review  -> {out / 'review_priority.txt'}")
    print(f"summary -> {out / 'bootstrap_summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap YOLO labels from the existing color detector")
    parser.add_argument("--src", default=str(_PKG_DIR.parent / "yolo_raw"), help="raw image source dir")
    parser.add_argument("--out", default=str(_THIS_DIR), help="yolo working dir (holds images/labels/overlays)")
    parser.add_argument("--min-area", type=float, default=500.0)
    parser.add_argument("--aspect-min", type=float, default=0.65)
    parser.add_argument("--aspect-max", type=float, default=1.55)
    parser.add_argument("--blue-min-v", type=float, default=140.0,
                        help="reject blue candidates whose interior median V is below this (checkerboard black squares)")
    parser.add_argument("--nms-iou", type=float, default=0.5,
                        help="suppress same-class boxes overlapping more than this IoU")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
