#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manual camera-to-robot calibration point collector.

This script never imports or commands the robot. It only opens the camera,
detects one color block, and asks the operator to type the known robot-frame
coordinates for that block. The output is compatible with calc_extrinsics.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from detect import ColorBlockDetector


def choose_detection(results: list[dict], color: str | None) -> dict:
    if color:
        results = [item for item in results if item.get("type") == color]
    if not results:
        label = color if color else "any color"
        raise RuntimeError(f"no detection found for {label}")
    if len(results) > 1:
        print(f"[warn] {len(results)} detections found; using the largest/first visible result")
    return results[0]


def parse_xyz(text: str) -> list[float] | None:
    text = text.strip()
    if text.lower() in {"q", "quit", "exit"}:
        return None
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("please enter exactly three numbers: x y z")
    return [float(v) for v in parts]


def load_existing(path: Path, append: bool) -> list[dict]:
    if not append or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("records") or [])


def estimate_affine(records: list[dict]) -> tuple[list[list[float]] | None, list[int] | None, float | None]:
    if len(records) < 3:
        return None, None, None
    camera_points = np.array([record["camera_center"] for record in records], dtype=np.float64)
    robot_points = np.array([record["robot_xy"] for record in records], dtype=np.float64)
    matrix, inliers = cv2.estimateAffine2D(camera_points, robot_points)
    if matrix is None:
        return None, None, None
    predicted = cv2.transform(camera_points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(predicted - robot_points, axis=1)
    rms_error = float(np.sqrt(np.mean(errors * errors)))
    return matrix.tolist(), inliers.reshape(-1).astype(int).tolist(), rms_error


def save_result(path: Path, records: list[dict]) -> None:
    matrix, inliers, rms_error = estimate_affine(records)
    output = {
        "camera_to_robot_affine_2x3": matrix,
        "rms_error_mm": rms_error,
        "records": records,
        "inliers": inliers,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    records = load_existing(output, args.append)
    detector = ColorBlockDetector(
        camera_index=args.camera,
        calib_path=args.camera_calib,
        width=args.width,
        height=args.height,
        min_area=args.min_area,
    )

    try:
        while args.count <= 0 or len(records) < args.count:
            next_index = len(records) + 1
            print()
            print(f"[point {next_index}] put the block at a known robot coordinate.")
            print("[input] type robot xyz in mm, for example: 190 -120 82")
            text = input("[robot xyz] > ")
            xyz = parse_xyz(text)
            if xyz is None:
                break

            print("[vision] camera window opened; press 'a' to accept detection, 's' to save raw image")
            detections = detector.recognize()
            detection = choose_detection(detections, args.color)
            center = detection["center"]
            record = {
                "index": next_index,
                "type": detection["type"],
                "robot_xyz": xyz,
                "robot_xy": xyz[:2],
                "camera_center": [float(center[0]), float(center[1])],
                "angle": float(detection.get("angle", 0.0)),
            }
            records.append(record)
            save_result(output, records)
            print(f"[record] {record}")
            print(f"[save] {output}")
    finally:
        detector.release()

    if len(records) < 4:
        print(f"[warn] only {len(records)} records saved; calc_extrinsics.py needs at least 4")
    else:
        print(f"[done] {len(records)} records saved. Now run: python calc_extrinsics.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manually collect camera/robot point pairs without moving the robot")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-calib", default="camera_calibration_result.json")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--min-area", type=float, default=500.0)
    parser.add_argument("--color", default=None, choices=["red", "green", "blue", "yellow"])
    parser.add_argument("--count", type=int, default=6, help="number of records to collect; <=0 means until quit")
    parser.add_argument("--output", default="calib_result.json")
    parser.add_argument("--append", action="store_true", help="append to an existing output file")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
