#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Teach camera-to-robot calibration pairs by hand-guiding the arm.

The script does not command motion trajectories. It can put the arm in free or
lock mode, then records:
    current robot pose from FK + current camera color-block center
into calib_result.json, compatible with calc_extrinsics.py.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detect import ColorBlockDetector


def capture_detection(detector: ColorBlockDetector, color: str | None) -> dict:
    last_results: list[dict] = []
    try:
        while True:
            ok, frame = detector.cap.read()
            if not ok:
                continue

            raw_frame = frame.copy()
            last_results, display = detector.detect_blocks(raw_frame)
            detector.add_robot_coordinates(last_results, display)
            cv2.putText(
                display,
                "s=save+accept  a=accept  q=cancel",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            detector.show(display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt("camera capture cancelled")
            if key in (ord("s"), ord("a")):
                detection = choose_detection(last_results, color)
                if key == ord("s"):
                    filename = f"teach_calib_{datetime.now():%Y%m%d_%H%M%S}.png"
                    cv2.imwrite(filename, raw_frame)
                    print(f"[image] saved {filename}")
                return detection
    finally:
        detector.close_window()


def choose_detection(results: list[dict], color: str | None) -> dict:
    if color:
        results = [item for item in results if item.get("type") == color]
    if not results:
        label = color if color else "any color"
        raise RuntimeError(f"no detection found for {label}")
    return results[0]


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


def read_robot_pose(ro: object, table_z: float | None) -> tuple[list[float], list[float]]:
    ro.detect_pose()
    x, y, z = [float(v) for v in ro.pl]
    pitch, yaw, roll = [float(v) for v in ro.theta_P_Y_R]
    if table_z is not None:
        z = float(table_z)
    return [x, y, z], [pitch, yaw, roll]


def run(args: argparse.Namespace) -> None:
    import arm_robot as robot

    output = Path(args.output)
    records = load_existing(output, args.append)

    ro = robot.arm_robot(com=args.com, uart_baudrate=args.baudrate)
    detector = ColorBlockDetector(
        camera_index=args.camera,
        calib_path=args.camera_calib,
        width=args.width,
        height=args.height,
        min_area=args.min_area,
        display_width=args.display_width,
    )

    if args.free_on_start:
        print("[robot] free mode: you can hand-guide the arm now")
        ro.free()

    try:
        while args.count <= 0 or len(records) < args.count:
            next_index = len(records) + 1
            print()
            if args.capture_first:
                print(f"[point {next_index}] keep the arm away from the camera view and keep the block fixed.")
                print("[keys] Enter=capture camera point, f=free, l=lock, q=quit")
            else:
                print(f"[point {next_index}] hand-guide the arm/TCP to the calibration block center.")
                print("[keys] Enter=record, f=free, l=lock, q=quit")
            cmd = input("[teach] > ").strip().lower()
            if cmd in {"q", "quit", "exit"}:
                break
            if cmd == "f":
                ro.free()
                print("[robot] free")
                continue
            if cmd == "l":
                ro.lock()
                print("[robot] lock")
                continue
            if cmd:
                print("[warn] unknown command")
                continue

            detection = None
            if args.capture_first:
                print("[vision] press 's' to save+accept, or 'a' to accept without saving")
                detection = capture_detection(detector, args.color)
                print(f"[camera] accepted {detection.get('type')} at {detection.get('center')}")
                print("[teach] now hand-guide the TCP/gripper center to the same physical block center.")
                print("[teach] keep the block fixed; press Enter when aligned, or q to cancel this point.")
                cmd = input("[robot pose] > ").strip().lower()
                if cmd in {"q", "quit", "exit"}:
                    break

            if args.lock_before_record:
                ro.lock()
                print("[robot] locked before recording")

            robot_xyz, robot_rpy = read_robot_pose(ro, args.table_z)
            if detection is None:
                print("[vision] press 's' to save+accept, or 'a' to accept without saving")
                detection = capture_detection(detector, args.color)
            center = detection["center"]
            record = {
                "index": next_index,
                "type": detection["type"],
                "robot_xyz": robot_xyz,
                "robot_xy": robot_xyz[:2],
                "robot_rpy": robot_rpy,
                "camera_center": [float(center[0]), float(center[1])],
                "angle": float(detection.get("angle", 0.0)),
            }
            records.append(record)
            save_result(output, records)
            print(f"[record] {record}")
            print(f"[save] {output}")

            if args.free_after_record:
                ro.free()
                print("[robot] free")
    finally:
        detector.release()

    if len(records) < 4:
        print(f"[warn] only {len(records)} records saved; calc_extrinsics.py needs at least 4")
    else:
        print(f"[done] {len(records)} records saved. Next: python calc_extrinsics.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect hand-taught camera/robot calibration pairs")
    parser.add_argument("--com", default="/dev/ttyACM1", help="robot serial port")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--camera", type=int, default=4)
    parser.add_argument("--camera-calib", default="camera_calibration_result.json")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--min-area", type=float, default=500.0)
    parser.add_argument(
        "--display-width",
        type=int,
        default=960,
        help="maximum visualization window image width; <=0 keeps original size",
    )
    parser.add_argument("--color", default=None, choices=["red", "green", "blue", "yellow"])
    parser.add_argument("--count", type=int, default=6, help="number of records to collect; <=0 means until quit")
    parser.add_argument("--table-z", type=float, default=82.0, help="record this fixed z value; use --robot-z to keep FK z")
    parser.add_argument("--robot-z", action="store_true", help="use robot FK z instead of fixed --table-z")
    parser.add_argument("--free-on-start", action="store_true", help="put robot in free mode at startup")
    parser.add_argument("--lock-before-record", action="store_true", help="lock robot before reading pose")
    parser.add_argument("--free-after-record", action="store_true", help="return to free mode after each record")
    parser.add_argument(
        "--capture-first",
        action="store_true",
        help="capture camera point before hand-guiding the robot, avoiding arm occlusion",
    )
    parser.add_argument("--output", default="calib_result.json")
    parser.add_argument("--append", action="store_true", help="append to an existing output file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.robot_z:
        args.table_z = None
    run(args)


if __name__ == "__main__":
    main()
