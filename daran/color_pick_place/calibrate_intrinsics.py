#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live chessboard calibration for camera intrinsics.

This script only opens the camera. It does not import or command the robot.
It writes camera_calibration_result.json in the format used by detect.py and
convert.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= float(square_size)
    return points


def find_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def show_resized(window_name: str, image: np.ndarray, display_width: int) -> None:
    shown = image
    if display_width > 0 and image.shape[1] > display_width:
        scale = display_width / float(image.shape[1])
        size = (display_width, int(round(image.shape[0] * scale)))
        shown = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    cv2.imshow(window_name, shown)


def calibrate(image_paths: list[Path], args: argparse.Namespace) -> None:
    pattern_size = (args.cols, args.rows)
    objp = object_points(args.cols, args.rows, args.square_size)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    last_image: np.ndarray | None = None

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ok, corners = find_corners(gray, pattern_size)
        if not ok or corners is None:
            print(f"[skip] no chessboard: {path}")
            continue
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        image_size = gray.shape[::-1]
        last_image = image

    if len(objpoints) < 3 or image_size is None:
        raise RuntimeError(f"only {len(objpoints)} valid images; capture more chessboard views")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    total_error_sq = 0.0
    total_points = 0
    per_view_errors = []
    for i, obj in enumerate(objpoints):
        projected, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        count = len(projected)
        per_view_errors.append(float(error / count))
        total_error_sq += float(error * error)
        total_points += count
    mean_error = float(np.sqrt(total_error_sq / total_points))

    output = {
        "image_size": list(image_size),
        "board_size": [args.cols, args.rows],
        "square_size": float(args.square_size),
        "valid_image_count": len(objpoints),
        "rms_reprojection_error": float(rms),
        "mean_reprojection_error_px": mean_error,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "rvecs": [r.reshape(-1).tolist() for r in rvecs],
        "tvecs": [t.reshape(-1).tolist() for t in tvecs],
        "per_view_error_px": per_view_errors,
    }
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if last_image is not None:
        undistorted = cv2.undistort(last_image, camera_matrix, dist_coeffs)
        cv2.imwrite(str(Path(args.output).with_name("undistorted_example.jpg")), undistorted)

    print("[calib] complete")
    print(f"[calib] valid images: {len(objpoints)}")
    print(f"[calib] RMS reprojection error: {rms:.4f}")
    print(f"[calib] mean reprojection error: {mean_error:.4f} px")
    print(f"[calib] saved: {args.output}")


def collect_live(args: argparse.Namespace) -> list[Path]:
    pattern_size = (args.cols, args.rows)
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"camera open failed: {args.camera}")

    saved: list[Path] = []
    auto = bool(args.auto)
    last_auto = 0.0
    print("[camera] s=save detected frame, a=toggle auto, c=calibrate, q=quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_corners(gray, pattern_size)
            display = frame.copy()
            color = (0, 200, 0) if found else (0, 0, 255)
            if found and corners is not None:
                cv2.drawChessboardCorners(display, pattern_size, corners, found)
            cv2.putText(
                display,
                f"{'detected' if found else 'not detected'} saved={len(saved)} auto={'on' if auto else 'off'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )
            show_resized("intrinsics calibration", display, args.display_width)

            now = time.time()
            if found and auto and now - last_auto >= args.auto_interval:
                path = image_dir / f"calib_{len(saved) + 1:03d}.jpg"
                cv2.imwrite(str(path), frame)
                saved.append(path)
                last_auto = now
                print(f"[save] {path}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                auto = not auto
                print(f"[auto] {'on' if auto else 'off'}")
            if key == ord("s"):
                if not found:
                    print("[warn] chessboard not detected; not saved")
                    continue
                path = image_dir / f"calib_{len(saved) + 1:03d}.jpg"
                cv2.imwrite(str(path), frame)
                saved.append(path)
                print(f"[save] {path}")
            if key == ord("c"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics from a live chessboard")
    parser.add_argument("--camera", type=int, default=4)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--display-width",
        type=int,
        default=960,
        help="maximum visualization window image width; <=0 keeps original size",
    )
    parser.add_argument("--cols", type=int, default=7, help="inner corners per row")
    parser.add_argument("--rows", type=int, default=7, help="inner corners per column")
    parser.add_argument("--square-size", type=float, default=20.0, help="square size, usually mm")
    parser.add_argument("--image-dir", default="intrinsics_images")
    parser.add_argument("--from-images", default="", help="calibrate from an existing image directory")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--auto-interval", type=float, default=1.5)
    parser.add_argument("--output", default="camera_calibration_result.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.from_images:
        image_paths = sorted(Path(args.from_images).glob("*.jpg")) + sorted(Path(args.from_images).glob("*.png"))
    else:
        image_paths = collect_live(args)
    calibrate(image_paths, args)


if __name__ == "__main__":
    main()
