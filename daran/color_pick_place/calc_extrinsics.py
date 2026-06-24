#!/usr/bin/env python3
"""Compute camera-to-robot extrinsics from saved calibration records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CAMERA_CALIB_PATH = "camera_calibration_result.json"
DEFAULT_CALIB_RESULT_PATH = "calib_result.json"
DEFAULT_OUTPUT_PATH = "extrinsics.json"


def load_camera_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs


def load_calibration_records(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = list(data.get("records") or [])
    if len(records) < 4:
        raise ValueError("calib_result.json must contain at least 4 records")
    return records


def calculate_extrinsics(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    records: list[dict],
) -> tuple[np.ndarray, float]:
    robot_points = np.array([record["robot_xyz"] for record in records], dtype=np.float64)
    image_points = np.array([record["camera_center"] for record in records], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        robot_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed while estimating camera/robot extrinsics")

    rotation_camera_robot, _ = cv2.Rodrigues(rvec)
    translation_camera_robot = tvec.reshape(3)
    rotation_robot_camera = rotation_camera_robot.T
    translation_robot_camera = -rotation_robot_camera @ translation_camera_robot

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_robot_camera
    transform[:3, 3] = translation_robot_camera

    errors = []
    for record in records:
        pixel = np.array([[record["camera_center"]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pixel, camera_matrix, dist_coeffs)
        x_norm, y_norm = undistorted[0, 0]
        ray_camera = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
        ray_robot = rotation_robot_camera @ ray_camera
        scale = (float(record["robot_xyz"][2]) - translation_robot_camera[2]) / ray_robot[2]
        point_robot = rotation_robot_camera @ (scale * ray_camera) + translation_robot_camera
        errors.append(float(np.linalg.norm(point_robot[:2] - np.array(record["robot_xy"], dtype=np.float64))))

    rms_error_mm = float(np.sqrt(np.mean(np.square(errors))))
    return transform, rms_error_mm


def save_extrinsics(path: str | Path, transform: np.ndarray, rms_error_mm: float) -> None:
    output = {
        "T_robot_camera": transform.tolist(),
        "rms_error_mm": rms_error_mm,
        "source": {
            "camera_calib": DEFAULT_CAMERA_CALIB_PATH,
            "calib_result": DEFAULT_CALIB_RESULT_PATH,
        },
    }
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute T_robot_camera and save it to extrinsics.json")
    parser.add_argument("--camera-calib", default=DEFAULT_CAMERA_CALIB_PATH)
    parser.add_argument("--calib-result", default=DEFAULT_CALIB_RESULT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    camera_matrix, dist_coeffs = load_camera_calibration(args.camera_calib)
    records = load_calibration_records(args.calib_result)
    transform, rms_error_mm = calculate_extrinsics(camera_matrix, dist_coeffs, records)
    save_extrinsics(args.output, transform, rms_error_mm)

    print(f"[extrinsics] saved: {args.output}")
    print("[extrinsics] T_robot_camera:")
    print(transform)
    print(f"[extrinsics] RMS XY error: {rms_error_mm:.3f} mm")


if __name__ == "__main__":
    main()
