#!/usr/bin/env python3
"""Convert one camera observation to a robot 6-DoF pose with standard calibration."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np


DEFAULT_CAMERA_CALIB_PATH = "camera_calibration_result.json"
DEFAULT_EXTRINSICS_PATH = "extrinsics.json"
TABLE_Z_MM = 80.0
BASE_PITCH_DEG = 0.0
BASE_YAW_DEG = -90.0
BASE_ROLL_DEG = 180.0


def camera_observation_to_robot_pose(
    observation: Mapping[str, object] | Sequence[float],
    table_z_mm: float = TABLE_Z_MM,
    base_orientation: Sequence[float] = (BASE_PITCH_DEG, BASE_YAW_DEG, BASE_ROLL_DEG),
    angle_gain: float = -1.0,
    angle_offset_deg: float = 0.0,
    camera_calib_path: str | Path = DEFAULT_CAMERA_CALIB_PATH,
    extrinsics_path: str | Path = DEFAULT_EXTRINSICS_PATH,
) -> tuple[float, float, float, float, float, float]:
    """Return robot-frame (x, y, z, pitch, yaw, roll) for one observation.

    observation can be either:
        (u, v)
        (u, v, angle_deg)
        {"center": [u, v], "angle": angle_deg, ...}

    The transform is standard pinhole camera geometry:
        1. load camera intrinsics and distortion,
        2. load T_robot_camera from extrinsics.json,
        3. undistort the pixel into a camera ray,
        4. intersect that ray with the robot table plane z=table_z_mm.

    The output orientation keeps the default downward grasp pose
    (0, -90, 180), then applies the observed in-plane angle to yaw.
    """

    x_robot, y_robot = camera_observation_to_robot_xy(
        observation,
        table_z_mm=table_z_mm,
        camera_calib_path=camera_calib_path,
        extrinsics_path=extrinsics_path,
    )
    angle_deg = _observation_angle(observation)
    pitch, yaw, roll = [float(v) for v in base_orientation]
    rotation_delta = _wrap_square_degrees(angle_gain * angle_deg + angle_offset_deg)
    yaw = _wrap_degrees(yaw + rotation_delta)
    return float(x_robot), float(y_robot), float(table_z_mm), pitch, yaw, roll


def camera_observation_to_robot_xy(
    observation: Mapping[str, object] | Sequence[float],
    table_z_mm: float = TABLE_Z_MM,
    camera_calib_path: str | Path = DEFAULT_CAMERA_CALIB_PATH,
    extrinsics_path: str | Path = DEFAULT_EXTRINSICS_PATH,
) -> tuple[float, float]:
    """Return robot-frame (x, y) for a camera pixel observation on z=85mm."""

    u, v = _observation_center(observation)
    camera_matrix, dist_coeffs = _load_camera_calibration(str(camera_calib_path))
    transform = _load_extrinsics(str(extrinsics_path))
    rotation_robot_camera = transform[:3, :3]
    translation_robot_camera = transform[:3, 3]

    pixel = np.array([[[u, v]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pixel, camera_matrix, dist_coeffs)
    x_norm, y_norm = undistorted[0, 0]

    ray_camera = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
    ray_robot = rotation_robot_camera @ ray_camera
    if abs(float(ray_robot[2])) < 1e-9:
        raise RuntimeError("camera ray is parallel to the robot table plane")

    scale = (float(table_z_mm) - translation_robot_camera[2]) / ray_robot[2]
    point_robot = rotation_robot_camera @ (scale * ray_camera) + translation_robot_camera
    return float(point_robot[0]), float(point_robot[1])


def _observation_center(observation: Mapping[str, object] | Sequence[float]) -> tuple[float, float]:
    if isinstance(observation, Mapping):
        if "center" in observation:
            observation = observation["center"]  # type: ignore[assignment]
        elif "camera_center" in observation:
            observation = observation["camera_center"]  # type: ignore[assignment]
        else:
            raise KeyError('observation dict must contain key "center" or "camera_center"')
    if len(observation) < 2:  # type: ignore[arg-type]
        raise ValueError("observation must contain at least u and v")
    return float(observation[0]), float(observation[1])  # type: ignore[index]


def _observation_angle(observation: Mapping[str, object] | Sequence[float]) -> float:
    if isinstance(observation, Mapping):
        return float(observation.get("angle", 0.0))
    if len(observation) >= 3:
        return float(observation[2])
    return 0.0


def _wrap_degrees(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _wrap_square_degrees(angle: float) -> float:
    return (float(angle) + 45.0) % 90.0 - 45.0


@lru_cache(maxsize=8)
def _load_camera_calibration(
    camera_calib_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    with Path(camera_calib_path).open("r", encoding="utf-8") as f:
        camera_data = json.load(f)
    camera_matrix = np.array(camera_data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(camera_data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs


@lru_cache(maxsize=8)
def _load_extrinsics(extrinsics_path: str) -> np.ndarray:
    with Path(extrinsics_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    transform = np.array(data["T_robot_camera"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("T_robot_camera must be a 4x4 matrix")
    return transform


if __name__ == "__main__":
    sample = {"center": [854, 538], "angle": 90.0}
    print(camera_observation_to_robot_pose(sample))
