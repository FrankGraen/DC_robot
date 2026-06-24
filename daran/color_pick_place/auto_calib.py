#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Move one block through robot points and calibrate camera-to-robot XY."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from detect import ColorBlockDetector
from joint_action_motion import JointActionMotion, MotionConfig


STANDBY_JOINTS = [3.200, 129.500, -106.900, -28.600, 87.800, 2.400]
IK_INITIAL_JOINTS = [32.2, 51.5, -32.9, -24.0, 84.7, 28.4]
DEFAULT_ORIENTATION = [0.0, -90.0, 180.0]


@dataclass(frozen=True)
class RobotPoint:
    x: float
    y: float
    z: float

    def xy(self) -> list[float]:
        return [self.x, self.y]

    def xyz(self) -> list[float]:
        return [self.x, self.y, self.z]


def build_pick_place_pose_actions(
    pick: RobotPoint,
    place: RobotPoint,
    approach_height: float,
    orientation: Sequence[float],
) -> list[list[float]]:
    pitch, yaw, roll = [float(v) for v in orientation]
    pick_up = [pick.x, pick.y, pick.z + approach_height, pitch, yaw, roll]
    pick_down = [pick.x, pick.y, pick.z, pitch, yaw, roll]
    place_up = [place.x, place.y, place.z + approach_height, pitch, yaw, roll]
    place_down = [place.x, place.y, place.z, pitch, yaw, roll]
    return [
        pick_up + [0],
        pick_down + [1],
        pick_up + [1],
        place_up + [1],
        place_down + [0],
        place_up + [0],
    ]


def make_grid_points(args: argparse.Namespace) -> list[RobotPoint]:
    points: list[RobotPoint] = []
    row_mid = (args.rows - 1) / 2.0
    col_mid = (args.cols - 1) / 2.0
    for row in range(args.rows):
        for col in range(args.cols):
            x = args.grid_center_x + (row - row_mid) * args.spacing_x
            y = args.grid_center_y + (col - col_mid) * args.spacing_y
            points.append(RobotPoint(float(x), float(y), float(args.table_z)))
    return points


def move_to_standby(motion: JointActionMotion, joints: Sequence[float], speed: float) -> None:
    if motion.ro is None:
        motion.ro = motion._make_robot()
    print(f"[standby] joints={[round(float(v), 2) for v in joints]}")
    ok = motion.ro.set_arm_joints(angle_list=[float(v) for v in joints], speed=speed)
    if ok is False:
        raise RuntimeError("standby joint command failed")
    motion.ro.pose_done()


def prepend_standby_action(joint_actions: list[list[float]]) -> list[list[float]]:
    if not joint_actions:
        return joint_actions
    first_grip = int(joint_actions[0][6])
    return [list(STANDBY_JOINTS) + [first_grip]] + joint_actions


def choose_detection(results: list[dict], color: str | None) -> dict:
    if color:
        filtered = [item for item in results if item.get("type") == color]
    else:
        filtered = results
    if not filtered:
        label = color if color else "any color"
        raise RuntimeError(f"no detection found for {label}")
    return filtered[0]


def estimate_affine(records: list[dict]) -> tuple[np.ndarray, np.ndarray, float]:
    camera_points = np.array([record["camera_center"] for record in records], dtype=np.float64)
    robot_points = np.array([record["robot_xy"] for record in records], dtype=np.float64)
    matrix, inliers = cv2.estimateAffine2D(camera_points, robot_points)
    if matrix is None:
        raise RuntimeError("cv2.estimateAffine2D failed")

    predicted = cv2.transform(camera_points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(predicted - robot_points, axis=1)
    rms_error = float(np.sqrt(np.mean(errors * errors)))
    return matrix, inliers, rms_error


def save_result(
    path: str | Path,
    records: list[dict],
    matrix: np.ndarray | None = None,
    inliers: np.ndarray | None = None,
    rms_error: float | None = None,
) -> None:
    output = {
        "camera_to_robot_affine_2x3": matrix.tolist() if matrix is not None else None,
        "rms_error_mm": rms_error,
        "records": records,
        "inliers": inliers.reshape(-1).astype(int).tolist() if inliers is not None else None,
    }
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def run_calibration(args: argparse.Namespace) -> list[dict]:
    grid_points = make_grid_points(args)
    if len(grid_points) != 6:
        raise ValueError("this calibration program expects exactly 6 points; use --rows 2 --cols 3")

    cfg = MotionConfig(
        joint_speed=args.joint_speed,
        gripper_open_width=args.gripper_open_width,
        gripper_close_width=args.gripper_close_width,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
        settle=args.settle,
    )
    motion = JointActionMotion(
        com=args.com,
        baudrate=args.baudrate,
        cfg=cfg,
        ik_ilimit=args.ik_ilimit,
        ik_slimit=args.ik_slimit,
        connect=True,
    )
    detector = ColorBlockDetector(
        camera_index=args.camera,
        calib_path=args.camera_calib,
        width=args.width,
        height=args.height,
        min_area=args.min_area,
    )

    records: list[dict] = []
    current_pick = RobotPoint(args.start_x, args.start_y, args.start_z)
    planned_moves: list[tuple[int, RobotPoint, list[list[float]]]] = []
    total_points = len(grid_points)

    print(f"[precheck] solving all {total_points} point moves before real execution...", flush=True)
    for index, target in enumerate(grid_points, start=1):
        print(f"[precheck {index}/{total_points}] {current_pick.xyz()} -> {target.xyz()}", flush=True)
        pose_actions = build_pick_place_pose_actions(
            current_pick,
            target,
            approach_height=args.approach_height,
            orientation=args.orientation,
        )
        joint_actions = motion.solve(pose_actions, initial_joints=args.ik_initial_joints)
        planned_moves.append((index, target, joint_actions))
        current_pick = target
    print("[precheck] all point moves are reachable", flush=True)

    try:
        for index, target, joint_actions in planned_moves:
            if index == 1:
                print("[standby] move to standby before first point", flush=True)
                move_to_standby(motion, STANDBY_JOINTS, args.joint_speed)
                joint_actions = prepend_standby_action(joint_actions)
            print(f"[point {index}/{total_points}] execute move to {target.xyz()}", flush=True)
            print("[sim] close the simulation window to execute the real motion", flush=True)
            motion.sim(joint_actions)
            print("[run] executing real motion...", flush=True)
            motion.run(joint_actions)

            move_to_standby(motion, STANDBY_JOINTS, args.joint_speed)
            print("[vision] press 'a' in the detection window to accept this point")
            detections = detector.recognize()
            detection = choose_detection(detections, args.color)
            center = detection["center"]
            record = {
                "index": index,
                "type": detection["type"],
                "robot_xyz": target.xyz(),
                "robot_xy": target.xy(),
                "camera_center": [float(center[0]), float(center[1])],
                "angle": float(detection["angle"]),
            }
            records.append(record)
            save_result(args.output, records)
            print(f"[record] {record}")
            print(f"[save] partial result saved: {args.output}")
            time.sleep(args.after_record_settle)
    finally:
        detector.release()

    matrix, inliers, rms_error = estimate_affine(records)
    save_result(args.output, records, matrix, inliers, rms_error)
    print(f"[calib] saved: {args.output}")
    print(f"[calib] affine 2x3:\n{matrix}")
    print(f"[calib] RMS error: {rms_error:.3f} mm")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="6-point camera-to-robot calibration by moving one color block")
    parser.add_argument("--com", default="/dev/cu.usbmodem94869FA301481", help="robot serial port")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-calib", default="camera_calibration_result.json")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--min-area", type=float, default=500.0)
    parser.add_argument("--color", default=None, choices=["red", "green", "blue", "yellow"])

    parser.add_argument("--start-x", type=float, default=226)
    parser.add_argument("--start-y", type=float, default=203)
    parser.add_argument("--start-z", type=float, default=82)
    parser.add_argument("--table-z", type=float, default=82)
    parser.add_argument("--grid-center-x", type=float, default=230.0)
    parser.add_argument("--grid-center-y", type=float, default=-60.0)
    parser.add_argument("--spacing-x", type=float, default=80.0)
    parser.add_argument("--spacing-y", type=float, default=60.0)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--approach-height", type=float, default=35.0)
    parser.add_argument("--orientation", type=float, nargs=3, default=DEFAULT_ORIENTATION, metavar=("P", "Y", "R"))
    parser.add_argument("--ik-initial-joints", type=float, nargs=6, default=IK_INITIAL_JOINTS)

    parser.add_argument("--joint-speed", type=float, default=2.0)
    parser.add_argument("--ik-ilimit", type=int, default=200)
    parser.add_argument("--ik-slimit", type=int, default=200)
    parser.add_argument("--gripper-open-width", type=float, default=5.0)
    parser.add_argument("--gripper-close-width", type=float, default=0.0)
    parser.add_argument("--gripper-speed", type=float, default=10.0)
    parser.add_argument("--gripper-force", type=float, default=80.0)
    parser.add_argument("--settle", type=float, default=0.3)
    parser.add_argument("--after-record-settle", type=float, default=0.2)
    parser.add_argument("--output", default="calib_result.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_calibration(args)


if __name__ == "__main__":
    main()
