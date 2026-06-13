#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recognize color blocks and stack selected colors with the robot arm."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from convert import camera_observation_to_robot_pose
from detect import ColorBlockDetector
from joint_action_motion import JointActionMotion, MotionConfig


DEFAULT_WEIGHTS = str(Path(__file__).resolve().parent / "yolo" / "runs" / "blocks" / "weights" / "best.pt")


def build_detector(args: argparse.Namespace):
    """Build the chosen block detector. Both expose recognize()/release()."""
    if args.detector == "yolo":
        from yolo.yolo_detector import YoloBlockDetector

        return YoloBlockDetector(
            weights=args.weights,
            camera_index=args.camera,
            calib_path=args.camera_calib,
            width=args.width,
            height=args.height,
            min_area=args.min_area,
            conf=args.conf,
            device=args.device,
            display_width=args.display_width,
        )
    return ColorBlockDetector(
        camera_index=args.camera,
        calib_path=args.camera_calib,
        width=args.width,
        height=args.height,
        min_area=args.min_area,
        display_width=args.display_width,
    )


STANDBY_JOINTS = [0.0, 90.0, 0.0, 0.0, 0.0, 0.0]
IK_INITIAL_JOINTS = [32.2, 51.5, -32.9, -24.0, 84.7, 28.4]
DEFAULT_ORIENTATION = [0.0, -90.0, 180.0]
TARGET_POSITIONS = {
    "red": [180.0, 172.0, 90.0],
    "yellow": [180.0, 172.0, 90.0],
    "green": [125.0, 172.0, 90.0],
    "blue": [50.0, 172.0, 90.0],
}
DEFAULT_COLORS = ["red", "yellow"]
IGNORE_TARGET_RADIUS_MM = 40.0
IGNORE_ROBOT_Y_GT = 172.0
STACK_HEIGHT_STEP_MM = 32.0
PLACE_TRANSIT_XY = [135.0, 120.0]


@dataclass(frozen=True)
class RobotPose:
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float

    def up(self, approach_height: float) -> list[float]:
        return [self.x, self.y, self.z + approach_height, self.pitch, self.yaw, self.roll]

    def down(self) -> list[float]:
        return [self.x, self.y, self.z, self.pitch, self.yaw, self.roll]


def move_to_standby(motion: JointActionMotion, speed: float) -> None:
    if motion.ro is None:
        motion.ro = motion._make_robot()
    print(f"[standby] joints={STANDBY_JOINTS}", flush=True)
    ok = motion.ro.set_arm_joints(angle_list=STANDBY_JOINTS, speed=speed)
    if ok is False:
        raise RuntimeError("standby joint command failed")
    motion.ro.pose_done()


def build_pick_place_actions(
    pick: RobotPose,
    place: RobotPose,
    approach_height: float,
) -> list[list[float]]:
    transit_z = max(pick.z, place.z) + approach_height
    transit_pose = [
        PLACE_TRANSIT_XY[0],
        PLACE_TRANSIT_XY[1],
        transit_z,
        place.pitch,
        place.yaw,
        place.roll,
    ]
    return [
        pick.up(approach_height) + [0],
        pick.down() + [1],
        pick.up(approach_height) + [1],
        transit_pose + [1],
        place.up(approach_height) + [1],
        place.down() + [0],
        place.up(approach_height) + [0],
    ]


def prepend_standby_action(joint_actions: list[list[float]]) -> list[list[float]]:
    if not joint_actions:
        return joint_actions
    first_grip = int(joint_actions[0][6])
    return [list(STANDBY_JOINTS) + [first_grip]] + joint_actions


def parse_robot_pose(values: Sequence[float], orientation: Sequence[float]) -> RobotPose:
    if len(values) == 3:
        pitch, yaw, roll = [float(v) for v in orientation]
        return RobotPose(float(values[0]), float(values[1]), float(values[2]), pitch, yaw, roll)
    if len(values) == 6:
        return RobotPose(*(float(v) for v in values))
    raise ValueError("target pose must have 3 values (x y z) or 6 values (x y z pitch yaw roll)")


def make_targets(args: argparse.Namespace) -> dict[str, RobotPose]:
    configured_targets = {
        "red": parse_robot_pose(args.red_target, args.orientation),
        "yellow": parse_robot_pose(args.yellow_target, args.orientation),
        "green": parse_robot_pose(args.green_target, args.orientation),
        "blue": parse_robot_pose(args.blue_target, args.orientation),
    }
    return {color: configured_targets[color] for color in args.colors}


def is_in_target_area(pose: RobotPose, targets: Mapping[str, RobotPose], radius_mm: float) -> bool:
    for target in targets.values():
        distance = math.hypot(pose.x - target.x, pose.y - target.y)
        if distance <= radius_mm:
            return True
    return False


def choose_detection(
    args: argparse.Namespace,
    results: list[dict],
    targets: Mapping[str, RobotPose],
    color_counts: Mapping[str, int],
) -> tuple[dict, RobotPose] | tuple[None, None]:
    candidates: list[tuple[int, dict, RobotPose]] = []
    for result in results:
        color = str(result.get("type"))
        if color not in targets:
            continue
        pick_pose = observation_to_pick_pose(args, result)
        if pick_pose.y > args.ignore_robot_y_gt:
            print(
                f"[ignore] {color} at robot=({pick_pose.x:.1f}, {pick_pose.y:.1f}) "
                f"has y > {args.ignore_robot_y_gt:.1f}",
                flush=True,
            )
            continue
        if is_in_target_area(pick_pose, targets, args.ignore_target_radius):
            print(
                f"[ignore] {color} at robot=({pick_pose.x:.1f}, {pick_pose.y:.1f}) "
                f"is inside target area",
                flush=True,
            )
            continue
        candidates.append((int(color_counts.get(color, 0)), result, pick_pose))
    if not candidates:
        return None, None
    _, result, pick_pose = min(candidates, key=lambda item: item[0])
    return result, pick_pose


def stacked_place_pose(base_pose: RobotPose, count: int, stack_height_step: float) -> RobotPose:
    return RobotPose(
        base_pose.x,
        base_pose.y,
        base_pose.z + count * stack_height_step,
        base_pose.pitch,
        base_pose.yaw,
        base_pose.roll,
    )


def observation_to_pick_pose(args: argparse.Namespace, observation: Mapping[str, object]) -> RobotPose:
    x, y, z, pitch, yaw, roll = camera_observation_to_robot_pose(
        observation,
        table_z_mm=args.table_z,
        base_orientation=args.orientation,
        angle_gain=args.angle_gain,
        angle_offset_deg=args.angle_offset,
        camera_calib_path=args.camera_calib,
    )
    return RobotPose(x, y, z, pitch, yaw, roll)


def run_sorting(args: argparse.Namespace) -> None:
    cfg = MotionConfig(
        joint_speed=args.joint_speed,
        gripper_open_width=args.gripper_open_width,
        gripper_close_width=args.gripper_close_width,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
        gripper_open_loop_wait=args.gripper_open_loop_wait,
        settle=args.settle,
        execution_points_per_segment=args.execution_points_per_segment,
        wait_intermediate_pose_done=args.wait_intermediate_pose_done,
        intermediate_settle=args.intermediate_settle,
    )
    motion = JointActionMotion(
        com=args.com,
        baudrate=args.baudrate,
        cfg=cfg,
        ik_ilimit=args.ik_ilimit,
        ik_slimit=args.ik_slimit,
        connect=not args.dry_run,
    )
    targets = make_targets(args)

    if args.dry_run:
        print("[dry-run] robot connection and real motion are disabled", flush=True)
    else:
        print("[init] move to standby before opening camera", flush=True)
        move_to_standby(motion, args.joint_speed)

    detector = build_detector(args)

    try:
        count = 0
        color_counts = {color: 0 for color in targets}
        while True:
            if args.dry_run:
                print("[dry-run] keep the real robot clear of the camera view", flush=True)
            else:
                print("[loop] move to standby before recognition", flush=True)
                move_to_standby(motion, args.joint_speed)
            print("[vision] press 'a' to accept detections; empty result stops the program", flush=True)
            detections = detector.recognize()
            detection, pick_pose = choose_detection(args, detections, targets, color_counts)
            if detection is None:
                print(f"[done] no {'/'.join(targets)} block detected outside target areas")
                break

            color = str(detection["type"])
            place_pose = stacked_place_pose(targets[color], count, args.stack_height_step)
            print(
                f"[pick] {color}: camera={detection.get('center')} angle={float(detection.get('angle', 0.0)):.1f} "
                f"-> robot=({pick_pose.x:.1f}, {pick_pose.y:.1f}, {pick_pose.z:.1f})",
                flush=True,
            )
            print(
                f"[place] {color}: ({place_pose.x:.1f}, {place_pose.y:.1f}, {place_pose.z:.1f}), "
                f"stack_index={count}, color_count={color_counts[color]}",
                flush=True,
            )

            pose_actions = build_pick_place_actions(pick_pose, place_pose, args.approach_height)
            joint_actions = motion.solve(pose_actions, initial_joints=args.ik_initial_joints)
            joint_actions = prepend_standby_action(joint_actions)

            if args.dry_run:
                print("[sim] dry-run preview only; closing the simulation window will not move the robot", flush=True)
            else:
                print("[sim] close the simulation window, then press Enter in the terminal to execute real motion", flush=True)
            motion.sim(joint_actions)

            if args.dry_run:
                print("[dry-run] planned joint actions:")
                for i, row in enumerate(joint_actions):
                    grip = int(row[6])
                    width = cfg.gripper_close_width if grip else cfg.gripper_open_width
                    print(
                        f"  {i}: {[round(float(v), 3) for v in row[:6]]} "
                        f"grip={grip} gripper_width={width:.1f}mm"
                    )
                break

            confirm = input("[confirm] Press Enter to execute real robot motion, or type q then Enter to abort: ")
            if confirm.strip().lower() in {"q", "quit", "abort", "n", "no"}:
                print("[abort] real motion skipped", flush=True)
                break

            print("[run] executing pick and place...", flush=True)
            motion.run(joint_actions)
            color_counts[color] += 1
            count += 1
            print(f"[loop] finished {count} block(s), counts={color_counts}", flush=True)
            if args.max_blocks > 0 and count >= args.max_blocks:
                print(f"[done] reached max blocks: {args.max_blocks}", flush=True)
                break
    finally:
        detector.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect selected color blocks and stack them")
    parser.add_argument("--com", default="/dev/cu.usbmodem94869FA301481")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--camera", type=int, default=0)
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

    parser.add_argument("--detector", choices=["yolo", "color"], default="yolo",
                        help="yolo = trained YOLO11n + ROI angle; color = legacy HSV detector")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO weights path (for --detector yolo)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--device", default="cpu", help="YOLO inference device: cpu or 0")

    parser.add_argument("--table-z", type=float, default=80.0)
    parser.add_argument("--approach-height", type=float, default=35.0)
    parser.add_argument("--orientation", type=float, nargs=3, default=DEFAULT_ORIENTATION, metavar=("P", "Y", "R"))
    parser.add_argument("--angle-gain", type=float, default=-1.0)
    parser.add_argument("--angle-offset", type=float, default=0.0)

    parser.add_argument(
        "--colors",
        nargs="+",
        choices=["red", "yellow", "green", "blue"],
        default=DEFAULT_COLORS,
        help="colors to pick; default is red yellow",
    )
    parser.add_argument("--max-blocks", type=int, default=2, help="maximum blocks to pick; <=0 means until none")
    parser.add_argument("--red-target", type=float, nargs="+", default=TARGET_POSITIONS["red"])
    parser.add_argument("--yellow-target", type=float, nargs="+", default=TARGET_POSITIONS["yellow"])
    parser.add_argument("--green-target", type=float, nargs="+", default=TARGET_POSITIONS["green"])
    parser.add_argument("--blue-target", type=float, nargs="+", default=TARGET_POSITIONS["blue"])
    parser.add_argument("--ignore-target-radius", type=float, default=IGNORE_TARGET_RADIUS_MM)
    parser.add_argument("--ignore-robot-y-gt", type=float, default=IGNORE_ROBOT_Y_GT)
    parser.add_argument("--stack-height-step", type=float, default=STACK_HEIGHT_STEP_MM)

    parser.add_argument("--joint-speed", type=float, default=2.0)
    parser.add_argument("--ik-initial-joints", type=float, nargs=6, default=IK_INITIAL_JOINTS)
    parser.add_argument("--ik-ilimit", type=int, default=200)
    parser.add_argument("--ik-slimit", type=int, default=200)
    parser.add_argument("--execution-points-per-segment", type=int, default=10)
    parser.add_argument("--wait-intermediate-pose-done", action="store_true")
    parser.add_argument("--intermediate-settle", type=float, default=0.0)
    parser.add_argument("--gripper-open-width", type=float, default=5.0)
    parser.add_argument("--gripper-close-width", type=float, default=0.0)
    parser.add_argument("--gripper-speed", type=float, default=10.0)
    parser.add_argument("--gripper-force", type=float, default=80.0)
    parser.add_argument("--gripper-open-loop-wait", type=float, default=1.0)
    parser.add_argument("--settle", type=float, default=0.3)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="detect and simulate one pick/place without robot motion")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_sorting(args)


if __name__ == "__main__":
    main()
