#!/usr/bin/env python3
"""Vision-guided grasp demo for task3.

Pipeline:
    camera image -> colored target pixel -> workspace plane point
    -> robot base point -> inverse kinematics -> optional hardware motion

The script is intentionally conservative:
    * Default mode only detects, computes and prints a plan.
    * Real robot motion requires --execute.
    * A workspace transform is required for --execute, so board coordinates are
      not accidentally treated as robot-base coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np.set_printoptions(precision=3, suppress=True)


# Same standard-DH model family used in task1/task2.
LINK_PARAMS = (
    dict(alpha=np.pi / 2),
    dict(a=0.15),
    dict(a=0.15),
    dict(d=-0.05494, alpha=np.pi / 2, offset=np.pi / 2),
    dict(d=0.068, alpha=-np.pi / 2),
    dict(d=0.033),
)
QMIN_DEG = [-160, -40, -160, -160, -180, -180]
QMAX_DEG = [160, 180, 160, 160, 180, 180]
PARK_POSE_DEG = [0, 90, 0, 0, 0, 0]
DEFAULT_ORIENTATION_Q_DEG = [0, -60, 60, 180, 0, 0]

DESK_Z_MIN_M = -0.02
JOINT_SPEED_RPM = 2.0
SEG_WAYPOINTS = 40

GRIPPER_ID = 7
GRIPPER_GEAR_D_MM = 10
GRIPPER_DEG_PER_MM = -180.0 / (np.pi * GRIPPER_GEAR_D_MM)
GRIPPER_OPEN_WIDTH_MM = 6.0
GRIPPER_CLOSE_WIDTH_MM = 1.8
GRIPPER_SPEED_MM_S = 10.0
GRIPPER_OPEN_FORCE_N = 20.0
GRIPPER_CLOSE_FORCE_N = 40.0

CAN_BRIDGE_PORT = "/dev/serial/by-id/usb-Dr-Tech_DR-USB_CAN_9A856B82094B-if00"
SERIAL_BAUD = 115200
ARM_WAIT_TIMEOUT_S = 25.0
WAIT_POLL_PERIOD_S = 0.08

HSV_PRESETS = {
    "red": [((0, 80, 60), (10, 255, 255)), ((170, 80, 60), (180, 255, 255))],
    "green": [((35, 60, 50), (90, 255, 255))],
    "blue": [((90, 60, 50), (130, 255, 255))],
    "yellow": [((18, 80, 80), (38, 255, 255))],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Detect a colored block and plan a grasp.")
    parser.add_argument("--camera", type=int, default=4)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--extrinsics", default="calibration_output/camera_extrinsic.json")
    parser.add_argument("--workspace-transform", default="calibration_output/workspace_transform.json")
    parser.add_argument("--color", choices=sorted(HSV_PRESETS), default="red")
    parser.add_argument("--min-area", type=float, default=500.0)
    parser.add_argument("--plane-z", type=float, default=0.0, help="Workspace plane z in board coordinates, meters.")
    parser.add_argument("--target-height-mm", type=float, default=10.0, help="Object top height above robot base plane.")
    parser.add_argument("--approach-height-mm", type=float, default=70.0)
    parser.add_argument("--grasp-offset-mm", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--orientation-q-deg", nargs=6, type=float, default=DEFAULT_ORIENTATION_Q_DEG)
    parser.add_argument("--execute", action="store_true", help="Actually move the robot and gripper.")
    parser.add_argument("--dry-run-image", default="", help="Use one image instead of live camera.")

    sub = parser.add_subparsers(dest="command")
    make_tf = sub.add_parser("make-workspace-transform", help="Create board->robot-base transform from 3 measured points.")
    make_tf.add_argument("--origin-base-mm", nargs=3, type=float, required=True)
    make_tf.add_argument("--x-point-base-mm", nargs=3, type=float, required=True)
    make_tf.add_argument("--y-point-base-mm", nargs=3, type=float, required=True)
    make_tf.add_argument("--x-point-board-mm", nargs=3, type=float, default=[8.0, 0.0, 0.0])
    make_tf.add_argument("--y-point-board-mm", nargs=3, type=float, default=[0.0, 8.0, 0.0])
    make_tf.add_argument("--output", default="calibration_output/workspace_transform.json")
    return parser.parse_args()


def require_roboticstoolbox():
    try:
        import roboticstoolbox as rtb
        from roboticstoolbox import DHRobot, RevoluteDH
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "roboticstoolbox is required for inverse kinematics. "
            "Please run this script in the same environment that can run task1/task2 notebooks."
        ) from exc
    return rtb, DHRobot, RevoluteDH


def make_dfarm():
    _, DHRobot, RevoluteDH = require_roboticstoolbox()
    links = [
        RevoluteDH(qlim=np.deg2rad([QMIN_DEG[i], QMAX_DEG[i]]), **kw)
        for i, kw in enumerate(LINK_PARAMS)
    ]
    return DHRobot(links, name="DFarm_StdDH")


def verify_within_limits(q_deg, label=""):
    q = np.asarray(q_deg, dtype=float)
    bad = [
        f"J{i + 1}={q[i]:+.2f} not in [{QMIN_DEG[i]:+.1f}, {QMAX_DEG[i]:+.1f}]"
        for i in range(6)
        if q[i] < QMIN_DEG[i] or q[i] > QMAX_DEG[i]
    ]
    if bad:
        raise ValueError(f"[{label}] joint limit error: " + "; ".join(bad))


def verify_above_desk(dfarm, q_deg, label=""):
    for i, T in enumerate(dfarm.fkine_all(np.deg2rad(q_deg))[1:], start=1):
        if float(T.t[2]) < DESK_Z_MIN_M:
            raise ValueError(f"[{label}] frame F{i} below desk guard: z={float(T.t[2]) * 1000:.1f} mm")


def load_extrinsics(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        np.array(data["camera_matrix"], dtype=np.float64),
        np.array(data["dist_coeffs"], dtype=np.float64),
        np.array(data["world_to_camera"], dtype=np.float64),
    )


def load_workspace_transform(path):
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["board_to_base"], dtype=np.float64)


def save_workspace_transform(args):
    origin_base = np.asarray(args.origin_base_mm, dtype=float) / 1000.0
    x_base = np.asarray(args.x_point_base_mm, dtype=float) / 1000.0
    y_base = np.asarray(args.y_point_base_mm, dtype=float) / 1000.0
    x_board = np.asarray(args.x_point_board_mm, dtype=float) / 1000.0
    y_board = np.asarray(args.y_point_board_mm, dtype=float) / 1000.0

    bx = x_board - np.zeros(3)
    by = y_board - np.zeros(3)
    rx = x_base - origin_base
    ry = y_base - origin_base
    if np.linalg.norm(bx) < 1e-9 or np.linalg.norm(by) < 1e-9:
        raise ValueError("Board reference points must be non-zero.")
    if np.linalg.norm(rx) < 1e-9 or np.linalg.norm(ry) < 1e-9:
        raise ValueError("Robot-base reference points must be non-zero.")

    x_axis = rx / np.linalg.norm(rx)
    y_hint = ry / np.linalg.norm(ry)
    z_axis = np.cross(x_axis, y_hint)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)

    board_x_len = np.linalg.norm(bx)
    board_y_len = np.linalg.norm(by)
    base_x_len = np.linalg.norm(rx)
    base_y_len = np.linalg.norm(ry)
    scale_x = base_x_len / board_x_len
    scale_y = base_y_len / board_y_len
    if abs(scale_x - scale_y) > 0.08 * max(scale_x, scale_y):
        print(f"Warning: x/y scale differ: {scale_x:.4f} vs {scale_y:.4f}")
    scale = (scale_x + scale_y) / 2.0

    T = np.eye(4)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis]) * scale
    T[:3, 3] = origin_base

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "description": "Maps board/workspace coordinates from camera_extrinsic.json to robot base coordinates.",
        "unit": "meter",
        "origin_base_mm": args.origin_base_mm,
        "x_point_base_mm": args.x_point_base_mm,
        "y_point_base_mm": args.y_point_base_mm,
        "x_point_board_mm": args.x_point_board_mm,
        "y_point_board_mm": args.y_point_board_mm,
        "board_to_base": T.tolist(),
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved workspace transform: {out}")


def pixel_to_plane(pixel, z_plane, camera_matrix, dist_coeffs, world_to_camera):
    camera_to_world = np.linalg.inv(world_to_camera)
    pts = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
    norm = cv2.undistortPoints(pts, camera_matrix, dist_coeffs).reshape(2)
    ray_camera = np.array([norm[0], norm[1], 1.0])
    ray_origin_world = camera_to_world[:3, 3]
    ray_dir_world = camera_to_world[:3, :3] @ ray_camera
    if abs(ray_dir_world[2]) < 1e-12:
        raise RuntimeError("Camera ray is parallel to the target plane.")
    s = (z_plane - ray_origin_world[2]) / ray_dir_world[2]
    return ray_origin_world + s * ray_dir_world


def board_to_base(point_board, transform):
    ph = np.array([point_board[0], point_board[1], point_board[2], 1.0])
    return (transform @ ph)[:3]


def make_mask(frame, color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_PRESETS[color]:
        mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_target(frame, color, min_area):
    mask = make_mask(frame, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area:
        return None, mask
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-9:
        return None, mask
    u = moments["m10"] / moments["m00"]
    v = moments["m01"] / moments["m00"]
    rect = cv2.minAreaRect(contour)
    return {"pixel": np.array([u, v]), "area": area, "rect": rect, "contour": contour}, mask


def acquire_target(args):
    if args.dry_run_image:
        frame = cv2.imread(args.dry_run_image)
        if frame is None:
            raise RuntimeError(f"Cannot read image: {args.dry_run_image}")
        target, _ = detect_target(frame, args.color, args.min_area)
        if target is None:
            raise RuntimeError("No colored target found in dry-run image.")
        return target, frame

    cap = cv2.VideoCapture(args.camera)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    locked = None
    locked_frame = None
    print("Keys: s=lock current target, q=quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            target, mask = detect_target(frame, args.color, args.min_area)
            preview = frame.copy()
            if target is not None:
                contour = target["contour"]
                cv2.drawContours(preview, [contour], -1, (0, 255, 0), 2)
                u, v = target["pixel"]
                cv2.circle(preview, (int(u), int(v)), 5, (0, 0, 255), -1)
                msg = f"{args.color} target u={u:.1f}, v={v:.1f}, area={target['area']:.0f}"
                color = (0, 200, 0)
            else:
                msg = f"no {args.color} target"
                color = (0, 0, 255)
            cv2.putText(preview, msg + " | s lock, q quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.imshow("visual grasp", preview)
            cv2.imshow("target mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and target is not None:
                locked = target
                locked_frame = frame.copy()
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if locked is None:
        raise RuntimeError("No target was locked.")
    return locked, locked_frame


def make_target_pose(dfarm, q_orientation_deg, xyz_m):
    T = np.array(dfarm.fkine(np.deg2rad(q_orientation_deg)))
    T[:3, 3] = xyz_m
    return T


def solve_ik(dfarm, target_T, q0_deg, label):
    sol = dfarm.ikine_LM(
        target_T,
        q0=np.deg2rad(q0_deg),
        ilimit=5000,
        slimit=5000,
        joint_limits=True,
        tol=0.001,
    )
    if not sol.success:
        raise RuntimeError(f"{label}: IK failed: {sol.reason}")
    q_deg = np.rad2deg(sol.q)
    verify_within_limits(q_deg, label)
    verify_above_desk(dfarm, q_deg, label)
    return q_deg, sol


def jtraj_deg(q0, q1, n=SEG_WAYPOINTS):
    rtb, _, _ = require_roboticstoolbox()
    return np.rad2deg(rtb.jtraj(np.deg2rad(q0), np.deg2rad(q1), n).q)


def build_plan(dfarm, target_base_m, args):
    offset_m = np.asarray(args.grasp_offset_mm, dtype=float) / 1000.0
    grasp_xyz = np.asarray(target_base_m, dtype=float) + offset_m
    grasp_xyz[2] = args.target_height_mm / 1000.0 + offset_m[2]
    safe_xyz = grasp_xyz.copy()
    safe_xyz[2] += args.approach_height_mm / 1000.0

    q_ref = np.asarray(args.orientation_q_deg, dtype=float)
    q_safe, safe_sol = solve_ik(dfarm, make_target_pose(dfarm, q_ref, safe_xyz), q_ref, "target_safe")
    q_grasp, grasp_sol = solve_ik(dfarm, make_target_pose(dfarm, q_ref, grasp_xyz), q_safe, "target_grasp")

    segments = [
        ("Park -> target_safe", "move", np.asarray(PARK_POSE_DEG, dtype=float), q_safe),
        ("Open gripper", "gripper_open", None, None),
        ("target_safe -> target_grasp", "move", q_safe, q_grasp),
        ("Close gripper", "gripper_close", None, None),
        ("target_grasp -> target_safe", "move", q_grasp, q_safe),
        ("target_safe -> Park", "move", q_safe, np.asarray(PARK_POSE_DEG, dtype=float)),
    ]
    return {
        "grasp_xyz_m": grasp_xyz,
        "safe_xyz_m": safe_xyz,
        "q_safe_deg": q_safe,
        "q_grasp_deg": q_grasp,
        "safe_residual": float(safe_sol.residual),
        "grasp_residual": float(grasp_sol.residual),
        "segments": segments,
    }


def check_trajectories(dfarm, plan):
    for label, kind, q0, q1 in plan["segments"]:
        if kind != "move":
            continue
        path = jtraj_deg(q0, q1)
        for k, q in enumerate(path):
            verify_within_limits(q, f"{label}/k={k}")
            verify_above_desk(dfarm, q, f"{label}/k={k}")


def _wait_positions_done(arm, id_list, timeout_s, label):
    t0 = time.time()
    pending = set(id_list)
    while pending:
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"{label}: timeout, pending joints={sorted(pending)}")
        done = []
        for jid in list(pending):
            try:
                v = arm.read_property(id_num=jid, property="dr.controller.position_done")
            except Exception:
                v = None
            if v == 1:
                done.append(jid)
        for jid in done:
            pending.discard(jid)
        time.sleep(WAIT_POLL_PERIOD_S)


def move_arm(arm, q_deg, label):
    print(f"[move] {label}: {np.round(q_deg, 2).tolist()}")
    if arm.set_arm_joints(angle_list=list(np.asarray(q_deg, dtype=float)), speed=JOINT_SPEED_RPM) is False:
        raise RuntimeError(f"{label}: set_arm_joints returned False")
    _wait_positions_done(arm, arm.ID_list, ARM_WAIT_TIMEOUT_S, label)


def send_gripper(arm, width_mm, force_n, label):
    angle_deg = float(width_mm) * GRIPPER_DEG_PER_MM
    r = GRIPPER_GEAR_D_MM / 2.0
    rpm = GRIPPER_SPEED_MM_S / r / (2 * np.pi) * 60
    torque_nm = float(force_n) * r / 1000.0
    print(f"[grip] {label}: width={width_mm:.2f} mm, angle={angle_deg:+.2f} deg")
    arm.set_angle_adaptive(id_num=GRIPPER_ID, angle=angle_deg, speed=rpm, torque=torque_nm)
    time.sleep(1.0)


def execute_plan(plan):
    import arm_robot as robot

    arm = robot.arm_robot(
        L_p=0,
        L_p_mass_center=0,
        MAX_list_temp=QMAX_DEG,
        MIN_list_temp=QMIN_DEG,
        G_p=0,
        com=CAN_BRIDGE_PORT,
        uart_baudrate=SERIAL_BAUD,
    )

    move_arm(arm, PARK_POSE_DEG, "park")
    for label, kind, q0, q1 in plan["segments"]:
        if kind == "move":
            move_arm(arm, q1, label)
        elif kind == "gripper_open":
            send_gripper(arm, GRIPPER_OPEN_WIDTH_MM, GRIPPER_OPEN_FORCE_N, label)
        elif kind == "gripper_close":
            send_gripper(arm, GRIPPER_CLOSE_WIDTH_MM, GRIPPER_CLOSE_FORCE_N, label)


def print_plan(pixel, board_point, base_point, plan, transform_loaded):
    print("\n=== Vision Result ===")
    print(f"Pixel: u={pixel[0]:.2f}, v={pixel[1]:.2f}")
    print(f"Board/world point (m): {np.round(board_point, 6).tolist()}")
    print(f"Robot base point (m): {np.round(base_point, 6).tolist()}")
    if not transform_loaded:
        print("WARNING: workspace transform not found; base point currently equals board point.")
        print("         Use make-workspace-transform before executing on real hardware.")

    print("\n=== Grasp Plan ===")
    print(f"grasp xyz mm: {np.round(plan['grasp_xyz_m'] * 1000, 2).tolist()}")
    print(f"safe  xyz mm: {np.round(plan['safe_xyz_m'] * 1000, 2).tolist()}")
    print(f"q_safe  deg: {np.round(plan['q_safe_deg'], 2).tolist()} residual={plan['safe_residual']:.6f}")
    print(f"q_grasp deg: {np.round(plan['q_grasp_deg'], 2).tolist()} residual={plan['grasp_residual']:.6f}")
    for i, (label, kind, _q0, q1) in enumerate(plan["segments"], start=1):
        if kind == "move":
            print(f"{i}. [move] {label:28s} -> {np.round(q1, 2).tolist()}")
        else:
            print(f"{i}. [{kind}] {label}")


def main():
    args = parse_args()
    if args.command == "make-workspace-transform":
        save_workspace_transform(args)
        return

    camera_matrix, dist_coeffs, world_to_camera = load_extrinsics(args.extrinsics)
    board_to_base_T = load_workspace_transform(args.workspace_transform)
    transform_loaded = board_to_base_T is not None
    if board_to_base_T is None:
        board_to_base_T = np.eye(4)

    target, frame = acquire_target(args)
    pixel = target["pixel"]
    board_point = pixel_to_plane(pixel, args.plane_z, camera_matrix, dist_coeffs, world_to_camera)
    base_point = board_to_base(board_point, board_to_base_T)

    dfarm = make_dfarm()
    plan = build_plan(dfarm, base_point, args)
    check_trajectories(dfarm, plan)
    print_plan(pixel, board_point, base_point, plan, transform_loaded)

    out_dir = Path(args.extrinsics).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "visual_grasp_locked.jpg"), frame)

    if args.execute:
        if not transform_loaded:
            raise RuntimeError("Refusing to execute without workspace_transform.json.")
        print("\n=== Execute Real Robot ===")
        execute_plan(plan)
    else:
        print("\nDry run only. Add --execute after checking the numbers and workspace transform.")


if __name__ == "__main__":
    main()
