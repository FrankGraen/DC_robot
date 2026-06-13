#!/usr/bin/env python3
"""Reusable joint-action simulation and execution helpers.

Each action row is:
    [j1, j2, j3, j4, j5, j6, grip]

Joint angles are model angles in degrees. grip is 0 for open and 1 for
closed. The execution semantic is: move to the six joint angles first, then
send the gripper command for that row.

Real execution interpolates between joint action rows in joint space. The
gripper command is still sent only after reaching each key row.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np


Action = Sequence[float]
DEFAULT_JOINT_LIMITS_MIN = [-132.4, 17.7, -153.8, -100.6, -85.6, -219.2]
DEFAULT_JOINT_LIMITS_MAX = [131.3, 182.9, 42.4, 130.2, 226.6, 153.5]


@dataclass(frozen=True)
class KinematicsContext:
    """Standard-DH kinematic configuration.

    Length units are millimeters; angles are radians.
    """

    dh_a: Sequence[float]
    dh_alpha: Sequence[float]
    dh_d: Sequence[float]
    dh_offset: Sequence[float]

    @classmethod
    def default_daran(cls) -> "KinematicsContext":
        return cls(
            dh_a=[0, 150, 150, 0, 0, 0],
            dh_alpha=[math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0],
            dh_d=[0, 0, 0, -54.94, 68, 33],
            dh_offset=[0, 0, 0, math.pi / 2, 0, 0],
        )


@dataclass
class MotionConfig:
    joint_speed: float = 2.0
    gripper_open_width: float = 5.0
    gripper_close_width: float = 0.0
    gripper_speed: float = 10.0
    gripper_force: float = 80.0
    gripper_wait_timeout: float = 5.0
    gripper_poll: float = 0.1
    gripper_pre_wait: float = 0.5
    gripper_open_loop_wait: float = 1.0
    settle: float = 0.3
    wait_pose_done: bool = True
    joint_wait_timeout: float = 8.0
    joint_poll: float = 0.1
    joint_tolerance: float = 2.0
    joint_retry_count: int = 3
    wait_gripper_done: bool = False
    execution_points_per_segment: int = 10
    wait_intermediate_pose_done: bool = False
    intermediate_settle: float = 0.0

    def __post_init__(self) -> None:
        if self.gripper_open_width <= 0:
            raise ValueError("gripper_open_width must be > 0 mm")
        if self.gripper_close_width < 0:
            raise ValueError("gripper_close_width must be >= 0 mm")
        if self.gripper_speed <= 0:
            raise ValueError("gripper_speed must be > 0 mm/s")
        if self.gripper_force <= 0:
            raise ValueError("gripper_force must be > 0 N")


class JointActionMotion:
    """Bundle simulation and execution resources for joint action lists."""

    def __init__(
        self,
        com: str | None = None,
        baudrate: int = 115200,
        tool_length: float = 0.0,
        tool_mass_center: float = 0.0,
        payload: float = 0.0,
        joint_limits_max: Sequence[float] | None = DEFAULT_JOINT_LIMITS_MAX,
        joint_limits_min: Sequence[float] | None = DEFAULT_JOINT_LIMITS_MIN,
        kin: KinematicsContext | None = None,
        cfg: MotionConfig | None = None,
        points_per_segment: int = 40,
        fps: int = 30,
        ik_tol: float = 1e-8,
        ik_ilimit: int = 1000,
        ik_slimit: int = 1000,
        connect: bool | None = None,
    ) -> None:
        self.kin = kin if kin is not None else KinematicsContext.default_daran()
        self.cfg = cfg if cfg is not None else MotionConfig()
        self.points_per_segment = points_per_segment
        self.fps = fps
        self.ik_tol = ik_tol
        self.ik_ilimit = ik_ilimit
        self.ik_slimit = ik_slimit
        self.com = com
        self.baudrate = baudrate
        self.tool_length = tool_length
        self.tool_mass_center = tool_mass_center
        self.payload = payload
        self.joint_limits_max = joint_limits_max
        self.joint_limits_min = joint_limits_min
        self.ro: object | None = None

        should_connect = bool(com) if connect is None else connect
        if should_connect:
            self.ro = self._make_robot()

    def _make_robot(self) -> object:
        import arm_robot as robot

        kwargs = {
            "L_p": self.tool_length,
            "L_p_mass_center": self.tool_mass_center,
            "G_p": self.payload,
            "com": self.com or "",
            "uart_baudrate": self.baudrate,
        }
        if self.joint_limits_max is not None:
            kwargs["MAX_list_temp"] = list(self.joint_limits_max)
        if self.joint_limits_min is not None:
            kwargs["MIN_list_temp"] = list(self.joint_limits_min)
        return robot.arm_robot(**kwargs)

    def sim(
        self,
        actions: Sequence[Action],
        points_per_segment: int | None = None,
        fps: int | None = None,
    ) -> None:
        _show_sim(
            actions,
            self.kin,
            points_per_segment=self.points_per_segment if points_per_segment is None else points_per_segment,
            fps=self.fps if fps is None else fps,
        )

    def run(self, actions: Sequence[Action]) -> None:
        if self.ro is None:
            self.ro = self._make_robot()
        _execute_actions(actions, self.ro, self.cfg)

    def solve(
        self,
        pose_actions: Sequence[Action],
        initial_joints: Sequence[float],
    ) -> list[list[float]]:
        """Convert [x,y,z,pitch,yaw,roll,grip] rows to [j1..j6,grip] rows."""

        from roboticstoolbox import DHRobot, RevoluteDH
        from spatialmath import SE3

        rows = _validate_actions(pose_actions)
        if len(initial_joints) != 6:
            raise ValueError("initial_joints must contain exactly 6 joint angles")

        qlim = None
        if self.joint_limits_min is not None and self.joint_limits_max is not None:
            qlim = np.array([self.joint_limits_min, self.joint_limits_max], dtype=float).T * math.pi / 180.0

        links = []
        for i, (a, alpha, d, offset) in enumerate(
            zip(self.kin.dh_a, self.kin.dh_alpha, self.kin.dh_d, self.kin.dh_offset)
        ):
            limit = None if qlim is None else qlim[i]
            links.append(RevoluteDH(a=a / 1000.0, d=d / 1000.0, alpha=alpha, offset=offset, qlim=limit))
        ik_robot = DHRobot(links, name="JointActionMotionIK")

        solved: list[list[float]] = []
        previous_q = np.deg2rad(np.array(initial_joints, dtype=float))
        seed_offsets = np.deg2rad(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [15, 0, 0, 0, 0, 0],
                    [-15, 0, 0, 0, 0, 0],
                    [0, 15, -15, 0, 0, 0],
                    [0, -15, 15, 0, 0, 0],
                    [0, 0, 0, 0, 20, 20],
                    [0, 0, 0, 0, -20, -20],
                ],
                dtype=float,
            )
        )

        for i, row in enumerate(rows):
            x, y, z, pitch, yaw, roll = [float(v) for v in row[:6]]
            target = np.eye(4)
            target[:3, :3] = _rpy_to_rot(math.radians(roll), math.radians(pitch), math.radians(yaw))
            target[:3, 3] = np.array([x, y, z], dtype=float) / 1000.0

            candidates: list[tuple[float, np.ndarray, float]] = []
            for q0 in [previous_q + offset for offset in seed_offsets]:
                sol = ik_robot.ikine_LM(
                    SE3(target),
                    q0=q0,
                    ilimit=self.ik_ilimit,
                    slimit=self.ik_slimit,
                    joint_limits=qlim is not None,
                    tol=self.ik_tol,
                )
                if sol.success:
                    q = np.asarray(sol.q, dtype=float)
                    delta = (q - previous_q + math.pi) % (2 * math.pi) - math.pi
                    diff = float(np.linalg.norm(delta))
                    residual = float(sol.residual if sol.residual is not None else 0.0)
                    candidates.append((diff, q, residual))
            if not candidates:
                raise RuntimeError(f"pose action {i} IK failed")

            _, previous_q, residual = min(candidates, key=lambda item: (item[0], item[2]))
            solved.append([round(math.degrees(v), 3) for v in previous_q] + [int(row[6])])
            print(f"[solve] pose {i}: residual={residual:.6g}, joints={solved[-1][:6]}")

        return solved


def _validate_actions(actions: Sequence[Action]) -> list[list[float]]:
    rows = [list(row) for row in actions]
    if not rows:
        raise ValueError("actions must contain at least one row")
    for i, row in enumerate(rows):
        if len(row) != 7:
            raise ValueError(f"action {i} must have 7 values: j1..j6, grip")
        grip = int(row[6])
        if grip not in (0, 1):
            raise ValueError(f"action {i} grip must be 0 or 1, got {row[6]}")
    return rows


def _rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _fk_frames(joints_deg: Sequence[float], kin: KinematicsContext) -> list[np.ndarray]:
    """Return base + six joint frames for one model-joint pose."""

    if len(joints_deg) != 6:
        raise ValueError("joints_deg must contain exactly 6 joint angles")

    frames = [np.eye(4)]
    transform = np.eye(4)
    for a, alpha, d, q_deg, offset in zip(
        kin.dh_a,
        kin.dh_alpha,
        kin.dh_d,
        joints_deg,
        kin.dh_offset,
    ):
        q = math.radians(float(q_deg))
        cq, sq = math.cos(q + offset), math.sin(q + offset)
        ca, sa = math.cos(alpha), math.sin(alpha)

        rz = np.eye(4)
        rz[:3, :3] = [[cq, -sq, 0], [sq, cq, 0], [0, 0, 1]]

        tz = np.eye(4)
        tz[2, 3] = d

        tx = np.eye(4)
        tx[0, 3] = a

        rx = np.eye(4)
        rx[:3, :3] = [[1, 0, 0], [0, ca, -sa], [0, sa, ca]]

        transform = transform @ rz @ tz @ tx @ rx
        frames.append(transform.copy())
    return frames


def _link_points(joints_deg: Sequence[float], kin: KinematicsContext) -> np.ndarray:
    return np.array([frame[:3, 3] for frame in _fk_frames(joints_deg, kin)])


def _sim_frames(
    actions: Sequence[Action],
    points_per_segment: int,
) -> tuple[np.ndarray, list[int], list[str]]:
    rows = _validate_actions(actions)
    if points_per_segment < 1:
        raise ValueError("points_per_segment must be >= 1")

    frames: list[np.ndarray] = []
    grips: list[int] = []
    labels: list[str] = []

    first = np.array(rows[0][:6], dtype=float)
    frames.append(first)
    grips.append(int(rows[0][6]))
    labels.append("action 0")

    hold_frames = max(1, points_per_segment // 5)
    for i in range(1, len(rows)):
        q0 = np.array(rows[i - 1][:6], dtype=float)
        q1 = np.array(rows[i][:6], dtype=float)
        moving_grip = int(rows[i - 1][6])
        arrival_grip = int(rows[i][6])

        for s in np.linspace(0.0, 1.0, points_per_segment + 1)[1:]:
            frames.append((1.0 - s) * q0 + s * q1)
            grips.append(moving_grip)
            labels.append(f"action {i - 1} -> {i}")

        for _ in range(hold_frames):
            frames.append(q1.copy())
            grips.append(arrival_grip)
            labels.append(f"action {i} grip")

    return np.array(frames), grips, labels


def _show_sim(
    actions: Sequence[Action],
    kin: KinematicsContext,
    points_per_segment: int = 40,
    fps: int = 30,
) -> None:
    """Show an animated 3D joint-space preview window."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  side-effect import for 3D

    q_frames, grip_frames, labels = _sim_frames(actions, points_per_segment)
    link_frames = [_link_points(q, kin) for q in q_frames]
    ee_xyz = np.array([points[-1] for points in link_frames])

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    all_link_points = np.concatenate(link_frames, axis=0)
    margin = 40.0
    x_lo = min(float(all_link_points[:, 0].min()), 0.0) - margin
    x_hi = max(float(all_link_points[:, 0].max()), 0.0) + margin
    y_lo = min(float(all_link_points[:, 1].min()), 0.0) - margin
    y_hi = max(float(all_link_points[:, 1].max()), 0.0) + margin
    z_lo = min(float(all_link_points[:, 2].min()), 0.0) - 20.0
    z_hi = max(float(all_link_points[:, 2].max()) + margin, 250.0)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Joint Action Motion Preview")
    ax.view_init(elev=25, azim=-60)

    xx, yy = np.meshgrid(np.linspace(x_lo, x_hi, 7), np.linspace(y_lo, y_hi, 7))
    ax.plot_wireframe(xx, yy, np.zeros_like(xx), color="lightgray", linewidth=0.4, alpha=0.6)
    ax.scatter([0], [0], [0], c="black", s=40, marker="s")

    for i, row in enumerate(_validate_actions(actions)):
        xyz = _link_points(row[:6], kin)[-1]
        ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], s=45, depthshade=False)
        ax.text(xyz[0], xyz[1], xyz[2] + 8, f"{i}", fontsize=9)

    (arm_line,) = ax.plot([], [], [], "-o", color="tab:blue", lw=3, ms=4, label="arm")
    (trail_line,) = ax.plot([], [], [], "-", color="tab:purple", lw=1.2, alpha=0.7, label="EE trail")
    (ee_marker,) = ax.plot([], [], [], "o", color="black", ms=6)
    frame_text = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, fontsize=10, va="top")
    grip_text = ax.text2D(0.02, 0.92, "", transform=ax.transAxes, fontsize=10, va="top")
    ax.legend(loc="upper right", fontsize=8)

    def init():
        arm_line.set_data([], [])
        arm_line.set_3d_properties([])
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        ee_marker.set_data([], [])
        ee_marker.set_3d_properties([])
        frame_text.set_text("")
        grip_text.set_text("")
        return arm_line, trail_line, ee_marker, frame_text, grip_text

    def update(i: int):
        points = link_frames[i]
        arm_line.set_data(points[:, 0], points[:, 1])
        arm_line.set_3d_properties(points[:, 2])
        trail_line.set_data(ee_xyz[: i + 1, 0], ee_xyz[: i + 1, 1])
        trail_line.set_3d_properties(ee_xyz[: i + 1, 2])
        ee_marker.set_data([ee_xyz[i, 0]], [ee_xyz[i, 1]])
        ee_marker.set_3d_properties([ee_xyz[i, 2]])
        frame_text.set_text(f"frame {i + 1}/{len(q_frames)} | {labels[i]}")
        grip_text.set_text(f"gripper: {'CLOSED' if grip_frames[i] else 'OPEN'}")
        grip_text.set_color("tab:red" if grip_frames[i] else "tab:green")
        return arm_line, trail_line, ee_marker, frame_text, grip_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(q_frames),
        init_func=init,
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )
    plt.show()
    _ = anim


def _wait_gripper_done(ro: object, cfg: MotionConfig) -> bool:
    deadline = time.time() + cfg.gripper_wait_timeout
    while time.time() < deadline:
        done = ro.read_property(7, "dr.controller.position_done")
        if done == 1:
            return True
        time.sleep(cfg.gripper_poll)
    return False


def _wait_joints_reached(ro: object, joints: Sequence[float], cfg: MotionConfig, label: str) -> tuple[np.ndarray, np.ndarray] | None:
    target = np.array([float(v) for v in joints], dtype=float)
    deadline = time.time() + cfg.joint_wait_timeout
    last_angles: np.ndarray | None = None
    last_error: np.ndarray | None = None

    while time.time() < deadline:
        servo_angles = ro.read_joints()
        if servo_angles is not False:
            current = np.array(ro.servo_to_model(servo_angle_list=servo_angles), dtype=float)
            error = np.abs(current - target)
            last_angles = current
            last_error = error
            if bool(np.all(error <= cfg.joint_tolerance)):
                return None
        time.sleep(cfg.joint_poll)

    if last_angles is None or last_error is None:
        raise RuntimeError(f"{label} joint wait failed: could not read joints")
    return last_angles, last_error


def _ensure_joints_reached(ro: object, joints: Sequence[float], cfg: MotionConfig, label: str) -> None:
    wait_result = _wait_joints_reached(ro, joints, cfg, label)
    if wait_result is None:
        return
    angles, error = wait_result
    raise RuntimeError(
        f"{label} joint check failed before gripper; "
        f"current={[round(v, 2) for v in angles.tolist()]}, "
        f"target={[round(float(v), 2) for v in joints]}, "
        f"error={[round(v, 2) for v in error.tolist()]}"
    )


def _execute_joint_waypoint(
    ro: object,
    joints: Sequence[float],
    speed: float,
    label: str,
    wait_done: bool,
    cfg: MotionConfig,
) -> None:
    target = [float(v) for v in joints]
    attempts = max(1, int(cfg.joint_retry_count) + 1) if wait_done else 1
    last_angles: np.ndarray | None = None
    last_error: np.ndarray | None = None

    for attempt in range(1, attempts + 1):
        ok = ro.set_arm_joints(angle_list=target, speed=speed)
        if ok is False:
            raise RuntimeError(f"{label} joint command failed")
        if not wait_done:
            return

        wait_result = _wait_joints_reached(ro, target, cfg, label)
        if wait_result is None:
            if attempt > 1:
                print(f"[motion] {label}: reached after retry {attempt - 1}/{attempts - 1}")
            return

        last_angles, last_error = wait_result
        if attempt < attempts:
            print(
                f"[warn] {label}: joint wait timed out, retry {attempt}/{attempts - 1}; "
                f"error={[round(v, 2) for v in last_error.tolist()]}"
            )

    if last_angles is None or last_error is None:
        raise RuntimeError(f"{label} joint wait failed")
    raise RuntimeError(
        f"{label} joint wait timed out after {cfg.joint_wait_timeout:.1f}s x {attempts}; "
        f"current={[round(v, 2) for v in last_angles.tolist()]}, "
        f"target={[round(v, 2) for v in target]}, "
        f"error={[round(v, 2) for v in last_error.tolist()]}"
    )


def _execute_actions(actions: Sequence[Action], ro: object, cfg: MotionConfig) -> None:
    """Execute key actions without interpolation between rows."""

    rows = _validate_actions(actions)

    first_grip = int(rows[0][6])
    first_width = cfg.gripper_close_width if first_grip else cfg.gripper_open_width
    if first_width < 0:
        raise ValueError(f"initial gripper width must be >= 0 mm, got {first_width}")
    print(f"[gripper] initial: grip={first_grip}, width={first_width:.1f}mm")
    ro.grasp(wideth=first_width, speed=cfg.gripper_speed, force=cfg.gripper_force)
    if cfg.wait_gripper_done and not _wait_gripper_done(ro, cfg):
        print(f"[warn] initial gripper wait timed out after {cfg.gripper_wait_timeout:.1f}s")
    elif not cfg.wait_gripper_done and cfg.gripper_open_loop_wait > 0:
        time.sleep(cfg.gripper_open_loop_wait)

    for i, row in enumerate(rows):
        joints = [float(v) for v in row[:6]]
        grip = int(row[6])
        width = cfg.gripper_close_width if grip else cfg.gripper_open_width

        print(f"[motion] action {i}: joints={[round(v, 2) for v in joints]}, grip={grip}")
        _execute_joint_waypoint(
            ro,
            joints,
            cfg.joint_speed,
            label=f"action {i}",
            wait_done=cfg.wait_pose_done,
            cfg=cfg,
        )
        _ensure_joints_reached(ro, joints, cfg, label=f"action {i}")
        if cfg.gripper_pre_wait > 0:
            time.sleep(cfg.gripper_pre_wait)

        if width < 0:
            raise ValueError(f"action {i} gripper width must be >= 0 mm, got {width}")
        print(f"[gripper] action {i}: width={width:.1f}mm")
        ro.grasp(wideth=width, speed=cfg.gripper_speed, force=cfg.gripper_force)
        if cfg.wait_gripper_done and not _wait_gripper_done(ro, cfg):
            print(f"[warn] action {i}: gripper wait timed out after {cfg.gripper_wait_timeout:.1f}s")
        elif not cfg.wait_gripper_done and cfg.gripper_open_loop_wait > 0:
            time.sleep(cfg.gripper_open_loop_wait)

        if cfg.settle > 0:
            time.sleep(cfg.settle)


if __name__ == "__main__":
    # test_actions = [
    #     [32.2, 51.5, -32.9, -24.0, 84.7, 28.4, 0],
    #     [32.7, 58.7, -66.9, 3.3, 84.3, 33.5, 1],
    #     [32.2, 51.5, -32.9, -24.0, 84.7, 28.4, 1],
    #     [-6.1, 61.1, -46.4, -17.8, 84.7, -2.9, 1],
    #     [-5.4, 64.5, -76.7, 8.6, 83.4, -1.8, 0],
    #     [-6.1, 61.1, -46.4, -17.8, 84.7, -2.9, 0],
    # ]
    # motion = JointActionMotion(com=None)
    # motion.sim(test_actions)
    test_pose_actions = [
        [226, 203, 120, 0, -90, 180, 0],
        [226, 203, 90, 0, -90, 180, 1],
        [226, 203, 120, 0, -90, 180, 1],
        [267, 19, 120, 0, -90, 180, 1],
        [267, 19, 90, 0, -90, 180, 0],
        [267, 19, 120, 0, -90, 180, 0],
    ]
    motion = JointActionMotion(com="/dev/cu.usbmodem94869FA301481")
    if motion.ro is None:
        raise RuntimeError("robot connection was not initialized")
    current_joints = motion.ro.detect_joints()
    if current_joints is False or len(current_joints) != 6:
        raise RuntimeError("failed to read current robot joints")
    current_joints = [float(v) for v in current_joints]
    print(f"[motion] current joints={[round(v, 2) for v in current_joints]}")

    test_actions = motion.solve(
        test_pose_actions,
        initial_joints=current_joints,
    )
    test_actions = [current_joints + [int(test_pose_actions[0][6])]] + test_actions
    motion.sim(test_actions)
    motion.run(test_actions)
