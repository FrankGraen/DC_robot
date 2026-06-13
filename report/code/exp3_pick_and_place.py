import numpy as np

PARK_POSE_DEG = [0, 90, 0, 0, 0, 0]
GRIPPER_OPEN_WIDTH_MM = 6.0
GRIPPER_CLOSE_WIDTH_MM = 1.8
GRIPPER_DEG_PER_MM = -180.0 / (np.pi * 10.0)


def build_plan(waypoints):
    A_safe = waypoints[0][1]
    A_grasp = waypoints[1][1]
    B_safe = waypoints[2][1]
    B_place = waypoints[3][1]

    return [
        ("park -> A_safe", "move", PARK_POSE_DEG, A_safe),
        ("open gripper", "gripper", GRIPPER_OPEN_WIDTH_MM, None),
        ("A_safe -> A_grasp", "move", A_safe, A_grasp),
        ("close gripper", "gripper", GRIPPER_CLOSE_WIDTH_MM, None),
        ("A_grasp -> A_safe", "move", A_grasp, A_safe),
        ("A_safe -> B_safe", "move", A_safe, B_safe),
        ("B_safe -> B_place", "move", B_safe, B_place),
        ("open gripper", "gripper", GRIPPER_OPEN_WIDTH_MM, None),
        ("B_place -> B_safe", "move", B_place, B_safe),
        ("B_safe -> park", "move", B_safe, PARK_POSE_DEG),
    ]


def gripper_width_to_angle(width_mm: float) -> float:
    return float(width_mm) * GRIPPER_DEG_PER_MM
