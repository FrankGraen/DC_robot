import math
import xml.etree.ElementTree as ET

import numpy as np


def rx(theta):
    c, s = math.cos(theta), math.sin(theta)
    t = np.eye(4)
    t[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    return t


def ry(theta):
    c, s = math.cos(theta), math.sin(theta)
    t = np.eye(4)
    t[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return t


def rz(theta):
    c, s = math.cos(theta), math.sin(theta)
    t = np.eye(4)
    t[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return t


def tx(distance):
    t = np.eye(4)
    t[0, 3] = distance
    return t


def tz(distance):
    t = np.eye(4)
    t[2, 3] = distance
    return t


def rpy_to_transform(xyz, rpy):
    t = np.eye(4)
    t[:3, 3] = xyz
    t = t @ rz(rpy[2]) @ ry(rpy[1]) @ rx(rpy[0])
    return t


def standard_dh_fk(q):
    """Standard DH table from report.md, in millimeters."""
    a = [0, 150, 150, 0, 0, 0]
    alpha = [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]
    d = [0, 0, 0, -54.94, 68, 33]
    offset = [0, 0, 0, math.pi / 2, 0, 0]

    t = np.eye(4)
    for ai, alphai, di, qi, oi in zip(a, alpha, d, q, offset):
        t = t @ rz(qi + oi) @ tz(di) @ tx(ai) @ rx(alphai)
    return t


def report_modified_dh_fk(q):
    """The modified DH table currently written in report.md, in millimeters."""
    a_prev = [0, 150, 150, 0, 0, 0]
    alpha_prev = [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]
    d = [0, 0, 0, -54.94, 68, 33]
    offset = [0, 0, 0, math.pi / 2, 0, 0]

    t = np.eye(4)
    for ai, alphai, di, qi, oi in zip(a_prev, alpha_prev, d, q, offset):
        t = t @ rx(alphai) @ tx(ai) @ rz(qi + oi) @ tz(di)
    return t


def analytical_position(q):
    """Position formula used by arm_six_axis.py::forward_kinematics_pose."""
    theta1, theta2, theta3, theta4, theta5, _theta6 = q
    l1, l2, l3, d3, d4 = 150, 150, 68, 54.94, 33
    theta23 = theta2 + theta3
    theta234 = theta23 + theta4

    x = (
        d4
        * (
            math.cos(theta5) * math.sin(theta1)
            + math.sin(theta5) * math.cos(theta1) * math.sin(theta234)
        )
        - d3 * math.sin(theta1)
        + l3 * math.cos(theta1) * math.cos(theta234)
        + l1 * math.cos(theta1) * (math.cos(theta2) + math.cos(theta23))
    )
    y = (
        -d4
        * (
            math.cos(theta5) * math.cos(theta1)
            - math.sin(theta5) * math.sin(theta1) * math.sin(theta234)
        )
        + d3 * math.cos(theta1)
        + l3 * math.sin(theta1) * math.cos(theta234)
        + l1 * math.sin(theta1) * (math.cos(theta2) + math.cos(theta23))
    )
    z = (
        -d4 * math.sin(theta5) * math.cos(theta234)
        + l1 * math.sin(theta2)
        + l2 * math.sin(theta23)
        + l3 * math.sin(theta234)
    )
    return np.array([x, y, z], dtype=float)


def urdf_zero_pose(path):
    root = ET.parse(path).getroot()
    t = np.eye(4)
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        xyz = np.fromstring(origin.attrib["xyz"], sep=" ")
        rpy = np.fromstring(origin.attrib["rpy"], sep=" ")
        t = t @ rpy_to_transform(xyz, rpy)
    return t


def print_vector(name, vector):
    print(f"{name}: [{vector[0]: .6f}, {vector[1]: .6f}, {vector[2]: .6f}]")


def main():
    tests_deg = [
        [0, 0, 0, 0, 0, 0],
        [30, 45, 60, 90, 45, 30],
        [10, -20, 35, 15, -25, 40],
    ]

    print("1) 标准 DH vs arm_six_axis.py 解析位置公式")
    for deg in tests_deg:
        q = np.deg2rad(deg)
        dh_pos = standard_dh_fk(q)[:3, 3]
        analytical_pos = analytical_position(q)
        error = np.linalg.norm(dh_pos - analytical_pos)
        print(f"q(deg)={deg}, position_error_mm={error:.12g}")

    print("\n2) report.md 当前改进 DH 表 vs 标准 DH 表")
    for deg in tests_deg:
        q = np.deg2rad(deg)
        std = standard_dh_fk(q)
        mdh = report_modified_dh_fk(q)
        pos_error = np.linalg.norm(std[:3, 3] - mdh[:3, 3])
        max_matrix_error = np.max(np.abs(std - mdh))
        print(
            f"q(deg)={deg}, position_error_mm={pos_error:.6f}, "
            f"max_matrix_error={max_matrix_error:.6f}"
        )

    print("\n3) CAD/URDF 零位尺寸对照")
    urdf = urdf_zero_pose("assets/Dr_arm_6_desk/urdf/Dr_arm_6_desk.urdf")
    dh_zero = standard_dh_fk(np.zeros(6))
    print_vector("standard_dh_zero_pos_mm", dh_zero[:3, 3])
    print_vector("urdf_zero_pos_mm", urdf[:3, 3] * 1000)
    print(
        "urdf_minus_dh_zero_mm="
        f"{np.linalg.norm(urdf[:3, 3] * 1000 - dh_zero[:3, 3]):.6f}"
    )


if __name__ == "__main__":
    main()
