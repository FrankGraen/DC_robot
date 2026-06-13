import cv2
import numpy as np


def camera_observation_to_robot_xy(pixel, table_z_mm, camera_matrix, dist_coeffs, T_robot_camera):
    u, v = pixel
    rotation_robot_camera = T_robot_camera[:3, :3]
    translation_robot_camera = T_robot_camera[:3, 3]

    pts = np.array([[[u, v]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs)
    x_norm, y_norm = undistorted[0, 0]

    ray_camera = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
    ray_robot = rotation_robot_camera @ ray_camera
    if abs(ray_robot[2]) < 1e-9:
        raise RuntimeError("camera ray is parallel to the robot table plane")

    scale = (table_z_mm - translation_robot_camera[2]) / ray_robot[2]
    point_robot = rotation_robot_camera @ (scale * ray_camera) + translation_robot_camera
    return float(point_robot[0]), float(point_robot[1])
