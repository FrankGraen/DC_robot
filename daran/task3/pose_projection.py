#!/usr/bin/env python3
"""Estimate camera pose and project points with a calibrated camera.

This script is the practical version of "space point projection":

1. Put a chessboard on the working plane.
2. Estimate the camera extrinsic matrix from the chessboard.
3. Project 3D points to image pixels, or back-project image pixels to the plane.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Camera extrinsic calibration and point projection.")
    parser.add_argument(
        "--intrinsics",
        default="calibration_output/camera_calibration.json",
        help="Path to camera_calibration.json from camera_calibration.py.",
    )
    parser.add_argument("--extrinsics", default="calibration_output/camera_extrinsic.json")

    subparsers = parser.add_subparsers(dest="command", required=True)

    pose_live = subparsers.add_parser("pose-live", help="Estimate chessboard pose from live camera.")
    pose_live.add_argument("--camera", type=int, default=4)
    pose_live.add_argument("--width", type=int, default=0)
    pose_live.add_argument("--height", type=int, default=0)
    pose_live.add_argument("--cols", type=int, default=7, help="Inner corners per row.")
    pose_live.add_argument("--rows", type=int, default=7, help="Inner corners per column.")
    pose_live.add_argument("--square-size", type=float, default=0.008, help="Chessboard square size in meters.")

    pose_image = subparsers.add_parser("pose-image", help="Estimate chessboard pose from one image.")
    pose_image.add_argument("--image", required=True)
    pose_image.add_argument("--cols", type=int, default=7)
    pose_image.add_argument("--rows", type=int, default=7)
    pose_image.add_argument("--square-size", type=float, default=0.008)

    project = subparsers.add_parser("project", help="Project one 3D world point to a pixel.")
    project.add_argument("--point", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))

    back_project = subparsers.add_parser("pixel-to-plane", help="Back-project one pixel to a world Z plane.")
    back_project.add_argument("--pixel", nargs=2, type=float, required=True, metavar=("U", "V"))
    back_project.add_argument("--z", type=float, default=0.0, help="World plane height in meters.")

    return parser.parse_args()


def load_intrinsics(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    return camera_matrix, dist_coeffs


def make_object_points(cols, rows, square_size):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def find_corners(gray, pattern_size):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def transform_from_rvec_tvec(rvec, tvec):
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    return transform


def reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    error = cv2.norm(image_points, projected, cv2.NORM_L2) / len(projected)
    return float(error)


def draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, square_size):
    axis_len = square_size * 3.0
    axis = np.float32(
        [
            [0, 0, 0],
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, -axis_len],
        ]
    )
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    pts = imgpts.reshape(-1, 2).astype(int)
    origin = tuple(pts[0])
    frame = cv2.line(frame, origin, tuple(pts[1]), (0, 0, 255), 3)
    frame = cv2.line(frame, origin, tuple(pts[2]), (0, 255, 0), 3)
    frame = cv2.line(frame, origin, tuple(pts[3]), (255, 0, 0), 3)
    cv2.putText(frame, "X", tuple(pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "Y", tuple(pts[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "Z", tuple(pts[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame


def save_extrinsics(path, camera_matrix, dist_coeffs, rvec, tvec, object_points, image_points, cols, rows, square_size):
    world_to_camera = transform_from_rvec_tvec(rvec, tvec)
    camera_to_world = np.linalg.inv(world_to_camera)
    error = reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)

    data = {
        "description": "world is the chessboard coordinate system; origin is the first detected inner corner",
        "pattern_size": [cols, rows],
        "square_size": square_size,
        "reprojection_error_px": error,
        "rvec": rvec.reshape(3).tolist(),
        "tvec": tvec.reshape(3).tolist(),
        "world_to_camera": world_to_camera.tolist(),
        "camera_to_world": camera_to_world.tolist(),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    np.savez(
        out.with_suffix(".npz"),
        world_to_camera=world_to_camera,
        camera_to_world=camera_to_world,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvec=rvec,
        tvec=tvec,
    )
    print(f"Saved extrinsics: {out}")
    print(f"Saved extrinsics: {out.with_suffix('.npz')}")
    print(f"Reprojection error: {error:.6f} px")


def estimate_pose_from_frame(frame, camera_matrix, dist_coeffs, cols, rows, square_size):
    pattern_size = (cols, rows)
    object_points = make_object_points(cols, rows, square_size)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = find_corners(gray, pattern_size)
    if not found:
        return False, None, None, None, object_points

    ok, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
    if not ok:
        return False, None, None, corners, object_points
    return True, rvec, tvec, corners, object_points


def run_pose_live(args, camera_matrix, dist_coeffs):
    cap = cv2.VideoCapture(args.camera)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    print("Put the chessboard flat on the workspace.")
    print("Keys: s=save current extrinsic, q=quit")

    last_pose = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            found, rvec, tvec, corners, object_points = estimate_pose_from_frame(
                frame, camera_matrix, dist_coeffs, args.cols, args.rows, args.square_size
            )
            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, (args.cols, args.rows), corners, found)
                preview = draw_axes(preview, camera_matrix, dist_coeffs, rvec, tvec, args.square_size)
                error = reprojection_error(object_points, corners, rvec, tvec, camera_matrix, dist_coeffs)
                cv2.putText(
                    preview,
                    f"detected | error {error:.3f}px | s save, q quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 0),
                    2,
                )
                last_pose = (rvec, tvec, corners, object_points, preview)
            else:
                cv2.putText(
                    preview,
                    "not detected | s save, q quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("camera pose", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if last_pose is None:
                    print("No valid pose to save.")
                    continue
                rvec, tvec, corners, object_points, saved_preview = last_pose
                save_extrinsics(
                    args.extrinsics,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    object_points,
                    corners,
                    args.cols,
                    args.rows,
                    args.square_size,
                )
                preview_path = Path(args.extrinsics).with_name("camera_extrinsic_preview.jpg")
                cv2.imwrite(str(preview_path), saved_preview)
                print(f"Saved preview: {preview_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_pose_image(args, camera_matrix, dist_coeffs):
    frame = cv2.imread(args.image)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    found, rvec, tvec, corners, object_points = estimate_pose_from_frame(
        frame, camera_matrix, dist_coeffs, args.cols, args.rows, args.square_size
    )
    if not found:
        raise RuntimeError("Chessboard was not detected in this image.")

    save_extrinsics(
        args.extrinsics,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        object_points,
        corners,
        args.cols,
        args.rows,
        args.square_size,
    )
    preview = frame.copy()
    cv2.drawChessboardCorners(preview, (args.cols, args.rows), corners, True)
    preview = draw_axes(preview, camera_matrix, dist_coeffs, rvec, tvec, args.square_size)
    preview_path = Path(args.extrinsics).with_name("camera_extrinsic_preview.jpg")
    cv2.imwrite(str(preview_path), preview)
    print(f"Saved preview: {preview_path}")


def load_extrinsics(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    world_to_camera = np.array(data["world_to_camera"], dtype=np.float64)
    return camera_matrix, dist_coeffs, world_to_camera


def project_point(args):
    camera_matrix, dist_coeffs, world_to_camera = load_extrinsics(args.extrinsics)
    point_world = np.array(args.point, dtype=np.float64).reshape(1, 1, 3)
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    rvec, _ = cv2.Rodrigues(rotation)

    image_points, _ = cv2.projectPoints(point_world, rvec, translation, camera_matrix, dist_coeffs)
    u, v = image_points.reshape(2)

    point_h = np.array([args.point[0], args.point[1], args.point[2], 1.0])
    point_camera = world_to_camera @ point_h
    print(f"World point: {args.point}")
    print(f"Camera point: {point_camera[:3].tolist()}")
    print(f"Pixel: u={u:.3f}, v={v:.3f}")


def pixel_to_plane(args):
    camera_matrix, dist_coeffs, world_to_camera = load_extrinsics(args.extrinsics)
    camera_to_world = np.linalg.inv(world_to_camera)

    pixel = np.array(args.pixel, dtype=np.float64).reshape(1, 1, 2)
    normalized = cv2.undistortPoints(pixel, camera_matrix, dist_coeffs).reshape(2)
    ray_camera = np.array([normalized[0], normalized[1], 1.0])

    ray_origin_world = camera_to_world[:3, 3]
    ray_dir_world = camera_to_world[:3, :3] @ ray_camera

    if abs(ray_dir_world[2]) < 1e-12:
        raise RuntimeError("The ray is parallel to the target plane.")
    scale = (args.z - ray_origin_world[2]) / ray_dir_world[2]
    point_world = ray_origin_world + scale * ray_dir_world

    print(f"Pixel: {args.pixel}")
    print(f"World plane z: {args.z}")
    print(f"World point: x={point_world[0]:.6f}, y={point_world[1]:.6f}, z={point_world[2]:.6f}")


def main():
    args = parse_args()
    camera_matrix, dist_coeffs = load_intrinsics(args.intrinsics)

    if args.command == "pose-live":
        run_pose_live(args, camera_matrix, dist_coeffs)
    elif args.command == "pose-image":
        run_pose_image(args, camera_matrix, dist_coeffs)
    elif args.command == "project":
        project_point(args)
    elif args.command == "pixel-to-plane":
        pixel_to_plane(args)


if __name__ == "__main__":
    main()
