#!/usr/bin/env python3
"""Live chessboard camera calibration with OpenCV.

Usage example:
    python3 camera_calibration.py --camera 0 --cols 7 --rows 7 --square-size 0.008

During live capture:
    s  save current detected chessboard frame
    c  calibrate with saved frames
    q  quit
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate a USB camera with a chessboard target.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0 or 1.")
    parser.add_argument("--width", type=int, default=0, help="Optional capture width.")
    parser.add_argument("--height", type=int, default=0, help="Optional capture height.")
    parser.add_argument("--cols", type=int, default=7, help="Number of inner chessboard corners per row.")
    parser.add_argument("--rows", type=int, default=7, help="Number of inner chessboard corners per column.")
    parser.add_argument("--square-size", type=float, default=0.008, help="Chessboard square size in meters.")
    parser.add_argument("--min-samples", type=int, default=15, help="Recommended minimum valid frames.")
    parser.add_argument("--output-dir", default="calibration_output", help="Directory for images and results.")
    parser.add_argument("--from-images", default="", help="Calibrate from an existing image directory instead of camera.")
    parser.add_argument("--auto", action="store_true", help="Automatically save detected chessboard frames.")
    parser.add_argument("--auto-interval", type=float, default=2.0, help="Seconds between automatic saves.")
    return parser.parse_args()


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


def collect_from_camera(args, output_dir, pattern_size):
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}. Try --camera 1 if this is an external USB camera.")

    samples = []
    auto_capture = args.auto
    last_auto_save = 0.0
    print("Camera opened.")
    print("Move the chessboard to different positions and angles.")
    print("Keys: s=save detected frame, a=toggle auto save, c=calibrate, q=quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_corners(gray, pattern_size)
            preview = frame.copy()

            if found:
                cv2.drawChessboardCorners(preview, pattern_size, corners, found)
                status = "detected"
                color = (0, 200, 0)
            else:
                status = "not detected"
                color = (0, 0, 255)

            cv2.putText(
                preview,
                f"{status} | saved: {len(samples)} | auto: {'on' if auto_capture else 'off'} | s/a/c/q",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
            cv2.imshow("camera calibration", preview)

            now = time.time()
            if found and auto_capture and now - last_auto_save >= args.auto_interval:
                filename = image_dir / f"calib_{len(samples) + 1:03d}.jpg"
                cv2.imwrite(str(filename), frame)
                samples.append(filename)
                last_auto_save = now
                print(f"Auto saved {filename}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                auto_capture = not auto_capture
                print(f"Auto save {'enabled' if auto_capture else 'disabled'}.")
            if key == ord("s"):
                if not found:
                    print("Chessboard not detected; frame was not saved.")
                    continue
                filename = image_dir / f"calib_{len(samples) + 1:03d}.jpg"
                cv2.imwrite(str(filename), frame)
                samples.append(filename)
                print(f"Saved {filename}")
            if key == ord("c"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(samples) < args.min_samples:
        print(f"Warning: only {len(samples)} valid frames saved. {args.min_samples}+ is recommended.")
    return samples


def collect_from_images(image_dir, pattern_size):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for ext in exts:
        images.extend(sorted(Path(image_dir).glob(ext)))
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    valid_images = []
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, _ = find_corners(gray, pattern_size)
        if found:
            valid_images.append(path)
            print(f"Detected chessboard: {path}")
        else:
            print(f"Skipped, no chessboard: {path}")
    return valid_images


def calibrate(image_paths, args, output_dir, pattern_size):
    if not image_paths:
        raise RuntimeError("No valid calibration images.")

    objp = make_object_points(args.cols, args.rows, args.square_size)
    objpoints = []
    imgpoints = []
    image_size = None
    last_image = None

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray, pattern_size)
        if not found:
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners)
        image_size = gray.shape[::-1]
        last_image = img

    if len(objpoints) < 3:
        raise RuntimeError(f"Only {len(objpoints)} valid images. Capture more chessboard views.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    per_view_errors = []
    total_error = 0.0
    total_points = 0
    for i, object_points in enumerate(objpoints):
        projected, _ = cv2.projectPoints(object_points, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        point_count = len(projected)
        per_view_errors.append(error / point_count)
        total_error += error * error
        total_points += point_count
    mean_error = float(np.sqrt(total_error / total_points))

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, image_size, 1, image_size
    )

    result = {
        "image_size": list(image_size),
        "pattern_size": [args.cols, args.rows],
        "square_size": args.square_size,
        "valid_image_count": len(objpoints),
        "rms_reprojection_error": float(rms),
        "mean_reprojection_error_px": mean_error,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "new_camera_matrix": new_camera_matrix.tolist(),
        "roi": [int(v) for v in roi],
        "per_view_error_px": [float(v) for v in per_view_errors],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "camera_calibration.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    np.savez(
        output_dir / "camera_calibration.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        new_camera_matrix=new_camera_matrix,
        roi=np.array(roi),
        image_size=np.array(image_size),
    )

    if last_image is not None:
        undistorted = cv2.undistort(last_image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y : y + h, x : x + w]
        cv2.imwrite(str(output_dir / "undistorted_example.jpg"), undistorted)

    print("\nCalibration complete.")
    print(f"Valid images: {len(objpoints)}")
    print(f"RMS reprojection error: {rms:.6f}")
    print(f"Mean reprojection error: {mean_error:.6f} px")
    print(f"Saved: {output_dir / 'camera_calibration.json'}")
    print(f"Saved: {output_dir / 'camera_calibration.npz'}")
    print(f"Saved: {output_dir / 'undistorted_example.jpg'}")
    return result


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    pattern_size = (args.cols, args.rows)

    if args.from_images:
        image_paths = collect_from_images(args.from_images, pattern_size)
    else:
        image_paths = collect_from_camera(args, output_dir, pattern_size)

    calibrate(image_paths, args, output_dir, pattern_size)


if __name__ == "__main__":
    main()
