#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train yolo11n on the color-block dataset.

Key choice: hsv_h=0.0 -- the class IS the color, so hue must NOT be augmented
(shifting hue would turn a red block into orange and corrupt the label). Geometry
augmentation is kept generous because the camera is top-down and block orientation
is arbitrary.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

THIS_DIR = Path(__file__).resolve().parent


def main() -> None:
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(THIS_DIR / "data.yaml"),
        epochs=120,
        imgsz=960,
        batch=16,
        device=0,
        seed=0,
        patience=40,
        # augmentation
        hsv_h=0.0,      # do NOT shift hue: color == class
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=180.0,  # top-down, any in-plane rotation is valid
        fliplr=0.5,
        flipud=0.5,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        # output
        project=str(THIS_DIR / "runs"),
        name="blocks",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
