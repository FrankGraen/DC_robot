#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an ultralytics train/val split from yolo/images + yolo/labels.

Produces (under yolo/dataset):
    images/train images/val   -> symlinks to the real raw PNGs
    labels/train labels/val   -> copies of the YOLO .txt labels
    ../data.yaml              -> dataset config for `yolo detect train`

Images with no label file get an empty label (treated as background).
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def run(args: argparse.Namespace) -> None:
    images_dir = THIS_DIR / "images"
    labels_dir = THIS_DIR / "labels"
    dataset_dir = THIS_DIR / "dataset"

    stems = sorted(p.stem for p in images_dir.glob("*.png"))
    if not stems:
        raise SystemExit(f"no images in {images_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(stems)
    n_val = max(1, round(len(stems) * args.val_frac))
    val_set = set(stems[:n_val])

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for stem in stems:
        split = "val" if stem in val_set else "train"
        # symlink the real PNG (resolve through the yolo/images symlink)
        src_png = (images_dir / f"{stem}.png").resolve()
        link = dataset_dir / "images" / split / f"{stem}.png"
        link.symlink_to(src_png)
        # copy the label (empty file if missing -> background image)
        src_txt = labels_dir / f"{stem}.txt"
        dst_txt = dataset_dir / "labels" / split / f"{stem}.txt"
        if src_txt.exists():
            shutil.copyfile(src_txt, dst_txt)
        else:
            dst_txt.write_text("", encoding="utf-8")
        counts[split] += 1

    data_yaml = THIS_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: red\n"
        f"  1: green\n"
        f"  2: blue\n",
        encoding="utf-8",
    )

    print(f"train: {counts['train']}  val: {counts['val']}  (seed={args.seed}, val_frac={args.val_frac})")
    print(f"data.yaml -> {data_yaml}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make train/val split + data.yaml for YOLO")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
