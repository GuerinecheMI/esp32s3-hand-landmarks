#!/usr/bin/env python3
"""
prepare_dataset.py – Extract and convert hand pose datasets (e.g. Ultralytics
hand-keypoints or YOLO pose format) into 96x96 grayscale crops with normalized
21-point landmark coordinates for ESP32-S3 model training.

Usage:
    python prepare_dataset.py --zip hand-keypoints.zip --max-samples 1500 --output dataset/
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

INPUT_SIZE = 96
NUM_LANDMARKS = 21


def pad_and_crop(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[np.ndarray, int, int]:
    """Crop image to [x1:x2, y1:y2] with zero-padding if coordinates exceed borders."""
    h, w = image.shape[:2]
    crop_w = x2 - x1
    crop_h = y2 - y1

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    cropped = image[src_y1:src_y2, src_x1:src_x2]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        if cropped.ndim == 3:
            padded = np.zeros((crop_h, crop_w, cropped.shape[2]), dtype=image.dtype)
            padded[pad_top : pad_top + (src_y2 - src_y1), pad_left : pad_left + (src_x2 - src_x1)] = cropped
        else:
            padded = np.zeros((crop_h, crop_w), dtype=image.dtype)
            padded[pad_top : pad_top + (src_y2 - src_y1), pad_left : pad_left + (src_x2 - src_x1)] = cropped
        return padded, x1, y1

    return cropped, x1, y1


def process_zip_dataset(
    zip_path: Path,
    output_dir: Path,
    max_samples: int = 1500,
    margin: float = 1.25,
) -> int:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "landmarks.csv"

    print(f"Opening archive: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()

        # Find label files (.txt) inside labels/train or labels/
        label_files = [
            n for n in namelist if n.endswith(".txt") and "labels" in n and not n.endswith("classes.txt")
        ]

        if not label_files:
            raise FileNotFoundError(f"No YOLO label files found in {zip_path}")

        print(f"Found {len(label_files)} annotation files in archive.")

        # Find corresponding images
        sample_count = 0
        written_rows = []

        pbar = tqdm(label_files[: max_samples * 2], desc="Extracting & cropping")
        for label_name in pbar:
            if sample_count >= max_samples:
                break

            # Find matching image
            base_stem = Path(label_name).stem
            # Usually image is in images/train/<stem>.jpg or .png
            candidates = [
                label_name.replace("labels", "images").replace(".txt", ext)
                for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]
            ]
            img_name = None
            for c in candidates:
                if c in zf.namelist():
                    img_name = c
                    break

            if not img_name:
                continue

            # Read label
            raw_label = zf.read(label_name).decode("utf-8").strip()
            if not raw_label:
                continue

            lines = raw_label.splitlines()
            # Parse first valid hand
            for line in lines:
                parts = line.strip().split()
                # YOLO Pose format: class_id cx cy w h kx0 ky0 v0 kx1 ky1 v1 ...
                # 5 bbox params + 21 * 3 = 68 tokens
                if len(parts) < 5 + NUM_LANDMARKS * 2:
                    continue

                try:
                    vals = list(map(float, parts[1:]))
                except ValueError:
                    continue

                cx_norm, cy_norm, bw_norm, bh_norm = vals[0], vals[1], vals[2], vals[3]

                # Check if coordinates have 3 values (x, y, v) or 2 values (x, y)
                kpt_tokens = vals[4:]
                if len(kpt_tokens) >= NUM_LANDMARKS * 3:
                    coords_raw = []
                    for i in range(NUM_LANDMARKS):
                        coords_raw.append((kpt_tokens[i * 3], kpt_tokens[i * 3 + 1]))
                elif len(kpt_tokens) >= NUM_LANDMARKS * 2:
                    coords_raw = []
                    for i in range(NUM_LANDMARKS):
                        coords_raw.append((kpt_tokens[i * 2], kpt_tokens[i * 2 + 1]))
                else:
                    continue

                # Read and decode image
                img_bytes = zf.read(img_name)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                img_h, img_w = img.shape[:2]

                # Convert bbox to pixel coordinates
                cx = cx_norm * img_w
                cy = cy_norm * img_h
                bw = bw_norm * img_w
                bh = bh_norm * img_h

                side = int(max(bw, bh) * margin)
                if side < 20:
                    continue

                x1 = int(cx - side / 2.0)
                y1 = int(cy - side / 2.0)
                x2 = x1 + side
                y2 = y1 + side

                # Crop and pad
                crop, crop_x1, crop_y1 = pad_and_crop(img, x1, y1, x2, y2)
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                resized_crop = cv2.resize(
                    gray_crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA
                )

                # Map landmarks to crop [0, 1]
                norm_pts = []
                for kx, ky in coords_raw:
                    px = kx * img_w
                    py = ky * img_h
                    norm_x = np.clip((px - crop_x1) / float(side), 0.0, 1.0)
                    norm_y = np.clip((py - crop_y1) / float(side), 0.0, 1.0)
                    norm_pts.extend([round(float(norm_x), 4), round(float(norm_y), 4)])

                out_fname = f"hand_{sample_count:05d}.png"
                cv2.imwrite(str(images_dir / out_fname), resized_crop)
                written_rows.append([out_fname] + norm_pts)
                sample_count += 1
                pbar.set_postfix({"samples": sample_count})
                break

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(written_rows)

    print(f"\n✓ Extracted {sample_count} samples into {output_dir}")
    print(f"  Images: {images_dir}")
    print(f"  CSV:    {csv_path}")
    return sample_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare hand dataset for training")
    parser.add_argument("--zip", type=Path, default=Path("hand-keypoints.zip"))
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--max-samples", type=int, default=1500)
    parser.add_argument("--margin", type=float, default=1.25)
    args = parser.parse_args()

    process_zip_dataset(args.zip, args.output, args.max_samples, args.margin)


if __name__ == "__main__":
    main()
