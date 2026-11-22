#!/usr/bin/env python3
"""
collect_dataset.py – Webcam / image-folder hand-landmark dataset collector.

Usage
-----
Webcam (interactive annotation):
    python collect_dataset.py --source webcam --output dataset/

Image folder (requires pre-existing bbox labels):
    python collect_dataset.py --source folder --images path/to/images --labels path/to/labels.csv --output dataset/

The output directory is organised as::

    dataset/
    ├── images/          # 96×96 grayscale crops (.png)
    └── landmarks.csv    # <filename>, x0, y0, x1, y1, … , x20, y20
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

from roi_utils import (
    INPUT_SIZE,
    crop_and_resize,
    draw_landmarks,
    largest_contour_bbox,
    skin_mask,
)

# ---------------------------------------------------------------------------
# Globals / state for the interactive annotator
# ---------------------------------------------------------------------------
_current_points: list[tuple[int, int]] = []
_clone: np.ndarray | None = None

NUM_LANDMARKS = 21


def _mouse_callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
    global _clone
    if event == cv2.EVENT_LBUTTONDOWN and len(_current_points) < NUM_LANDMARKS:
        _current_points.append((x, y))
        if _clone is not None:
            cv2.circle(_clone, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("annotate", _clone)


# ---------------------------------------------------------------------------
# Webcam collector
# ---------------------------------------------------------------------------

def collect_from_webcam(output_dir: Path, cam_index: int = 0) -> None:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "landmarks.csv"

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_index}")

    sample_idx = 0
    print("Controls: [SPACE] capture & annotate | [S] skip | [Q] quit")

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            mask = skin_mask(frame)
            bbox = largest_contour_bbox(mask)
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("capture", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" ") and bbox is not None:
                crop = crop_and_resize(frame, bbox)
                _annotate_crop(crop, bbox, frame, writer, images_dir, sample_idx)
                sample_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {sample_idx} samples → {output_dir}")


def _annotate_crop(
    crop: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame: np.ndarray,
    writer: csv.writer,
    images_dir: Path,
    idx: int,
) -> None:
    """Show the crop and let the user click 21 landmarks."""
    global _current_points, _clone

    display_size = 384
    big = cv2.resize(crop, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    _clone = big.copy() if big.ndim == 3 else cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    _current_points = []
    scale = display_size / INPUT_SIZE

    cv2.imshow("annotate", _clone)
    cv2.setMouseCallback("annotate", _mouse_callback)

    print(f"  Sample {idx}: click {NUM_LANDMARKS} landmarks, then press ENTER (or ESC to skip)")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC → skip
            print("    ↳ skipped")
            return
        if key == 13 and len(_current_points) == NUM_LANDMARKS:  # ENTER
            break

    # Normalise points to [0, 1] relative to crop
    norm_pts = [(x / scale / INPUT_SIZE, y / scale / INPUT_SIZE) for x, y in _current_points]
    flat = [coord for pt in norm_pts for coord in pt]

    fname = f"sample_{idx:05d}.png"
    cv2.imwrite(str(images_dir / fname), crop)
    writer.writerow([fname] + flat)
    print(f"    ↳ saved {fname}")


# ---------------------------------------------------------------------------
# Folder collector (batch from pre-labelled data)
# ---------------------------------------------------------------------------

def collect_from_folder(
    images_path: Path,
    labels_path: Path,
    output_dir: Path,
) -> None:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "landmarks.csv"

    with open(labels_path) as lf:
        reader = csv.reader(lf)
        rows = list(reader)

    count = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            img_name = row[0]
            coords = list(map(float, row[1:]))
            if len(coords) != NUM_LANDMARKS * 2:
                print(f"Skipping {img_name}: expected {NUM_LANDMARKS * 2} coords, got {len(coords)}")
                continue

            img = cv2.imread(str(images_path / img_name))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)

            out_name = f"sample_{count:05d}.png"
            cv2.imwrite(str(images_dir / out_name), resized)
            writer.writerow([out_name] + coords)
            count += 1

    print(f"Processed {count} samples → {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Hand landmark dataset collector")
    parser.add_argument("--source", choices=["webcam", "folder"], default="webcam")
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--camera", type=int, default=0, help="Camera index (webcam mode)")
    parser.add_argument("--images", type=Path, help="Image folder (folder mode)")
    parser.add_argument("--labels", type=Path, help="Labels CSV (folder mode)")
    args = parser.parse_args()

    if args.source == "webcam":
        collect_from_webcam(args.output, cam_index=args.camera)
    else:
        if not args.images or not args.labels:
            parser.error("--images and --labels are required for folder mode")
        collect_from_folder(args.images, args.labels, args.output)


if __name__ == "__main__":
    main()
