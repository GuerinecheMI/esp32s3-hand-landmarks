#!/usr/bin/env python3
"""
collect_dataset.py – Webcam / image-folder hand landmark dataset collector.

Usage:
------
1. Automatic annotation using MediaPipe (recommended):
    python collect_dataset.py --source webcam --auto --output dataset/

2. Manual annotation from webcam:
    python collect_dataset.py --source webcam --output dataset/

3. Import from pre-labelled image folder:
    python collect_dataset.py --source folder --images path/to/images --labels path/to/labels.csv --output dataset/
"""

from __future__ import annotations

import argparse
import csv
import sys
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

NUM_LANDMARKS = 21

# ---------------------------------------------------------------------------
# Manual interactive annotation state
# ---------------------------------------------------------------------------
_current_points: list[tuple[int, int]] = []
_display_img: np.ndarray | None = None
_base_crop: np.ndarray | None = None
_scale: float = 4.0


def _redraw_annotator() -> None:
    global _display_img
    if _base_crop is None:
        return
    _display_img = _base_crop.copy()
    for idx, (x, y) in enumerate(_current_points):
        cv2.circle(_display_img, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(
            _display_img, str(idx), (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
        )
    cv2.imshow("annotate", _display_img)


def _mouse_callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
    if event == cv2.EVENT_LBUTTONDOWN and len(_current_points) < NUM_LANDMARKS:
        _current_points.append((x, y))
        _redraw_annotator()


# ---------------------------------------------------------------------------
# Auto-collector using MediaPipe
# ---------------------------------------------------------------------------

def collect_with_mediapipe(output_dir: Path, cam_index: int = 0, interval: int = 5) -> None:
    try:
        import mediapipe as mp
    except ImportError:
        print("MediaPipe is not installed. Run: pip install mediapipe")
        sys.exit(1)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "landmarks.csv"

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_index}")

    count = 0
    frame_idx = 0
    print("\nMediaPipe Auto-Collector Active!")
    print("Hold your hand in front of the camera.")
    print("Controls: [SPACE] toggle pause | [Q] quit\n")

    paused = False
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            display = frame.copy()

            if res.multi_hand_landmarks and not paused:
                hand_lms = res.multi_hand_landmarks[0]
                pts = np.array([(lm.x * w, lm.y * h) for lm in hand_lms.landmark])

                # Compute square bounding box with 20% margin
                min_x, min_y = np.min(pts, axis=0)
                max_x, max_y = np.max(pts, axis=0)
                bw, bh = max_x - min_x, max_y - min_y
                side = max(bw, bh) * 1.3
                cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

                x1 = int(max(0, cx - side / 2.0))
                y1 = int(max(0, cy - side / 2.0))
                x2 = int(min(w, cx + side / 2.0))
                y2 = int(min(h, cy + side / 2.0))

                if (x2 - x1) > 40 and (y2 - y1) > 40:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Capture every 'interval' frames
                    if frame_idx % interval == 0:
                        crop = frame[y1:y2, x1:x2]
                        crop_h, crop_w = crop.shape[:2]
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)

                        # Normalise landmarks relative to the cropped patch
                        norm_coords = []
                        for lm in hand_lms.landmark:
                            px, py = lm.x * w, lm.y * h
                            norm_x = np.clip((px - x1) / crop_w, 0.0, 1.0)
                            norm_y = np.clip((py - y1) / crop_h, 0.0, 1.0)
                            norm_coords.extend([round(float(norm_x), 4), round(float(norm_y), 4)])

                        fname = f"auto_{count:05d}.png"
                        cv2.imwrite(str(images_dir / fname), resized)
                        writer.writerow([fname] + norm_coords)
                        f.flush()
                        count += 1

            frame_idx += 1
            status_text = f"Samples: {count} {'[PAUSED]' if paused else '[RECORDING]'}"
            cv2.putText(display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("MediaPipe Collector", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nAuto-collected {count} samples -> {output_dir}")


# ---------------------------------------------------------------------------
# Manual Webcam Collector
# ---------------------------------------------------------------------------

def collect_from_webcam(output_dir: Path, cam_index: int = 0) -> None:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "landmarks.csv"

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_index}")

    sample_idx = 0
    print("Controls: [SPACE] capture & annotate | [Q] quit")

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
                _annotate_crop(crop, writer, images_dir, sample_idx)
                sample_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def _annotate_crop(
    crop: np.ndarray,
    writer: csv.writer,
    images_dir: Path,
    idx: int,
) -> None:
    global _current_points, _base_crop, _scale

    display_size = 384
    _scale = display_size / INPUT_SIZE
    big = cv2.resize(crop, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    _base_crop = big.copy() if big.ndim == 3 else cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    _current_points = []

    cv2.namedWindow("annotate")
    cv2.setMouseCallback("annotate", _mouse_callback)
    _redraw_annotator()

    print(f"\nSample {idx}: click {NUM_LANDMARKS} points in order: Wrist -> Thumb (1-4) -> Index (5-8) -> ...")
    print("Controls: [ENTER] save | [U] undo last point | [ESC] skip")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC -> skip
            print("  -> Skipped")
            break
        elif key in (ord("u"), ord("U")):  # Undo
            if _current_points:
                _current_points.pop()
                _redraw_annotator()
        elif key == 13 and len(_current_points) == NUM_LANDMARKS:  # ENTER
            norm_pts = [(x / _scale / INPUT_SIZE, y / _scale / INPUT_SIZE) for x, y in _current_points]
            flat = [round(float(coord), 4) for pt in norm_pts for coord in pt]

            fname = f"sample_{idx:05d}.png"
            cv2.imwrite(str(images_dir / fname), crop)
            writer.writerow([fname] + flat)
            print(f"  -> Saved {fname}")
            break

    cv2.destroyWindow("annotate")


# ---------------------------------------------------------------------------
# Folder collector
# ---------------------------------------------------------------------------

def collect_from_folder(images_path: Path, labels_path: Path, output_dir: Path) -> None:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "landmarks.csv"

    with open(labels_path, newline="") as lf:
        rows = list(csv.reader(lf))

    count = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            if not row:
                continue
            img_name = row[0]
            coords = list(map(float, row[1:]))
            if len(coords) != NUM_LANDMARKS * 2:
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

    print(f"Processed {count} samples -> {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Hand landmark dataset collector")
    parser.add_argument("--source", choices=["webcam", "folder"], default="webcam")
    parser.add_argument("--auto", action="store_true", help="Use MediaPipe for automatic landmark labeling")
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--images", type=Path, help="Images directory (folder mode)")
    parser.add_argument("--labels", type=Path, help="Labels CSV file (folder mode)")
    args = parser.parse_args()

    if args.source == "webcam":
        if args.auto:
            collect_with_mediapipe(args.output, cam_index=args.camera)
        else:
            collect_from_webcam(args.output, cam_index=args.camera)
    else:
        if not args.images or not args.labels:
            parser.error("--images and --labels required for folder mode")
        collect_from_folder(args.images, args.labels, args.output)


if __name__ == "__main__":
    main()
