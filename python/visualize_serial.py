#!/usr/bin/env python3
"""
visualize_serial.py – Live PC visualizer for the ESP32-S3 Hand Landmark Tracker.

Connects to the ESP32 via USB/Serial, reads the live landmark JSON stream,
and renders the tracked hand skeleton with MediaPipe color-coding in real time.

Usage:
    python visualize_serial.py --port COM3 --baud 115200
    # Linux / macOS:
    python visualize_serial.py --port /dev/ttyUSB0 --baud 115200
"""

from __future__ import annotations

import argparse
import json
import time
from typing import List, Tuple

import cv2
import numpy as np
import serial

# MediaPipe Hand Landmark Connections
CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm base knuckles
    (5, 9), (9, 13), (13, 17),
]

# Color palette (BGR) for distinct finger joints
FINGER_COLORS = [
    (0, 0, 255),    # 0: Wrist (Red)
    (0, 165, 255),  # 1-4: Thumb (Orange)
    (0, 165, 255),
    (0, 165, 255),
    (0, 165, 255),
    (0, 255, 0),    # 5-8: Index (Green)
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (255, 255, 0),  # 9-12: Middle (Cyan)
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0),
    (255, 0, 0),    # 13-16: Ring (Blue)
    (255, 0, 0),
    (255, 0, 0),
    (255, 0, 0),
    (255, 0, 255),  # 17-20: Pinky (Magenta)
    (255, 0, 255),
    (255, 0, 255),
    (255, 0, 255),
]


def render_skeleton(
    canvas: np.ndarray,
    landmarks: list[list[float]],
    roi: list[float] | None = None,
    fps: float = 0.0,
) -> None:
    h, w = canvas.shape[:2]

    # Draw camera aspect ratio guideline
    cam_w, cam_h = 160, 120
    scale_x = w / cam_w
    scale_y = h / cam_h

    # Draw dynamic tracked ROI bounding box if available
    if roi and len(roi) == 3:
        rx, ry, rsize = roi
        px1 = int(rx * scale_x)
        py1 = int(ry * scale_y)
        px2 = int((rx + rsize) * scale_x)
        py2 = int((ry + rsize) * scale_y)
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 255, 255), 2)
        cv2.putText(
            canvas, "Tracking ROI", (px1 + 4, py1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
        )

    # Convert normalized coordinates [0, 1] to canvas pixel coords
    points = []
    for lm in landmarks:
        x = int(np.clip(lm[0] * w, 0, w - 1))
        y = int(np.clip(lm[1] * h, 0, h - 1))
        points.append((x, y))

    # Draw skeleton bones
    for start_idx, end_idx in CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            pt1 = points[start_idx]
            pt2 = points[end_idx]
            cv2.line(canvas, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA)

    # Draw joint points
    for idx, (px, py) in enumerate(points):
        color = FINGER_COLORS[idx] if idx < len(FINGER_COLORS) else (0, 255, 0)
        radius = 6 if idx in (4, 8, 12, 16, 20) else 4  # Larger circles for fingertips
        cv2.circle(canvas, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

    # UI Information Overlay
    cv2.putText(
        canvas, f"ESP32-S3 Stream: {fps:.1f} FPS", (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    cv2.putText(
        canvas, "21 MediaPipe Landmarks", (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live ESP32-S3 Hand Landmark Visualizer")
    parser.add_argument("--port", type=str, default="COM3", help="Serial COM port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--width", type=int, default=640, help="Window display width")
    parser.add_argument("--height", type=int, default=480, help="Window display height")
    args = parser.parse_args()

    print(f"Connecting to ESP32 on {args.port} at {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
    except serial.SerialException as e:
        print(f"\nCould not open serial port {args.port}: {e}")
        print("Tip: Check Device Manager for your COM port or pass --port COMx")
        return

    print("Connected! Listening for landmark stream. Press [Q] to exit.")
    window_name = "ESP32-S3 Hand Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Look for JSON payload containing landmarks
            if "landmarks" in line and line.startswith("{"):
                try:
                    data = json.loads(line)
                    landmarks = data.get("landmarks", [])
                    roi = data.get("roi", None)

                    now = time.time()
                    dt = now - prev_time
                    prev_time = now
                    if dt > 0:
                        fps = 0.8 * fps + 0.2 * (1.0 / dt)

                    canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    render_skeleton(canvas, landmarks, roi=roi, fps=fps)
                    cv2.imshow(window_name, canvas)

                except json.JSONDecodeError:
                    pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        ser.close()
        cv2.destroyAllWindows()
        print("\nVisualizer closed.")


if __name__ == "__main__":
    main()
