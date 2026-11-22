"""
roi_utils.py – Hand Region-of-Interest extraction utilities.

Provides helpers to detect, crop, and normalise the hand bounding box
from a full camera frame so the landmark model receives a clean 96×96 input.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_SIZE: int = 96  # landmark model spatial resolution


# ---------------------------------------------------------------------------
# Skin-colour segmentation (YCrCb)
# ---------------------------------------------------------------------------
_LOWER_SKIN = np.array([0, 133, 77], dtype=np.uint8)
_UPPER_SKIN = np.array([255, 173, 127], dtype=np.uint8)


def skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Return a binary mask highlighting skin-coloured pixels."""
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, _LOWER_SKIN, _UPPER_SKIN)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


# ---------------------------------------------------------------------------
# Bounding-box extraction
# ---------------------------------------------------------------------------

def largest_contour_bbox(
    mask: np.ndarray,
    min_area: int = 3000,
    padding: float = 0.15,
) -> tuple[int, int, int, int] | None:
    """Find the bounding box of the largest contour in *mask*.

    Returns ``(x, y, w, h)`` with *padding* fraction added on each side,
    or ``None`` when no contour exceeds *min_area*.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    x, y, w, h = cv2.boundingRect(largest)

    # Pad & make square
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x = max(x - pad_x, 0)
    y = max(y - pad_y, 0)
    w = w + 2 * pad_x
    h = h + 2 * pad_y

    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    x = max(cx - side // 2, 0)
    y = max(cy - side // 2, 0)

    return x, y, side, side


# ---------------------------------------------------------------------------
# Crop & preprocess for the landmark model
# ---------------------------------------------------------------------------

def crop_and_resize(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_size: int = INPUT_SIZE,
) -> np.ndarray:
    """Crop *frame_bgr* to *bbox* and resize to ``(target_size, target_size)`` grayscale."""
    x, y, w, h = bbox
    h_img, w_img = frame_bgr.shape[:2]
    x2 = min(x + w, w_img)
    y2 = min(y + h, h_img)
    crop = frame_bgr[y:y2, x:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def preprocess_for_model(gray_96: np.ndarray) -> np.ndarray:
    """Normalise a ``(96, 96)`` uint8 grayscale image to ``(1, 1, 96, 96)`` float32."""
    img = gray_96.astype(np.float32) / 255.0
    return img[np.newaxis, np.newaxis, :, :]


# ---------------------------------------------------------------------------
# Landmark coordinate mapping
# ---------------------------------------------------------------------------

def map_landmarks_to_frame(
    landmarks_norm: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Convert normalised ``(21, 2)`` landmarks back to full-frame pixel coords."""
    x, y, w, h = bbox
    landmarks_px = landmarks_norm.copy()
    landmarks_px[:, 0] = landmarks_norm[:, 0] * w + x
    landmarks_px[:, 1] = landmarks_norm[:, 1] * h + y
    return landmarks_px.astype(np.int32)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# Hand skeleton connectivity (parent → child)
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),              # palm
]


def draw_landmarks(
    frame: np.ndarray,
    landmarks_px: np.ndarray,
    radius: int = 3,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw landmarks and skeleton on *frame* (mutated in-place and returned)."""
    for start, end in HAND_CONNECTIONS:
        pt1 = tuple(landmarks_px[start])
        pt2 = tuple(landmarks_px[end])
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    for pt in landmarks_px:
        cv2.circle(frame, tuple(pt), radius, (0, 0, 255), -1, cv2.LINE_AA)

    return frame
