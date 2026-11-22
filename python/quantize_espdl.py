#!/usr/bin/env python3
"""
quantize_espdl.py – Convert a trained PyTorch checkpoint to an ESP-DL
quantised model via ONNX.

Pipeline::

    PyTorch (.pth) → ONNX → INT8 calibration → ESP-DL quantised model (.espdl)

Usage
-----
    python quantize_espdl.py \\
        --checkpoint checkpoints/best.pth \\
        --calibration-dir dataset/images \\
        --output ../firmware/main/models/hand_landmarks.espdl

Requires the ``esp-ppq`` quantisation toolkit from Espressif.
See: https://github.com/espressif/esp-ppq
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from model import HandLandmarkNet, build_model, export_onnx

# ---------------------------------------------------------------------------
# Calibration dataset loader
# ---------------------------------------------------------------------------

def load_calibration_images(
    cal_dir: Path,
    num_samples: int = 200,
    input_size: int = 96,
) -> np.ndarray:
    """Load grayscale images as a float32 array ``(N, 1, 96, 96)``."""
    paths = sorted(cal_dir.glob("*.png"))[:num_samples]
    if not paths:
        raise FileNotFoundError(f"No .png calibration images found in {cal_dir}")

    images = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (input_size, input_size))
        images.append(img.astype(np.float32) / 255.0)

    batch = np.stack(images)[:, np.newaxis, :, :]  # (N, 1, 96, 96)
    print(f"Loaded {batch.shape[0]} calibration images from {cal_dir}")
    return batch


# ---------------------------------------------------------------------------
# Quantisation (ESP-PPQ)
# ---------------------------------------------------------------------------

def quantize_to_espdl(
    onnx_path: str,
    calibration_data: np.ndarray,
    output_path: str,
    target_chip: str = "esp32s3",
) -> None:
    """Quantise an ONNX model to INT8 using ESP-PPQ and export ``.espdl``."""
    try:
        from espdl import export_model_to_espdl
        from ppq import QuantizationSettingFactory, TargetPlatform
        from ppq.api import espdl_quantize_onnx
    except ImportError:
        print(
            "ERROR: esp-ppq is not installed.\n"
            "Install it with:\n"
            "  pip install git+https://github.com/espressif/esp-ppq.git\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Calibration data as list of dicts
    calib_loader = [{"input": calibration_data[i : i + 1]} for i in range(calibration_data.shape[0])]

    setting = QuantizationSettingFactory.espdl_setting()
    quantized = espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        calib_dataloader=calib_loader,
        calib_steps=min(len(calib_loader), 128),
        input_shape=[1, 1, 96, 96],
        target=target_chip,
        setting=setting,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    export_model_to_espdl(quantized, output_path)
    print(f"Quantised model exported → {output_path}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    checkpoint: Path,
    calibration_dir: Path,
    output: Path,
    num_cal_samples: int = 200,
) -> None:
    """End-to-end: load checkpoint → ONNX → quantise → .espdl."""
    # 1. Load & export ONNX
    model = build_model()
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    onnx_path = str(checkpoint.with_suffix(".onnx"))
    export_onnx(model, onnx_path)

    # 2. Load calibration data
    cal_data = load_calibration_images(calibration_dir, num_samples=num_cal_samples)

    # 3. Quantise
    quantize_to_espdl(onnx_path, cal_data, str(output))

    print("\n✓ Pipeline complete.")
    print(f"  ONNX  : {onnx_path}")
    print(f"  ESPDL : {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Quantise hand landmark model for ESP-DL")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pth")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("dataset/images"),
        help="Directory with calibration PNGs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../firmware/main/models/hand_landmarks.espdl"),
        help="Output .espdl path",
    )
    parser.add_argument("--num-cal-samples", type=int, default=200)
    args = parser.parse_args()

    run_pipeline(
        checkpoint=args.checkpoint,
        calibration_dir=args.calibration_dir,
        output=args.output,
        num_cal_samples=args.num_cal_samples,
    )


if __name__ == "__main__":
    main()
