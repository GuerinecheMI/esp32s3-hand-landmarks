# ESP32-S3 21-Point Hand Landmark Tracker with ESP-DL

Real-time 21-point hand landmark detection running entirely on-device on an ESP32-S3 using Espressif's [ESP-DL](https://github.com/espressif/esp-dl) inference framework.

## Project Structure

```
esp32s3-hand-landmarks/
├── python/                   # Training & quantisation pipeline
│   ├── requirements.txt
│   ├── roi_utils.py          # Hand ROI extraction helpers
│   ├── collect_dataset.py    # Webcam auto (MediaPipe) & manual collector
│   ├── model.py              # PyTorch landmark model (optimized ~150k params)
│   ├── train.py              # Training loop with clean train/val split
│   └── quantize_espdl.py     # ONNX → ESP-DL INT8 quantised model
├── firmware/                 # ESP-IDF application
│   ├── CMakeLists.txt
│   ├── sdkconfig.defaults
│   ├── partitions.csv        # Custom 8MB flash partition table
│   └── main/
│       ├── CMakeLists.txt
│       ├── idf_component.yml
│       ├── models/
│       │   └── hand_landmarks.espdl # Place quantised model here
│       ├── hand_tracker.hpp
│       ├── hand_tracker.cpp  # Fast 256-entry LUT quantization + ESP-DL inference
│       └── main.cpp
└── README.md
```

## Quick Start

### 1. Collect dataset & train (host PC)

```bash
cd python
pip install -r requirements.txt

# Option A: Auto-annotate using webcam with MediaPipe (fastest)
python collect_dataset.py --source webcam --auto --output dataset/

# Option B: Manual interactive annotation
python collect_dataset.py --source webcam --output dataset/

# Train model
python train.py --data dataset/ --epochs 120

# Quantize to INT8 ESP-DL model (requires esp-ppq)
pip install git+https://github.com/espressif/esp-ppq.git
python quantize_espdl.py --checkpoint checkpoints/best.pth --output ../firmware/main/models/hand_landmarks.espdl
```

### 2. Build & flash the firmware

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p COM_PORT flash monitor
```

## Model Architecture & Optimizations

- **Input:** 96 × 96 grayscale crop
- **Encoder:** 4 Conv2d-BatchNorm-ReLU-MaxPool blocks (16 → 32 → 64 → 128 filters)
- **Bottleneck:** `AdaptiveAvgPool2d((2, 2))` pooling down to 512 features, eliminating large fully-connected weight overhead
- **Parameters:** ~150,000 parameters (~150 KB in INT8) — fits inside ESP32-S3 internal SRAM/cache
- **On-chip Quantization:** 256-element precomputed lookup table (LUT) eliminates 9,216 floating-point calculations per frame

| Landmark Index | Joint           |
|---------------|-----------------|
| 0             | Wrist           |
| 1–4           | Thumb           |
| 5–8           | Index finger    |
| 9–12          | Middle finger   |
| 13–16         | Ring finger     |
| 17–20         | Pinky finger    |

## Hardware

- **MCU:** ESP32-S3 (dual-core Xtensa LX7 @ 240MHz, 8MB Flash, PSRAM)
- **Camera:** OV2640 / OV5640 via DVP interface (ESP-S3-EYE, Freenove, or AI-Thinker)
- **Output:** Normalised landmark coordinates JSON streamed over UART

## License

MIT
