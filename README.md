# ESP32-S3 21-Point Hand Landmark Tracker with ESP-DL

Real-time 21-point hand landmark detection and tracking running on an ESP32-S3 using Espressif's [ESP-DL](https://github.com/espressif/esp-dl) framework.

Features **adaptive field-of-view scanning and temporal tracking** (160×120 QQVGA) — the hand does not need to be locked into an exact position; the on-chip tracker automatically locates, centers, and follows the hand dynamically across the camera's view.

## Project Structure

```
esp32s3-hand-landmarks/
├── python/                   # Training & tooling pipeline
│   ├── requirements.txt
│   ├── roi_utils.py          # Hand ROI extraction helpers
│   ├── prepare_dataset.py    # Auto-extracts 96x96 crops & 21-point keypoints
│   ├── collect_dataset.py    # Webcam auto (MediaPipe) & manual collector
│   ├── model.py              # PyTorch landmark model (~150k params)
│   ├── train.py              # Training loop with clean train/val split
│   ├── quantize_espdl.py     # ONNX → ESP-DL INT8 quantised model
│   └── visualize_serial.py   # Live PC visualizer for UART landmark stream
├── firmware/                 # ESP-IDF application
│   ├── CMakeLists.txt
│   ├── sdkconfig.defaults
│   ├── partitions.csv        # Custom 8MB flash partition table
│   └── main/
│       ├── CMakeLists.txt
│       ├── idf_component.yml
│       ├── models/
│       │   └── hand_landmarks.espdl # Quantised INT8 model
│       ├── hand_tracker.hpp
│       ├── hand_tracker.cpp  # Fast 256-entry LUT quantization + ESP-DL inference
│       └── main.cpp          # Adaptive ROI scanner & temporal landmark tracking
└── README.md
```

## How It Works: Adaptive Wide-View Tracking

Unlike rigid single-patch systems, this firmware uses a **2-phase temporal tracking loop**:

1. **Acquisition / Scan:** If no hand is currently tracked, the ESP32 performs a fast subsampled scan across the full 160×120 QQVGA camera scene to find the hand's center of mass.
2. **Crop & Downscale:** Crops a square bounding box around the hand and downscales it to 96×96 in <0.2ms using integer fixed-point scaling.
3. **Inference:** Runs the 150k-parameter INT8 CNN via ESP-DL hardware acceleration.
4. **Temporal Landmark Feedback:** The min/max spread of the detected 21 landmarks updates the bounding box center and size for the next frame using an Exponential Moving Average (EMA). The tracking box smoothly follows your hand as you move it left, right, up, down, closer, or farther away.
5. **UART Output:** Normalised landmark coordinates mapped to the full camera frame are streamed over USB Serial as JSON.

## Quick Start

### 1. Training (host PC)

```bash
cd python
pip install -r requirements.txt

# Extract dataset (from hand-keypoints or auto-collect with webcam)
python collect_dataset.py --source webcam --auto --output dataset/

# Train model
python train.py --data dataset/ --epochs 25

# Quantize to INT8 ESP-DL model (requires esp-ppq)
pip install git+https://github.com/espressif/esp-ppq.git
python quantize_espdl.py --checkpoint checkpoints/best.pth --output ../firmware/main/models/hand_landmarks.espdl
```

### 2. Build & flash the firmware

```bash
cd firmware
idf.py build
idf.py -p COM_PORT flash monitor
```

### 3. Live PC Skeleton Visualizer

Run the visualizer to view the tracked MediaPipe hand skeleton in real time:

```bash
cd python
python visualize_serial.py --port COM3
```

## Model Architecture & Optimizations

- **Camera Stream:** 160 × 120 QQVGA (grayscale)
- **Model Input:** 96 × 96 grayscale crop
- **Encoder:** 4 Conv2d-BatchNorm-ReLU-MaxPool blocks (16 → 32 → 64 → 128 filters)
- **Bottleneck:** `AdaptiveAvgPool2d((2, 2))` pooling down to 512 features
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
- **Camera:** OV2640 / OV3660 / OV5640 via DVP interface (ESP-S3-EYE, Freenove, etc.)
- **Output:** Live JSON streamed over UART (115200 baud)

## License

MIT
