# ESP32-S3 21-Point Hand Landmark Tracker with ESP-DL

Real-time 21-point hand landmark detection and tracking running entirely on an ESP32-S3 microcontroller using Espressif's [ESP-DL](https://github.com/espressif/esp-dl) deep learning framework.

Features **adaptive wide-view scanning and temporal tracking** ($160 \times 120$ QQVGA) — the hand does not need to stay in a fixed location. The on-chip tracker automatically acquires, centers, and follows the hand dynamically across the scene.

---

## Project Structure

```
esp32s3-hand-landmarks/
├── python/                     # Training & tooling pipeline
│   ├── requirements.txt
│   ├── roi_utils.py            # Hand ROI extraction & skeleton drawing
│   ├── prepare_dataset.py      # Extracts 96x96 crops & 21 keypoints from YOLO/Kaggle datasets
│   ├── collect_dataset.py      # Webcam auto-collector (MediaPipe) & manual collector
│   ├── model.py                # PyTorch landmark model (~150k params)
│   ├── train.py                # Training loop with clean train/val split
│   ├── quantize_espdl.py       # ONNX → ESP-DL INT8 quantized model exporter
│   ├── visualize_serial.py     # Live PC visualizer for UART landmark stream
│   ├── checkpoints/            # PyTorch checkpoints (e.g. best.pth)
│   └── exports/                # Exported models (hand_landmarks.onnx)
├── firmware/                   # ESP-IDF firmware application
│   ├── CMakeLists.txt
│   ├── sdkconfig.defaults
│   ├── partitions.csv          # Custom 8MB flash partition table
│   └── main/
│       ├── CMakeLists.txt
│       ├── idf_component.yml
│       ├── models/
│       │   └── hand_landmarks.espdl # Place quantized INT8 model here
│       ├── hand_tracker.hpp
│       ├── hand_tracker.cpp    # Fast 256-entry LUT quantization + ESP-DL inference
│       └── main.cpp            # Adaptive ROI scanner & temporal landmark tracking
└── README.md
```

---

## Models & Weights

The project follows a 3-stage model lifecycle:

```
[PyTorch .pth] ────(export_onnx)────> [ONNX .onnx] ────(esp-ppq)────> [ESP-DL .espdl]
  Training                             Evaluation                      On-Device MCU
```

### 1. PyTorch Model (`python/checkpoints/best.pth`)
* **Framework:** PyTorch 2.x
* **File Format:** Standard PyTorch state dictionary (`torch.save(model.state_dict())`)
* **Input Shape:** `(B, 1, 96, 96)` — Grayscale hand crop normalized to $[0.0, 1.0]$
* **Output Shape:** `(B, 42)` — Flattened $(x, y)$ coordinates for the 21 joints $[0.0, 1.0]$
* **Weights Size:** ~600 KB (Float32)

### 2. ONNX Model (`python/exports/hand_landmarks.onnx`)
* **Standard:** ONNX Opset 13
* **Verification:** Validated with `onnx.checker` and runnable directly with `onnxruntime` on PC for verification and desktop debugging.
* **Weights Size:** ~600 KB

### 3. ESP-DL Quantized Model (`firmware/main/models/hand_landmarks.espdl`)
* **Target:** ESP32-S3 hardware accelerator (Xtensa PIE vector instructions)
* **Format:** FlatBuffers / ESP-DL INT8 quantized format
* **Calibration:** 8-bit Post-Training Quantization (PTQ) via Espressif's `esp-ppq`
* **Model Size:** **~150 KB** (fits directly into the ESP32-S3's internal SRAM/cache, enabling 20+ FPS inference without PSRAM memory bottlenecks)

---

## Detailed Model Architecture

The CNN is engineered specifically for microcontrollers with tight SRAM budgets:

| Stage | Layer Type | Output Dimension | Parameters | Details |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Image | $(1, 96, 96)$ | — | Single-channel grayscale |
| **Block 1** | Conv2d + BN + ReLU + MaxPool | $(16, 48, 48)$ | 144 | $3 \times 3$ kernel, stride 1, pad 1 |
| **Block 2** | Conv2d + BN + ReLU + MaxPool | $(32, 24, 24)$ | 4,608 | $3 \times 3$ kernel, stride 1, pad 1 |
| **Block 3** | Conv2d + BN + ReLU + MaxPool | $(64, 12, 12)$ | 18,432 | $3 \times 3$ kernel, stride 1, pad 1 |
| **Block 4** | Conv2d + BN + ReLU + MaxPool | $(128, 6, 6)$ | 73,728 | $3 \times 3$ kernel, stride 1, pad 1 |
| **Bottleneck** | `AdaptiveAvgPool2d((2, 2))` | $(128, 2, 2)$ | 0 | Compresses $128 \times 36 \to 128 \times 4$ |
| **Dense 1** | Flatten + Linear + ReLU | $128$ | 65,664 | $512 \to 128$ + Dropout(0.2) |
| **Dense 2** | Linear + ReLU | $64$ | 8,256 | $128 \to 64$ |
| **Output** | Linear + Sigmoid | $42$ | 2,730 | $64 \to 42$ coordinates $[0, 1]$ |
| **Total** | | | **~150,000** | **~150 KB INT8 / 600 KB FP32** |

---

## How It Works: Adaptive Wide-View Tracking

Unlike static single-patch systems, this firmware uses a **closed-loop temporal tracking pipeline**:

1. **Wide-View Capture:** Camera runs at **$160 \times 120$ QQVGA** (capturing a wide field of view).
2. **Auto-Acquisition:** If no hand is currently tracked, the firmware scans subsampled pixel intensities across the scene to locate the hand's center of mass.
3. **Integer Crop & Rescale (<0.2 ms):** Crops a square bounding box around the hand and scales it down to $96 \times 96$ using fixed-point integer math (zero floating-point operations).
4. **Fast Quantization (LUT):** Converts 8-bit camera pixels to INT8 model inputs using a precomputed 256-byte lookup table in a single memory pass.
5. **Inference:** Executes the forward pass using ESP-DL vector DSP acceleration.
6. **Temporal Landmark Feedback:** The physical spread ($\min / \max$) of the 21 predicted landmarks updates the tracking box center and scale for the next frame using an Exponential Moving Average (EMA). The box smoothly follows your hand as you move it around the frame.
7. **UART Output:** Normalised landmark coordinates mapped to the full camera frame are streamed over USB Serial as compact JSON.

---

## Quick Start

### 1. Training (Host PC)

```bash
cd python
pip install -r requirements.txt

# Option A: Auto-annotate custom dataset from webcam with MediaPipe
python collect_dataset.py --source webcam --auto --output dataset/

# Option B: Extract from downloaded Kaggle / YOLO hand-keypoints dataset
python prepare_dataset.py --zip hand-keypoints.zip --max-samples 1500 --output dataset/

# Train model (saves best checkpoint to checkpoints/best.pth and exports ONNX)
python train.py --data dataset/ --epochs 25 --batch-size 32

# Quantize ONNX model to INT8 ESP-DL format (requires esp-ppq)
pip install git+https://github.com/espressif/esp-ppq.git
python quantize_espdl.py --checkpoint checkpoints/best.pth --output ../firmware/main/models/hand_landmarks.espdl
```

### 2. Build & Flash Firmware (ESP32-S3)

```bash
cd firmware
idf.py build
idf.py -p COM_PORT flash monitor
```

### 3. Live PC Skeleton Visualizer

Run the real-time visualizer on your PC to view the tracked skeleton and dynamic bounding box:

```bash
cd python
python visualize_serial.py --port COM3
```

---

## Landmark Joint Indices (MediaPipe Compatible)

| Index | Joint Name | Anatomical Location |
| :---: | :--- | :--- |
| **0** | `Wrist` | Wrist center |
| **1 – 4** | `Thumb` | CMC $\to$ MCP $\to$ IP $\to$ Tip |
| **5 – 8** | `Index Finger` | MCP $\to$ PIP $\to$ DIP $\to$ Tip |
| **9 – 12** | `Middle Finger` | MCP $\to$ PIP $\to$ DIP $\to$ Tip |
| **13 – 16** | `Ring Finger` | MCP $\to$ PIP $\to$ DIP $\to$ Tip |
| **17 – 20** | `Pinky Finger` | MCP $\to$ PIP $\to$ DIP $\to$ Tip |

---

## Hardware Compatibility

- **MCU:** ESP32-S3 (Xtensa dual-core LX7 @ 240MHz, 8MB Flash, Octal/Quad PSRAM recommended)
- **Supported Modules:** ESP-S3-EYE, Freenove ESP32-S3 WROOM CAM, Seeed Xiao ESP32S3 Sense, AI-Thinker ESP32-S3 CAM
- **Camera Sensors:** OV2640, OV3660, OV5640 via 8-bit DVP interface

---

## Standalone Library

Looking to integrate hand landmark tracking into an existing project? Check out the companion standalone C++ / Arduino / ESP-IDF library:
👉 **[esp32s3-hand-landmarks-lib](https://github.com/GuerinecheMI/esp32s3-hand-landmarks-lib)**

---

## License

MIT
