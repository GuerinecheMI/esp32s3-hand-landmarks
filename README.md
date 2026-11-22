# ESP32-S3 21-Point Hand Landmark Tracker with ESP-DL

Real-time 21-point hand landmark detection running entirely on-device on an ESP32-S3 using Espressif's [ESP-DL](https://github.com/espressif/esp-dl) inference framework.

## Project Structure

```
esp32s3-hand-landmarks/
├── python/                   # Training & quantisation pipeline
│   ├── requirements.txt
│   ├── roi_utils.py          # Hand ROI extraction helpers
│   ├── collect_dataset.py    # Webcam / image dataset collector
│   ├── model.py              # PyTorch landmark regression model
│   ├── train.py              # Training loop
│   └── quantize_espdl.py     # ONNX → ESP-DL quantised model
├── firmware/                 # ESP-IDF application
│   ├── CMakeLists.txt
│   ├── sdkconfig.defaults
│   └── main/
│       ├── CMakeLists.txt
│       ├── idf_component.yml
│       ├── models/.gitkeep   # Place quantised .espdl model here
│       ├── hand_tracker.hpp
│       ├── hand_tracker.cpp
│       └── main.cpp
└── README.md
```

## Quick Start

### 1. Train the model (host PC)

```bash
cd python
pip install -r requirements.txt
python collect_dataset.py --source webcam --output dataset/
python train.py --data dataset/ --epochs 120
python quantize_espdl.py --checkpoint checkpoints/best.pth --output ../firmware/main/models/hand_landmarks.espdl
```

### 2. Build & flash the firmware

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

## Model

The landmark model is a lightweight CNN that takes a **96 × 96** grayscale crop of the hand region and regresses **21 × 2 = 42** normalised (x, y) coordinates corresponding to the standard hand skeleton.

| Landmark Index | Joint           |
|---------------|-----------------|
| 0             | Wrist           |
| 1–4           | Thumb           |
| 5–8           | Index finger    |
| 9–12          | Middle finger   |
| 13–16         | Ring finger     |
| 17–20         | Pinky finger    |

## Hardware

- **MCU:** ESP32-S3 (dual-core Xtensa LX7, 512 KB SRAM, PSRAM recommended)
- **Camera:** OV2640 or OV5640 via DVP / SPI interface
- **Display (optional):** SPI LCD for overlay visualisation

## License

MIT
