"""
model.py – Lightweight CNN for 21-point hand landmark regression on ESP32-S3.

Architecture
------------
Input : (B, 1, 96, 96) – 96x96 grayscale crop
Encoder: 4 Conv-BN-ReLU-Pool blocks (16 -> 32 -> 64 -> 128 channels)
Pooling: AdaptiveAvgPool2d((2, 2)) bottleneck (512 features)
Head   : Fully-connected layers predicting 42 coordinates (21 joints x 2)
Output : (B, 42) normalised (x, y) coordinates in [0, 1]

Parameter count: ~150,000 (~150 KB in INT8) — fits easily in ESP32-S3 SRAM.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        pool: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HandLandmarkNet(nn.Module):
    """Tiny, fast landmark regression network optimized for ESP32-S3 and ESP-DL."""

    NUM_LANDMARKS: int = 21

    def __init__(self, in_channels: int = 1, base_filters: int = 16) -> None:
        super().__init__()
        f = base_filters

        # Encoder: 96x96 -> 48x48 -> 24x24 -> 12x12 -> 6x6
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, f, 3, pool=True),       # 96 -> 48
            ConvBlock(f, f * 2, 3, pool=True),              # 48 -> 24
            ConvBlock(f * 2, f * 4, 3, pool=True),          # 24 -> 12
            ConvBlock(f * 4, f * 8, 3, pool=True),          # 12 -> 6
        )

        # Spatial bottleneck: reduce (B, 128, 6, 6) -> (B, 128, 2, 2)
        # Slashes dense layer parameters from 1.18M down to ~65k
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        flat_dim = (f * 8) * 2 * 2  # 128 * 4 = 512

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.NUM_LANDMARKS * 2),
            nn.Sigmoid(),  # constrain output to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        pooled = self.pool(feat)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def build_model(**kwargs) -> HandLandmarkNet:
    """Instantiate and return the model."""
    return HandLandmarkNet(**kwargs)


def export_onnx(
    model: HandLandmarkNet,
    path: str = "hand_landmarks.onnx",
    opset: int = 13,
) -> None:
    """Export model to ONNX for downstream ESP-DL / PPQ quantisation."""
    model.eval()
    dummy = torch.randn(1, 1, 96, 96)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["landmarks"],
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"ONNX model exported -> {path}")


if __name__ == "__main__":
    net = build_model()
    total = sum(p.numel() for p in net.parameters())
    print(f"HandLandmarkNet parameters: {total:,} (~{total / 1024:.1f} KB in INT8)")
    x = torch.randn(2, 1, 96, 96)
    y = net(x)
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(y.shape)}")
