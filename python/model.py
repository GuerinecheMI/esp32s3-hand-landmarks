"""
model.py – Lightweight CNN for 21-point hand landmark regression.

Architecture
------------
A compact encoder (4 conv blocks) followed by a fully-connected head that
outputs 42 values (21 landmarks × 2 coordinates, normalised to [0, 1]).

Input : (B, 1, 96, 96)  – grayscale crop
Output: (B, 42)          – flattened (x, y) pairs
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → optional MaxPool."""

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
    """Tiny landmark regression network targeting ESP32-S3 (< 500 KB)."""

    NUM_LANDMARKS: int = 21

    def __init__(self, in_channels: int = 1, base_filters: int = 16) -> None:
        super().__init__()
        f = base_filters

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, f, 3, pool=True),       # 96→48
            ConvBlock(f, f * 2, 3, pool=True),              # 48→24
            ConvBlock(f * 2, f * 4, 3, pool=True),          # 24→12
            ConvBlock(f * 4, f * 8, 3, pool=True),          # 12→6
        )

        # After encoder: (B, f*8, 6, 6)
        flat_dim = f * 8 * 6 * 6

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.NUM_LANDMARKS * 2),
            nn.Sigmoid(),  # constrain to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


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
    """Export *model* to ONNX for downstream quantisation."""
    model.eval()
    dummy = torch.randn(1, 1, 96, 96)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["landmarks"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "landmarks": {0: "batch"}},
    )
    print(f"ONNX model exported → {path}")


if __name__ == "__main__":
    net = build_model()
    total = sum(p.numel() for p in net.parameters())
    print(f"HandLandmarkNet — {total:,} parameters")
    x = torch.randn(2, 1, 96, 96)
    y = net(x)
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(y.shape)}")
