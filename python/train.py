#!/usr/bin/env python3
"""
train.py – Training loop for HandLandmarkNet.

Usage
-----
    python train.py --data dataset/ --epochs 120 --lr 1e-3

After training the best checkpoint is saved to ``checkpoints/best.pth`` and
an ONNX export is written to ``exports/hand_landmarks.onnx``.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from model import HandLandmarkNet, build_model, export_onnx

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HandLandmarkDataset(Dataset):
    """Reads crops and landmark CSVs produced by ``collect_dataset.py``."""

    def __init__(self, root: Path, augment: bool = False) -> None:
        self.images_dir = root / "images"
        self.augment = augment
        self.samples: list[tuple[str, np.ndarray]] = []

        csv_path = root / "landmarks.csv"
        with open(csv_path) as f:
            for row in csv.reader(f):
                fname = row[0]
                coords = np.array(list(map(float, row[1:])), dtype=np.float32)
                self.samples.append((fname, coords))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fname, coords = self.samples[idx]
        img = cv2.imread(str(self.images_dir / fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {self.images_dir / fname}")

        # Simple augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                coords_xy = coords.reshape(-1, 2)
                coords_xy[:, 0] = 1.0 - coords_xy[:, 0]
                coords = coords_xy.flatten()

            # Brightness jitter
            delta = np.random.randint(-30, 31)
            img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

        tensor_img = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # (1, 96, 96)
        tensor_lm = torch.from_numpy(coords)  # (42,)
        return tensor_img, tensor_lm


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_dir: Path,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.15,
    checkpoint_dir: Path = Path("checkpoints"),
    export_dir: Path = Path("exports"),
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = HandLandmarkDataset(data_dir, augment=True)
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    # Disable augmentation for validation split (approximation)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= train_len

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                loss = criterion(preds, targets)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= val_len

        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr_now:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = checkpoint_dir / "best.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ best model saved ({val_loss:.6f})")

    # Final ONNX export
    model.load_state_dict(torch.load(checkpoint_dir / "best.pth", map_location=device))
    onnx_path = str(export_dir / "hand_landmarks.onnx")
    export_onnx(model, onnx_path)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train HandLandmarkNet")
    parser.add_argument("--data", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    train(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
