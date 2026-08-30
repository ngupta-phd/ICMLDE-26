"""Lightweight U-Net for optional tumor-region localization (not the real-time default)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import config


class LightweightUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        b = self.bottleneck(self.pool(x1))
        u = self.up(b)
        return self.dec1(torch.cat([u, x1], dim=1))


def load_unet_model(weights_path: Path | str | None = None, device: torch.device | str = "cpu") -> LightweightUNet:
    weights_path = Path(weights_path) if weights_path else config.UNET_WEIGHTS
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"U-Net weights not found: {weights_path}\n"
            "Place unet_brain_mri.pth in models/ or omit --segment."
        )

    model = LightweightUNet()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
