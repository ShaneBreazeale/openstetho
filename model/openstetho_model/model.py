"""Murmur classifier — small Conv2D backbone on log-mel-spec frames.

Input shape  : (B, 1, T, N_MELS)   T=62 frames per 4-s window @ 4 kHz
Output shape : (B,)                logit for murmur-present probability

Design goals:
  * Small enough to run on the Apple Neural Engine via Core ML.
  * All ops are ANE-supported (Conv2D, BatchNorm, ReLU, AvgPool, Linear).
  * No attention / RNN — keep first pass shippable; revisit with AST later.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .preprocess import N_MELS


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class MurmurCNN(nn.Module):
    """Tiny VGG-style CNN. ~74k trainable params."""

    def __init__(self, n_classes: int = 1):
        super().__init__()
        self.b1 = conv_block(1, 16)   # (B, 16, T/2,  N/2)
        self.b2 = conv_block(16, 32)  # (B, 32, T/4,  N/4)
        self.b3 = conv_block(32, 64)  # (B, 64, T/8,  N/8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, N) → (B, 1, T, N) on the fly so callers can pass either.
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.pool(x)
        x = self.head(x)
        return x.squeeze(-1)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = MurmurCNN()
    x = torch.randn(2, 62, N_MELS)
    print("input ", tuple(x.shape))
    print("output", tuple(m(x).shape))
    print("params", count_parameters(m))
