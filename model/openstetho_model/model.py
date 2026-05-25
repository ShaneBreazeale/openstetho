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
from torch.nn.parameter import UninitializedParameter

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

    def __init__(self, n_classes: int = 1, in_channels: int = 1):
        super().__init__()
        self.b1 = conv_block(in_channels, 16)   # (B, 16, T/2,  N/2)
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


class MurmurCNNBiGRU(nn.Module):
    """CNN frontend with a bidirectional GRU temporal head.

    This keeps frequency pooling aggressive while preserving a reduced time
    axis for the recurrent layer. It is intended for offline experiments
    before any export/ANE work; the current production Core ML model remains
    `MurmurCNN`.
    """

    def __init__(self, n_classes: int = 1, hidden_size: int = 48, in_channels: int = 1):
        super().__init__()
        self.b1 = conv_block_pool(in_channels, 24, pool=(2, 2))
        self.b2 = conv_block_pool(24, 48, pool=(1, 2))
        self.b3 = conv_block_pool(48, 96, pool=(1, 2))
        self.freq_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.gru = nn.GRU(
            input_size=96,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_size * 2, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(48, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.freq_pool(x).squeeze(-1)  # (B, C, T')
        x = x.transpose(1, 2)              # (B, T', C)
        seq, _ = self.gru(x)
        pooled = seq.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class MurmurScatteringCNN1D(nn.Module):
    """Small 1D CNN for wavelet-scattering feature maps.

    Input shape is `(B, T, C)` where `C` is the scattering coefficient axis.
    The network treats scattering coefficients as 1D channels and convolves
    along time, matching the low-risk part of scattering+1D-CNN papers while
    staying usable with the existing recording-level aggregation loop.
    """

    def __init__(self, n_classes: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=5, padding=2, bias=False),
            nn.LazyBatchNorm1d(),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(48, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"expected (B,T,C) scattering input, got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        x = self.features(x)
        return self.head(x).squeeze(-1)


class S3CNN(nn.Module):
    """Bigger-capacity backbone for S3 detection.

    Three VGG-style conv blocks at 32 → 64 → 128 channels (~580k params,
    ~8× MurmurCNN). Stays inside the ANE-friendly op set: Conv2D, BatchNorm,
    ReLU, MaxPool, AdaptiveAvgPool, Linear, Dropout. No attention/RNN.

    Input shape  : (B, 1, T, N_MELS) — variable T, N_MELS fixed at 32.
    Output shape : (B,) — single logit (binary S3 head).

    Dropout is raised to 0.4 vs MurmurCNN's 0.3 — larger model, same data
    budget, more regularization needed to avoid overfitting on the synthetic
    label distribution.
    """

    def __init__(self, n_classes: int = 1):
        super().__init__()
        self.b1 = conv_block(1, 32)    # (B, 32, T/2, N/2)
        self.b2 = conv_block(32, 64)   # (B, 64, T/4, N/4)
        self.b3 = conv_block(64, 128)  # (B, 128, T/8, N/8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.pool(x)
        x = self.head(x)
        return x.squeeze(-1)


def conv_block_pool(in_ch: int, out_ch: int, pool: tuple[int, int] = (2, 2)) -> nn.Sequential:
    """Variant of `conv_block` with an explicit `(t, f)` pool shape so callers
    can freeze the time axis (`pool=(1, 2)`) while still downsampling
    frequency.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool),
    )


class S3CNN_v2(nn.Module):
    """Timing-aware backbone for S3/S4 discrimination.

    The original `S3CNN` collapses both time and frequency via
    `AdaptiveAvgPool2d(1)` — that destroys the temporal cue that
    distinguishes S3 (early diastolic) from S4 (late diastolic). This
    variant pools frequency only, then runs a small 1D conv head over
    the temporal axis so the network can learn *where* in the cycle a
    low-frequency event sits, not just whether one exists.

    Pool schedule: block 1 reduces both axes (T/2, F/2). Blocks 2-3
    reduce only frequency, so T stays at T/2 through the conv stack.
    With our 1.5-second cycle (T≈23 mel frames) the temporal head sees
    ~11 frames — enough to localise S3 (early diastole) vs S4 (late
    diastole) with frame-level accuracy.

    Stays ANE-friendly: Conv2D, Conv1D, BN, ReLU, MaxPool,
    AdaptiveAvgPool, Dropout, Linear. No attention or RNN.

    Roughly 4× MurmurCNN params (~290 k). The extra capacity went into
    the temporal head, not channel width.
    """

    def __init__(self, n_classes: int = 1):
        super().__init__()
        self.b1 = conv_block_pool(1, 32, pool=(2, 2))    # (B, 32, T/2,  F/2)
        self.b2 = conv_block_pool(32, 64, pool=(1, 2))   # (B, 64, T/2,  F/4)
        self.b3 = conv_block_pool(64, 128, pool=(1, 2))  # (B, 128, T/2, F/8)
        # Pool frequency only — keep the temporal axis intact for the head.
        self.freq_pool = nn.AdaptiveAvgPool2d((None, 1))  # (B, 128, T/2, 1)
        self.temporal = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # (B, 32, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.freq_pool(x).squeeze(-1)  # (B, 128, T/2)
        x = self.temporal(x)               # (B, 32, 1)
        return self.head(x).squeeze(-1)


class S3CNN_v3(nn.Module):
    """Wider timing-aware backbone targeting AUPRC > 0.9.

    Same topology as `S3CNN_v2` (freq-only pooling after block 1, time
    preserved through the conv stack, 1D temporal head) but widened to
    64 → 128 → 256 channels and deepened in the temporal head from 2 to
    3 conv1d layers. About 2.4 M parameters, ~7.5× S3CNN_v2.

    Dropout is raised to 0.5 / 0.3 to compensate for the capacity bump —
    our supervision is still synthetic so we are extra cautious about
    fitting noise.

    Stays ANE-friendly.
    """

    def __init__(self, n_classes: int = 1):
        super().__init__()
        self.b1 = conv_block_pool(1, 64, pool=(2, 2))      # (B, 64, T/2,  F/2)
        self.b2 = conv_block_pool(64, 128, pool=(1, 2))    # (B, 128, T/2, F/4)
        self.b3 = conv_block_pool(128, 256, pool=(1, 2))   # (B, 256, T/2, F/8)
        self.freq_pool = nn.AdaptiveAvgPool2d((None, 1))   # (B, 256, T/2, 1)
        self.temporal = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.freq_pool(x).squeeze(-1)  # (B, 256, T/2)
        x = self.temporal(x)               # (B, 32, 1)
        return self.head(x).squeeze(-1)


def count_parameters(m: nn.Module) -> int:
    return sum(
        p.numel()
        for p in m.parameters()
        if p.requires_grad and not isinstance(p, UninitializedParameter)
    )


if __name__ == "__main__":
    m = MurmurCNN()
    x = torch.randn(2, 62, N_MELS)
    print("input ", tuple(x.shape))
    print("output", tuple(m(x).shape))
    print("params", count_parameters(m))
