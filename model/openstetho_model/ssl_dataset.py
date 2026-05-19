"""Contrastive (SimCLR-style) view-pair dataset.

Wraps an `S3CycleDataset` configured with rich augmentation (different
freq-mask, time-mask, and S3-injection random draws each call) so that two
consecutive accesses to the same cycle index produce *different* mel-spec
realizations — a valid positive pair for NT-Xent contrastive learning.

Labels are ignored; only the mel tensor is used.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .s3_dataset import S3CycleDataset


class ContrastivePCGDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Pairs of mel-spec views drawn from an underlying `S3CycleDataset`."""

    def __init__(
        self,
        wavs: Sequence[str | Path],
        freq_mask_max_width: int = 12,
        time_mask_max_width: int = 4,
        min_segment_confidence: float = 0.3,
        seed: int = 0,
        segmenter: str = "heuristic",
        crop_anchor: str = "s2",
        apply_pcg_augment: bool = True,
    ):
        # CRITICAL: cardiac content must be IDENTICAL across the two views
        # of a cycle, otherwise the contrastive loss rewards features that
        # are invariant to the very physiology we want the encoder to learn
        # (S3 / S4 / confounder presence). All event injections are disabled
        # here; the only differences between views are physio-level aug
        # (recording-condition jitter) and spec masks (frequency/time
        # dropout). Events are injected later during supervised fine-tune.
        self.inner = S3CycleDataset(
            wavs,
            positive_rate=0.0,           # no S3 injection
            min_segment_confidence=min_segment_confidence,
            seed=seed,
            prob_multi=0.0,
            prob_s4=0.0,
            prob_split_s2=0.0,
            prob_opening_snap=0.0,
            prob_ejection_click=0.0,
            freq_mask_max_width=freq_mask_max_width,
            time_mask_max_width=time_mask_max_width,
            apply_spec_masks=True,
            crop_anchor=crop_anchor,
            emit_multiclass=False,
            segmenter=segmenter,
            apply_pcg_augment=apply_pcg_augment,
        )

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Two consecutive accesses — the augmentation rng inside the inner
        # dataset is freshly seeded each call (`np.random.default_rng()` with
        # no seed) so we get two different views of the same underlying cycle.
        view_a, _ = self.inner[idx]
        view_b, _ = self.inner[idx]
        return view_a, view_b
