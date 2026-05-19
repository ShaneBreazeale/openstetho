"""Smoke tests for the cycle-level S3 dataset."""
from __future__ import annotations

from pathlib import Path

import torch

from openstetho_model.s3_dataset import (
    CYCLE_WINDOW_FRAMES,
    S3CycleDataset,
    write_synthetic_pcg_wav,
)


def _make_corpus(tmp_path: Path, n_wavs: int = 4, seed_base: int = 0) -> list[Path]:
    wavs = []
    for i in range(n_wavs):
        p = tmp_path / f"clip_{i:02d}.wav"
        write_synthetic_pcg_wav(p, n_cycles=6, seed=seed_base + i)
        wavs.append(p)
    return wavs


def test_dataset_non_empty(tmp_path: Path):
    wavs = _make_corpus(tmp_path)
    ds = S3CycleDataset(wavs, positive_rate=0.5, seed=0)
    assert len(ds) > 0


def test_class_balance_near_target(tmp_path: Path):
    wavs = _make_corpus(tmp_path, n_wavs=8)
    ds = S3CycleDataset(wavs, positive_rate=0.5, seed=42)
    balance = ds.class_balance()
    total = balance["negative"] + balance["positive"]
    assert abs(balance["positive"] / total - 0.5) < 0.20


def test_item_shape_is_fixed(tmp_path: Path):
    wavs = _make_corpus(tmp_path)
    ds = S3CycleDataset(wavs, positive_rate=0.5, seed=1)
    for k in range(min(len(ds), 6)):
        mel, label = ds[k]
        assert isinstance(mel, torch.Tensor)
        assert mel.shape[1] == 32  # N_MELS
        assert mel.shape[0] == CYCLE_WINDOW_FRAMES
        assert label in (0, 1)


def test_seed_positive_yields_label_one(tmp_path: Path):
    wavs = _make_corpus(tmp_path)
    ds = S3CycleDataset(wavs, positive_rate=1.0, seed=2)
    # Every cycle was seeded positive; require >= 70 % return label 1
    # (some may fail to inject due to short diastolic windows).
    labels = [ds[i][1] for i in range(len(ds))]
    assert sum(labels) / len(labels) >= 0.70


def test_seed_negative_yields_label_zero(tmp_path: Path):
    wavs = _make_corpus(tmp_path)
    ds = S3CycleDataset(wavs, positive_rate=0.0, seed=3)
    labels = [ds[i][1] for i in range(len(ds))]
    assert all(label == 0 for label in labels)


def test_s4_negative_mining_keeps_label_zero(tmp_path: Path):
    wavs = _make_corpus(tmp_path, n_wavs=4)
    # All cycles seeded negative, but prob_s4=1.0 forces S4 events into the
    # audio for every negative cycle. Labels must remain 0 — S4 is a
    # confounder, not a positive class.
    ds_s4 = S3CycleDataset(wavs, positive_rate=0.0, seed=11, prob_s4=1.0)
    labels = [ds_s4[i][1] for i in range(len(ds_s4))]
    assert labels and all(label == 0 for label in labels)


def test_s4_modifies_audio_path(tmp_path: Path):
    import numpy as np
    wavs = _make_corpus(tmp_path, n_wavs=4)
    ds_plain = S3CycleDataset(wavs, positive_rate=0.0, seed=12, prob_s4=0.0, apply_spec_masks=False)
    ds_s4 = S3CycleDataset(wavs, positive_rate=0.0, seed=12, prob_s4=1.0, apply_spec_masks=False)
    diffs = sum(
        1 for i in range(min(len(ds_s4), 6))
        if not np.array_equal(ds_plain[i][0].numpy(), ds_s4[i][0].numpy())
    )
    assert diffs >= 1, "S4 injection did not modify any mel sample"


def test_spec_masks_zero_some_bins(tmp_path: Path):
    wavs = _make_corpus(tmp_path)
    ds_mask = S3CycleDataset(
        wavs, positive_rate=0.0, seed=5,
        freq_mask_max_width=16, time_mask_max_width=4, apply_spec_masks=True,
    )
    ds_plain = S3CycleDataset(wavs, positive_rate=0.0, seed=5, apply_spec_masks=False)
    # With aggressive masks at least one access should produce a contiguous
    # all-equal band (the mask fill region). Compare to the un-masked dataset
    # to confirm the mask code is actually running, not the segmenter producing
    # accidentally-uniform mel bins.
    diffs = 0
    for i in range(min(len(ds_mask), 8)):
        a = ds_mask[i][0].numpy()
        b = ds_plain[i][0].numpy()
        if not (a == b).all():
            diffs += 1
    assert diffs >= 1, "freq/time masks did not modify any sample"


def test_low_confidence_recordings_dropped(tmp_path: Path):
    import numpy as np
    import soundfile as sf

    # Pure noise → segmenter confidence ≈ 0 → recording skipped.
    noise_path = tmp_path / "noise.wav"
    sf.write(str(noise_path), np.random.default_rng(0).normal(0, 0.01, 4 * 4000).astype("float32"), 4000)

    clean = _make_corpus(tmp_path, n_wavs=2, seed_base=10)
    ds = S3CycleDataset([noise_path] + clean, positive_rate=0.5, seed=0, min_segment_confidence=0.3)
    # Index should contain only cycles from `clean`; no negative index entries
    # tied to the noise file.
    assert all(e.wav != noise_path for e in ds._index)
