from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from openstetho_model.calibration import nll
from openstetho_model.dataset import MurmurAugmentationConfig, _augment_channel
from openstetho_model.preprocess import N_MELS, SAMPLE_RATE
from openstetho_model.recording_mix import (
    MixedRecordingBatchSampler,
    SyntheticConfig,
    SyntheticRecordings,
    mixed_train_prior,
    murmur_sources,
)
from openstetho_model.train import validation_selection

WINDOW = 5 * SAMPLE_RATE
HOP = WINDOW // 2


def _write_source(tmp_path, name: str, seconds: float = 12.0, with_tsv: bool = True):
    """A fake normal recording: clicks at S1/S2 plus a matching CirCor tsv."""
    n = int(seconds * SAMPLE_RATE)
    rng = np.random.default_rng(0)
    audio = 0.01 * rng.standard_normal(n).astype(np.float32)
    rows = []
    t = 0.0
    while t + 0.8 <= seconds:
        rows += [(t, t + 0.1, 1), (t + 0.1, t + 0.35, 2), (t + 0.35, t + 0.45, 3), (t + 0.45, t + 0.8, 4)]
        for start, end, state in rows[-4:]:
            if state in (1, 3):
                a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
                audio[a:b] += 0.3 * np.hanning(b - a).astype(np.float32)
        t += 0.8
    wav = tmp_path / f"{name}.wav"
    sf.write(wav, audio, SAMPLE_RATE)
    if with_tsv:
        (tmp_path / f"{name}.tsv").write_text("".join(f"{a:.3f}\t{b:.3f}\t{s}\n" for a, b, s in rows))
    return wav


def test_sampler_mixes_exact_synthetic_count_and_covers_every_real_once():
    sampler = MixedRecordingBatchSampler(n_real=100, n_synth=1000, batch_size=16, synth_fraction=0.3, seed=0)
    assert sampler.k_synth == 4 and sampler.k_real == 12
    assert sampler.synthetic_share == pytest.approx(0.25)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 100 // 12
    real_seen = []
    for batch in batches:
        assert len(batch) == 16
        synth = [i for i in batch if i >= 100]
        assert len(synth) == 4 and all(i < 1100 for i in synth)
        real_seen += [i for i in batch if i < 100]
    assert len(real_seen) == len(set(real_seen)) == 12 * len(batches)


def test_sampler_reshuffles_each_epoch():
    sampler = MixedRecordingBatchSampler(n_real=50, n_synth=500, batch_size=10, synth_fraction=0.2, seed=1)
    assert list(sampler) != list(sampler)


def test_sampler_rejects_all_synthetic():
    with pytest.raises(ValueError):
        MixedRecordingBatchSampler(n_real=10, n_synth=10, batch_size=4, synth_fraction=1.0, seed=0)


def test_mixed_train_prior():
    assert mixed_train_prior(0.2, 0.25, 0.5) == pytest.approx(0.75 * 0.2 + 0.25 * 0.5)
    assert mixed_train_prior(0.2, 0.0, 0.5) == pytest.approx(0.2)


def test_murmur_sources_requires_tsv_and_a_full_window(tmp_path):
    good = _write_source(tmp_path, "good")
    no_tsv = _write_source(tmp_path, "no_tsv", with_tsv=False)
    short = _write_source(tmp_path, "short", seconds=3.0)
    assert murmur_sources([good, no_tsv, short], WINDOW) == [good]


def test_synthetic_murmur_recording_is_labelled_windowed_and_deterministic(tmp_path):
    src = _write_source(tmp_path, "src")
    ds = SyntheticRecordings(
        [src], SyntheticConfig(murmur_share=1.0, seed=3),
        window_samples=WINDOW, hop_samples=HOP, feature_mode="logmel",
    )
    mels, label = ds[7]
    again, _ = ds[7]
    assert label == 1
    assert mels.ndim == 3 and mels.shape[-1] == N_MELS
    assert mels.shape[0] == 1 + (12 * SAMPLE_RATE - WINDOW) // HOP
    assert np.array_equal(mels.numpy(), again.numpy())
    assert not np.array_equal(mels.numpy(), ds[8][0].numpy())


def test_synthetic_no_heart_needs_no_sources():
    ds = SyntheticRecordings(
        [], SyntheticConfig(murmur_share=0.0, seed=0),
        window_samples=WINDOW, hop_samples=HOP, feature_mode="logmel",
    )
    for k in range(6):
        mels, label = ds[k]
        assert label == 0 and mels.shape[0] >= 1


def test_synthetic_murmur_without_sources_is_rejected():
    with pytest.raises(ValueError):
        SyntheticRecordings(
            [], SyntheticConfig(murmur_share=0.5),
            window_samples=WINDOW, hop_samples=HOP, feature_mode="logmel",
        )


def test_channel_flags_enable_augmentation_and_change_audio():
    cfg = MurmurAugmentationConfig(eq_prob=1.0)
    assert cfg.enabled and cfg.channel_enabled
    assert not MurmurAugmentationConfig().channel_enabled
    x = np.random.default_rng(0).standard_normal(8000).astype(np.float32)
    y = _augment_channel(x, cfg, np.random.default_rng(0))
    assert y.shape == x.shape and y.dtype == np.float32 and not np.allclose(x, y)
    assert _augment_channel(x, MurmurAugmentationConfig(), np.random.default_rng(0)) is x


def test_nll_selection_is_negated_shifted_nll():
    labels = np.array([0, 0, 1, 1, 0])
    probs = np.array([0.1, 0.4, 0.8, 0.6, 0.3])
    z = np.log(probs / (1 - probs))
    score, row = validation_selection(labels, probs, "nll")
    assert score == pytest.approx(-nll(z, labels), abs=1e-6)
    shifted, row = validation_selection(labels, probs, "nll", logit_offset=-0.7)
    assert shifted == pytest.approx(-nll(z - 0.7, labels), abs=1e-6)
    assert row["logit_offset"] == -0.7
