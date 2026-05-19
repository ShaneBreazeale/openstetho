"""Smoke tests for physio augmentation helpers."""
from __future__ import annotations

import numpy as np

from openstetho_model import pcg_augment as pa

SR = 4000


def _signal():
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 0.1, size=SR * 4).astype(np.float32)


def test_respiration_adds_lf_energy():
    x = _signal()
    y = pa.add_respiration_noise(x, np.random.default_rng(1), amp_rel_range=(0.10, 0.10), f_range_hz=(0.3, 0.3))
    spec_x = np.abs(np.fft.rfft(x))
    spec_y = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / SR)
    lf_mask = (freqs > 0.0) & (freqs < 1.0)
    assert spec_y[lf_mask].sum() > spec_x[lf_mask].sum() * 1.5


def test_baseline_drift_modifies_signal():
    x = _signal()
    y = pa.add_baseline_drift(x, np.random.default_rng(2))
    assert not np.array_equal(x, y)
    # Drift content < 1 Hz; check power in that band rose.
    spec_x = np.abs(np.fft.rfft(x))
    spec_y = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / SR)
    band = (freqs > 0.05) & (freqs < 1.0)
    assert spec_y[band].sum() >= spec_x[band].sum()


def test_random_eq_returns_same_length():
    x = _signal()
    y = pa.random_eq(x, np.random.default_rng(3))
    assert len(y) == len(x)


def test_random_attenuation_lowers_amplitude():
    x = _signal()
    rng = np.random.default_rng(4)
    y = pa.random_attenuation(x, rng, gain_db_range=(-12.0, -12.0))
    # Gain reduction = -12 dB ≈ 0.25× → energy ratio ≤ 1/16. Allow LP filter slack.
    assert (y * y).mean() < (x * x).mean() * 0.30


def test_time_stretch_changes_length():
    x = _signal()
    y = pa.time_stretch_cycle(x, np.random.default_rng(5), stretch_range=(0.5, 0.5))
    # stretch 0.5 -> half-length (within polyphase rounding).
    assert abs(len(y) - len(x) // 2) <= 4


def test_random_pcg_augment_returns_same_length():
    x = _signal()
    y = pa.random_pcg_augment(x, np.random.default_rng(6))
    assert len(y) == len(x)
