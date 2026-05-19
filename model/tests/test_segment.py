"""Heuristic PCG segmenter — synthetic-cycle validation."""
from __future__ import annotations

import numpy as np

from openstetho_model import segment as seg

SR = 4000


def _synthetic_pcg(
    n_cycles: int = 8,
    cycle_period_s: float = 0.857,  # 70 bpm
    systole_frac: float = 0.35,
    s1_amp: float = 1.0,
    s2_amp: float = 0.6,
    noise_rms: float = 0.005,
    seed: int = 0,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Generate fake PCG: damped-sinusoid S1/S2 with realistic spacing.

    Returns (waveform, s1_indices, s2_indices) in samples.
    """
    rng = np.random.default_rng(seed)
    n = int(n_cycles * cycle_period_s * SR) + SR
    x = rng.normal(0.0, noise_rms, size=n).astype(np.float32)

    def thump(f0: float, dur_s: float, amp: float, tau_s: float = 0.025) -> np.ndarray:
        m = int(dur_s * SR)
        t = np.arange(m, dtype=np.float32) / SR
        return (amp * np.exp(-t / tau_s) * np.sin(2 * np.pi * f0 * t)).astype(np.float32)

    s1_wave = thump(60.0, 0.10, s1_amp)
    s2_wave = thump(80.0, 0.08, s2_amp)
    s1s, s2s = [], []
    for k in range(n_cycles):
        s1_idx = int((k * cycle_period_s + 0.20) * SR)
        s2_idx = int((k * cycle_period_s + 0.20 + systole_frac * cycle_period_s) * SR)
        x[s1_idx : s1_idx + len(s1_wave)] += s1_wave
        x[s2_idx : s2_idx + len(s2_wave)] += s2_wave
        s1s.append(s1_idx)
        s2s.append(s2_idx)
    return x, s1s, s2s


def test_short_audio_returns_empty():
    out = seg.segment(np.zeros(1000, dtype=np.float32))
    assert out.cycles == []
    assert out.confidence == 0.0


def test_recovers_cycle_period_within_5_percent():
    x, _, _ = _synthetic_pcg(n_cycles=10, cycle_period_s=0.857)
    out = seg.segment(x)
    assert abs(out.cycle_period_s - 0.857) / 0.857 < 0.05


def test_detects_most_cycles():
    x, s1s, _ = _synthetic_pcg(n_cycles=10, cycle_period_s=0.857)
    out = seg.segment(x)
    # Allow off-by-one at edges; require >= 7 of 9 candidate cycles.
    assert len(out.cycles) >= 7


def test_s1_before_s2_in_each_cycle():
    x, _, _ = _synthetic_pcg(n_cycles=10)
    out = seg.segment(x)
    for c in out.cycles:
        assert c.s1_idx < c.s2_idx < c.next_s1_idx


def test_diastole_window_inside_cycle():
    x, _, _ = _synthetic_pcg(n_cycles=10)
    out = seg.segment(x)
    for c in out.cycles:
        d0, d1 = c.diastole
        assert c.s2_idx <= d0 < d1 < c.next_s1_idx


def test_confidence_higher_for_clean_signal():
    clean, _, _ = _synthetic_pcg(noise_rms=0.001, seed=1)
    noisy = clean + np.random.default_rng(2).normal(0.0, 0.10, size=len(clean)).astype(np.float32)
    a = seg.segment(clean)
    b = seg.segment(noisy)
    assert a.confidence > b.confidence
