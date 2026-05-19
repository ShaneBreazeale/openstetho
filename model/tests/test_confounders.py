"""Smoke tests for the split-S2 / opening-snap / ejection-click confounders."""
from __future__ import annotations

import numpy as np

from openstetho_model import s3_inject as inj
from openstetho_model import segment as seg

SR = 4000


def _synthetic_pcg(seed: int = 0, n_cycles: int = 8):
    rng = np.random.default_rng(seed)
    cycle_period_s = 0.857
    n = int(n_cycles * cycle_period_s * SR) + SR
    x = rng.normal(0.0, 0.005, size=n).astype(np.float32)

    def thump(f0, dur_s, amp, tau_s=0.025):
        m = int(dur_s * SR)
        t = np.arange(m, dtype=np.float32) / SR
        return (amp * np.exp(-t / tau_s) * np.sin(2 * np.pi * f0 * t)).astype(np.float32)

    s1_wave = thump(60.0, 0.10, 1.0)
    s2_wave = thump(80.0, 0.08, 0.6)
    for k in range(n_cycles):
        s1_idx = int((k * cycle_period_s + 0.20) * SR)
        s2_idx = int((k * cycle_period_s + 0.20 + 0.35 * cycle_period_s) * SR)
        x[s1_idx : s1_idx + len(s1_wave)] += s1_wave
        x[s2_idx : s2_idx + len(s2_wave)] += s2_wave
    return x, seg.segment(x)


def test_split_s2_zero_prob_is_noop():
    x, s = _synthetic_pcg()
    y = inj.split_s2_inject(x, s, np.random.default_rng(0), prob_per_cycle=0.0)
    np.testing.assert_array_equal(x, y)


def test_split_s2_full_prob_modifies_audio():
    x, s = _synthetic_pcg()
    y = inj.split_s2_inject(x, s, np.random.default_rng(7), prob_per_cycle=1.0, snr_db_range=(15.0, 15.0))
    assert not np.array_equal(x, y)


def test_opening_snap_lands_after_s2():
    x, s = _synthetic_pcg()
    y = inj.opening_snap_inject(x, s, np.random.default_rng(3), prob_per_cycle=1.0, snr_db_range=(15.0, 15.0))
    diff = np.abs(y - x)
    nonzero = np.flatnonzero(diff > 1e-7)
    if len(nonzero) == 0:
        return
    for cycle in s.cycles:
        cycle_zone = (nonzero >= cycle.s2_idx) & (nonzero <= cycle.next_s1_idx)
        in_cycle = nonzero[cycle_zone]
        if len(in_cycle) == 0:
            continue
        # Opening-snap timing: 60-120ms after S2.
        min_offset = int(0.060 * SR)
        assert in_cycle.min() - cycle.s2_idx >= min_offset - 1


def test_ejection_click_stays_in_systole():
    x, s = _synthetic_pcg()
    y = inj.ejection_click_inject(x, s, np.random.default_rng(5), prob_per_cycle=1.0, snr_db_range=(15.0, 15.0))
    diff = np.abs(y - x)
    nonzero = np.flatnonzero(diff > 1e-7)
    for cycle in s.cycles:
        cycle_zone = (nonzero >= cycle.s1_idx) & (nonzero <= cycle.next_s1_idx)
        in_cycle = nonzero[cycle_zone]
        if len(in_cycle) == 0:
            continue
        # All injected samples must lie between S1 and S2 (systole).
        assert (in_cycle <= cycle.s2_idx).all()
