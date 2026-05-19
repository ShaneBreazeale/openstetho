"""Unit tests for synthetic S3 injection."""
from __future__ import annotations

import numpy as np

from openstetho_model import s3_inject as inj
from openstetho_model import segment as seg
from openstetho_model.preprocess import apply_s3_preset

SR = 4000


def _synthetic_pcg(seed: int = 0, n_cycles: int = 8) -> tuple[np.ndarray, seg.Segmentation]:
    """Build a clean synthetic PCG and segment it, returning both."""
    rng = np.random.default_rng(seed)
    cycle_period_s = 0.857
    n = int(n_cycles * cycle_period_s * SR) + SR
    x = rng.normal(0.0, 0.005, size=n).astype(np.float32)

    def thump(f0: float, dur_s: float, amp: float, tau_s: float = 0.025) -> np.ndarray:
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


def test_all_negative_leaves_audio_unchanged():
    x, s = _synthetic_pcg()
    rng = np.random.default_rng(7)
    y, records = inj.s3_inject(x, s, rng, prob_per_cycle=0.0)
    np.testing.assert_array_equal(x, y)
    assert all(not r.positive for r in records)


def test_all_positive_changes_audio():
    x, s = _synthetic_pcg()
    rng = np.random.default_rng(9)
    y, records = inj.s3_inject(x, s, rng, prob_per_cycle=1.0)
    assert not np.array_equal(x, y)
    # Most cycles should successfully inject (some may be skipped at edges).
    n_pos = sum(1 for r in records if r.positive)
    assert n_pos >= max(1, int(0.7 * len(records)))


def test_modifications_stay_within_diastolic_window():
    x, s = _synthetic_pcg()
    rng = np.random.default_rng(11)
    y, records = inj.s3_inject(x, s, rng, prob_per_cycle=1.0)
    diff = np.abs(y - x)
    # Any non-zero diff must lie inside *some* cycle's diastolic window.
    nonzero = np.flatnonzero(diff > 1e-7)
    diastolic_ranges = [r.cycle.diastole for r in records if r.positive]
    for idx in nonzero:
        assert any(d0 <= idx < d1 for d0, d1 in diastolic_ranges), (
            f"sample {idx} modified outside any diastolic window"
        )


def test_snr_target_approximately_met():
    # SNR convention: RMS of the injected S3 measured over its own time
    # footprint vs RMS of the diastolic noise measured over the (pre-injection)
    # diastolic window. This matches what the detector observes per mel frame.
    x, s = _synthetic_pcg(seed=3)
    rng = np.random.default_rng(13)
    y, records = inj.s3_inject(x, s, rng, prob_per_cycle=1.0, snr_db_range=(10.0, 10.0))
    diff = y - x
    achieved = []
    for r in records:
        if not r.positive:
            continue
        d0, d1 = r.cycle.diastole
        noise = x[d0:d1]
        n_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2) + 1e-12))
        # Find the S3 footprint inside this cycle's diastole.
        cycle_diff = diff[d0:d1]
        active = np.flatnonzero(np.abs(cycle_diff) > 1e-7)
        if len(active) == 0 or n_rms <= 0:
            continue
        footprint = cycle_diff[active[0] : active[-1] + 1]
        s_rms = float(np.sqrt(np.mean(footprint.astype(np.float64) ** 2) + 1e-12))
        achieved.append(20.0 * np.log10(s_rms / n_rms))
    assert achieved, "expected at least one successful injection"
    mean = float(np.mean(achieved))
    assert abs(mean - 10.0) < 2.0, f"mean SNR off target: {mean:.2f} dB"


def test_deterministic_per_seed():
    x, s = _synthetic_pcg()
    a, _ = inj.s3_inject(x, s, np.random.default_rng(42), prob_per_cycle=0.5)
    b, _ = inj.s3_inject(x, s, np.random.default_rng(42), prob_per_cycle=0.5)
    np.testing.assert_array_equal(a, b)


def test_s4_inject_places_events_in_late_diastole():
    x, s = _synthetic_pcg(seed=21)
    from openstetho_model.s3_inject import s4_inject

    y = s4_inject(x, s, np.random.default_rng(31), prob_per_cycle=1.0, snr_db_range=(10.0, 10.0))
    diff = y - x
    # Every modified sample must lie in the *late* half of its cycle's diastole,
    # closer to the next S1 than to S2.
    for cycle in s.cycles:
        d0, d1 = cycle.diastole
        cycle_diff = diff[d0:d1]
        active = np.flatnonzero(np.abs(cycle_diff) > 1e-7)
        if len(active) == 0:
            continue
        mid = (d1 - d0) // 2
        # All injected samples should be past the midpoint (late diastole).
        assert active.min() >= mid, (
            f"S4 injection landed in early diastole: cycle {cycle}, active range {active.min()}-{active.max()}, mid {mid}"
        )


def test_s4_inject_zero_prob_is_noop():
    x, s = _synthetic_pcg()
    from openstetho_model.s3_inject import s4_inject

    y = s4_inject(x, s, np.random.default_rng(0), prob_per_cycle=0.0)
    np.testing.assert_array_equal(x, y)


def test_prob_multi_one_adds_more_energy_than_single():
    # With prob_multi=1.0 every positive cycle gets two S3 events; total
    # diff energy should exceed the single-event case at the same SNR target.
    x, s = _synthetic_pcg(seed=4)
    rng_single = np.random.default_rng(101)
    y_single, _ = inj.s3_inject(x, s, rng_single, prob_per_cycle=1.0, snr_db_range=(10.0, 10.0))
    rng_multi = np.random.default_rng(101)
    y_multi, _ = inj.s3_inject(x, s, rng_multi, prob_per_cycle=1.0, snr_db_range=(10.0, 10.0), prob_multi=1.0)
    e_single = float(((y_single - x) ** 2).sum())
    e_multi = float(((y_multi - x) ** 2).sum())
    assert e_multi > e_single * 1.3


def test_s3_band_energy_increases_after_injection():
    x, s = _synthetic_pcg(seed=5)
    rng = np.random.default_rng(17)
    y, _ = inj.s3_inject(x, s, rng, prob_per_cycle=1.0, snr_db_range=(10.0, 15.0))
    xf = apply_s3_preset(x)
    yf = apply_s3_preset(y)
    # S3-band-filtered energy must be higher post-injection (S3 lives at
    # 30–70 Hz, the HP15→LP120 preset preserves that range).
    assert float((yf**2).mean()) > float((xf**2).mean()) * 1.05
