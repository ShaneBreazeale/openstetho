"""Unit tests for synthetic S3 generator."""
from __future__ import annotations

import numpy as np

from openstetho_model import s3_synth


def test_length_matches_duration():
    w = s3_synth.synth_s3(sample_rate=4000, duration_s=0.060)
    assert w.dtype == np.float32
    assert len(w) == 240  # 0.060 * 4000


def test_zero_duration_returns_empty():
    w = s3_synth.synth_s3(duration_s=0.0)
    assert w.shape == (0,)


def test_starts_at_zero_no_click():
    w = s3_synth.synth_s3()
    assert abs(w[0]) < 1e-6


def test_peak_frequency_within_band():
    w = s3_synth.synth_s3(sample_rate=4000, f0_hz=45.0, duration_s=0.5, tau_s=10.0, amp=1.0)
    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), d=1.0 / 4000)
    peak_hz = freqs[int(np.argmax(spec))]
    assert abs(peak_hz - 45.0) < 3.0


def test_envelope_decays_monotonically_at_zero_crossings():
    w = s3_synth.synth_s3(f0_hz=50.0, duration_s=0.080, tau_s=0.020, amp=1.0)
    peaks = []
    for i in range(1, len(w) - 1):
        if w[i] > w[i - 1] and w[i] > w[i + 1] and w[i] > 0:
            peaks.append(w[i])
    assert len(peaks) >= 3
    for a, b in zip(peaks, peaks[1:]):
        assert b < a


def test_amp_scales_linearly():
    w1 = s3_synth.synth_s3(amp=0.1)
    w2 = s3_synth.synth_s3(amp=0.4)
    np.testing.assert_allclose(w2, 4.0 * w1, rtol=1e-5, atol=1e-7)


def test_random_within_clinical_ranges():
    rng = np.random.default_rng(0)
    for _ in range(50):
        _, p = s3_synth.synth_s3_random(rng)
        assert s3_synth.F0_MIN_HZ <= p.f0_hz <= s3_synth.F0_MAX_HZ
        assert s3_synth.TAU_MIN_S <= p.tau_s <= s3_synth.TAU_MAX_S
        assert s3_synth.DURATION_MIN_S <= p.duration_s <= s3_synth.DURATION_MAX_S
        assert s3_synth.AMP_MIN <= p.amp <= s3_synth.AMP_MAX


def test_random_is_deterministic_per_seed():
    a, _ = s3_synth.synth_s3_random(np.random.default_rng(42))
    b, _ = s3_synth.synth_s3_random(np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
