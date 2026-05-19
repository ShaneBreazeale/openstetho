"""Unit tests for synthetic S4 generator."""
from __future__ import annotations

import numpy as np

from openstetho_model import s4_synth


def test_length_matches_duration():
    w = s4_synth.synth_s4(sample_rate=4000, duration_s=0.040)
    assert w.dtype == np.float32
    assert len(w) == 160  # 0.040 * 4000


def test_peak_freq_within_band():
    w = s4_synth.synth_s4(sample_rate=4000, f0_hz=30.0, duration_s=0.4, tau_s=10.0, amp=1.0)
    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), d=1.0 / 4000)
    peak_hz = freqs[int(np.argmax(spec))]
    assert abs(peak_hz - 30.0) < 3.0


def test_s4_distinct_from_s3_band():
    # S4 dominant peak should land below the S3 nominal peak (40 Hz) with
    # the lower-bound f0=20 Hz draw.
    w = s4_synth.synth_s4(sample_rate=4000, f0_hz=20.0, duration_s=0.5, tau_s=10.0, amp=1.0)
    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), d=1.0 / 4000)
    peak_hz = freqs[int(np.argmax(spec))]
    assert peak_hz < 35.0


def test_random_within_clinical_ranges():
    rng = np.random.default_rng(0)
    for _ in range(40):
        _, p = s4_synth.synth_s4_random(rng)
        assert s4_synth.F0_MIN_HZ <= p.f0_hz <= s4_synth.F0_MAX_HZ
        assert s4_synth.TAU_MIN_S <= p.tau_s <= s4_synth.TAU_MAX_S
        assert s4_synth.DURATION_MIN_S <= p.duration_s <= s4_synth.DURATION_MAX_S
        assert s4_synth.AMP_MIN <= p.amp <= s4_synth.AMP_MAX
