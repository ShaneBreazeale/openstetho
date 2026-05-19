"""Synthetic third-heart-sound (S3) generator.

S3 acoustic morphology (Shaver 1985; Marcus 2006):
  * Low-frequency damped oscillation, dominant 30–70 Hz, peak ~40 Hz.
  * Duration ~40–80 ms (longer than S1/S2 transients).
  * Onset 100–200 ms after S2 (early diastolic rapid-filling phase).
  * Amplitude 5–40 % of S1 in pathologic gallop; lower in physiologic S3.

This module produces *synthetic* S3 events for data augmentation. Real S3 is
not yet labeled in our public corpora — see docs/s3_annotation_protocol.md.

The generator is deterministic given (sample_rate, f0, tau_s, duration_s, amp).
A `synth_s3_random` helper draws all four from clinically-plausible ranges
using a caller-supplied `np.random.Generator` for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 4000

# Clinically-plausible ranges. Tight enough to avoid producing artifacts that
# look nothing like S3; loose enough to span normal/pathologic morphology.
F0_MIN_HZ = 30.0
F0_MAX_HZ = 70.0
TAU_MIN_S = 0.020
TAU_MAX_S = 0.040
DURATION_MIN_S = 0.040
DURATION_MAX_S = 0.080
AMP_MIN = 0.10
AMP_MAX = 0.50


@dataclass(frozen=True)
class S3Params:
    f0_hz: float
    tau_s: float
    duration_s: float
    amp: float


def synth_s3(
    sample_rate: int = SAMPLE_RATE,
    f0_hz: float = 40.0,
    tau_s: float = 0.030,
    duration_s: float = 0.060,
    amp: float = 0.3,
) -> np.ndarray:
    """Single damped sinusoid: `amp * exp(-t/tau) * sin(2π f0 t)`.

    Returns float32 array of length `round(duration_s * sample_rate)`. The
    waveform starts at zero (phase=0) so it can be added cleanly to PCG audio
    without a click at injection onset.
    """
    if duration_s <= 0.0:
        return np.zeros(0, dtype=np.float32)
    n = int(round(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float32) / sample_rate
    envelope = np.exp(-t / tau_s).astype(np.float32)
    return (amp * envelope * np.sin(2.0 * np.pi * f0_hz * t)).astype(np.float32)


def synth_s3_random(rng: np.random.Generator, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, S3Params]:
    """Draw plausible S3 params, return `(waveform, params)`.

    Caller controls reproducibility by passing a seeded `np.random.Generator`.
    Amplitude here is the raw scalar in the formula; calibrate to the host
    recording's RMS at injection time (see `s3_inject`).
    """
    params = S3Params(
        f0_hz=float(rng.uniform(F0_MIN_HZ, F0_MAX_HZ)),
        tau_s=float(rng.uniform(TAU_MIN_S, TAU_MAX_S)),
        duration_s=float(rng.uniform(DURATION_MIN_S, DURATION_MAX_S)),
        amp=float(rng.uniform(AMP_MIN, AMP_MAX)),
    )
    wave = synth_s3(
        sample_rate=sample_rate,
        f0_hz=params.f0_hz,
        tau_s=params.tau_s,
        duration_s=params.duration_s,
        amp=params.amp,
    )
    return wave, params
