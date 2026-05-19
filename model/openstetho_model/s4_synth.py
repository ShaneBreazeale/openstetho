"""Synthetic fourth-heart-sound (S4) generator.

S4 ("atrial gallop") arises from atrial contraction against a stiff
ventricle. Clinically reported with:

  * Frequency 20–45 Hz (lower and slightly tighter than S3).
  * Duration 30–50 ms (shorter than S3).
  * Onset *late* in diastole — immediately precedes the next S1 (often
    cited at 60–120 ms before S1).
  * Amplitude typically quieter than S3.

S4 is a classic confounder for S3 detection: both are low-frequency
diastolic sounds. The detector needs to distinguish them by *timing*
(early vs late diastolic) and morphology, not just by spectral content.

Used by `s3_inject.s4_inject` to plant S4 events in *negative* cycles —
the cycle stays labeled 0, but the model is forced to learn that "low
frequency diastolic sound != S3" unless it is in the early-diastolic
window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 4000

F0_MIN_HZ = 20.0
F0_MAX_HZ = 45.0
TAU_MIN_S = 0.018
TAU_MAX_S = 0.030
DURATION_MIN_S = 0.030
DURATION_MAX_S = 0.050
AMP_MIN = 0.08
AMP_MAX = 0.35


@dataclass(frozen=True)
class S4Params:
    f0_hz: float
    tau_s: float
    duration_s: float
    amp: float


def synth_s4(
    sample_rate: int = SAMPLE_RATE,
    f0_hz: float = 30.0,
    tau_s: float = 0.022,
    duration_s: float = 0.040,
    amp: float = 0.2,
) -> np.ndarray:
    if duration_s <= 0.0:
        return np.zeros(0, dtype=np.float32)
    n = int(round(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float32) / sample_rate
    envelope = np.exp(-t / tau_s).astype(np.float32)
    return (amp * envelope * np.sin(2.0 * np.pi * f0_hz * t)).astype(np.float32)


def synth_s4_random(rng: np.random.Generator, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, S4Params]:
    params = S4Params(
        f0_hz=float(rng.uniform(F0_MIN_HZ, F0_MAX_HZ)),
        tau_s=float(rng.uniform(TAU_MIN_S, TAU_MAX_S)),
        duration_s=float(rng.uniform(DURATION_MIN_S, DURATION_MAX_S)),
        amp=float(rng.uniform(AMP_MIN, AMP_MAX)),
    )
    wave = synth_s4(
        sample_rate=sample_rate,
        f0_hz=params.f0_hz,
        tau_s=params.tau_s,
        duration_s=params.duration_s,
        amp=params.amp,
    )
    return wave, params
