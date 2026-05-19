"""Acoustic augmentations for PCG self-supervision.

Per [[feedback-ssl-must-teach-physiology]], the synthetic augmentation suite
must vary the recording at the physiology level so that contrastive SSL does
not collapse to learning "recording fingerprint" features. Each helper here
mutates a 4 kHz mono PCG waveform in a single physiologically-plausible way
and returns a fresh array (callers should not mutate the input).
"""
from __future__ import annotations

import numpy as np
import scipy.signal as sps

from .preprocess import SAMPLE_RATE


def add_respiration_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    sample_rate: int = SAMPLE_RATE,
    amp_rel_range: tuple[float, float] = (0.02, 0.15),
    f_range_hz: tuple[float, float] = (0.20, 0.50),
) -> np.ndarray:
    """Add a slow low-frequency oscillation modeling breathing-induced
    envelope modulation of the PCG signal.
    """
    amp = float(rng.uniform(*amp_rel_range)) * (np.abs(x).max() + 1e-9)
    f = float(rng.uniform(*f_range_hz))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    t = np.arange(len(x), dtype=np.float32) / sample_rate
    return (x + amp * np.sin(2.0 * np.pi * f * t + phase)).astype(np.float32)


def add_baseline_drift(
    x: np.ndarray,
    rng: np.random.Generator,
    sample_rate: int = SAMPLE_RATE,
    amp_rel_range: tuple[float, float] = (0.02, 0.20),
    f_cutoff_hz: float = 1.0,
) -> np.ndarray:
    """Random walk filtered to below f_cutoff_hz — models DC wander from
    contact pressure changes or vagal envelope drift.
    """
    walk = rng.standard_normal(len(x)).astype(np.float32).cumsum()
    nyq = sample_rate / 2.0
    sos = sps.butter(2, f_cutoff_hz / nyq, btype="low", output="sos")
    drift = sps.sosfiltfilt(sos, walk).astype(np.float32)
    drift -= drift.mean()
    drift /= max(np.abs(drift).max(), 1e-9)
    amp = float(rng.uniform(*amp_rel_range)) * (np.abs(x).max() + 1e-9)
    return (x + amp * drift).astype(np.float32)


def random_eq(
    x: np.ndarray,
    rng: np.random.Generator,
    sample_rate: int = SAMPLE_RATE,
    boost_db_range: tuple[float, float] = (-6.0, 6.0),
    center_hz_range: tuple[float, float] = (40.0, 300.0),
    q_range: tuple[float, float] = (0.5, 3.0),
) -> np.ndarray:
    """Apply a single peaking biquad with random center / gain / Q. Models
    the bandpass character of different stethoscope / mic combinations.
    """
    center = float(rng.uniform(*center_hz_range))
    gain_db = float(rng.uniform(*boost_db_range))
    q = float(rng.uniform(*q_range))
    # Peaking EQ via scipy.signal.iirpeak doesn't take gain; build manually.
    w0 = 2.0 * np.pi * center / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    A = 10.0 ** (gain_db / 40.0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return sps.lfilter(b, a, x).astype(np.float32)


def random_attenuation(
    x: np.ndarray,
    rng: np.random.Generator,
    sample_rate: int = SAMPLE_RATE,
    gain_db_range: tuple[float, float] = (-12.0, 0.0),
    rolloff_hz_range: tuple[float, float] = (200.0, 1000.0),
) -> np.ndarray:
    """Model body-habitus signal attenuation: lower gain + steeper LP
    rolloff (high frequencies attenuate first as fat thickness increases).
    """
    gain_lin = 10.0 ** (float(rng.uniform(*gain_db_range)) / 20.0)
    cutoff = float(rng.uniform(*rolloff_hz_range))
    nyq = sample_rate / 2.0
    sos = sps.butter(2, cutoff / nyq, btype="low", output="sos")
    y = sps.sosfilt(sos, x).astype(np.float32) * gain_lin
    return y.astype(np.float32)


def time_stretch_cycle(
    x: np.ndarray,
    rng: np.random.Generator,
    stretch_range: tuple[float, float] = (0.7, 1.4),
) -> np.ndarray:
    """Resample to simulate HR change. `stretch < 1` = faster heart rate
    (compresses the signal). Implemented via `scipy.signal.resample_poly`.
    """
    stretch = float(rng.uniform(*stretch_range))
    # up / down ratios — quantise to small integers so resample_poly stays
    # cheap. Step of 0.05 in the stretch is fine for our purpose.
    up = max(1, int(round(100 * stretch)))
    down = 100
    return sps.resample_poly(x, up, down).astype(np.float32)


def random_pcg_augment(
    x: np.ndarray,
    rng: np.random.Generator,
    sample_rate: int = SAMPLE_RATE,
    prob_respiration: float = 0.7,
    prob_drift: float = 0.5,
    prob_eq: float = 0.6,
    prob_attenuation: float = 0.4,
) -> np.ndarray:
    """Apply a stack of random augmentations to a PCG waveform. Each
    augmentation fires with the given probability. The returned array has
    the same dtype + sample rate as the input.

    Time-stretch is NOT included here — it changes the array length and
    must be applied at cycle-segmentation time, not at the final waveform
    step.
    """
    y = x.astype(np.float32, copy=True)
    if rng.random() < prob_respiration:
        y = add_respiration_noise(y, rng, sample_rate)
    if rng.random() < prob_drift:
        y = add_baseline_drift(y, rng, sample_rate)
    if rng.random() < prob_eq:
        y = random_eq(y, rng, sample_rate)
    if rng.random() < prob_attenuation:
        y = random_attenuation(y, rng, sample_rate)
    return y
