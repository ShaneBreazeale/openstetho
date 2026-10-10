"""Synthetic signals with ground truth known by construction.

Two jobs:

* `insert_murmur` adds a band-limited, enveloped noise murmur to a *real*
  CirCor recording, only inside one annotated cardiac phase, at an exact
  SNR relative to that recording's S1 + S2. The heart underneath is real;
  the label, timing and loudness of the murmur are known exactly.
* `no_heart` produces inputs with no heart in them (silence, coloured
  noise, mains hum, rubbing bursts) — the out-of-distribution cases that
  currently read as murmur.

Everything here comes from public formulas and CirCor's own segmentation
annotations. No teacher outputs, no vendor data.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preprocess import SAMPLE_RATE

# CirCor `.tsv` segmentation states.
STATE_UNANNOTATED = 0
STATE_S1 = 1
STATE_SYSTOLE = 2
STATE_S2 = 3
STATE_DIASTOLE = 4

MURMUR_SHAPES = ("crescendo_decrescendo", "plateau", "decrescendo", "crescendo")
NO_HEART_KINDS = ("zeros", "white", "pink", "brown", "hum", "rub")

# Segments shorter than this (8 ms) are left alone: too short to carry a
# band-limited murmur and too short to matter at a 64 ms STFT frame.
MIN_SEGMENT_SAMPLES = 32


@dataclass(frozen=True)
class Segment:
    start: float  # seconds
    end: float  # seconds
    state: int


def load_segments(tsv: str | Path) -> list[Segment]:
    """Read a CirCor segmentation file: one `start end state` row per line."""
    segs = []
    for line in Path(tsv).read_text().splitlines():
        parts = line.split()
        if parts:
            segs.append(Segment(float(parts[0]), float(parts[1]), int(float(parts[2]))))
    return segs


def segment_slice(seg: Segment, n: int, sr: int = SAMPLE_RATE) -> slice:
    """Sample range covered by `seg` in an n-sample recording. Rounded to
    the nearest sample and clipped to the audio (a few CirCor annotations
    run past the end of their WAV)."""
    a = min(max(int(round(seg.start * sr)), 0), n)
    b = min(max(int(round(seg.end * sr)), a), n)
    return slice(a, b)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def _band_bins(n: int, low_hz: float, high_hz: float, sr: int) -> np.ndarray:
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return (f >= low_hz) & (f <= high_hz)


def band_noise(
    n: int, low_hz: float, high_hz: float, rng: np.random.Generator, sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Gaussian noise restricted to [low_hz, high_hz] by zeroing FFT bins,
    scaled to unit RMS. Because the mask is applied to the exact n-point
    spectrum, every bit of power lies inside the band."""
    if not 0.0 <= low_hz < high_hz <= sr / 2:
        raise ValueError(f"band must satisfy 0 <= low < high <= {sr / 2}, got ({low_hz}, {high_hz})")
    keep = _band_bins(n, low_hz, high_hz, sr)
    if not keep.any():
        raise ValueError(f"no FFT bin of a {n}-sample signal falls in ({low_hz}, {high_hz}) Hz")
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[~keep] = 0.0
    x = np.fft.irfft(spec, n=n)
    return x / _rms(x)


def murmur_envelope(shape: str, n: int, *, peak: float = 0.5) -> np.ndarray:
    """Amplitude envelope for one murmur segment of `n` samples.

    Shapes follow the clinical murmur profiles:
      crescendo_decrescendo  mid-systolic ejection (aortic / pulmonic
                             stenosis, innocent flow): rises then falls,
                             peak at fraction `peak` of the segment
                             (mild AS peaks early, severe AS late),
                             ~0 at both ends
      plateau                holosystolic regurgitant (MR, TR, VSD):
                             flat, with a 10 % raised-cosine taper at
                             each edge
      decrescendo            early diastolic (AR, PR): loudest at the
                             start, decays exponentially
      crescendo              presystolic / late systolic (MS, MVP):
                             builds exponentially to the end

    Every shape is smooth: a hard corner would add broadband clicks the
    model could learn instead of the murmur. Peak is exactly 1.
    """
    if shape not in MURMUR_SHAPES:
        raise ValueError(f"unknown murmur shape {shape!r}; expected one of {MURMUR_SHAPES}")
    if not 0.0 < peak < 1.0:
        raise ValueError(f"peak must lie strictly inside (0, 1), got {peak}")
    t = (np.arange(n) + 0.5) / n  # sample centres in (0, 1); n = 1 gives t = 0.5
    if shape == "crescendo_decrescendo":
        # Piecewise-linear time warp sends t = peak to 0.5, so sin² peaks there.
        u = np.where(t < peak, 0.5 * t / peak, 0.5 + 0.5 * (t - peak) / (1.0 - peak))
        e = np.sin(np.pi * u) ** 2
    elif shape == "plateau":
        ramp = np.clip(np.minimum(t, 1.0 - t) / 0.1, 0.0, 1.0)
        e = np.sin(0.5 * np.pi * ramp) ** 2
    elif shape == "decrescendo":
        e = np.exp(-3.0 * t)
    else:  # crescendo
        e = np.exp(-3.0 * (1.0 - t))
    return e / e.max()


def insert_murmur(
    audio: np.ndarray,
    segments: Sequence[Segment],
    *,
    state: int,
    shape: str,
    snr_db: float,
    rng: np.random.Generator,
    band: tuple[float, float] = (100.0, 400.0),
    sr: int = SAMPLE_RATE,
    envelope: Callable[[str, int], np.ndarray] | None = None,
) -> np.ndarray:
    """Return a copy of `audio` with a synthetic murmur added inside every
    segment of `state` (STATE_SYSTOLE or STATE_DIASTOLE).

    Each segment gets fresh `band_noise` times `envelope(shape, len)`. The
    sum is then scaled so its RMS over the active samples is `snr_db`
    relative to the RMS of the recording's S1 + S2 samples: grade a
    murmur against the heart sounds, not against silence. Samples outside
    the target segments are bit-identical to the input.
    """
    env = murmur_envelope if envelope is None else envelope
    x = np.asarray(audio, dtype=np.float64)
    n = len(x)

    heart = np.zeros(n, dtype=bool)
    for s in segments:
        if s.state in (STATE_S1, STATE_S2):
            heart[segment_slice(s, n, sr)] = True
    ref = _rms(x[heart])
    if ref == 0.0:
        raise ValueError("recording has no annotated S1/S2 energy to set the SNR against")

    component = np.zeros(n)
    active = np.zeros(n, dtype=bool)
    for s in segments:
        if s.state != state:
            continue
        sl = segment_slice(s, n, sr)
        m = sl.stop - sl.start
        if m < MIN_SEGMENT_SAMPLES or not _band_bins(m, *band, sr).any():
            continue
        component[sl] = band_noise(m, *band, rng, sr) * env(shape, m)
        active[sl] = True
    if not active.any():
        raise ValueError(f"no usable segment with state {state}")

    level = _rms(component[active])
    if level == 0.0:
        raise ValueError(f"envelope {shape!r} is zero on every segment")
    out = x + component * (ref * 10.0 ** (snr_db / 20.0) / level)
    return out.astype(np.asarray(audio).dtype, copy=False)


def _coloured(n: int, exponent: float, rng: np.random.Generator, sr: int) -> np.ndarray:
    """Noise with power spectral density proportional to f^-exponent
    (0 white, 1 pink, 2 brown): shape white noise's amplitude spectrum by
    f^(-exponent/2). DC is dropped."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    gain = np.zeros_like(f)
    gain[1:] = f[1:] ** (-exponent / 2.0)
    return np.fft.irfft(spec * gain, n=n)


def _hum(n: int, rng: np.random.Generator, sr: int) -> np.ndarray:
    mains = float(rng.choice((50.0, 60.0)))
    t = np.arange(n) / sr
    x = sum(np.sin(2 * np.pi * k * mains * t + rng.uniform(0, 2 * np.pi)) / k for k in range(1, 5))
    return x + 0.01 * rng.standard_normal(n)


def _rub(n: int, rng: np.random.Generator, sr: int) -> np.ndarray:
    """Sparse Hann-windowed broadband bursts (20–200 ms), like a hand or
    clothing brushing the diaphragm, over a faint noise floor."""
    x = 1e-3 * rng.standard_normal(n)
    for _ in range(max(1, rng.poisson(1.5 * n / sr))):
        m = min(int(rng.uniform(0.02, 0.2) * sr), n)
        start = int(rng.integers(0, n - m + 1))
        burst = band_noise(m, 20.0, 1500.0, rng, sr) * np.hanning(m)
        x[start:start + m] += rng.lognormal(0.0, 0.5) * burst
    return x


def no_heart(
    kind: str, n: int, rng: np.random.Generator, *, rms: float = 0.05, sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """An n-sample float32 signal with no heart in it, scaled to `rms`
    (except "zeros", which stays exactly zero)."""
    if kind == "zeros":
        return np.zeros(n, dtype=np.float32)
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind == "pink":
        x = _coloured(n, 1.0, rng, sr)
    elif kind == "brown":
        x = _coloured(n, 2.0, rng, sr)
    elif kind == "hum":
        x = _hum(n, rng, sr)
    elif kind == "rub":
        x = _rub(n, rng, sr)
    else:
        raise ValueError(f"unknown kind {kind!r}; expected one of {NO_HEART_KINDS}")
    return (x * (rms / _rms(x))).astype(np.float32)
