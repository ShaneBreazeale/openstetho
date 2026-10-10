"""Channel simulation: what happens to a heart sound between the chest and
the model on real hardware.

* `random_eq_curve` / `apply_eq` -- a random smooth magnitude response,
  standing in for the unknown frequency response of a given stethoscope,
  diaphragm, coupling pressure or app-side filter. Training through many of
  them is the clean-room way to get device robustness without fitting to any
  one device.
* `adpcm_encode` / `adpcm_decode` / `adpcm_round_trip` -- the IMA ADPCM codec
  on the BLE link. The decoder is a line-for-line port of
  `stetho-core/src/codec/adpcm.rs` and is pinned to it bit for bit by a
  fixture the Rust crate writes (`examples/dump_adpcm_parity.rs`). The step
  and index tables are the public IMA/DVI standard.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from .preprocess import SAMPLE_RATE

# ─── random smooth EQ ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class EqCurve:
    """Magnitude response given at control frequencies (Hz, increasing) in
    dB. Between control points the gain is interpolated monotonically (PCHIP)
    on a log-frequency axis, so it never rings past the control gains; beyond
    the end points it stays flat."""

    freqs_hz: np.ndarray
    gains_db: np.ndarray

    def gain_db(self, f: np.ndarray) -> np.ndarray:
        f = np.clip(np.asarray(f, dtype=np.float64), self.freqs_hz[0], self.freqs_hz[-1])
        return PchipInterpolator(np.log(self.freqs_hz), self.gains_db)(np.log(f))


def random_eq_curve(
    rng: np.random.Generator,
    *,
    n_points: int = 8,
    f_lo: float = 20.0,
    f_hi: float = 2000.0,
    sigma_db: float = 6.0,
    max_db: float = 12.0,
) -> EqCurve:
    """Control points evenly spaced in log frequency (about one per octave
    with the defaults), gains drawn from N(0, sigma_db) and clipped to
    ±max_db."""
    freqs = np.geomspace(f_lo, f_hi, n_points)
    gains = np.clip(rng.normal(0.0, sigma_db, n_points), -max_db, max_db)
    return EqCurve(freqs, gains)


def apply_eq(audio: np.ndarray, curve: EqCurve, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Zero-phase filter `audio` by `curve`: scale each rFFT bin by the
    curve's gain at that frequency. Real, length- and dtype-preserving."""
    x = np.asarray(audio, dtype=np.float64)
    spec = np.fft.rfft(x)
    gain = 10.0 ** (curve.gain_db(np.fft.rfftfreq(len(x), 1.0 / sr)) / 20.0)
    return np.fft.irfft(spec * gain, n=len(x)).astype(np.asarray(audio).dtype, copy=False)


# ─── IMA ADPCM ──────────────────────────────────────────────────────────────

_STEP_TABLE = (
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
    2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
)
_INDEX_TABLE = (-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8)


def _decode_nibble(n: int, predictor: int, index: int) -> tuple[int, int]:
    """One step of `AdpcmDecoder::decode_nibble` in stetho-core."""
    step = _STEP_TABLE[index]
    diff = step >> 3
    if n & 4:
        diff += step
    if n & 2:
        diff += step >> 1
    if n & 1:
        diff += step >> 2
    predictor = predictor - diff if n & 8 else predictor + diff
    predictor = min(max(predictor, -32768), 32767)
    index = min(max(index + _INDEX_TABLE[n], 0), 88)
    return predictor, index


def adpcm_decode(nibbles: np.ndarray) -> np.ndarray:
    """Decode 4-bit codes to int16 PCM from a cold decoder (predictor and
    step index both 0, as after a codec reset)."""
    out = np.empty(len(nibbles), dtype=np.int16)
    predictor = index = 0
    for i, n in enumerate(np.asarray(nibbles, dtype=np.uint8).tolist()):
        predictor, index = _decode_nibble(n & 0x0F, predictor, index)
        out[i] = predictor
    return out


def adpcm_encode(pcm: np.ndarray) -> np.ndarray:
    """Standard IMA ADPCM encoder. Picks each code by successive
    approximation against step, step/2, step/4, then advances its state with
    the decoder itself, so encoder and decoder never drift apart."""
    out = np.empty(len(pcm), dtype=np.uint8)
    predictor = index = 0
    for i, sample in enumerate(np.asarray(pcm, dtype=np.int16).tolist()):
        step = _STEP_TABLE[index]
        diff = sample - predictor
        n = 0
        if diff < 0:
            n, diff = 8, -diff
        if diff >= step:
            n |= 4
            diff -= step
        if diff >= step >> 1:
            n |= 2
            diff -= step >> 1
        if diff >= step >> 2:
            n |= 1
        out[i] = n
        predictor, index = _decode_nibble(n, predictor, index)
    return out


def adpcm_round_trip(audio: np.ndarray, *, peak_dbfs: float = -12.0) -> np.ndarray:
    """Pass float audio through the BLE codec: scale so its peak sits at
    `peak_dbfs` of int16 full scale, encode, decode, and scale back. The
    level matters: IMA ADPCM's noise floor is fixed in int16 units, so quiet
    input loses more. Returns float32 at the input's level."""
    x = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak == 0.0:
        return np.zeros(len(x), dtype=np.float32)
    scale = 32767.0 * 10.0 ** (peak_dbfs / 20.0) / peak
    pcm = np.clip(np.round(x * scale), -32768, 32767).astype(np.int16)
    return (adpcm_decode(adpcm_encode(pcm)) / scale).astype(np.float32)


def apply_channel(
    audio: np.ndarray,
    rng: np.random.Generator,
    *,
    eq_prob: float,
    adpcm_prob: float,
    adpcm_peak_dbfs: tuple[float, float] = (-30.0, -6.0),
) -> np.ndarray:
    """One random channel draw: a smooth EQ with probability `eq_prob`, then
    the ADPCM codec with probability `adpcm_prob` at a random input level."""
    x = audio
    if rng.random() < eq_prob:
        x = apply_eq(x, random_eq_curve(rng))
    if rng.random() < adpcm_prob:
        x = adpcm_round_trip(x, peak_dbfs=float(rng.uniform(*adpcm_peak_dbfs)))
    return np.asarray(x, dtype=np.float32)
