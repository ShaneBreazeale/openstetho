"""Heuristic PCG segmentation: locate S1 / S2 boundaries from waveform.

Springer's HSMM segmentation is the academic standard; this is a lightweight
substitute used to place synthetic S3 events in plausible diastolic windows.
For real S3 detection on labeled data, swap for HSMM or learned segmentation.

Pipeline:
  1. Bandpass 25–150 Hz (where S1/S2 thumps dominate; suppresses murmur band).
  2. Hilbert envelope, downsampled to 200 Hz for cheap autocorrelation.
  3. Autocorrelation peak in (0.4, 1.5) s lag → cycle period (HR 40–150 bpm).
  4. Prominent envelope peaks with min spacing = 0.4 × cycle period.
  5. Alternate S1 / S2 by relative amplitude in pairs:
     S1 typically louder + sharper than S2 at apex; pick the pattern that
     gives more consistent S1→S2 → S1 intervals (systole shorter than diastole
     except at high HR).
  6. Diastolic window = [S2_n + 0.0 s, S1_{n+1} − 0.05 s].

Returns boundaries in *samples* at the input sample rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as sps

from .preprocess import SAMPLE_RATE


@dataclass(frozen=True)
class Cycle:
    s1_idx: int  # sample index of S1 peak
    s2_idx: int  # sample index of S2 peak
    next_s1_idx: int  # sample index of S1 in the following cycle

    @property
    def systole(self) -> tuple[int, int]:
        return (self.s1_idx, self.s2_idx)

    @property
    def diastole(self) -> tuple[int, int]:
        # Trim final 50 ms to keep S3 candidates clear of the next S1.
        margin = int(0.050 * SAMPLE_RATE)
        return (self.s2_idx, max(self.s2_idx + 1, self.next_s1_idx - margin))


@dataclass(frozen=True)
class Segmentation:
    cycles: list[Cycle]
    cycle_period_s: float
    confidence: float  # 0..1, based on autocorr peak prominence + spacing CV


def _bandpass(x: np.ndarray, sr: int, lo: float = 25.0, hi: float = 150.0) -> np.ndarray:
    nyq = sr / 2.0
    hi = min(hi, nyq * 0.99)
    sos = sps.butter(2, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sps.sosfiltfilt(sos, x).astype(np.float32, copy=False)


def _envelope(x: np.ndarray) -> np.ndarray:
    analytic = sps.hilbert(x)
    return np.abs(analytic).astype(np.float32)


def _estimate_period_s(envelope: np.ndarray, sr: int) -> tuple[float, float]:
    """Returns (period_seconds, autocorr_peak_prominence_0..1)."""
    e = envelope - envelope.mean()
    n = len(e)
    if n < int(2.0 * sr):
        return (0.0, 0.0)
    ac = np.correlate(e, e, mode="full")[n - 1 :]
    ac = ac / max(ac[0], 1e-9)
    lo_lag = int(0.40 * sr)  # 150 bpm
    hi_lag = int(1.50 * sr)  # 40 bpm
    if hi_lag >= len(ac):
        hi_lag = len(ac) - 1
    window = ac[lo_lag:hi_lag]
    if len(window) == 0:
        return (0.0, 0.0)
    peak_rel = int(np.argmax(window))
    peak_val = float(window[peak_rel])
    period_samples = lo_lag + peak_rel
    return (period_samples / sr, max(0.0, min(1.0, peak_val)))


def _find_peaks(envelope: np.ndarray, sr: int, period_s: float) -> np.ndarray:
    # Spacing must admit both S1→S2 (≥80 ms at highest HR) and S2→S1
    # (≥350 ms at lowest HR). Use a floor well below the shortest plausible
    # S1→S2 gap, with a soft tie to cycle period for very slow rhythms.
    min_distance = max(int(0.080 * sr), int(0.15 * period_s * sr))
    height = np.percentile(envelope, 75)
    peaks, _ = sps.find_peaks(envelope, distance=max(min_distance, 1), height=height)
    return peaks


def _label_s1_s2(peaks: np.ndarray, envelope: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split peaks into S1 / S2 by alternating with the systole-shorter-than-
    diastole heuristic. Returns (s1_indices, s2_indices) into the peaks array.
    """
    if len(peaks) < 4:
        return (np.array([], dtype=int), np.array([], dtype=int))
    intervals = np.diff(peaks)
    # Try both phasings (even-as-S1 vs odd-as-S1) and pick whichever yields a
    # tighter systole<diastole pattern.
    best_score = -np.inf
    best = (peaks[0::2], peaks[1::2])
    for offset in (0, 1):
        s1 = peaks[offset::2]
        s2 = peaks[1 - offset :: 2]
        # Align lengths.
        m = min(len(s1), len(s2))
        if m < 2:
            continue
        s1, s2 = s1[:m], s2[:m]
        if s1[0] > s2[0]:
            s2 = s2[1:]
            s1 = s1[: len(s2)]
        m = min(len(s1), len(s2))
        if m < 2:
            continue
        systole = (s2[:m] - s1[:m]).astype(float)
        diastole = (s1[1:m] - s2[: m - 1]).astype(float)
        if len(diastole) == 0:
            continue
        # Score: fraction of cycles where systole < diastole.
        score = float((systole[: len(diastole)] < diastole).mean())
        if score > best_score:
            best_score = score
            best = (s1, s2)
    return best


def segment_unified(audio: np.ndarray, method: str = "heuristic", sample_rate: int = SAMPLE_RATE) -> "Segmentation":
    """Dispatch to the requested segmenter. Returns a `Segmentation` either
    way so callers don't need to know which engine produced it.

    method ∈ {"heuristic", "hsmm"}.
    """
    if method == "heuristic":
        return segment(audio, sample_rate=sample_rate)
    if method == "hsmm":
        from .springer import cycles_from_hsmm, segment_hsmm

        hsmm = segment_hsmm(audio, sample_rate=sample_rate)
        cycles = [
            Cycle(s1_idx=s1, s2_idx=s2, next_s1_idx=ns1)
            for s1, s2, ns1 in cycles_from_hsmm(hsmm)
        ]
        # Confidence reuses the heuristic notion (peak-prominence × period-CV);
        # for HSMM we synthesize confidence from cycle count and period stability.
        if not cycles:
            return Segmentation(cycles=[], cycle_period_s=hsmm.cycle_period_s, confidence=0.0)
        periods = np.array([c.next_s1_idx - c.s1_idx for c in cycles], dtype=float)
        cv = float(periods.std() / max(periods.mean(), 1.0))
        confidence = float(max(0.0, 1.0 - cv))
        return Segmentation(cycles=cycles, cycle_period_s=hsmm.cycle_period_s, confidence=confidence)
    raise ValueError(f"unknown segmentation method: {method}")


def segment(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Segmentation:
    """Locate S1/S2 in a mono PCG waveform. Returns `Segmentation`.

    `confidence` is the geometric mean of (autocorr peak prominence,
    cycle-period coefficient-of-variation goodness). Use it to drop unreliable
    recordings from training.
    """
    if len(audio) < int(2.0 * sample_rate):
        return Segmentation(cycles=[], cycle_period_s=0.0, confidence=0.0)

    bp = _bandpass(audio, sample_rate)
    env = _envelope(bp)
    period_s, autocorr_q = _estimate_period_s(env, sample_rate)
    if period_s <= 0.0:
        return Segmentation(cycles=[], cycle_period_s=0.0, confidence=0.0)

    peaks = _find_peaks(env, sample_rate, period_s)
    s1s, s2s = _label_s1_s2(peaks, env)
    if len(s1s) < 2 or len(s2s) < 1:
        return Segmentation(cycles=[], cycle_period_s=period_s, confidence=0.0)

    cycles: list[Cycle] = []
    for i in range(len(s1s) - 1):
        s1 = int(s1s[i])
        next_s1 = int(s1s[i + 1])
        s2_candidates = s2s[(s2s > s1) & (s2s < next_s1)]
        if len(s2_candidates) == 0:
            continue
        s2 = int(s2_candidates[0])
        cycles.append(Cycle(s1_idx=s1, s2_idx=s2, next_s1_idx=next_s1))

    if not cycles:
        return Segmentation(cycles=[], cycle_period_s=period_s, confidence=0.0)

    periods = np.array([c.next_s1_idx - c.s1_idx for c in cycles], dtype=float)
    cv = float(periods.std() / max(periods.mean(), 1.0))
    cv_q = float(max(0.0, 1.0 - cv))
    confidence = float(np.sqrt(autocorr_q * cv_q))

    return Segmentation(cycles=cycles, cycle_period_s=period_s, confidence=confidence)
