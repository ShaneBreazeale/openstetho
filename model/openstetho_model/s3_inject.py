"""Inject synthetic S3 events into real PCG audio for augmentation.

Given a segmented waveform, decide per-cycle whether to add an S3 (drawn from
`s3_synth.synth_s3_random`), place it in the early-diastolic window with
amplitude calibrated to a target SNR against the local diastolic noise floor,
and return per-cycle labels for downstream cycle-level training.

The injection point is **after** any pre-existing diastolic content; we do not
attempt to remove a real S3 if it happens to be there. For CirCor / PhysioNet
2016, real S3 prevalence is unknown but low — treat the injected label as
"likely S3" rather than "ground truth". See [[s3-annotation-protocol]].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .s3_synth import S3Params, synth_s3_random
from .s4_synth import S4Params, synth_s4_random
from .segment import Cycle, Segmentation

SAMPLE_RATE = 4000

# Early-diastolic onset window for S3 relative to S2 peak (seconds).
ONSET_MIN_S = 0.100
ONSET_MAX_S = 0.200

# SNR range targets the gallop's amplitude relative to diastolic noise RMS.
# Lower end models faint pathologic-but-quiet S3; upper end models clear S3.
SNR_DB_MIN = -3.0
SNR_DB_MAX = 15.0


@dataclass(frozen=True)
class InjectedCycle:
    cycle: Cycle
    positive: bool
    params: S3Params | None
    onset_idx: int | None
    snr_db: float | None


def s3_inject(
    audio: np.ndarray,
    segmentation: Segmentation,
    rng: np.random.Generator,
    prob_per_cycle: float = 0.5,
    snr_db_range: tuple[float, float] = (SNR_DB_MIN, SNR_DB_MAX),
    sample_rate: int = SAMPLE_RATE,
    prob_multi: float = 0.0,
    multi_min_gap_s: float = 0.080,
) -> tuple[np.ndarray, list[InjectedCycle]]:
    """Return `(audio_with_s3, per_cycle_records)`.

    `audio` is not mutated. Each cycle in `segmentation` produces one record
    (positive or negative). Negative cycles are untouched; positive cycles get
    one synthetic S3 placed at `S2 + uniform(ONSET_MIN_S, ONSET_MAX_S)` with
    amplitude calibrated so the S3 RMS equals `noise_rms * 10**(snr_db/20)`,
    where `noise_rms` is measured on the diastolic window of *this* cycle.

    Multi-event support: with probability `prob_multi`, a positive cycle gets
    a *second* S3 placed at least `multi_min_gap_s` after the first (still
    inside the same diastolic window). This models split-S3 and persistent
    gallop scenarios. The cycle is still labeled positive.

    If the diastolic window is too short to fit the chosen S3 duration the
    waveform is truncated; if truncation leaves <20 % of the original samples
    the cycle is recorded as negative (label fidelity over augmentation rate).
    """
    out = audio.astype(np.float32, copy=True)
    records: list[InjectedCycle] = []

    for cycle in segmentation.cycles:
        positive = rng.random() < prob_per_cycle
        if not positive:
            records.append(InjectedCycle(cycle, False, None, None, None))
            continue

        d0, d1 = cycle.diastole
        if d1 - d0 < int(0.080 * sample_rate):
            records.append(InjectedCycle(cycle, False, None, None, None))
            continue

        noise_seg = audio[d0:d1]
        noise_rms = float(np.sqrt(np.mean(noise_seg.astype(np.float64) ** 2) + 1e-12))
        if noise_rms < 1e-8:
            # Silent diastole — no usable SNR target. Skip.
            records.append(InjectedCycle(cycle, False, None, None, None))
            continue

        wave, params = synth_s3_random(rng, sample_rate=sample_rate)
        if len(wave) == 0:
            records.append(InjectedCycle(cycle, False, None, None, None))
            continue
        s3_rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2) + 1e-12))

        snr_db = float(rng.uniform(*snr_db_range))
        target_rms = noise_rms * 10.0 ** (snr_db / 20.0)
        wave = (wave * (target_rms / s3_rms)).astype(np.float32)

        onset_offset = int(rng.uniform(ONSET_MIN_S, ONSET_MAX_S) * sample_rate)
        start = cycle.s2_idx + onset_offset
        end = start + len(wave)

        if start >= d1:
            records.append(InjectedCycle(cycle, False, None, None, None))
            continue
        if end > d1:
            keep = d1 - start
            if keep < int(0.20 * len(wave)):
                records.append(InjectedCycle(cycle, False, None, None, None))
                continue
            wave = wave[:keep]
            end = start + len(wave)

        out[start:end] += wave
        records.append(
            InjectedCycle(
                cycle=cycle,
                positive=True,
                params=params,
                onset_idx=start,
                snr_db=snr_db,
            )
        )

        # Optional second S3 in the same diastolic window for multi-gallop
        # augmentation. Placed strictly after the first event, with at least
        # `multi_min_gap_s` of separation.
        if prob_multi > 0.0 and rng.random() < prob_multi:
            second_wave, second_params = synth_s3_random(rng, sample_rate=sample_rate)
            if len(second_wave) == 0:
                continue
            second_rms = float(np.sqrt(np.mean(second_wave.astype(np.float64) ** 2) + 1e-12))
            second_snr_db = float(rng.uniform(*snr_db_range))
            target_rms2 = noise_rms * 10.0 ** (second_snr_db / 20.0)
            second_wave = (second_wave * (target_rms2 / second_rms)).astype(np.float32)

            min_start = end + int(multi_min_gap_s * sample_rate)
            if min_start >= d1:
                continue
            second_start = int(rng.integers(min_start, d1))
            second_end = second_start + len(second_wave)
            if second_end > d1:
                keep = d1 - second_start
                if keep < int(0.20 * len(second_wave)):
                    continue
                second_wave = second_wave[:keep]
                second_end = second_start + len(second_wave)
            out[second_start:second_end] += second_wave
            # The previous record stays as the canonical entry for this cycle;
            # the second event is implicit (still label 1). If callers need
            # per-event detail they should rebuild this API.

    return out, records


# ─── S4 confounder injection (negatives only) ───────────────────────────────

# S4 onset is *late* diastolic — immediately before the next S1.
S4_PRE_S1_MIN_S = 0.060
S4_PRE_S1_MAX_S = 0.120


def s4_inject(
    audio: np.ndarray,
    segmentation: Segmentation,
    rng: np.random.Generator,
    prob_per_cycle: float = 0.0,
    snr_db_range: tuple[float, float] = (SNR_DB_MIN, SNR_DB_MAX),
    sample_rate: int = SAMPLE_RATE,
    return_flags: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[bool]]:
    """Plant late-diastolic S4 events into a subset of cycles.

    Returns a new audio array; the per-cycle label remains 0. The
    detector should learn to reject these as "not S3" because of timing —
    S4 sits within ~60–120 ms before the next S1, whereas S3 sits within
    100–200 ms after S2.

    Used for negative mining only — call separately from `s3_inject`.
    """
    flags: list[bool] = []
    if prob_per_cycle <= 0.0:
        out = audio.astype(np.float32, copy=False)
        if return_flags:
            return out, [False] * len(segmentation.cycles)
        return out

    out = audio.astype(np.float32, copy=True)
    for cycle in segmentation.cycles:
        if rng.random() >= prob_per_cycle:
            flags.append(False)
            continue

        # S4 placement is anchored to the *next* S1, not S2. Use cycle.next_s1_idx.
        pre = int(rng.uniform(S4_PRE_S1_MIN_S, S4_PRE_S1_MAX_S) * sample_rate)
        end = cycle.next_s1_idx - 1
        start = cycle.next_s1_idx - pre
        if start <= cycle.s2_idx + int(0.020 * sample_rate):
            # Diastole too short — skip rather than land on S2.
            flags.append(False)
            continue

        d0, d1 = cycle.diastole
        noise_seg = audio[d0:d1]
        if len(noise_seg) < int(0.080 * sample_rate):
            flags.append(False)
            continue
        noise_rms = float(np.sqrt(np.mean(noise_seg.astype(np.float64) ** 2) + 1e-12))
        if noise_rms < 1e-8:
            flags.append(False)
            continue

        wave, _ = synth_s4_random(rng, sample_rate=sample_rate)
        if len(wave) == 0:
            flags.append(False)
            continue
        s4_rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2) + 1e-12))

        snr_db = float(rng.uniform(*snr_db_range))
        target_rms = noise_rms * 10.0 ** (snr_db / 20.0)
        wave = (wave * (target_rms / s4_rms)).astype(np.float32)

        wave_end = start + len(wave)
        if wave_end > end:
            keep = end - start
            if keep < int(0.20 * len(wave)):
                flags.append(False)
                continue
            wave = wave[:keep]
            wave_end = start + len(wave)
        out[start:wave_end] += wave
        flags.append(True)

    if return_flags:
        return out, flags
    return out


# ─── Hard-negative confounders ──────────────────────────────────────────────
#
# These plant heart-sound events that overlap S3 in spectral content but
# differ in *timing* and *morphology*. Cycles with these injections stay
# labeled 0 (non-S3) for the classifier; they exist to push the model away
# from learning "any low-frequency diastolic blip = S3".

# Split-S2 — two close peaks at S2 timing (A2 then P2, ~30 ms apart).
SPLIT_S2_GAP_MIN_S = 0.020
SPLIT_S2_GAP_MAX_S = 0.060

# Opening snap — post-S2 high-frequency click, ~80 ms after S2.
OS_OFFSET_MIN_S = 0.060
OS_OFFSET_MAX_S = 0.120
OS_F0_MIN_HZ = 100.0
OS_F0_MAX_HZ = 200.0

# Ejection click — post-S1 click, ~50 ms after S1 (systolic).
EJ_OFFSET_MIN_S = 0.030
EJ_OFFSET_MAX_S = 0.080
EJ_F0_MIN_HZ = 120.0
EJ_F0_MAX_HZ = 220.0


def _damped_burst(rng: np.random.Generator, f0: float, duration_s: float, sample_rate: int) -> np.ndarray:
    n = int(round(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float32) / sample_rate
    tau = duration_s * 0.4
    return (np.exp(-t / tau) * np.sin(2.0 * np.pi * f0 * t)).astype(np.float32)


def _calibrated_inject(
    out: np.ndarray,
    audio_for_rms: np.ndarray,
    rng: np.random.Generator,
    wave: np.ndarray,
    start: int,
    end_bound: int,
    snr_db_range: tuple[float, float],
) -> bool:
    """Helper: scale `wave` to a calibrated SNR vs local audio RMS and add
    it into `out` starting at `start` (truncated to `end_bound`). Returns
    True if the write actually happened.
    """
    if start < 0 or end_bound <= start or len(wave) == 0:
        return False
    lo = max(0, start - 100)
    hi = min(len(audio_for_rms), end_bound + 100)
    if hi - lo < 32:
        return False
    rms = float(np.sqrt(np.mean(audio_for_rms[lo:hi].astype(np.float64) ** 2) + 1e-12))
    if rms < 1e-8:
        return False
    snr_db = float(rng.uniform(*snr_db_range))
    target_rms = rms * 10.0 ** (snr_db / 20.0)
    wave_rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2) + 1e-12))
    wave = (wave * (target_rms / wave_rms)).astype(np.float32)
    wave_end = start + len(wave)
    if wave_end > end_bound:
        keep = end_bound - start
        if keep < int(0.20 * len(wave)):
            return False
        wave = wave[:keep]
        wave_end = start + len(wave)
    out[start:wave_end] += wave
    return True


def split_s2_inject(
    audio: np.ndarray,
    segmentation: Segmentation,
    rng: np.random.Generator,
    prob_per_cycle: float = 0.0,
    snr_db_range: tuple[float, float] = (SNR_DB_MIN, SNR_DB_MAX),
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Add a second S2 component (P2) shortly after the existing S2 to
    simulate a split S2 ("fixed split" pattern). Keeps the cycle label 0.
    """
    if prob_per_cycle <= 0.0:
        return audio
    out = audio.astype(np.float32, copy=True)
    for cycle in segmentation.cycles:
        if rng.random() >= prob_per_cycle:
            continue
        gap_s = float(rng.uniform(SPLIT_S2_GAP_MIN_S, SPLIT_S2_GAP_MAX_S))
        gap = int(gap_s * sample_rate)
        start = cycle.s2_idx + gap
        wave = _damped_burst(rng, f0=80.0 + float(rng.uniform(0, 20)), duration_s=0.060, sample_rate=sample_rate)
        end_bound = cycle.next_s1_idx - int(0.020 * sample_rate)
        _calibrated_inject(out, audio, rng, wave, start, end_bound, snr_db_range)
    return out


def opening_snap_inject(
    audio: np.ndarray,
    segmentation: Segmentation,
    rng: np.random.Generator,
    prob_per_cycle: float = 0.0,
    snr_db_range: tuple[float, float] = (SNR_DB_MIN, SNR_DB_MAX),
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Add a high-frequency click ~80 ms after S2 (mitral opening snap).
    Higher F0 than S3 — should not fool a timing/freq-aware model but is a
    realistic confounder for spectral-only classifiers.
    """
    if prob_per_cycle <= 0.0:
        return audio
    out = audio.astype(np.float32, copy=True)
    for cycle in segmentation.cycles:
        if rng.random() >= prob_per_cycle:
            continue
        offset = int(float(rng.uniform(OS_OFFSET_MIN_S, OS_OFFSET_MAX_S)) * sample_rate)
        start = cycle.s2_idx + offset
        wave = _damped_burst(rng, f0=float(rng.uniform(OS_F0_MIN_HZ, OS_F0_MAX_HZ)), duration_s=0.040, sample_rate=sample_rate)
        end_bound = cycle.next_s1_idx - int(0.020 * sample_rate)
        _calibrated_inject(out, audio, rng, wave, start, end_bound, snr_db_range)
    return out


def ejection_click_inject(
    audio: np.ndarray,
    segmentation: Segmentation,
    rng: np.random.Generator,
    prob_per_cycle: float = 0.0,
    snr_db_range: tuple[float, float] = (SNR_DB_MIN, SNR_DB_MAX),
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Add a high-frequency click ~50 ms after S1 (systolic ejection click).
    Sits in systole, so it should never be confused with S3 — but it must
    appear in training so the model learns to ignore systolic transients.
    """
    if prob_per_cycle <= 0.0:
        return audio
    out = audio.astype(np.float32, copy=True)
    for cycle in segmentation.cycles:
        if rng.random() >= prob_per_cycle:
            continue
        offset = int(float(rng.uniform(EJ_OFFSET_MIN_S, EJ_OFFSET_MAX_S)) * sample_rate)
        start = cycle.s1_idx + offset
        wave = _damped_burst(rng, f0=float(rng.uniform(EJ_F0_MIN_HZ, EJ_F0_MAX_HZ)), duration_s=0.030, sample_rate=sample_rate)
        end_bound = cycle.s2_idx - int(0.020 * sample_rate)
        _calibrated_inject(out, audio, rng, wave, start, end_bound, snr_db_range)
    return out
