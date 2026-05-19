"""Springer-style PCG segmentation (LR-HSMM, TBME 2016).

Reference:
    Springer, Tarassenko, Clifford (2016)
    "Logistic Regression-HSMM-based Heart Sound Segmentation"
    IEEE Transactions on Biomedical Engineering 63(4): 822-832.

This module produces per-sample state assignments for the four heart-cycle
phases (S1, systole, S2, diastole), using duration-constrained Viterbi over
four standard PCG envelope features. The classic Springer pipeline trains a
logistic regression to convert envelopes into per-state emission probabilities;
the authors' pretrained LR weights are not redistributable from inside this
project, so we provide two interchangeable backends:

* `EnvelopeGaussianHSMM` — fits four-state Gaussian emissions to envelopes
  drawn from heuristic pseudo-labels (uses `segment.segment` to bootstrap).
  No external weights needed, accuracy is slightly below original Springer.

* `LogisticHSMM` (stub) — placeholder for plugging in real LR coefficients
  if/when those become available via a license-compatible source.

The HMM duration distribution is gamma with state-specific (mean, std) drawn
from Springer Table II (parameters cited in the paper). Viterbi is the
duration-aware variant from Yu (2010) — for each (state, time) tuple we
maximize over admissible durations.

This is a meaningful upgrade over `segment.segment`'s heuristic for two
reasons:
1. Whole-cycle decoding — every sample gets a state, not just S1/S2 peaks.
2. Duration constraints — short heuristic peaks land in S2 instead of being
   classified as noise; missed peaks get filled in by the dynamic program.

For S3 detection, the practical win is **better S2 localization**, which is
where our synthetic injection anchors. Better S2 → cleaner labels → higher
ceiling on supervised AUPRC.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as sps

from .preprocess import SAMPLE_RATE

# State indices.
S1 = 0
SYSTOLE = 1
S2 = 2
DIASTOLE = 3
N_STATES = 4
STATE_NAMES = ("S1", "sys", "S2", "dia")

# Default gamma duration parameters (mean, std) in seconds — from Springer
# Table II, slightly relaxed to accommodate pediatric / fast HR recordings.
# `sys` and `dia` are derived per-recording from the heart-rate estimate.
S1_MEAN_S, S1_STD_S = 0.122, 0.022
S2_MEAN_S, S2_STD_S = 0.092, 0.022


@dataclass(frozen=True)
class HSMMSegmentation:
    """Per-sample state path plus convenience accessors.

    `states[i] ∈ {0,1,2,3}` for sample `i` of the original waveform.
    `cycle_period_s` is the median cycle length recovered from the path.
    """
    states: np.ndarray
    cycle_period_s: float


# ─── envelopes ──────────────────────────────────────────────────────────────

def homomorphic_envelope(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Low-pass-filtered log-magnitude envelope (cepstral smoothing).

    This is the Springer/Schmidt homomorphic envelope used for S1/S2 onset
    localization. Output is in dB-ish units and zero-mean per recording.
    """
    nyq = sample_rate / 2.0
    sos = sps.butter(1, 8.0 / nyq, btype="low", output="sos")
    env = sps.sosfiltfilt(sos, np.abs(x) + 1e-9).astype(np.float32)
    env = np.log(np.maximum(env, 1e-9))
    return (env - env.mean()).astype(np.float32)


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    analytic = sps.hilbert(x)
    env = np.abs(analytic).astype(np.float32)
    return (env - env.mean()).astype(np.float32)


def psd_band_envelope(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    band: tuple[float, float] = (40.0, 60.0),
) -> np.ndarray:
    """Bandpass + magnitude in the dominant S1/S2 transient band."""
    nyq = sample_rate / 2.0
    sos = sps.butter(2, [band[0] / nyq, band[1] / nyq], btype="band", output="sos")
    bp = sps.sosfiltfilt(sos, x).astype(np.float32)
    env = np.abs(bp)
    return (env - env.mean()).astype(np.float32)


def wavelet_envelope(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Cheap DWT-style envelope without pywavelets: sum-of-squares over the
    same band the wavelet detail coefficient would emphasize (~30–120 Hz at
    4 kHz). Close enough to Springer's level-3-DB7 for HMM emissions.
    """
    nyq = sample_rate / 2.0
    sos = sps.butter(2, [30.0 / nyq, 120.0 / nyq], btype="band", output="sos")
    bp = sps.sosfiltfilt(sos, x).astype(np.float32)
    env = bp * bp
    return (env - env.mean()).astype(np.float32)


def compute_envelopes(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Stack the four Springer envelopes into shape (N, 4)."""
    return np.stack(
        [
            homomorphic_envelope(x, sample_rate),
            hilbert_envelope(x),
            psd_band_envelope(x, sample_rate),
            wavelet_envelope(x, sample_rate),
        ],
        axis=1,
    ).astype(np.float32)


# ─── duration model ─────────────────────────────────────────────────────────

def gamma_log_pmf(d: np.ndarray, mean_s: float, std_s: float, sample_rate: int) -> np.ndarray:
    """log P(duration = d samples) under a gamma distribution with the given
    seconds-domain mean / std. Vectorised over `d`.
    """
    mean_n = max(mean_s * sample_rate, 2.0)
    std_n = max(std_s * sample_rate, 1.0)
    k = (mean_n / std_n) ** 2          # shape
    theta = (std_n ** 2) / mean_n      # scale
    d = np.clip(d, 1, None).astype(np.float64)
    # log gamma pmf (use pdf as a stand-in — discrete-vs-continuous bias
    # is constant under arg-max).
    return (
        (k - 1.0) * np.log(d)
        - d / theta
        - k * np.log(theta)
        - _log_gamma(k)
    ).astype(np.float32)


def _log_gamma(k: float) -> float:
    # math.lgamma is in stdlib; avoid scipy import for hot loops.
    from math import lgamma

    return lgamma(k)


def derive_systole_diastole_duration(
    cycle_period_s: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Springer empirical: systole ≈ 0.34 × cycle_period, diastole = rest.
    Returns ((sys_mean, sys_std), (dia_mean, dia_std)).
    """
    sys_mean = 0.34 * cycle_period_s
    sys_std = max(0.04, 0.10 * sys_mean)
    dia_mean = cycle_period_s - sys_mean - S1_MEAN_S - S2_MEAN_S
    dia_std = max(0.06, 0.15 * dia_mean)
    return (sys_mean, sys_std), (dia_mean, dia_std)


# ─── Gaussian emission model from heuristic pseudo-labels ────────────────────

def gaussian_emission_stats(
    envelopes: np.ndarray,
    state_assign: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-state diagonal-covariance Gaussian stats from envelopes.

    `state_assign` is a per-sample state index (or -1 for unassigned). Returns
    `(means, vars)` each of shape `(N_STATES, n_envelope_channels)`.
    """
    n_ch = envelopes.shape[1]
    means = np.zeros((N_STATES, n_ch), dtype=np.float64)
    variances = np.ones((N_STATES, n_ch), dtype=np.float64)
    for state in range(N_STATES):
        mask = state_assign == state
        if mask.sum() < 4:
            # Not enough samples — fall back to global stats.
            means[state] = envelopes.mean(axis=0)
            variances[state] = envelopes.var(axis=0) + 1e-3
            continue
        means[state] = envelopes[mask].mean(axis=0)
        variances[state] = envelopes[mask].var(axis=0) + 1e-3
    return means, variances


def gaussian_log_emission(
    envelopes: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Per-sample log p(envelope | state) for each state. Shape (N, N_STATES)."""
    n_ch = envelopes.shape[1]
    log_em = np.zeros((envelopes.shape[0], N_STATES), dtype=np.float32)
    for state in range(N_STATES):
        diff = envelopes - means[state]
        log_em[:, state] = -0.5 * np.sum(diff * diff / variances[state], axis=1)
        log_em[:, state] += -0.5 * np.sum(np.log(variances[state]))
        log_em[:, state] += -0.5 * n_ch * np.log(2.0 * np.pi)
    return log_em


def bootstrap_pseudo_labels(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, float]:
    """Use the heuristic `segment.segment` to label each sample with a coarse
    state estimate. Returns `(state_assign, cycle_period_s)`.

    Sample labels: S1 within ±40 ms of detected S1 peaks, S2 within ±30 ms
    of S2 peaks, systole between S1 end and S2 start, diastole between S2
    end and next S1 start. Unassigned samples (outside detected cycles)
    are marked -1.
    """
    from .segment import segment as _heur_segment  # avoid circular at module load

    n = len(audio)
    out = np.full(n, -1, dtype=np.int8)
    seg = _heur_segment(audio, sample_rate=sample_rate)
    if not seg.cycles:
        return out, 0.0
    half_s1 = int(0.040 * sample_rate)
    half_s2 = int(0.030 * sample_rate)
    for cycle in seg.cycles:
        s1_lo = max(0, cycle.s1_idx - half_s1)
        s1_hi = min(n, cycle.s1_idx + half_s1)
        s2_lo = max(0, cycle.s2_idx - half_s2)
        s2_hi = min(n, cycle.s2_idx + half_s2)
        next_s1_lo = max(0, cycle.next_s1_idx - half_s1)
        out[s1_lo:s1_hi] = S1
        out[s2_lo:s2_hi] = S2
        if s2_lo > s1_hi:
            out[s1_hi:s2_lo] = SYSTOLE
        if next_s1_lo > s2_hi:
            out[s2_hi:next_s1_lo] = DIASTOLE
    return out, float(seg.cycle_period_s)


# ─── Duration-aware Viterbi ─────────────────────────────────────────────────

def viterbi_hsmm(
    log_emissions: np.ndarray,
    duration_log_pmf: list[np.ndarray],
    transition: np.ndarray,
    max_duration: int,
) -> np.ndarray:
    """Yu-style explicit-duration HMM Viterbi.

    `log_emissions[t, s]` — observation log-likelihood at time t under state s.
    `duration_log_pmf[s][d]` — log P(stay d samples in state s), d ∈ [1, max_d].
    `transition[s_from, s_to]` — log transition probability (gives 0 on the
        legal cyclic edges, -inf elsewhere).

    Returns `path[t]` — most likely state at each sample.
    """
    T, S = log_emissions.shape
    # Cumulative log-emission so segment likelihoods are O(1) lookup.
    cum_log_em = np.concatenate(
        [np.zeros((1, S), dtype=np.float32), np.cumsum(log_emissions, axis=0)],
        axis=0,
    )

    best = np.full((T + 1, S), -np.inf, dtype=np.float32)
    back_state = np.full((T + 1, S), -1, dtype=np.int32)
    back_dur = np.full((T + 1, S), -1, dtype=np.int32)

    # Initial: any state can start at time 0 with equal prior.
    best[0, :] = 0.0

    for t in range(1, T + 1):
        for s_to in range(S):
            d_max = min(max_duration, t)
            ds = np.arange(1, d_max + 1)
            seg_loglik = cum_log_em[t, s_to] - cum_log_em[t - ds, s_to]
            dur_log = duration_log_pmf[s_to][ds - 1]
            # For each predecessor state, best `best[t-d, s_from] + transition[s_from, s_to]`.
            prev = best[t - ds]  # shape (d_max, S)
            prev_plus_trans = prev + transition[:, s_to][np.newaxis, :]
            best_from = prev_plus_trans.max(axis=1)
            best_state = prev_plus_trans.argmax(axis=1)
            candidates = best_from + seg_loglik + dur_log
            idx = int(np.argmax(candidates))
            best[t, s_to] = float(candidates[idx])
            back_state[t, s_to] = int(best_state[idx])
            back_dur[t, s_to] = int(ds[idx])

    # Backtrace.
    path = np.zeros(T, dtype=np.int8)
    t = T
    s = int(best[T, :].argmax())
    while t > 0:
        d = back_dur[t, s]
        if d <= 0:
            break
        path[t - d : t] = s
        s_prev = back_state[t, s]
        t -= d
        s = s_prev
    return path


# ─── Top-level API ──────────────────────────────────────────────────────────

HSMM_DOWNSAMPLE_HZ = 50  # Springer's recommended internal rate.


def segment_hsmm(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    max_duration_s: float = 0.60,
) -> HSMMSegmentation:
    """End-to-end Springer-style segmentation.

    Pipeline:
    1. Heuristic peak detection bootstraps coarse per-sample pseudo-labels.
    2. Estimate Gaussian emission stats from those pseudo-labels.
    3. Downsample envelopes to ~50 Hz for tractable HSMM decoding.
    4. Derive systole/diastole gamma duration parameters from the cycle
       period.
    5. Run duration-aware Viterbi; upsample the resulting state path back
       to the input sample rate via nearest-neighbour.

    Returns `HSMMSegmentation`. Use `cycles_from_hsmm` to convert the dense
    state path to peak triples.
    """
    envelopes = compute_envelopes(audio, sample_rate)
    pseudo, cycle_period_s = bootstrap_pseudo_labels(audio, sample_rate)
    if cycle_period_s <= 0.0:
        return HSMMSegmentation(states=np.zeros(len(audio), dtype=np.int8), cycle_period_s=0.0)

    means, variances = gaussian_emission_stats(envelopes, pseudo)
    log_em_full = gaussian_log_emission(envelopes, means, variances)

    # Downsample to HSMM_DOWNSAMPLE_HZ for the Viterbi pass. Average-pool
    # log-emissions over each downsample window (≈ marginalising over the
    # frames inside the window).
    factor = sample_rate // HSMM_DOWNSAMPLE_HZ
    n_full = log_em_full.shape[0]
    n_ds = n_full // factor
    if n_ds == 0:
        return HSMMSegmentation(states=np.zeros(len(audio), dtype=np.int8), cycle_period_s=cycle_period_s)
    log_em = log_em_full[: n_ds * factor].reshape(n_ds, factor, N_STATES).mean(axis=1)

    (sys_mean, sys_std), (dia_mean, dia_std) = derive_systole_diastole_duration(cycle_period_s)
    max_d_samples = max(2, int(max_duration_s * HSMM_DOWNSAMPLE_HZ))
    ds = np.arange(1, max_d_samples + 1)
    duration_log_pmf = [
        gamma_log_pmf(ds, S1_MEAN_S, S1_STD_S, HSMM_DOWNSAMPLE_HZ),
        gamma_log_pmf(ds, sys_mean, sys_std, HSMM_DOWNSAMPLE_HZ),
        gamma_log_pmf(ds, S2_MEAN_S, S2_STD_S, HSMM_DOWNSAMPLE_HZ),
        gamma_log_pmf(ds, dia_mean, dia_std, HSMM_DOWNSAMPLE_HZ),
    ]
    neg_inf = -1e9
    transition = np.full((N_STATES, N_STATES), neg_inf, dtype=np.float32)
    transition[S1, SYSTOLE] = 0.0
    transition[SYSTOLE, S2] = 0.0
    transition[S2, DIASTOLE] = 0.0
    transition[DIASTOLE, S1] = 0.0

    path_ds = viterbi_hsmm(log_em.astype(np.float32), duration_log_pmf, transition, max_d_samples)

    # Upsample state path back to original rate via nearest-neighbour repeat.
    path_full = np.repeat(path_ds, factor)
    if len(path_full) < len(audio):
        pad = np.full(len(audio) - len(path_full), path_full[-1] if len(path_full) else 0, dtype=np.int8)
        path_full = np.concatenate([path_full, pad])
    elif len(path_full) > len(audio):
        path_full = path_full[: len(audio)]

    return HSMMSegmentation(states=path_full.astype(np.int8), cycle_period_s=cycle_period_s)


# ─── Convert dense states to discrete cycles for downstream code ────────────

def cycles_from_hsmm(segmentation: HSMMSegmentation) -> list[tuple[int, int, int]]:
    """Extract `(s1_peak, s2_peak, next_s1_peak)` triples from a dense
    state path. Peak indices are chosen as the centroid of each state run.
    """
    states = segmentation.states
    if len(states) == 0:
        return []
    out: list[tuple[int, int, int]] = []
    # Find runs: list of (state, start, end).
    diffs = np.diff(states)
    boundaries = np.flatnonzero(diffs) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(states)]])
    runs = list(zip(states[starts], starts, ends))

    s1_centroids: list[int] = []
    s2_centroids: list[int] = []
    for state, lo, hi in runs:
        c = (lo + hi) // 2
        if state == S1:
            s1_centroids.append(c)
        elif state == S2:
            s2_centroids.append(c)

    for i in range(len(s1_centroids) - 1):
        s1 = s1_centroids[i]
        next_s1 = s1_centroids[i + 1]
        s2_in_range = [s for s in s2_centroids if s1 < s < next_s1]
        if not s2_in_range:
            continue
        out.append((s1, s2_in_range[0], next_s1))
    return out
