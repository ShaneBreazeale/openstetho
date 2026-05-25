"""Audio preprocessing — keep numerically aligned with stetho-core::dsp.

Pipeline shape (matches Rust):
  raw mono float32 → resample to 4 kHz → cardiac biquad chain
  (HP 35 → HP 55 → LP 100, Butterworth Q=√2/2)
  → window into 4 s frames @ 50 % overlap
  → 256-point STFT, Hann window, no overlap inside the STFT (hop=256)
  → 32-band Slaney mel, f_min=20, f_max≈1000 (top of 65-bin window)
  → log10 × 10, floor 1e-10
  → per-frame z-score, clipped at -80 dB / 80 = -1.0

The mel filterbank is regenerated from Slaney's standard formula
(Apple Technical Report #45, 1998).
"""
from __future__ import annotations

import numpy as np
import scipy.fft as spfft
import scipy.signal as sps
import soundfile as sf

SAMPLE_RATE = 4000
WINDOW_SECONDS = 4.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)  # 16000
HOP_FRACTION = 0.5
WINDOW_HOP = int(WINDOW_SAMPLES * HOP_FRACTION)  # 8000

N_FFT = 256
STFT_HOP = 256  # no overlap, mirrors stetho-core::dsp::mel
N_MELS = 32
MEL_FFT_BINS = 65  # lower-half-of-spectrum window
F_MIN = 20.0
F_MAX = (MEL_FFT_BINS - 1) * SAMPLE_RATE / N_FFT  # ≈ 1000 Hz
LOG_FLOOR = 1e-10
NORM_CLIP = -1.0  # = -80 dB / 80
FEATURE_MODE_LOGMEL = "logmel"
FEATURE_MODE_MFCC = "mfcc"
FEATURE_MODE_SCATTERING = "scattering"
FEATURE_MODE_MULTI = "multi"
FEATURE_MODES = (FEATURE_MODE_LOGMEL, FEATURE_MODE_MFCC, FEATURE_MODE_SCATTERING, FEATURE_MODE_MULTI)
SCATTERING_J = 8
SCATTERING_Q = 4


# ─── filters ────────────────────────────────────────────────────────────────

def cardiac_sos(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Cascaded second-order sections for HP35 → HP55 → LP100 Butterworth."""
    hp35 = sps.butter(2, 35.0, btype="high", fs=sample_rate, output="sos")
    hp55 = sps.butter(2, 55.0, btype="high", fs=sample_rate, output="sos")
    lp100 = sps.butter(2, 100.0, btype="low", fs=sample_rate, output="sos")
    return np.vstack([hp35, hp55, lp100])


def apply_cardiac(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply the cardiac chain causally (sosfilt, not sosfiltfilt). Matches the
    real-time path the model will see in production."""
    return sps.sosfilt(cardiac_sos(sample_rate), x).astype(np.float32, copy=False)


def s3_sos(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Cascaded SOS for HP15 → LP120 Butterworth. Preserves S3 band (30–70 Hz)
    while attenuating mains hum and above-cardiac noise. Use for the S3
    detector path — `cardiac_sos` would attenuate ~20 dB at 40 Hz and erase
    most of the gallop energy."""
    hp15 = sps.butter(2, 15.0, btype="high", fs=sample_rate, output="sos")
    lp120 = sps.butter(2, 120.0, btype="low", fs=sample_rate, output="sos")
    return np.vstack([hp15, lp120])


def apply_s3_preset(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply the S3-detector filter chain causally."""
    return sps.sosfilt(s3_sos(sample_rate), x).astype(np.float32, copy=False)


def bandpass_sos(
    low_hz: float,
    high_hz: float,
    sample_rate: int = SAMPLE_RATE,
    order: int = 4,
) -> np.ndarray:
    """Butterworth bandpass for offline benchmark experiments."""
    return sps.butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")


def apply_bandpass(
    x: np.ndarray,
    low_hz: float,
    high_hz: float,
    sample_rate: int = SAMPLE_RATE,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase offline bandpass. Not used by the realtime path."""
    sos = bandpass_sos(low_hz, high_hz, sample_rate=sample_rate, order=order)
    return sps.sosfiltfilt(sos, x).astype(np.float32, copy=False)


# ─── resample + load ────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load WAV (any sr / any channel count) → mono float32 at target_sr."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        # `polyphase` is faster than `fft` for typical SR ratios and is the
        # default scipy recommends for audio.
        data = sps.resample_poly(data, target_sr, sr).astype(np.float32, copy=False)
    return data


# ─── mel-spec ───────────────────────────────────────────────────────────────

def _hz_to_mel_slaney(f: float) -> float:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return min_log_mel + np.log(f / min_log_hz) / logstep if f >= min_log_hz else f / f_sp


def _mel_to_hz_slaney(m: float) -> float:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return min_log_hz * np.exp(logstep * (m - min_log_mel)) if m >= min_log_mel else m * f_sp


def mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    mel_fft_bins: int = MEL_FFT_BINS,
    f_min: float = F_MIN,
    f_max: float | None = None,
) -> np.ndarray:
    """`n_mels × mel_fft_bins` triangular filterbank, librosa-Slaney style."""
    if f_max is None:
        f_max = (mel_fft_bins - 1) * sample_rate / n_fft
    m_min = _hz_to_mel_slaney(f_min)
    m_max = _hz_to_mel_slaney(f_max)
    mel_pts = np.array([
        _mel_to_hz_slaney(m_min + (m_max - m_min) * i / (n_mels + 1))
        for i in range(n_mels + 2)
    ])
    bin_freq = np.arange(mel_fft_bins) * sample_rate / n_fft
    bank = np.zeros((n_mels, mel_fft_bins), dtype=np.float32)
    for m in range(n_mels):
        lower, center, upper = mel_pts[m], mel_pts[m + 1], mel_pts[m + 2]
        norm = 2.0 / max(upper - lower, 1e-9)
        for b, f in enumerate(bin_freq):
            if f < lower or f > upper:
                continue
            w = (f - lower) / max(center - lower, 1e-9) if f <= center else (upper - f) / max(upper - center, 1e-9)
            bank[m, b] = w * norm
    return bank


_FILTERBANK_CACHE: dict[tuple, np.ndarray] = {}
_SCATTERING_CACHE: dict[tuple[int, int, int], object] = {}


def _get_filterbank() -> np.ndarray:
    key = (SAMPLE_RATE, N_MELS, N_FFT, MEL_FFT_BINS, F_MIN, F_MAX)
    bank = _FILTERBANK_CACHE.get(key)
    if bank is None:
        bank = mel_filterbank()
        _FILTERBANK_CACHE[key] = bank
    return bank


def _stft_magnitude(audio: np.ndarray) -> np.ndarray:
    """Frame audio and return lower STFT magnitudes, shape `(T, MEL_FFT_BINS)`."""
    if audio.ndim != 1:
        raise ValueError("expected mono 1D input")

    # Frame into N_FFT-length blocks with hop=STFT_HOP (no overlap).
    n = (len(audio) // STFT_HOP) * STFT_HOP
    if n == 0:
        return np.zeros((0, MEL_FFT_BINS), dtype=np.float32)
    audio = audio[:n]
    frames = audio.reshape(-1, STFT_HOP)  # (T, N_FFT) when HOP == N_FFT

    # Mean-center per frame before the STFT.
    frames = frames - frames.mean(axis=1, keepdims=True)

    # Hann window.
    win = np.hanning(N_FFT).astype(np.float32)
    framed = frames * win

    # rFFT magnitude, keep lower MEL_FFT_BINS bins.
    return np.abs(np.fft.rfft(framed, n=N_FFT, axis=1))[:, :MEL_FFT_BINS].astype(np.float32, copy=False)


def _normalize_feature_frames(x: np.ndarray) -> np.ndarray:
    """Per-frame z-score with the same lower clip used by log-mel."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True).clip(min=1e-9)
    x = (x - mean) / std
    x = np.clip(x, NORM_CLIP, None)
    return x.astype(np.float32, copy=False)


def _log_mel_db(audio: np.ndarray) -> np.ndarray:
    spec = _stft_magnitude(audio)
    if spec.shape[0] == 0:
        return np.zeros((0, N_MELS), dtype=np.float32)

    bank = _get_filterbank()
    mel = spec @ bank.T  # (T, N_MELS)
    return (10.0 * np.log10(np.maximum(mel, LOG_FLOOR))).astype(np.float32, copy=False)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """Compute the log-mel-spec for a single (already-filtered, 4 kHz mono)
    audio array. Returns `(n_frames, N_MELS)` float32."""
    return _normalize_feature_frames(_log_mel_db(audio))


def mfcc(audio: np.ndarray) -> np.ndarray:
    """MFCC channel derived from the same 32-band log-mel frames."""
    mel = _log_mel_db(audio)
    if mel.shape[0] == 0:
        return np.zeros((0, N_MELS), dtype=np.float32)
    coeffs = spfft.dct(mel, type=2, axis=1, norm="ortho")
    return _normalize_feature_frames(coeffs)


def stft_log_energy(audio: np.ndarray) -> np.ndarray:
    """Log-STFT energy channel compressed to `(n_frames, N_MELS)`.

    The lower 64 FFT bins are averaged in pairs so this channel has the same
    spatial shape as log-mel and MFCC while preserving linear-frequency cues.
    """
    spec = _stft_magnitude(audio)
    if spec.shape[0] == 0:
        return np.zeros((0, N_MELS), dtype=np.float32)
    log_spec = 10.0 * np.log10(np.maximum(spec[:, : N_MELS * 2], LOG_FLOOR))
    paired = log_spec.reshape(log_spec.shape[0], N_MELS, 2).mean(axis=2)
    return _normalize_feature_frames(paired)


def _get_scattering(length: int, j: int = SCATTERING_J, q: int = SCATTERING_Q):
    key = (length, j, q)
    scattering = _SCATTERING_CACHE.get(key)
    if scattering is None:
        try:
            from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D
        except ImportError as e:
            raise ImportError(
                "feature_mode='scattering' requires kymatio; run `uv sync --project model`"
            ) from e
        scattering = ScatteringNumPy1D(J=j, shape=length, Q=q)
        _SCATTERING_CACHE[key] = scattering
    return scattering


def scattering_features(audio: np.ndarray) -> np.ndarray:
    """Wavelet scattering coefficients as a `(time, coefficients)` feature map."""
    if audio.ndim != 1:
        raise ValueError("expected mono 1D input")
    if len(audio) == 0:
        return np.zeros((0, 1), dtype=np.float32)

    audio = audio.astype(np.float32, copy=False)
    audio = audio - float(audio.mean())
    scale = float(np.max(np.abs(audio)))
    if scale > 1e-9:
        audio = audio / scale
    scattering = _get_scattering(len(audio))
    coeffs = np.asarray(scattering(audio), dtype=np.float32)
    if coeffs.ndim != 2:
        raise ValueError(f"expected 2D scattering output, got shape {coeffs.shape}")
    coeffs = np.log1p(np.maximum(coeffs, 0.0))
    return _normalize_feature_frames(coeffs.T)


def feature_channels(mode: str) -> int:
    if mode in (FEATURE_MODE_LOGMEL, FEATURE_MODE_MFCC, FEATURE_MODE_SCATTERING):
        return 1
    if mode == FEATURE_MODE_MULTI:
        return 3
    raise ValueError(f"unknown feature mode {mode!r}; valid: {FEATURE_MODES}")


def window_features(audio: np.ndarray, mode: str = FEATURE_MODE_LOGMEL) -> np.ndarray:
    """Return model features for one window.

    `logmel` returns `(T, N_MELS)` to preserve the existing production path.
    `mfcc` returns `(T, N_MELS)` for a single-channel MFCC-only experiment.
    `scattering` returns `(T, C)` wavelet-scattering coefficients.
    `multi` returns `(3, T, N_MELS)` with log-mel, MFCC, and log-STFT energy.
    """
    if mode == FEATURE_MODE_LOGMEL:
        return log_mel(audio)
    if mode == FEATURE_MODE_MFCC:
        return mfcc(audio)
    if mode == FEATURE_MODE_SCATTERING:
        return scattering_features(audio)
    if mode == FEATURE_MODE_MULTI:
        return np.stack([log_mel(audio), mfcc(audio), stft_log_energy(audio)], axis=0)
    raise ValueError(f"unknown feature mode {mode!r}; valid: {FEATURE_MODES}")


# ─── 4-second windows ───────────────────────────────────────────────────────

def split_windows(
    audio: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    hop: int = WINDOW_HOP,
) -> np.ndarray:
    """Split into shape `(n_windows, window_samples)`, dropping tail."""
    if len(audio) < window_samples:
        return np.zeros((0, window_samples), dtype=audio.dtype)
    n_windows = 1 + (len(audio) - window_samples) // hop
    windows = np.stack(
        [audio[i * hop : i * hop + window_samples] for i in range(n_windows)],
        axis=0,
    )
    return windows


def preprocess_file(path: str) -> np.ndarray:
    """End-to-end: WAV path → `(n_windows, n_frames_per_window, N_MELS)`."""
    audio = load_audio(path)
    audio = apply_cardiac(audio)
    windows = split_windows(audio)
    if len(windows) == 0:
        return np.zeros((0, WINDOW_SAMPLES // STFT_HOP, N_MELS), dtype=np.float32)
    specs = np.stack([log_mel(w) for w in windows], axis=0)
    return specs
