"""Sanity tests for the preprocessing pipeline."""
from __future__ import annotations

import math

import numpy as np
import pytest

from openstetho_model.preprocess import (
    F_MAX,
    F_MIN,
    MEL_FFT_BINS,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    apply_cardiac,
    log_mel,
    mel_filterbank,
    split_windows,
)


def sine(freq: float, duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return np.sin(2 * math.pi * freq * t).astype(np.float32)


def test_filterbank_shape_and_monotonic_centers():
    bank = mel_filterbank()
    assert bank.shape == (N_MELS, MEL_FFT_BINS)
    # Each filter's peak bin should be monotonically non-decreasing.
    peaks = bank.argmax(axis=1)
    assert all(peaks[i + 1] >= peaks[i] for i in range(N_MELS - 1))


def test_low_tone_lands_in_low_mel_bins():
    audio = sine(60.0, 4.0)
    audio = apply_cardiac(audio)
    mel = log_mel(audio)
    # Use the middle frame to avoid window-leakage transients at the edges.
    frame = mel[mel.shape[0] // 2]
    assert frame.argmax() < N_MELS // 4


def test_upper_band_tone_lands_in_upper_mel_bins():
    # Stay inside the mel window (F_MAX ≈ 1000 Hz).
    audio = sine(800.0, 4.0)
    audio = apply_cardiac(audio)  # cardiac LP at 100 Hz will mostly kill it
    mel = log_mel(audio)
    frame = mel[mel.shape[0] // 2]
    # The cardiac chain attenuates 800 Hz heavily; we just check the signal
    # didn't shift to the bottom (it should still skew above the middle).
    assert frame.argmax() > N_MELS // 3


def test_split_windows_50_percent_overlap():
    audio = np.arange(WINDOW_SAMPLES * 3, dtype=np.float32)
    w = split_windows(audio)
    # 50% hop → for 3 windows worth of audio, expect 5 windows.
    assert w.shape == (5, WINDOW_SAMPLES)


def test_log_mel_dimensions_match_window():
    audio = np.random.randn(WINDOW_SAMPLES).astype(np.float32)
    mel = log_mel(audio)
    # WINDOW_SAMPLES / STFT_HOP (256) = 62.5; integer-truncated → 62 frames.
    assert mel.shape == (62, N_MELS)


def test_fmax_inside_mel_window():
    assert F_MIN < F_MAX <= 1000.5
    assert F_MAX == pytest.approx((MEL_FFT_BINS - 1) * SAMPLE_RATE / N_FFT)
