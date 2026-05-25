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
    feature_channels,
    log_mel,
    mel_filterbank,
    mfcc,
    scattering_features,
    split_windows,
    stft_log_energy,
    window_features,
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


def test_multi_channel_features_match_window_shape():
    audio = np.random.randn(WINDOW_SAMPLES).astype(np.float32)
    mel = log_mel(audio)
    mfccs = mfcc(audio)
    stft = stft_log_energy(audio)
    stacked = window_features(audio, "multi")
    assert mfccs.shape == (62, N_MELS)
    assert stft.shape == (62, N_MELS)
    assert window_features(audio, "mfcc").shape == (62, N_MELS)
    assert stacked.shape == (3, 62, N_MELS)
    np.testing.assert_allclose(stacked[0], mel)
    assert feature_channels("logmel") == 1
    assert feature_channels("mfcc") == 1
    assert feature_channels("scattering") == 1
    assert feature_channels("multi") == 3


def test_five_second_feature_dimensions():
    audio = np.random.randn(int(SAMPLE_RATE * 5.0)).astype(np.float32)
    assert log_mel(audio).shape == (78, N_MELS)
    assert mfcc(audio).shape == (78, N_MELS)


def test_scattering_features_are_2d_and_finite():
    audio = np.random.randn(WINDOW_SAMPLES).astype(np.float32)
    scattering = scattering_features(audio)
    assert scattering.ndim == 2
    assert scattering.shape[0] > 0
    assert scattering.shape[1] > 8
    assert np.isfinite(scattering).all()
    np.testing.assert_allclose(window_features(audio, "scattering"), scattering)


def test_fmax_inside_mel_window():
    assert F_MIN < F_MAX <= 1000.5
    assert F_MAX == pytest.approx((MEL_FFT_BINS - 1) * SAMPLE_RATE / N_FFT)
