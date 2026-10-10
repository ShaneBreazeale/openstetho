"""Tests for openstetho_model.channel — stethoscope / transport channel
simulation: random smooth EQ and the IMA ADPCM codec used on the BLE
link."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openstetho_model.channel import (
    EqCurve,
    adpcm_decode,
    adpcm_encode,
    adpcm_round_trip,
    apply_eq,
    random_eq_curve,
)
from openstetho_model.preprocess import SAMPLE_RATE

ADPCM_FIXTURE = Path(__file__).parent / "fixtures" / "adpcm_rust.txt"


def _tone(freq: float, n: int = SAMPLE_RATE, amp: float = 0.5) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * freq * np.arange(n) / SAMPLE_RATE)).astype(np.float32)


def _amp_db(x: np.ndarray, freq: float) -> float:
    """Magnitude of the FFT bin at `freq` (bin-exact for 1 s signals)."""
    spec = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64)))
    return float(20 * np.log10(spec[int(round(freq * len(x) / SAMPLE_RATE))]))


def _snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    err = np.asarray(test, dtype=np.float64) - ref
    return float(10 * np.log10(np.sum(ref**2) / np.sum(err**2)))


# ─── random smooth EQ ───────────────────────────────────────────────────────

def test_random_eq_curve_is_log_spaced_bounded_and_seeded():
    c = random_eq_curve(np.random.default_rng(0), n_points=8, f_lo=20.0, f_hi=2000.0, max_db=12.0)
    assert c.freqs_hz.shape == c.gains_db.shape == (8,)
    assert c.freqs_hz[0] == pytest.approx(20.0) and c.freqs_hz[-1] == pytest.approx(2000.0)
    ratios = c.freqs_hz[1:] / c.freqs_hz[:-1]
    assert np.allclose(ratios, ratios[0])
    assert np.all(np.abs(c.gains_db) <= 12.0)
    again = random_eq_curve(np.random.default_rng(0), n_points=8, f_lo=20.0, f_hi=2000.0, max_db=12.0)
    assert np.array_equal(c.gains_db, again.gains_db)


def test_random_eq_curves_differ_across_draws():
    rng = np.random.default_rng(0)
    a, b = random_eq_curve(rng), random_eq_curve(rng)
    assert not np.allclose(a.gains_db, b.gains_db)


def test_flat_eq_is_identity():
    x = np.random.default_rng(0).standard_normal(SAMPLE_RATE).astype(np.float32)
    flat = EqCurve(np.geomspace(20.0, 2000.0, 8), np.zeros(8))
    assert np.allclose(apply_eq(x, flat), x, atol=1e-6)


@pytest.mark.parametrize("freq,gain", [(80.0, 12.0), (320.0, -12.0), (1280.0, 6.0)])
def test_eq_scales_a_tone_at_a_control_point_by_its_gain(freq, gain):
    curve = EqCurve(np.array([20.0, 80.0, 320.0, 1280.0]), np.array([0.0, 12.0, -12.0, 6.0]))
    x = _tone(freq)
    assert _amp_db(apply_eq(x, curve), freq) - _amp_db(x, freq) == pytest.approx(gain, abs=0.05)


def test_eq_gain_is_flat_beyond_the_end_control_points():
    curve = EqCurve(np.array([40.0, 400.0]), np.array([-6.0, 9.0]))
    assert curve.gain_db(np.array([5.0, 20.0, 40.0])) == pytest.approx([-6.0, -6.0, -6.0])
    assert curve.gain_db(np.array([400.0, 1500.0])) == pytest.approx([9.0, 9.0])


def test_eq_interpolation_never_overshoots_the_control_gains():
    """Monotone (PCHIP) interpolation: no ringing between control points."""
    rng = np.random.default_rng(1)
    f = np.geomspace(1.0, 2000.0, 4000)
    for _ in range(50):
        c = random_eq_curve(rng)
        g = c.gain_db(f)
        assert g.max() <= c.gains_db.max() + 1e-9
        assert g.min() >= c.gains_db.min() - 1e-9


def test_eq_impulse_response_fits_inside_one_stft_frame():
    """A smooth magnitude response has a short impulse response, so the
    EQ colours the sound without smearing S1/S2 across mel frames
    (256-sample STFT frames, 64 ms)."""
    rng = np.random.default_rng(2)
    n, centre, half = SAMPLE_RATE, SAMPLE_RATE // 2, 128
    impulse = np.zeros(n, dtype=np.float32)
    impulse[centre] = 1.0
    for _ in range(50):
        y = apply_eq(impulse, random_eq_curve(rng)).astype(np.float64)
        energy = np.sum(y**2)
        assert np.sum(y[centre - half:centre + half] ** 2) / energy > 0.99


def test_apply_eq_keeps_length_dtype_and_input():
    x = np.random.default_rng(0).standard_normal(5001).astype(np.float32)
    before = x.copy()
    y = apply_eq(x, random_eq_curve(np.random.default_rng(0)))
    assert y.shape == x.shape and y.dtype == x.dtype
    assert np.array_equal(x, before)


# ─── IMA ADPCM ──────────────────────────────────────────────────────────────

def _load_adpcm_fixture() -> tuple[np.ndarray, np.ndarray]:
    lines = ADPCM_FIXTURE.read_text().split("\n")
    n = int(lines[0])
    rows = np.array([[int(v) for v in line.split()] for line in lines[1:] if line.strip()])
    assert rows.shape == (n, 2)
    return rows[:, 0].astype(np.uint8), rows[:, 1].astype(np.int16)


def test_python_decoder_matches_stetho_core_bit_for_bit():
    """Fixture from `cargo run -q -p stetho-core --example adpcm_vectors`;
    covers every nibble value and both predictor clamps."""
    nibbles, expected = _load_adpcm_fixture()
    assert expected.min() == -32768 and expected.max() == 32767
    assert np.array_equal(adpcm_decode(nibbles), expected)


def test_encoder_emits_one_nibble_per_sample():
    pcm = (8000 * np.sin(np.arange(1000) / 7.0)).astype(np.int16)
    nib = adpcm_encode(pcm)
    assert nib.shape == pcm.shape and nib.dtype == np.uint8
    assert nib.max() <= 15


def test_encode_decode_round_trip_tracks_a_heart_band_tone():
    pcm = (8000 * np.sin(2 * np.pi * 100 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)).astype(np.int16)
    out = adpcm_decode(adpcm_encode(pcm))
    settle = 200  # 50 ms for the step size to adapt up from index 0
    assert _snr_db(pcm[settle:], out[settle:]) > 20.0


def test_encoder_survives_full_scale_square_wave():
    pcm = np.where(np.arange(4000) % 40 < 20, 32767, -32768).astype(np.int16)
    out = adpcm_decode(adpcm_encode(pcm))
    assert out.dtype == np.int16 and out.shape == pcm.shape


def test_digital_silence_round_trips_to_exact_zeros():
    assert not adpcm_decode(adpcm_encode(np.zeros(4000, dtype=np.int16))).any()


@pytest.mark.parametrize("peak_dbfs", [-6.0, -20.0])
def test_float_round_trip_preserves_level_and_adds_only_codec_noise(peak_dbfs):
    rng = np.random.default_rng(0)
    x = (_tone(100.0, 2 * SAMPLE_RATE, 0.3) + 0.05 * rng.standard_normal(2 * SAMPLE_RATE)).astype(np.float32)
    y = adpcm_round_trip(x, peak_dbfs=peak_dbfs)
    assert y.shape == x.shape and y.dtype == np.float32
    settle = 200
    assert _snr_db(x[settle:], y[settle:]) > 15.0
    assert _snr_db(x[settle:], y[settle:]) < 60.0  # the codec really ran


def test_float_round_trip_of_silence_is_silence():
    assert not adpcm_round_trip(np.zeros(4000, dtype=np.float32)).any()
