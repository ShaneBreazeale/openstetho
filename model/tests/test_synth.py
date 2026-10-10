"""Tests for openstetho_model.synth — synthetic signals whose ground truth is
known by construction (inserted murmur location and SNR, noise spectra)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch
from scipy.stats import kurtosis

from openstetho_model.preprocess import SAMPLE_RATE
from openstetho_model.murmur_synth import (
    MURMUR_SHAPES,
    NO_HEART_KINDS,
    STATE_DIASTOLE,
    STATE_S1,
    STATE_S2,
    STATE_SYSTOLE,
    Segment,
    band_noise,
    insert_murmur,
    load_segments,
    murmur_envelope,
    no_heart,
    segment_slice,
)

CIRCOR = Path(__file__).resolve().parents[2] / "data" / "circor"
MEDIAN_SYSTOLE = 560  # samples; CirCor median systole is 0.14 s at 4 kHz


def _flat(shape: str, n: int) -> np.ndarray:
    return np.ones(n)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _toy_recording(cycles: int = 3) -> tuple[np.ndarray, list[Segment]]:
    """Unannotated lead-in, then S1 / systole / S2 / diastole cycles with
    50 Hz Hann-windowed tone bursts standing in for S1 and S2."""
    segs, t = [Segment(0.0, 0.2, 0)], 0.2
    for _ in range(cycles):
        for state, dur in ((STATE_S1, 0.1), (STATE_SYSTOLE, 0.15), (STATE_S2, 0.1), (STATE_DIASTOLE, 0.45)):
            segs.append(Segment(t, t + dur, state))
            t += dur
    audio = np.zeros(int(round(t * SAMPLE_RATE)), dtype=np.float32)
    for s in segs:
        if s.state in (STATE_S1, STATE_S2):
            sl = segment_slice(s, len(audio))
            n = sl.stop - sl.start
            tt = np.arange(n) / SAMPLE_RATE
            audio[sl] = 0.5 * np.sin(2 * np.pi * 50 * tt) * np.hanning(n)
    return audio, segs


def _mask(segs: list[Segment], state: int, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s in segs:
        if s.state == state:
            m[segment_slice(s, n)] = True
    return m


# ─── load_segments / segment_slice ──────────────────────────────────────────

def test_load_segments_parses_circor_tsv(tmp_path):
    tsv = tmp_path / "x.tsv"
    tsv.write_text("0\t0.88025\t0\n0.88025\t1.00025\t1\n\n1.00025\t1.151917\t2\n")
    assert load_segments(tsv) == [
        Segment(0.0, 0.88025, 0),
        Segment(0.88025, 1.00025, 1),
        Segment(1.00025, 1.151917, 2),
    ]


@pytest.mark.skipif(not CIRCOR.is_dir(), reason="needs data/circor")
def test_load_segments_on_real_circor_file():
    segs = load_segments(sorted(CIRCOR.glob("*.tsv"))[0])
    assert len(segs) > 10
    assert all(s.start < s.end for s in segs)
    assert all(a.start <= b.start for a, b in zip(segs, segs[1:]))


def test_segment_slice_rounds_to_samples_and_clips_to_audio():
    assert segment_slice(Segment(0.25, 0.5, 2), 4000) == slice(1000, 2000)
    # 6 CirCor annotations run past the end of their WAV
    assert segment_slice(Segment(0.9, 1.2, 4), 4000) == slice(3600, 4000)
    assert segment_slice(Segment(2.0, 3.0, 4), 4000) == slice(4000, 4000)


# ─── band_noise ─────────────────────────────────────────────────────────────

def test_band_noise_is_unit_rms_and_band_limited():
    x = band_noise(SAMPLE_RATE * 4, 100.0, 400.0, np.random.default_rng(0))
    assert x.shape == (SAMPLE_RATE * 4,)
    assert _rms(x) == pytest.approx(1.0, rel=1e-9)
    power = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    inside = (f >= 100.0) & (f <= 400.0)
    assert power[inside].sum() / power.sum() > 0.999


def test_band_noise_is_reproducible_from_seed():
    a = band_noise(4000, 100.0, 400.0, np.random.default_rng(7))
    b = band_noise(4000, 100.0, 400.0, np.random.default_rng(7))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("low,high", [(400.0, 100.0), (100.0, 2500.0), (-5.0, 100.0)])
def test_band_noise_rejects_bad_band(low, high):
    with pytest.raises(ValueError):
        band_noise(4000, low, high, np.random.default_rng(0))


def test_band_noise_rejects_band_with_no_fft_bins():
    with pytest.raises(ValueError):
        band_noise(8, 100.0, 110.0, np.random.default_rng(0))  # bins every 500 Hz


# ─── murmur_envelope ────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", MURMUR_SHAPES)
def test_envelope_contract(shape):
    e = murmur_envelope(shape, MEDIAN_SYSTOLE)
    assert e.shape == (MEDIAN_SYSTOLE,)
    assert np.all(np.isfinite(e))
    assert e.min() >= 0.0
    # insert_murmur rescales by RMS, so the exact peak only needs to be
    # close to 1 (np.hanning on an even length peaks at 0.999992).
    assert e.max() == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("shape", MURMUR_SHAPES)
@pytest.mark.parametrize("n", [1, 2, 3])
def test_envelope_handles_tiny_segments(shape, n):
    e = murmur_envelope(shape, n)
    assert e.shape == (n,)
    assert np.all(np.isfinite(e)) and e.min() >= 0.0 and e.max() <= 1.0


def test_crescendo_decrescendo_peaks_in_the_middle_third():
    e = murmur_envelope("crescendo_decrescendo", MEDIAN_SYSTOLE)
    assert MEDIAN_SYSTOLE / 3 <= int(np.argmax(e)) <= 2 * MEDIAN_SYSTOLE / 3
    assert e[0] < 0.2 and e[-1] < 0.2


@pytest.mark.parametrize("peak", [0.3, 0.5, 0.7])
def test_crescendo_decrescendo_peak_position_is_adjustable(peak):
    """Mild aortic stenosis peaks early in systole, severe AS late."""
    e = murmur_envelope("crescendo_decrescendo", MEDIAN_SYSTOLE, peak=peak)
    assert int(np.argmax(e)) / MEDIAN_SYSTOLE == pytest.approx(peak, abs=0.01)
    rise, fall = e[: int(np.argmax(e)) + 1], e[int(np.argmax(e)):]
    assert np.all(np.diff(rise) >= -1e-12) and np.all(np.diff(fall) <= 1e-12)


@pytest.mark.parametrize("peak", [0.0, 1.0, -0.2, 1.5])
def test_envelope_rejects_peak_outside_open_unit_interval(peak):
    with pytest.raises(ValueError):
        murmur_envelope("crescendo_decrescendo", MEDIAN_SYSTOLE, peak=peak)


def test_decrescendo_starts_at_peak_and_never_rises():
    e = murmur_envelope("decrescendo", MEDIAN_SYSTOLE)
    assert e[0] == pytest.approx(1.0)
    assert np.all(np.diff(e) <= 1e-12)


def test_crescendo_ends_at_peak_and_never_falls():
    e = murmur_envelope("crescendo", MEDIAN_SYSTOLE)
    assert e[-1] == pytest.approx(1.0)
    assert np.all(np.diff(e) >= -1e-12)


def test_plateau_is_mostly_flat():
    e = murmur_envelope("plateau", MEDIAN_SYSTOLE)
    assert np.mean(e >= 0.9) >= 0.8


def test_envelope_rejects_unknown_shape():
    with pytest.raises(ValueError):
        murmur_envelope("triangle", MEDIAN_SYSTOLE)


# ─── insert_murmur ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("state", [STATE_SYSTOLE, STATE_DIASTOLE])
def test_insert_murmur_only_touches_target_state(state):
    audio, segs = _toy_recording()
    out = insert_murmur(audio, segs, state=state, shape="plateau", snr_db=-6.0,
                        rng=np.random.default_rng(0), envelope=_flat)
    diff = out.astype(np.float64) - audio
    target = _mask(segs, state, len(audio))
    assert np.all(diff[~target] == 0.0)
    assert np.mean(diff[target] != 0.0) > 0.99


@pytest.mark.parametrize("snr_db", [-20.0, -6.0, 0.0, 6.0])
def test_insert_murmur_hits_requested_snr_against_s1_s2(snr_db):
    """SNR is murmur RMS over its active samples relative to the RMS of
    the S1 + S2 samples, the way murmur grade is judged by ear."""
    audio, segs = _toy_recording()
    out = insert_murmur(audio, segs, state=STATE_SYSTOLE, shape="plateau", snr_db=snr_db,
                        rng=np.random.default_rng(0), envelope=_flat)
    n = len(audio)
    heart = _mask(segs, STATE_S1, n) | _mask(segs, STATE_S2, n)
    active = _mask(segs, STATE_SYSTOLE, n)
    diff = out.astype(np.float64) - audio
    got = 20 * np.log10(_rms(diff[active]) / _rms(audio[heart]))
    assert got == pytest.approx(snr_db, abs=1e-3)


def test_insert_murmur_applies_the_envelope():
    audio, segs = _toy_recording()

    def ramp(shape, n):
        return np.linspace(0.0, 1.0, n)

    out = insert_murmur(audio, segs, state=STATE_SYSTOLE, shape="crescendo", snr_db=0.0,
                        rng=np.random.default_rng(0), envelope=ramp)
    sl = segment_slice(next(s for s in segs if s.state == STATE_SYSTOLE), len(audio))
    seg = (out.astype(np.float64) - audio)[sl]
    half = len(seg) // 2
    assert _rms(seg[half:]) > 2 * _rms(seg[:half])


def test_insert_murmur_component_stays_in_band():
    audio, segs = _toy_recording()
    out = insert_murmur(audio, segs, state=STATE_SYSTOLE, shape="plateau", snr_db=0.0,
                        band=(150.0, 350.0), rng=np.random.default_rng(0), envelope=_flat)
    sl = segment_slice(next(s for s in segs if s.state == STATE_SYSTOLE), len(audio))
    seg = (out.astype(np.float64) - audio)[sl]
    power = np.abs(np.fft.rfft(seg)) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / SAMPLE_RATE)
    assert power[(f >= 150.0) & (f <= 350.0)].sum() / power.sum() > 0.99


def test_insert_murmur_leaves_input_untouched_and_keeps_dtype():
    audio, segs = _toy_recording()
    before = audio.copy()
    out = insert_murmur(audio, segs, state=STATE_SYSTOLE, shape="plateau", snr_db=0.0,
                        rng=np.random.default_rng(0), envelope=_flat)
    assert np.array_equal(audio, before)
    assert out.dtype == audio.dtype and out.shape == audio.shape


def test_insert_murmur_is_reproducible_from_seed():
    audio, segs = _toy_recording()
    kw = dict(state=STATE_SYSTOLE, shape="plateau", snr_db=0.0, envelope=_flat)
    a = insert_murmur(audio, segs, rng=np.random.default_rng(3), **kw)
    b = insert_murmur(audio, segs, rng=np.random.default_rng(3), **kw)
    assert np.array_equal(a, b)


def test_insert_murmur_skips_segments_too_short_to_hold_the_band():
    audio, segs = _toy_recording()
    t = segs[1].start
    segs = [*segs, Segment(t + 0.001, t + 0.003, STATE_DIASTOLE)]  # 8 samples
    out = insert_murmur(audio, segs, state=STATE_DIASTOLE, shape="plateau", snr_db=0.0,
                        rng=np.random.default_rng(0), envelope=_flat)
    tiny = segment_slice(segs[-1], len(audio))
    assert np.array_equal(out[tiny], audio[tiny])


def test_insert_murmur_without_target_state_raises():
    audio, segs = _toy_recording()
    segs = [s for s in segs if s.state != STATE_DIASTOLE]
    with pytest.raises(ValueError, match="no usable"):
        insert_murmur(audio, segs, state=STATE_DIASTOLE, shape="plateau", snr_db=0.0,
                      rng=np.random.default_rng(0), envelope=_flat)


def test_insert_murmur_without_heart_sounds_raises():
    """One CirCor file has no S1/S2 annotation; SNR has no reference."""
    audio, segs = _toy_recording()
    segs = [s for s in segs if s.state not in (STATE_S1, STATE_S2)]
    with pytest.raises(ValueError, match="S1/S2"):
        insert_murmur(audio, segs, state=STATE_SYSTOLE, shape="plateau", snr_db=0.0,
                      rng=np.random.default_rng(0), envelope=_flat)


def test_insert_murmur_ignores_unknown_states():
    """One CirCor annotation row carries state 28."""
    audio, segs = _toy_recording()
    segs = [*segs, Segment(0.0, 0.1, 28)]
    out = insert_murmur(audio, segs, state=STATE_SYSTOLE, shape="plateau", snr_db=0.0,
                        rng=np.random.default_rng(0), envelope=_flat)
    assert np.array_equal(out[:400], audio[:400])


# ─── no_heart ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", NO_HEART_KINDS)
def test_no_heart_shape_dtype_and_level(kind):
    x = no_heart(kind, SAMPLE_RATE * 4, np.random.default_rng(0), rms=0.05)
    assert x.shape == (SAMPLE_RATE * 4,)
    assert x.dtype == np.float32
    assert np.all(np.isfinite(x))
    if kind == "zeros":
        assert not x.any()
    else:
        assert _rms(x) == pytest.approx(0.05, rel=1e-3)


def _psd_slope(x: np.ndarray, lo: float = 20.0, hi: float = 1000.0) -> float:
    f, p = welch(x, fs=SAMPLE_RATE, nperseg=1024)
    m = (f >= lo) & (f <= hi)
    return float(np.polyfit(np.log(f[m]), np.log(p[m]), 1)[0])


@pytest.mark.parametrize("kind,slope", [("white", 0.0), ("pink", -1.0), ("brown", -2.0)])
def test_coloured_noise_has_known_log_log_psd_slope(kind, slope):
    x = no_heart(kind, SAMPLE_RATE * 30, np.random.default_rng(0))
    assert _psd_slope(x) == pytest.approx(slope, abs=0.15)


def test_hum_peaks_at_mains_frequency():
    for seed in range(4):
        x = no_heart("hum", SAMPLE_RATE * 10, np.random.default_rng(seed))
        f, p = welch(x, fs=SAMPLE_RATE, nperseg=SAMPLE_RATE)  # 1 Hz bins
        assert f[np.argmax(p)] in (50.0, 60.0)


def test_rub_is_bursty():
    """Gaussian noise has excess kurtosis 0; sparse bursts are heavy-tailed."""
    x = no_heart("rub", SAMPLE_RATE * 10, np.random.default_rng(0))
    assert kurtosis(x) > 3.0


def test_no_heart_rejects_unknown_kind():
    with pytest.raises(ValueError):
        no_heart("thunder", 4000, np.random.default_rng(0))
