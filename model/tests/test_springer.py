"""Smoke tests for the Springer-style HSMM segmenter."""
from __future__ import annotations

import numpy as np

from openstetho_model import springer
from openstetho_model.s3_dataset import write_synthetic_pcg_wav
from openstetho_model.preprocess import load_audio
from openstetho_model.segment import segment_unified

SR = 4000


def _synth(tmp_path, n_cycles=10, seed=0):
    p = tmp_path / "x.wav"
    write_synthetic_pcg_wav(p, n_cycles=n_cycles, seed=seed)
    return load_audio(str(p))


def test_hsmm_runs_on_synthetic_pcg(tmp_path):
    audio = _synth(tmp_path)
    seg = springer.segment_hsmm(audio)
    assert len(seg.states) == len(audio)
    assert seg.cycle_period_s > 0
    assert set(np.unique(seg.states)).issubset({0, 1, 2, 3})


def test_hsmm_recovers_most_cycles(tmp_path):
    audio = _synth(tmp_path, n_cycles=10)
    seg = springer.segment_hsmm(audio)
    cycles = springer.cycles_from_hsmm(seg)
    assert len(cycles) >= 7  # allow misses at boundaries


def test_hsmm_cycle_period_within_5pct(tmp_path):
    audio = _synth(tmp_path)
    seg = springer.segment_hsmm(audio)
    assert abs(seg.cycle_period_s - 0.857) / 0.857 < 0.05


def test_segment_unified_dispatches_to_hsmm(tmp_path):
    audio = _synth(tmp_path)
    s = segment_unified(audio, method="hsmm")
    assert len(s.cycles) >= 7
    for c in s.cycles:
        assert c.s1_idx < c.s2_idx < c.next_s1_idx


def test_segment_unified_unknown_method_raises(tmp_path):
    audio = _synth(tmp_path)
    try:
        segment_unified(audio, method="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown method")
