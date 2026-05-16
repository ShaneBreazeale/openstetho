"""Python ↔ Rust mel-spec parity test.

`stetho-core/examples/dump_mel_parity.rs` writes the log-mel of a known
60 Hz sine to `model/tests/fixtures/mel_rust_60hz.txt`. This test feeds
the same synthetic signal through the Python preprocessor and asserts
the two outputs agree within a small tolerance.

Catches drift when either side changes mel-bank, FFT, or normalization
math — a class of bug that would otherwise silently disagree between
training-time features (Python) and inference-time features (Rust).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from openstetho_model.preprocess import N_MELS, SAMPLE_RATE, log_mel


FIXTURE = Path(__file__).parent / "fixtures" / "mel_rust_60hz.txt"


def _load_rust_fixture(path: Path) -> np.ndarray:
    with path.open() as f:
        header = f.readline().strip().split()
        n_frames, n_mels = int(header[0]), int(header[1])
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    assert len(rows) == n_frames
    arr = np.array(rows, dtype=np.float32)
    assert arr.shape == (n_frames, n_mels)
    return arr


def _python_reference() -> np.ndarray:
    sr = SAMPLE_RATE
    n = 16_000
    freq = 60.0
    amp = 10_000.0
    t = np.arange(n)
    audio = (amp * np.sin(2 * math.pi * freq * t / sr)).astype(np.float32)
    # NOTE: the Rust example does not apply the cardiac chain — it feeds
    # raw samples straight to LogMelSpectrogram, so we mirror that here.
    return log_mel(audio)


def test_mel_parity_shape_matches_rust():
    rust = _load_rust_fixture(FIXTURE)
    py = _python_reference()
    assert rust.shape == py.shape, f"rust {rust.shape} vs python {py.shape}"
    assert rust.shape[1] == N_MELS


def test_mel_parity_argmax_per_frame_matches():
    """Strongest claim: which mel bin wins on each frame should agree."""
    rust = _load_rust_fixture(FIXTURE)
    py = _python_reference()
    rust_arg = rust.argmax(axis=1)
    py_arg = py.argmax(axis=1)
    # Allow up to 1-bin drift per frame; equal majority of the time.
    matches = (rust_arg == py_arg).mean()
    close = (np.abs(rust_arg.astype(int) - py_arg.astype(int)) <= 1).mean()
    assert close > 0.95, f"argmax close-match rate {close:.2%} too low"
    assert matches > 0.70, f"argmax exact-match rate {matches:.2%} too low"


def test_mel_parity_per_frame_correlation():
    """Each Rust frame should correlate strongly with the Python frame at
    the same position (Pearson r > 0.95 average)."""
    rust = _load_rust_fixture(FIXTURE)
    py = _python_reference()
    rs = []
    for r, p in zip(rust, py, strict=True):
        if r.std() < 1e-9 or p.std() < 1e-9:
            continue
        r_cor = float(np.corrcoef(r, p)[0, 1])
        rs.append(r_cor)
    avg = float(np.mean(rs))
    assert avg > 0.95, f"avg per-frame correlation {avg:.3f} too low"
