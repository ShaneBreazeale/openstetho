"""Train-only synthetic recordings for recording-level (MIL) murmur training.

* `SyntheticRecordings` -- label 1: a murmur inserted into a train-split
  Murmur-Absent CirCor recording (`murmur_synth.insert_murmur`, real heart
  underneath); label 0: a no-heart signal (`murmur_synth.no_heart`). Each item
  is a deterministic function of its index over a large virtual index space,
  passed through a random channel (`channel.apply_channel`) and split into the
  same fixed windows as real recordings.
* `MixedRecordingBatchSampler` -- every batch holds exactly
  floor(synth_fraction * B) synthetic recordings and real recordings for the
  rest; each real recording is seen once per epoch.

Synthetic items never enter validation: only trainers build them, and only
from train-split recordings.
"""
from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from .channel import apply_channel
from .murmur_synth import (
    NO_HEART_KINDS,
    STATE_DIASTOLE,
    STATE_S1,
    STATE_S2,
    STATE_SYSTOLE,
    insert_murmur,
    load_segments,
    murmur_envelope,
    no_heart,
)
from .preprocess import SAMPLE_RATE, load_audio, split_windows, window_features

SYSTOLIC_SHAPES = ("crescendo_decrescendo", "plateau", "crescendo")
DIASTOLIC_SHAPES = ("decrescendo", "crescendo")


@dataclass(frozen=True)
class SyntheticConfig:
    murmur_share: float = 0.5  # of synthetic items, fraction with an inserted murmur
    snr_db: tuple[float, float] = (-15.0, 0.0)  # inserted murmur vs S1/S2 RMS
    no_heart_seconds: tuple[float, float] = (8.0, 20.0)
    eq_prob: float = 0.5
    adpcm_prob: float = 0.25
    seed: int = 0


def mixed_train_prior(real_prior: float, synthetic_share: float, murmur_share: float) -> float:
    """Positive share the model trains on once synthetic items take
    `synthetic_share` of every batch."""
    return (1.0 - synthetic_share) * real_prior + synthetic_share * murmur_share


def murmur_sources(wavs: Sequence[Path], min_samples: int) -> list[Path]:
    """Recordings usable as a murmur background: a `.tsv` with S1/S2
    annotations next to them and at least one full window of audio."""
    keep = []
    for wav in wavs:
        tsv = Path(wav).with_suffix(".tsv")
        if not tsv.exists():
            continue
        if not any(s.state in (STATE_S1, STATE_S2) for s in load_segments(tsv)):
            continue
        info = sf.info(str(wav))
        if info.frames * SAMPLE_RATE / info.samplerate < min_samples:
            continue
        keep.append(Path(wav))
    return keep


class SyntheticRecordings(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        sources: Sequence[Path],
        cfg: SyntheticConfig,
        *,
        window_samples: int,
        hop_samples: int,
        feature_mode: str,
        length: int = 1 << 20,
    ):
        if cfg.murmur_share > 0 and not sources:
            raise ValueError("murmur_share > 0 needs at least one source recording")
        self.sources = [Path(s) for s in sources]
        self.cfg = cfg
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self.feature_mode = feature_mode
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, k: int) -> tuple[torch.Tensor, int]:
        audio, label, _meta = self.render(k)
        windows = split_windows(audio, self.window_samples, self.hop_samples)
        mels = np.stack([window_features(w, self.feature_mode) for w in windows], axis=0)
        return torch.from_numpy(mels), label

    def render(self, k: int) -> tuple[np.ndarray, int, dict]:
        """(float32 recording after the channel, label, what was made)."""
        rng = np.random.default_rng((self.cfg.seed, 2, int(k)))
        if rng.random() < self.cfg.murmur_share:
            audio, label, meta = self._murmur(rng)
        else:
            audio, label, meta = self._no_heart(rng)
        audio = apply_channel(audio, rng, eq_prob=self.cfg.eq_prob, adpcm_prob=self.cfg.adpcm_prob)
        return audio, label, meta

    def _murmur(self, rng: np.random.Generator) -> tuple[np.ndarray, int, dict]:
        for _ in range(10):
            src = self.sources[int(rng.integers(len(self.sources)))]
            audio = load_audio(str(src))
            segs = load_segments(src.with_suffix(".tsv"))
            state = STATE_SYSTOLE if rng.random() < 0.75 else STATE_DIASTOLE
            shapes = SYSTOLIC_SHAPES if state == STATE_SYSTOLE else DIASTOLIC_SHAPES
            shape = shapes[int(rng.integers(len(shapes)))]
            peak = float(rng.uniform(0.3, 0.7))
            # Centre 100-400 Hz, 1-2 octaves wide: rumbles to harsh ejection murmurs.
            centre = math.exp(rng.uniform(math.log(100.0), math.log(400.0)))
            half = 2.0 ** (rng.uniform(1.0, 2.0) / 2.0)
            band = (max(25.0, centre / half), min(1000.0, centre * half))
            snr = float(rng.uniform(*self.cfg.snr_db))
            try:
                mixed = insert_murmur(
                    audio, segs, state=state, shape=shape, snr_db=snr, rng=rng, band=band,
                    envelope=lambda s, n, p=peak: murmur_envelope(s, n, peak=p),
                )
            except ValueError:
                continue
            meta = {"source": str(src), "state": state, "shape": shape, "peak": peak,
                    "band": band, "snr_db": snr}
            return np.asarray(mixed, dtype=np.float32), 1, meta
        raise RuntimeError("could not insert a murmur into any source after 10 attempts")

    def _no_heart(self, rng: np.random.Generator) -> tuple[np.ndarray, int, dict]:
        kind = NO_HEART_KINDS[int(rng.integers(len(NO_HEART_KINDS)))]
        rms = float(math.exp(rng.uniform(math.log(1e-3), math.log(0.3))))
        seconds = float(rng.uniform(*self.cfg.no_heart_seconds))
        n = max(self.window_samples, int(seconds * SAMPLE_RATE))
        return no_heart(kind, n, rng, rms=rms), 0, {"kind": kind, "rms": rms, "seconds": seconds}


class MixedRecordingBatchSampler(Sampler[list[int]]):
    """Batches over `ConcatDataset([real_train_subset, SyntheticRecordings])`.

    Real ids are `0..n_real-1`, each seen once per epoch; synthetic ids are
    offset by `n_real` and drawn at random. Drops the last partial batch, like
    the plain trainer's `drop_last=True`."""

    def __init__(self, *, n_real: int, n_synth: int, batch_size: int, synth_fraction: float, seed: int):
        if not 0.0 <= synth_fraction < 1.0:
            raise ValueError(f"synth_fraction must be in [0, 1), got {synth_fraction}")
        self.n_real = n_real
        self.n_synth = n_synth
        self.batch_size = batch_size
        self.k_synth = math.floor(synth_fraction * batch_size)
        self.k_real = batch_size - self.k_synth
        self.seed = seed
        self._epoch = 0

    @property
    def synthetic_share(self) -> float:
        return self.k_synth / self.batch_size

    def __len__(self) -> int:
        return self.n_real // self.k_real

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng((self.seed, self._epoch))
        self._epoch += 1
        order = rng.permutation(self.n_real)
        for b in range(len(self)):
            real = order[b * self.k_real:(b + 1) * self.k_real]
            synth = self.n_real + rng.integers(0, self.n_synth, self.k_synth)
            yield rng.permutation(np.concatenate([real, synth])).tolist()
