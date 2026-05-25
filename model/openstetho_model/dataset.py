"""PhysioNet CirCor DigiScope 2022 PyTorch Dataset.

Layout we expect on disk (after `scripts/download_circor.sh`):

    data/circor/
        training_data/
            <patient_id>_<location>.wav
            <patient_id>_<location>.tsv         (segmentation, optional)
            <patient_id>_<location>.hea
        training_data.csv                       (per-patient metadata + labels)

Per-patient label of interest:
    Murmur ∈ {Present, Absent, Unknown}

We drop Unknown rows for binary classification. A single patient may have
recordings at multiple auscultation locations (AV/MV/PV/TV/Phc); we treat
each recording as an independent sample and inherit the patient-level
murmur label.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess import (
    FEATURE_MODE_LOGMEL,
    HOP_FRACTION,
    SAMPLE_RATE,
    apply_cardiac,
    load_audio,
    split_windows,
    window_features,
)

log = logging.getLogger(__name__)


@dataclass
class CirCorSample:
    """One fixed-length window with a label."""
    patient_id: int
    location: str
    label: int       # 0 = Absent, 1 = Present
    audio: np.ndarray  # float32, cardiac-filtered, 4 kHz


def window_hop_samples(window_seconds: float, hop_seconds: float | None = None) -> tuple[int, int]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if hop_seconds is None:
        hop_seconds = window_seconds * HOP_FRACTION
    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive")
    return int(round(window_seconds * SAMPLE_RATE)), int(round(hop_seconds * SAMPLE_RATE))


def _load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Schema sanity.
    required = {"Patient ID", "Murmur", "Recording locations:"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CirCor metadata missing columns: {sorted(missing)}")
    return df


def _patient_recordings(root: Path, patient_id: int, locations: str) -> list[tuple[str, Path]]:
    """Resolve `<patient_id>_<loc>.wav` for every location listed for this
    patient. Locations is a `+`-separated string in CirCor's CSV.

    CirCor v1.0.3 ships the WAVs flat at the dataset root; we still try a
    `training_data/` subdir as a fallback in case the layout changes."""
    out: list[tuple[str, Path]] = []
    for loc in locations.split("+"):
        loc = loc.strip()
        for candidate in (root / f"{patient_id}_{loc}.wav", root / "training_data" / f"{patient_id}_{loc}.wav"):
            if candidate.exists():
                out.append((loc, candidate))
                break
    return out


class CirCorMurmurDataset(Dataset[tuple[torch.Tensor, int]]):
    """Binary murmur-present-vs-absent classifier dataset.

    Each `__getitem__` returns one preprocessed fixed-length window:
        mel: torch.Tensor (n_frames, N_MELS)  — log-mel spectrogram
             or (3, n_frames, N_MELS) in multi-channel research mode
        label: int                            — 0 absent, 1 present
    """

    def __init__(
        self,
        root: str | Path,
        patient_ids: Sequence[int] | None = None,
        apply_cardiac: bool = True,
        profile_fir_path: str | Path | None = None,
        feature_mode: str = FEATURE_MODE_LOGMEL,
        window_seconds: float = 4.0,
        hop_seconds: float | None = None,
    ):
        self.root = Path(root)
        self.window_samples, self.hop_samples = window_hop_samples(window_seconds, hop_seconds)
        # When False, audio goes straight to mel-spec with no biquad chain.
        # Matches the stetho-ui inference path which now also runs the model
        # on raw audio so training and deployment share the same preprocess.
        self.apply_cardiac_filter = apply_cardiac
        self.feature_mode = feature_mode

        # Optional spectral-profile FIR — convolves each loaded clip with
        # a 256-tap filter that shifts CirCor's mean magnitude spectrum
        # toward the target device's passband. Lets the model train on CirCor labels while
        # seeing audio statistics that look like the deployment domain.
        self._profile_fir: np.ndarray | None = None
        if profile_fir_path is not None:
            import json
            data = json.loads(Path(profile_fir_path).read_text())
            self._profile_fir = np.asarray(data["coefficients"], dtype=np.float32)
            log.info("CirCor training audio will be convolved with profile FIR (%d taps) from %s",
                     len(self._profile_fir), profile_fir_path)
        df = _load_metadata(self.root / "training_data.csv")
        df = df[df["Murmur"].isin(["Present", "Absent"])].copy()
        if patient_ids is not None:
            df = df[df["Patient ID"].isin(patient_ids)]
        self.metadata = df.reset_index(drop=True)
        self._index: list[tuple[int, str, Path, int, int]] = []
        # Pre-enumerate every (patient, recording, window) tuple so __len__
        # is O(1) and we can stratify-split deterministically.
        for _, row in self.metadata.iterrows():
            label = 1 if row["Murmur"] == "Present" else 0
            for loc, wav in _patient_recordings(self.root, int(row["Patient ID"]), row["Recording locations:"]):
                # Read header for length so we don't open WAVs at __init__.
                try:
                    audio_len = self._wav_frame_count(wav)
                except Exception as e:  # noqa: BLE001
                    log.warning("skip %s: %s", wav, e)
                    continue
                n_windows = max(0, 1 + (audio_len - self.window_samples) // self.hop_samples)
                for w in range(n_windows):
                    self._index.append((int(row["Patient ID"]), loc, wav, label, w))

    @staticmethod
    def _wav_frame_count(path: Path) -> int:
        import soundfile as sf
        info = sf.info(str(path))
        return int(info.frames * (4000 / info.samplerate))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        patient_id, loc, wav, label, w_idx = self._index[idx]
        audio = load_audio(str(wav))
        if self._profile_fir is not None:
            # scipy.signal.lfilter accepts (b, a, x); pure-FIR has a=[1].
            import scipy.signal as sps
            audio = sps.lfilter(self._profile_fir, [1.0], audio).astype(np.float32)
        if self.apply_cardiac_filter:
            audio = apply_cardiac(audio)
        windows = split_windows(audio, self.window_samples, self.hop_samples)
        if w_idx >= len(windows):
            # Length estimate was off; clamp to last window.
            w_idx = len(windows) - 1
        mel = window_features(windows[w_idx], self.feature_mode)
        return torch.from_numpy(mel), label

    def class_balance(self) -> dict[str, int]:
        absent = sum(1 for *_, label, _ in self._index if label == 0)
        present = len(self._index) - absent
        return {"absent": absent, "present": present}


class CirCorRecordingMurmurDataset(Dataset[tuple[torch.Tensor, int]]):
    """Recording-level murmur dataset.

    Each item is all fixed-length windows from one recording:
        mels: torch.Tensor (n_windows, n_frames, N_MELS)
              or (n_windows, 3, n_frames, N_MELS) in multi-channel mode
        label: int

    This supports multiple-instance learning where the model scores every
    window and training aggregates those logits to the recording label.
    """

    def __init__(
        self,
        root: str | Path,
        patient_ids: Sequence[int] | None = None,
        apply_cardiac: bool = True,
        profile_fir_path: str | Path | None = None,
        feature_mode: str = FEATURE_MODE_LOGMEL,
        window_seconds: float = 4.0,
        hop_seconds: float | None = None,
        cache_features: bool = False,
    ):
        self.root = Path(root)
        self.window_samples, self.hop_samples = window_hop_samples(window_seconds, hop_seconds)
        self.apply_cardiac_filter = apply_cardiac
        self.feature_mode = feature_mode
        self.cache_features = cache_features
        self._feature_cache: dict[int, torch.Tensor] = {}
        self._profile_fir: np.ndarray | None = None
        if profile_fir_path is not None:
            import json
            data = json.loads(Path(profile_fir_path).read_text())
            self._profile_fir = np.asarray(data["coefficients"], dtype=np.float32)
            log.info(
                "CirCor recording audio will be convolved with profile FIR (%d taps) from %s",
                len(self._profile_fir),
                profile_fir_path,
            )

        df = _load_metadata(self.root / "training_data.csv")
        df = df[df["Murmur"].isin(["Present", "Absent"])].copy()
        if patient_ids is not None:
            df = df[df["Patient ID"].isin(patient_ids)]
        self.metadata = df.reset_index(drop=True)
        self._records: list[tuple[int, str, Path, int]] = []
        for _, row in self.metadata.iterrows():
            label = 1 if row["Murmur"] == "Present" else 0
            for loc, wav in _patient_recordings(
                self.root,
                int(row["Patient ID"]),
                row["Recording locations:"],
            ):
                self._records.append((int(row["Patient ID"]), loc, wav, label))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        _patient_id, _loc, wav, label = self._records[idx]
        if self.cache_features and idx in self._feature_cache:
            return self._feature_cache[idx], label
        audio = load_audio(str(wav))
        if self._profile_fir is not None:
            import scipy.signal as sps
            audio = sps.lfilter(self._profile_fir, [1.0], audio).astype(np.float32)
        if self.apply_cardiac_filter:
            audio = apply_cardiac(audio)
        windows = split_windows(audio, self.window_samples, self.hop_samples)
        if len(windows) == 0:
            # Keep the model path defined for very short clips.
            padded = np.zeros(self.window_samples, dtype=np.float32)
            padded[: min(len(audio), self.window_samples)] = audio[: self.window_samples]
            windows = np.expand_dims(padded, axis=0)
        mels = np.stack([window_features(w, self.feature_mode) for w in windows], axis=0)
        tensor = torch.from_numpy(mels)
        if self.cache_features:
            self._feature_cache[idx] = tensor
        return tensor, label

    def class_balance(self) -> dict[str, int]:
        absent = sum(1 for *_, label in self._records if label == 0)
        present = len(self._records) - absent
        return {"absent": absent, "present": present}


__all__ = [
    "CirCorMurmurDataset",
    "CirCorRecordingMurmurDataset",
    "CirCorSample",
    "window_hop_samples",
]
