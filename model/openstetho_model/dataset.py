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

import copy
import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
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

AGE_BUCKETS = ("Neonate", "Infant", "Child", "Adolescent", "Young Adult")
SEX_BUCKETS = ("Female", "Male")
RECORDING_LOCATIONS = ("AV", "MV", "PV", "TV", "Phc")
WIDE_FEATURE_NAMES = (
    *(f"age_{age.lower().replace(' ', '_')}" for age in AGE_BUCKETS),
    "age_unknown",
    *(f"sex_{sex.lower()}" for sex in SEX_BUCKETS),
    "sex_unknown",
    "pregnancy_status",
    "height_cm",
    "height_missing",
    "weight_kg",
    "weight_missing",
    *(f"location_{loc.lower()}" for loc in RECORDING_LOCATIONS),
    "duration_s",
    "rms",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
)
RecordingItem = (
    tuple[torch.Tensor, int]
    | tuple[torch.Tensor, torch.Tensor, int]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
)


@dataclass
class CirCorSample:
    """One fixed-length window with a label."""
    patient_id: int
    location: str
    label: int       # 0 = Absent, 1 = Present
    audio: np.ndarray  # float32, cardiac-filtered, 4 kHz


@dataclass(frozen=True)
class MurmurAugmentationConfig:
    """Train-time augmentation knobs for CirCor murmur experiments."""

    audio_noise_snr_db: float | None = None
    audio_noise_prob: float = 0.0
    random_crop: bool = False
    window_jitter_seconds: float = 0.0
    time_shift_seconds: float = 0.0
    time_shift_prob: float = 0.0
    freq_mask_max_width: int = 0
    time_mask_max_width: int = 0

    @property
    def enabled(self) -> bool:
        return (
            (self.audio_noise_snr_db is not None and self.audio_noise_prob > 0.0)
            or self.random_crop
            or self.window_jitter_seconds > 0.0
            or (self.time_shift_seconds > 0.0 and self.time_shift_prob > 0.0)
            or self.freq_mask_max_width > 0
            or self.time_mask_max_width > 0
        )


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
        augmentation: MurmurAugmentationConfig | None = None,
        augment_seed: int | None = None,
    ):
        self.root = Path(root)
        self.window_samples, self.hop_samples = window_hop_samples(window_seconds, hop_seconds)
        # When False, audio goes straight to mel-spec with no biquad chain.
        # Matches the stetho-ui inference path which now also runs the model
        # on raw audio so training and deployment share the same preprocess.
        self.apply_cardiac_filter = apply_cardiac
        self.feature_mode = feature_mode
        self.augmentation = augmentation or MurmurAugmentationConfig()
        self._augment_rng = np.random.default_rng(augment_seed)

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
        window = _window_at_index(
            audio,
            w_idx,
            self.window_samples,
            self.hop_samples,
            self.augmentation,
            self._augment_rng,
        )
        window = _augment_audio(window, self.augmentation, self._augment_rng)
        mel = window_features(window, self.feature_mode)
        mel = _augment_features(mel, self.augmentation, self._augment_rng)
        return torch.from_numpy(mel), label

    def with_augmentation(
        self,
        augmentation: MurmurAugmentationConfig | None,
        augment_seed: int | None = None,
    ) -> "CirCorMurmurDataset":
        out = copy.copy(self)
        out.augmentation = augmentation or MurmurAugmentationConfig()
        out._augment_rng = np.random.default_rng(augment_seed)
        return out

    def class_balance(self) -> dict[str, int]:
        absent = sum(1 for *_, label, _ in self._index if label == 0)
        present = len(self._index) - absent
        return {"absent": absent, "present": present}


class CirCorRecordingMurmurDataset(Dataset[RecordingItem]):
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
        augmentation: MurmurAugmentationConfig | None = None,
        augment_seed: int | None = None,
        include_wide_features: bool = False,
        feature_cache_dir: str | Path | None = None,
        teacher_targets: dict[tuple[int, str], float] | None = None,
    ):
        self.root = Path(root)
        self.window_samples, self.hop_samples = window_hop_samples(window_seconds, hop_seconds)
        self.apply_cardiac_filter = apply_cardiac
        self.feature_mode = feature_mode
        self.cache_features = cache_features
        self._feature_cache: dict[int, torch.Tensor] = {}
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir is not None else None
        self.include_wide_features = include_wide_features
        self.teacher_targets = teacher_targets
        self._wide_cache: dict[int, np.ndarray] = {}
        self._wide_mean: np.ndarray | None = None
        self._wide_std: np.ndarray | None = None
        self.augmentation = augmentation or MurmurAugmentationConfig()
        self._augment_rng = np.random.default_rng(augment_seed)
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
        self._patient_rows = {
            int(row["Patient ID"]): row
            for _, row in self.metadata.iterrows()
        }
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

    @property
    def wide_feature_names(self) -> tuple[str, ...]:
        return WIDE_FEATURE_NAMES if self.include_wide_features else ()

    @property
    def wide_feature_dim(self) -> int:
        return len(self.wide_feature_names)

    def _load_recording_audio(self, wav: Path) -> np.ndarray:
        audio = load_audio(str(wav))
        if self._profile_fir is not None:
            import scipy.signal as sps
            audio = sps.lfilter(self._profile_fir, [1.0], audio).astype(np.float32)
        if self.apply_cardiac_filter:
            audio = apply_cardiac(audio)
        return audio

    def fit_wide_normalization(self, indices: Sequence[int]) -> None:
        if not self.include_wide_features:
            return
        if not indices:
            raise ValueError("cannot fit wide feature normalization on an empty split")
        features = np.stack([self._raw_wide_features(int(i)) for i in indices], axis=0)
        self._wide_mean = features.mean(axis=0).astype(np.float32)
        std = features.std(axis=0).astype(np.float32)
        self._wide_std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def _raw_wide_features(self, idx: int) -> np.ndarray:
        if idx not in self._wide_cache:
            patient_id, loc, wav, _label = self._records[idx]
            row = self._patient_rows[patient_id]
            audio = self._load_recording_audio(wav)
            self._wide_cache[idx] = wide_feature_vector(row, loc, audio)
        return self._wide_cache[idx]

    def _wide_features(self, idx: int) -> torch.Tensor:
        features = self._raw_wide_features(idx)
        if self._wide_mean is not None and self._wide_std is not None:
            features = (features - self._wide_mean) / self._wide_std
        return torch.from_numpy(features.astype(np.float32, copy=False))

    def _format_item(
        self,
        idx: int,
        tensor: torch.Tensor,
        label: int,
    ) -> RecordingItem:
        patient_id, loc, _wav, _label = self._records[idx]
        teacher = None
        if self.teacher_targets is not None:
            teacher = torch.tensor(
                self.teacher_targets.get((patient_id, loc), float("nan")),
                dtype=torch.float32,
            )
        if self.include_wide_features:
            wide = self._wide_features(idx)
            if teacher is not None:
                return tensor, wide, teacher, label
            return tensor, wide, label
        if teacher is not None:
            return tensor, teacher, label
        return tensor, label

    def __getitem__(
        self,
        idx: int,
    ) -> RecordingItem:
        _patient_id, _loc, wav, label = self._records[idx]
        if self.cache_features and idx in self._feature_cache:
            tensor = self._feature_cache[idx]
            return self._format_item(idx, tensor, label)
        cache_path = self._feature_cache_path(idx)
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path) as data:
                tensor = torch.from_numpy(data["features"].astype(np.float32, copy=False))
            if self.cache_features:
                self._feature_cache[idx] = tensor
            return self._format_item(idx, tensor, label)
        audio = self._load_recording_audio(wav)
        windows = _recording_windows(
            audio,
            self.window_samples,
            self.hop_samples,
            self.augmentation,
            self._augment_rng,
        )
        if len(windows) == 0:
            # Keep the model path defined for very short clips.
            padded = np.zeros(self.window_samples, dtype=np.float32)
            padded[: min(len(audio), self.window_samples)] = audio[: self.window_samples]
            windows = np.expand_dims(padded, axis=0)
        mels = np.stack([
            _augment_features(
                window_features(_augment_audio(w, self.augmentation, self._augment_rng), self.feature_mode),
                self.augmentation,
                self._augment_rng,
            )
            for w in windows
        ], axis=0)
        tensor = torch.from_numpy(mels)
        if self.cache_features and not self.augmentation.enabled:
            self._feature_cache[idx] = tensor
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")
            np.savez(tmp_path, features=mels.astype(np.float32, copy=False))
            tmp_path.replace(cache_path)
        return self._format_item(idx, tensor, label)

    def _feature_cache_path(self, idx: int) -> Path | None:
        if self.feature_cache_dir is None or self.augmentation.enabled:
            return None
        _patient_id, _loc, wav, _label = self._records[idx]
        profile = "profile" if self._profile_fir is not None else "no_profile"
        key = "|".join([
            str(self.root.resolve()),
            str(wav.resolve()),
            self.feature_mode,
            str(self.window_samples),
            str(self.hop_samples),
            "cardiac" if self.apply_cardiac_filter else "raw",
            profile,
            "v1",
        ])
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.feature_cache_dir / self.feature_mode / f"{digest}.npz"

    def with_augmentation(
        self,
        augmentation: MurmurAugmentationConfig | None,
        augment_seed: int | None = None,
    ) -> "CirCorRecordingMurmurDataset":
        out = copy.copy(self)
        out.augmentation = augmentation or MurmurAugmentationConfig()
        out._augment_rng = np.random.default_rng(augment_seed)
        out.cache_features = False
        out._feature_cache = {}
        out._wide_cache = self._wide_cache
        out.feature_cache_dir = None
        return out

    def class_balance(self) -> dict[str, int]:
        absent = sum(1 for *_, label in self._records if label == 0)
        present = len(self._records) - absent
        return {"absent": absent, "present": present}

    def teacher_target_coverage(self) -> dict[str, int]:
        if self.teacher_targets is None:
            return {"with_teacher": 0, "without_teacher": len(self._records)}
        with_teacher = sum(
            1
            for patient_id, loc, _wav, _label in self._records
            if (patient_id, loc) in self.teacher_targets
        )
        return {"with_teacher": with_teacher, "without_teacher": len(self._records) - with_teacher}


def load_recording_teacher_targets(
    csv_path: str | Path,
    prob_column: str = "onnx_prob",
    aggregation: Literal["mean", "max", "topk_mean"] = "max",
    topk: int = 3,
) -> dict[tuple[int, str], float]:
    """Load per-window teacher probabilities and aggregate to recording targets."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    required = {"patient_id", "location", prob_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"teacher predictions missing columns: {sorted(missing)}")
    df = df.dropna(subset=["patient_id", "location", prob_column]).copy()
    if df.empty:
        return {}

    targets: dict[tuple[int, str], float] = {}
    for (patient_id, loc), group in df.groupby(["patient_id", "location"]):
        values = group[prob_column].astype(float).to_numpy()
        if aggregation == "mean":
            value = float(np.mean(values))
        elif aggregation == "max":
            value = float(np.max(values))
        elif aggregation == "topk_mean":
            k = min(max(1, int(topk)), values.size)
            value = float(np.sort(values)[-k:].mean())
        else:
            raise ValueError(f"unknown teacher aggregation {aggregation!r}")
        targets[(int(patient_id), str(loc))] = float(np.clip(value, 0.0, 1.0))
    return targets


def _augment_audio(
    audio: np.ndarray,
    config: MurmurAugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    out = audio
    if config.time_shift_seconds > 0.0 and rng.random() < config.time_shift_prob:
        max_shift = int(round(config.time_shift_seconds * SAMPLE_RATE))
        if max_shift > 0:
            shift = int(rng.integers(-max_shift, max_shift + 1))
            if shift:
                out = np.roll(out, shift)

    if config.audio_noise_snr_db is not None and rng.random() < config.audio_noise_prob:
        out = out.astype(np.float32, copy=True)
        signal_power = float(np.mean(out**2))
        if signal_power > 1e-12:
            noise_power = signal_power / (10.0 ** (config.audio_noise_snr_db / 10.0))
            noise = rng.normal(0.0, np.sqrt(noise_power), size=out.shape).astype(np.float32)
            out = out + noise
    return out.astype(np.float32, copy=False)


def _window_at_index(
    audio: np.ndarray,
    window_idx: int,
    window_samples: int,
    hop_samples: int,
    config: MurmurAugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(audio) < window_samples:
        return np.zeros((0, window_samples), dtype=audio.dtype)[0]
    if config.random_crop:
        start = int(rng.integers(0, len(audio) - window_samples + 1))
        return audio[start : start + window_samples]
    base_start = window_idx * hop_samples
    if config.window_jitter_seconds > 0.0:
        max_jitter = int(round(config.window_jitter_seconds * SAMPLE_RATE))
        if max_jitter > 0:
            base_start += int(rng.integers(-max_jitter, max_jitter + 1))
    start = min(max(0, base_start), len(audio) - window_samples)
    return audio[start : start + window_samples]


def _recording_windows(
    audio: np.ndarray,
    window_samples: int,
    hop_samples: int,
    config: MurmurAugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if config.random_crop:
        if len(audio) < window_samples:
            padded = np.zeros(window_samples, dtype=np.float32)
            padded[: min(len(audio), window_samples)] = audio[:window_samples]
            return np.expand_dims(padded, axis=0)
        return np.expand_dims(_window_at_index(audio, 0, window_samples, hop_samples, config, rng), axis=0)
    if len(audio) < window_samples or config.window_jitter_seconds <= 0.0:
        return split_windows(audio, window_samples, hop_samples)
    n_windows = 1 + (len(audio) - window_samples) // hop_samples
    return np.stack([
        _window_at_index(audio, i, window_samples, hop_samples, config, rng)
        for i in range(n_windows)
    ], axis=0)


def _augment_features(
    features: np.ndarray,
    config: MurmurAugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if config.freq_mask_max_width <= 0 and config.time_mask_max_width <= 0:
        return features
    out = features.copy()
    if out.ndim == 2:
        _mask_2d_feature(out, config, rng)
    elif out.ndim == 3:
        # Multi-channel features use shape (C, T, F). Apply the same mask
        # geometry to every channel so the branches stay time/frequency aligned.
        mask = np.ones(out.shape[1:], dtype=bool)
        _mask_2d_feature(mask, config, rng, fill=False)
        fill = float(out.min())
        out[:, ~mask] = fill
    return out.astype(np.float32, copy=False)


def _mask_2d_feature(
    feature: np.ndarray,
    config: MurmurAugmentationConfig,
    rng: np.random.Generator,
    fill: float | bool | None = None,
) -> None:
    if feature.ndim != 2:
        raise ValueError(f"expected 2D feature map, got shape {feature.shape}")
    fill_value = feature.min() if fill is None else fill
    if config.freq_mask_max_width > 0 and feature.shape[1] > 0:
        width = int(rng.integers(0, config.freq_mask_max_width + 1))
        if 0 < width < feature.shape[1]:
            start = int(rng.integers(0, feature.shape[1] - width + 1))
            feature[:, start : start + width] = fill_value
    if config.time_mask_max_width > 0 and feature.shape[0] > 0:
        width = int(rng.integers(0, config.time_mask_max_width + 1))
        if 0 < width < feature.shape[0]:
            start = int(rng.integers(0, feature.shape[0] - width + 1))
            feature[start : start + width, :] = fill_value


def wide_feature_vector(row: pd.Series, location: str, audio: np.ndarray) -> np.ndarray:
    """Build HearHeart-style side-channel features for one recording."""
    values: list[float] = []

    age = str(row.get("Age", "")).strip()
    values.extend(1.0 if age == bucket else 0.0 for bucket in AGE_BUCKETS)
    values.append(0.0 if age in AGE_BUCKETS else 1.0)

    sex = str(row.get("Sex", "")).strip()
    values.extend(1.0 if sex == bucket else 0.0 for bucket in SEX_BUCKETS)
    values.append(0.0 if sex in SEX_BUCKETS else 1.0)

    pregnancy = row.get("Pregnancy status", False)
    values.append(1.0 if bool(pregnancy) else 0.0)

    height = row.get("Height", np.nan)
    height_missing = float(pd.isna(height))
    values.extend([0.0 if height_missing else float(height), height_missing])

    weight = row.get("Weight", np.nan)
    weight_missing = float(pd.isna(weight))
    values.extend([0.0 if weight_missing else float(weight), weight_missing])

    values.extend(1.0 if location == loc else 0.0 for loc in RECORDING_LOCATIONS)
    values.extend(_recording_stat_features(audio).tolist())
    return np.asarray(values, dtype=np.float32)


def _recording_stat_features(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(5, dtype=np.float32)
    duration = audio.size / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(audio**2)))
    signs = np.signbit(audio)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if audio.size > 1 else 0.0
    window = np.hamming(audio.size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(audio * window)).astype(np.float64)
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / SAMPLE_RATE)
    weight = float(spectrum.sum())
    if weight <= 1e-12:
        centroid = 0.0
        bandwidth = 0.0
    else:
        centroid = float((freqs * spectrum).sum() / weight)
        bandwidth = float(np.sqrt((((freqs - centroid) ** 2) * spectrum).sum() / weight))
    return np.asarray([duration, rms, zcr, centroid, bandwidth], dtype=np.float32)


__all__ = [
    "CirCorMurmurDataset",
    "CirCorRecordingMurmurDataset",
    "CirCorSample",
    "MurmurAugmentationConfig",
    "WIDE_FEATURE_NAMES",
    "load_recording_teacher_targets",
    "wide_feature_vector",
    "window_hop_samples",
]
