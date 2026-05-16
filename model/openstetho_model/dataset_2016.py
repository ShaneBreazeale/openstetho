"""PhysioNet/CinC Challenge 2016 PyTorch Dataset — binary normal/abnormal
heart-sound classification.

Layout we expect (after `scripts/download_physionet_2016.sh`):

    data/physionet_2016/
        training-a/  …f/                  WAV + .hea, REFERENCE.csv per subset
        validation/
        annotations/                       (optional, segmentation labels)

REFERENCE.csv per subset:
    <recording_id>,<label>          # label is -1 (normal) or 1 (abnormal)

Recordings are sampled at 2 kHz natively; our pipeline upsamples to 4 kHz
in `preprocess.load_audio` so the feature shape matches the CirCor pipeline.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess import (
    N_MELS,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    apply_cardiac,
    load_audio,
    log_mel,
    split_windows,
)

log = logging.getLogger(__name__)

SUBSETS = ("training-a", "training-b", "training-c", "training-d", "training-e", "training-f")


@dataclass(frozen=True)
class IndexEntry:
    subset: str
    record_id: str
    wav: Path
    label: int      # 0 = normal, 1 = abnormal
    w_idx: int


class PhysioNet2016Dataset(Dataset[tuple[torch.Tensor, int]]):
    """Window-level binary classifier dataset for PhysioNet 2016."""

    def __init__(
        self,
        root: str | Path,
        subsets: Sequence[str] = SUBSETS,
        include_validation: bool = False,
        apply_cardiac: bool = True,
    ):
        self.root = Path(root)
        self.apply_cardiac_filter = apply_cardiac
        self._index: list[IndexEntry] = []

        # Map recording_id → subset (training-a..f) using the updated/
        # per-subset REFERENCE files. WAVs themselves are flat at root.
        id_to_subset: dict[str, str] = {}
        for subset in list(subsets) + (["validation"] if include_validation else []):
            sub_ref = self.root / "updated" / subset / "REFERENCE_withSQI.csv"
            if not sub_ref.exists():
                continue
            try:
                df = pd.read_csv(sub_ref, header=None, names=["id", "label", "sqi"])
            except Exception as e:  # noqa: BLE001
                log.warning("read %s: %s", sub_ref, e)
                continue
            for rid in df["id"].astype(str).str.strip():
                id_to_subset[rid] = subset

        # Primary labels come from the flat root REFERENCE.csv (id,label).
        root_ref = self.root / "REFERENCE.csv"
        if not root_ref.exists():
            raise FileNotFoundError(f"missing {root_ref}")
        df = pd.read_csv(root_ref, header=None, names=["id", "label"])

        keep_subsets = set(subsets)
        if include_validation:
            keep_subsets.add("validation")

        for _, row in df.iterrows():
            record_id = str(row["id"]).strip()
            raw_label = int(row["label"])
            if raw_label not in (-1, 1):
                continue
            subset = id_to_subset.get(record_id, "unknown")
            if subset != "unknown" and subset not in keep_subsets:
                continue
            label = 1 if raw_label == 1 else 0
            wav = self.root / f"{record_id}.wav"
            if not wav.exists():
                continue
            try:
                audio_len = self._wav_frame_count_at_4khz(wav)
            except Exception as e:  # noqa: BLE001
                log.warning("skip %s: %s", wav, e)
                continue
            n_windows = max(0, 1 + (audio_len - WINDOW_SAMPLES) // (WINDOW_SAMPLES // 2))
            for w_idx in range(n_windows):
                self._index.append(IndexEntry(subset, record_id, wav, label, w_idx))

    @staticmethod
    def _wav_frame_count_at_4khz(path: Path) -> int:
        import soundfile as sf
        info = sf.info(str(path))
        return int(info.frames * (SAMPLE_RATE / info.samplerate))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        e = self._index[idx]
        audio = load_audio(str(e.wav))
        if self.apply_cardiac_filter:
            audio = apply_cardiac(audio)
        windows = split_windows(audio)
        if e.w_idx >= len(windows):
            return torch.zeros((WINDOW_SAMPLES // 256, N_MELS), dtype=torch.float32), e.label
        mel = log_mel(windows[e.w_idx])
        return torch.from_numpy(mel), e.label

    def class_balance(self) -> dict[str, int]:
        n_abnormal = sum(1 for e in self._index if e.label == 1)
        return {"normal": len(self._index) - n_abnormal, "abnormal": n_abnormal}

    def subset_balance(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {s: {"normal": 0, "abnormal": 0} for s in SUBSETS}
        for e in self._index:
            bucket = "abnormal" if e.label == 1 else "normal"
            out.setdefault(e.subset, {"normal": 0, "abnormal": 0})[bucket] += 1
        return out


def make_combined_dataset(
    circor_root: str | Path,
    pn2016_root: str | Path,
) -> Dataset[tuple[torch.Tensor, int]]:
    """Concatenate CirCor (murmur present/absent) and PhysioNet 2016
    (normal/abnormal) into a single binary dataset. Useful for the
    "device-diverse pre-train" step.

    Caveat: the labels mean slightly different things — CirCor's
    'Present' is specifically a clinically-confirmed murmur, while
    PN2016's 'abnormal' is any cardiac pathology (valve defects, CAD,
    arrhythmia). Treat them as a noisy single label and rely on
    fine-tuning to recover device-specific signal."""
    from torch.utils.data import ConcatDataset
    from .dataset import CirCorMurmurDataset

    circor = CirCorMurmurDataset(circor_root)
    pn2016 = PhysioNet2016Dataset(pn2016_root)
    return ConcatDataset([circor, pn2016])


__all__ = ["PhysioNet2016Dataset", "IndexEntry", "make_combined_dataset"]
