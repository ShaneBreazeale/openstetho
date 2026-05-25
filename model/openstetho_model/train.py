"""Train MurmurCNN on CirCor 2022.

Usage (from `model/`):

    uv run python -m openstetho_model.train \
        --data ../data/circor \
        --epochs 30 \
        --batch-size 64 \
        --out runs/circor_v1
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .dataset import (
    CirCorMurmurDataset,
    CirCorRecordingMurmurDataset,
    MurmurAugmentationConfig,
    load_recording_teacher_targets,
    window_hop_samples,
)
from .model import (
    MurmurCNN,
    MurmurCNNBiGRU,
    MurmurScatteringCNN1D,
    MurmurScatteringTransformer,
    count_parameters,
)
from .preprocess import FEATURE_MODES, FEATURE_MODE_LOGMEL, SAMPLE_RATE, feature_channels
from .thresholds import (
    SPECIFICITY_POLICY_TARGETS,
    specificity_constrained_row,
    sweep_thresholds,
    threshold_policy_row,
)

log = logging.getLogger("train")
LOSS_TYPES = ("bce", "focal_bce", "asymmetric_focal")
LOSS_LOGIT_CLIP = 30.0
SELECT_METRICS = ("auc", "f1", "youden_j", *SPECIFICITY_POLICY_TARGETS.keys(), "specificity_target")


def build_model(architecture: str, in_channels: int = 1, wide_feature_dim: int = 0) -> nn.Module:
    if wide_feature_dim > 0 and architecture != "cnn_bigru":
        raise ValueError("wide features are currently supported only for architecture='cnn_bigru'")
    if architecture == "cnn":
        return MurmurCNN(in_channels=in_channels)
    if architecture == "cnn_bigru":
        return MurmurCNNBiGRU(in_channels=in_channels, wide_feature_dim=wide_feature_dim)
    if architecture == "scattering_cnn1d":
        return MurmurScatteringCNN1D()
    if architecture == "scattering_transformer":
        return MurmurScatteringTransformer()
    raise ValueError(f"unknown architecture {architecture}")


def add_augmentation_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--audio-noise-snr-db",
        type=float,
        default=None,
        help="train-only Gaussian noise SNR in dB; unset disables audio noise",
    )
    p.add_argument(
        "--audio-noise-prob",
        type=float,
        default=0.0,
        help="probability of applying train-only audio noise",
    )
    p.add_argument(
        "--train-random-crop",
        action="store_true",
        help="for recording-level training, use one random window per recording; validation remains overlapped",
    )
    p.add_argument(
        "--window-jitter-seconds",
        type=float,
        default=0.0,
        help="train-only random jitter for fixed window/crop start times",
    )
    p.add_argument(
        "--time-shift-seconds",
        type=float,
        default=0.0,
        help="train-only circular time-shift jitter in seconds",
    )
    p.add_argument(
        "--time-shift-prob",
        type=float,
        default=0.0,
        help="probability of applying train-only time shift",
    )
    p.add_argument(
        "--freq-mask-max-width",
        type=int,
        default=0,
        help="train-only SpecAugment frequency-mask max width in feature bins",
    )
    p.add_argument(
        "--time-mask-max-width",
        type=int,
        default=0,
        help="train-only SpecAugment time-mask max width in frames",
    )


def add_teacher_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--teacher-predictions-csv",
        type=Path,
        default=None,
        help="optional per-window teacher predictions CSV for recording-level soft-target training",
    )
    p.add_argument(
        "--teacher-prob-column",
        default="onnx_prob",
        help="teacher probability column in --teacher-predictions-csv",
    )
    p.add_argument(
        "--teacher-aggregation",
        choices=["mean", "max", "topk_mean"],
        default="max",
        help="aggregate per-window teacher probabilities to a recording-level soft target",
    )
    p.add_argument("--teacher-topk", type=int, default=3)
    p.add_argument(
        "--teacher-distill-weight",
        type=float,
        default=0.0,
        help="extra BCE weight for finite recording-level teacher soft targets; 0 disables",
    )


def augmentation_config_from_args(args: argparse.Namespace) -> MurmurAugmentationConfig:
    return MurmurAugmentationConfig(
        audio_noise_snr_db=args.audio_noise_snr_db,
        audio_noise_prob=args.audio_noise_prob,
        random_crop=args.train_random_crop,
        window_jitter_seconds=args.window_jitter_seconds,
        time_shift_seconds=args.time_shift_seconds,
        time_shift_prob=args.time_shift_prob,
        freq_mask_max_width=args.freq_mask_max_width,
        time_mask_max_width=args.time_mask_max_width,
    )


def apply_train_augmentation(dataset, config: MurmurAugmentationConfig, seed: int):
    if not config.enabled:
        return dataset
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "with_augmentation"):
        return Subset(dataset.dataset.with_augmentation(config, seed), dataset.indices)
    if hasattr(dataset, "with_augmentation"):
        return dataset.with_augmentation(config, seed)
    return dataset


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.clamp(-LOSS_LOGIT_CLIP, LOSS_LOGIT_CLIP)
        labels = labels.float()
        bce = stable_bce_with_logits(logits, labels, self.pos_weight)
        probs = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probs, 1.0 - probs)
        return (((1.0 - pt).clamp_min(1e-6) ** self.gamma) * bce).mean()


class AsymmetricFocalBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
    ):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.clamp(-LOSS_LOGIT_CLIP, LOSS_LOGIT_CLIP)
        labels = labels.float()
        bce = stable_bce_with_logits(logits, labels, self.pos_weight)
        probs = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probs, 1.0 - probs)
        gamma = torch.where(
            labels == 1,
            torch.full_like(labels, self.gamma_pos),
            torch.full_like(labels, self.gamma_neg),
        )
        return (((1.0 - pt).clamp_min(1e-6) ** gamma) * bce).mean()


def build_loss_fn(
    loss: str,
    pos_weight: torch.Tensor,
    focal_gamma: float = 2.0,
    asymmetric_gamma_pos: float = 0.0,
    asymmetric_gamma_neg: float = 2.0,
) -> nn.Module:
    if loss == "bce":
        return StableBCEWithLogitsLoss(pos_weight=pos_weight)
    if loss == "focal_bce":
        return FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=focal_gamma)
    if loss == "asymmetric_focal":
        return AsymmetricFocalBCEWithLogitsLoss(
            pos_weight=pos_weight,
            gamma_pos=asymmetric_gamma_pos,
            gamma_neg=asymmetric_gamma_neg,
        )
    raise ValueError(f"unknown loss {loss!r}; valid: {LOSS_TYPES}")


class StableBCEWithLogitsLoss(nn.Module):
    """BCEWithLogits with defensive logit clipping for long-window MPS runs."""

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.clamp(-LOSS_LOGIT_CLIP, LOSS_LOGIT_CLIP)
        return stable_bce_with_logits(logits, labels.float(), self.pos_weight).mean()


def stable_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable BCE-with-logits formula with positive weighting."""
    max_val = (-logits).clamp_min(0)
    log_weight = 1.0 + (pos_weight - 1.0) * labels
    log_exp = torch.log(torch.exp(-max_val) + torch.exp(-logits - max_val))
    return (1.0 - labels) * logits + log_weight * (max_val + log_exp)


def labels_for_dataset(dataset) -> list[int]:
    if isinstance(dataset, Subset):
        base_labels = labels_for_dataset(dataset.dataset)
        return [base_labels[int(i)] for i in dataset.indices]
    if isinstance(dataset, CirCorRecordingMurmurDataset):
        return [int(label) for *_rest, label in dataset._records]
    if isinstance(dataset, CirCorMurmurDataset):
        return [int(label) for *_rest, label, _w in dataset._index]
    return [int(dataset[i][1]) for i in range(len(dataset))]


def positive_weighted_sampler(
    dataset,
    positive_sample_weight: float,
    seed: int,
) -> WeightedRandomSampler | None:
    if positive_sample_weight <= 1.0:
        return None
    labels = labels_for_dataset(dataset)
    weights = torch.tensor(
        [positive_sample_weight if label == 1 else 1.0 for label in labels],
        dtype=torch.double,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def patient_split(ds: CirCorMurmurDataset, val_fraction: float = 0.15, seed: int = 0):
    """Group-aware split: every window of a given patient lands in one side
    of the split. Prevents leakage at the patient level."""
    patient_labels = _patient_labels([(pid, label) for pid, *_rest, label, _w in ds._index])
    train_ids, val_ids = _stratified_patient_ids(patient_labels, val_fraction, seed)
    train_idx = [i for i, (pid, *_) in enumerate(ds._index) if pid in train_ids]
    val_idx = [i for i, (pid, *_) in enumerate(ds._index) if pid in val_ids]
    return Subset(ds, train_idx), Subset(ds, val_idx)


def _patient_labels(rows: list[tuple[int, int]]) -> dict[int, int]:
    labels: dict[int, int] = {}
    for pid, label in rows:
        prior = labels.setdefault(pid, label)
        if prior != label:
            raise ValueError(f"patient {pid} has mixed murmur labels")
    return labels


def _stratified_patient_ids(
    patient_labels: dict[int, int],
    val_fraction: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    rng = np.random.default_rng(seed)
    train_ids: set[int] = set()
    val_ids: set[int] = set()
    for label in (0, 1):
        ids = sorted(pid for pid, y in patient_labels.items() if y == label)
        rng.shuffle(ids)
        n_val = max(1, int(len(ids) * val_fraction)) if ids else 0
        val_ids.update(ids[:n_val])
        train_ids.update(ids[n_val:])
    return train_ids, val_ids


def recording_patient_split(
    ds: CirCorRecordingMurmurDataset,
    val_fraction: float = 0.15,
    seed: int = 0,
):
    """Patient-disjoint split for recording-level training."""
    patient_labels = _patient_labels([(pid, label) for pid, _loc, _wav, label in ds._records])
    train_ids, val_ids = _stratified_patient_ids(patient_labels, val_fraction, seed)
    train_idx = [i for i, (pid, *_) in enumerate(ds._records) if pid in train_ids]
    val_idx = [i for i, (pid, *_) in enumerate(ds._records) if pid in val_ids]
    return Subset(ds, train_idx), Subset(ds, val_idx)


def recording_collate(batch):
    if len(batch[0]) == 4:
        recordings = [mel for mel, _wide, _teacher, _label in batch]
        wides = torch.stack([wide for _mel, wide, _teacher, _label in batch])
        teachers = torch.stack([teacher for _mel, _wide, teacher, _label in batch])
        labels = torch.tensor([label for _mel, _wide, _teacher, label in batch], dtype=torch.float32)
        return recordings, wides, teachers, labels
    if len(batch[0]) == 3 and torch.as_tensor(batch[0][1]).ndim > 0:
        recordings = [mel for mel, _wide, _label in batch]
        wides = torch.stack([wide for _mel, wide, _label in batch])
        labels = torch.tensor([label for _mel, _wide, label in batch], dtype=torch.float32)
        return recordings, wides, labels
    if len(batch[0]) == 3:
        recordings = [mel for mel, _teacher, _label in batch]
        teachers = torch.stack([teacher for _mel, teacher, _label in batch])
        labels = torch.tensor([label for _mel, _teacher, label in batch], dtype=torch.float32)
        return recordings, teachers, labels
    recordings = [mel for mel, _label in batch]
    labels = torch.tensor([label for _mel, label in batch], dtype=torch.float32)
    return recordings, labels


def unpack_recording_batch(
    batch,
    dev: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if len(batch) == 4:
        recordings, wides, teachers, labels = batch
        return (
            recordings,
            labels.to(dev, non_blocking=True),
            wides.to(dev, non_blocking=True),
            teachers.to(dev, non_blocking=True),
        )
    if len(batch) == 3:
        recordings, aux, labels = batch
        aux = aux.to(dev, non_blocking=True)
        labels = labels.to(dev, non_blocking=True)
        if aux.ndim == 1:
            return recordings, labels, None, aux
        return recordings, labels, aux, None
    recordings, labels = batch
    return recordings, labels.to(dev, non_blocking=True), None, None


def aggregate_logits(logits: torch.Tensor, mode: str, topk: int) -> torch.Tensor:
    if logits.ndim != 1:
        logits = logits.reshape(-1)
    if mode == "mean":
        return logits.mean()
    if mode == "max":
        return logits.max()
    if mode == "topk_mean":
        k = min(max(1, topk), logits.numel())
        return logits.topk(k).values.mean()
    raise ValueError(f"unknown aggregation mode {mode}")


def recording_batch_logits(
    model: nn.Module,
    recordings: list[torch.Tensor],
    dev: torch.device,
    aggregation: str,
    topk: int,
    wides: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score a variable-window recording batch with one model call.

    `recording_collate` keeps recordings as a list because each recording can
    have a different number of windows. Concatenating all windows avoids one
    model invocation per recording, then `split` restores recording boundaries
    before mean/max/top-k aggregation.
    """
    counts = [int(mel.shape[0]) for mel in recordings]
    if any(count <= 0 for count in counts):
        raise ValueError(f"recording batch contains an empty recording: {counts}")
    all_windows = torch.cat([mel.to(dev, non_blocking=True) for mel in recordings], dim=0)
    if wides is None:
        window_logits = model(all_windows)
    else:
        repeated_wides = torch.repeat_interleave(wides, torch.tensor(counts, device=wides.device), dim=0)
        window_logits = model(all_windows, repeated_wides)
    per_recording = torch.split(window_logits.reshape(-1), counts)
    return torch.stack([aggregate_logits(logits, aggregation, topk) for logits in per_recording])


def run_epoch(
    model: nn.Module,
    loader: Iterable,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer | None,
    dev: torch.device,
    grad_clip_norm: float = 0.0,
) -> tuple[float, float]:
    model.train(opt is not None)
    losses: list[float] = []
    probs: list[float] = []
    labels: list[float] = []
    for mel, label in loader:
        batch_labels = [int(round(x)) for x in label.numpy().tolist()]
        mel = mel.to(dev, non_blocking=True)
        label = label.to(dev, non_blocking=True).float()
        with torch.set_grad_enabled(opt is not None):
            logit = model(mel)
            loss = loss_fn(logit, label)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                opt.step()
        losses.append(loss.item())
        probs.extend(torch.sigmoid(logit).detach().cpu().numpy().tolist())
        labels.extend(batch_labels)
    auc = float("nan")
    if len(set(labels)) == 2:
        auc = roc_auc_score(labels, probs)
    return float(np.mean(losses)), auc


def run_recording_epoch(
    model: nn.Module,
    loader: Iterable,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer | None,
    dev: torch.device,
    aggregation: str,
    topk: int,
    grad_clip_norm: float = 0.0,
    teacher_distill_weight: float = 0.0,
) -> tuple[float, float]:
    model.train(opt is not None)
    losses: list[float] = []
    probs: list[float] = []
    labels_out: list[float] = []
    teacher_pos_weight = torch.tensor([1.0], device=dev)
    for batch in loader:
        recordings, labels, wides, teachers = unpack_recording_batch(batch, dev)
        batch_labels = [int(round(x)) for x in labels.detach().cpu().numpy().tolist()]
        with torch.set_grad_enabled(opt is not None):
            batch_logits = recording_batch_logits(
                model,
                recordings,
                dev,
                aggregation,
                topk,
                wides=wides,
            )
            loss = loss_fn(batch_logits, labels)
            if (
                opt is not None
                and teacher_distill_weight > 0.0
                and teachers is not None
            ):
                mask = torch.isfinite(teachers)
                if mask.any():
                    teacher_loss = stable_bce_with_logits(
                        batch_logits[mask],
                        teachers[mask],
                        teacher_pos_weight,
                    ).mean()
                    loss = loss + teacher_distill_weight * teacher_loss
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                opt.step()
        losses.append(loss.item())
        probs.extend(torch.sigmoid(batch_logits).detach().cpu().numpy().tolist())
        labels_out.extend(batch_labels)
    auc = float("nan")
    if len(set(labels_out)) == 2:
        auc = roc_auc_score(labels_out, probs)
    return float(np.mean(losses)), auc


def predict_windows(
    model: nn.Module,
    loader: Iterable,
    dev: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for mel, label in loader:
            mel = mel.to(dev, non_blocking=True)
            logit = model(mel)
            probs.extend(torch.sigmoid(logit).detach().cpu().numpy().tolist())
            labels.extend(int(round(x)) for x in label.numpy().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(probs, dtype=np.float64)


def predict_recordings(
    model: nn.Module,
    loader: Iterable,
    dev: torch.device,
    aggregation: str,
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[float] = []
    labels_out: list[int] = []
    with torch.no_grad():
        for batch in loader:
            recordings, labels, wides, _teachers = unpack_recording_batch(batch, dev)
            logits = recording_batch_logits(
                model,
                recordings,
                dev,
                aggregation,
                topk,
                wides=wides,
            )
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            labels_out.extend(int(round(x)) for x in labels.detach().cpu().numpy().tolist())
    return np.asarray(labels_out, dtype=np.int64), np.asarray(probs, dtype=np.float64)


def validation_selection(
    labels: np.ndarray,
    probs: np.ndarray,
    metric: str,
    specificity_target: float = 0.95,
) -> tuple[float, dict[str, float]]:
    rows = sweep_thresholds(labels, probs)
    if metric == "f1":
        row = threshold_policy_row(rows, "best_f1")
        return float(row["f1"]), row
    if metric == "youden_j":
        row = threshold_policy_row(rows, "best_youden_j")
        return float(row["youden_j"]), row
    if metric in SPECIFICITY_POLICY_TARGETS:
        row = threshold_policy_row(rows, metric)
        if row.get("constraint_met", 0.0) >= 1.0:
            return float(row["sensitivity"]), row
        return float(row["specificity"]) - 1.0, row
    if metric == "specificity_target":
        row = specificity_constrained_row(rows, specificity_target)
        if row.get("constraint_met", 0.0) >= 1.0:
            return float(row["sensitivity"]), row
        return float(row["specificity"]) - 1.0, row
    raise ValueError(f"unknown select metric {metric!r}; valid: {SELECT_METRICS}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="CirCor root (training_data.csv lives here)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="optional on-disk recording feature cache for expensive feature modes",
    )
    p.add_argument("--out", type=Path, default=Path("runs/last"))
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window-seconds", type=float, default=4.0)
    p.add_argument(
        "--hop-seconds",
        type=float,
        default=None,
        help="window hop in seconds; defaults to 50 percent overlap",
    )
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--level", choices=["window", "recording"], default="window")
    p.add_argument(
        "--architecture",
        choices=["cnn", "cnn_bigru", "scattering_cnn1d", "scattering_transformer"],
        default="cnn",
    )
    p.add_argument(
        "--wide-features",
        action="store_true",
        help="add recording-level metadata/audio-stat feature branch; requires --level recording and cnn_bigru",
    )
    p.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default=FEATURE_MODE_LOGMEL,
        help="input representation: logmel production baseline or multi-channel research stack",
    )
    p.add_argument("--aggregation", choices=["mean", "topk_mean", "max"], default="mean")
    p.add_argument("--topk", type=int, default=3, help="k for --aggregation topk_mean")
    p.add_argument("--loss", choices=LOSS_TYPES, default="bce")
    p.add_argument(
        "--select-metric",
        choices=SELECT_METRICS,
        default="auc",
        help=(
            "checkpoint selection metric; specificity_* maximizes validation "
            "sensitivity among thresholds with the requested minimum specificity"
        ),
    )
    p.add_argument(
        "--select-specificity-target",
        type=float,
        default=0.95,
        help="minimum specificity for --select-metric specificity_target",
    )
    p.add_argument(
        "--pos-weight-multiplier",
        type=float,
        default=1.0,
        help="multiply the class-balanced BCE/focal positive weight",
    )
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--asymmetric-gamma-pos", type=float, default=0.0)
    p.add_argument("--asymmetric-gamma-neg", type=float, default=2.0)
    p.add_argument(
        "--positive-sample-weight",
        type=float,
        default=1.0,
        help="if >1, use replacement sampling that draws positive recordings/windows more often",
    )
    p.add_argument(
        "--lr-scheduler",
        choices=["none", "plateau"],
        default="none",
        help="optional validation-selection ReduceLROnPlateau scheduler",
    )
    p.add_argument("--plateau-patience", type=int, default=2)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="stop after N epochs without validation selection-metric improvement; 0 disables",
    )
    p.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    p.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="clip gradient norm after backward; 0 disables",
    )
    p.add_argument(
        "--no-cardiac",
        action="store_true",
        help="skip legacy cardiac filter; matches current stetho-ui preprocessing",
    )
    add_augmentation_args(p)
    add_teacher_args(p)
    args = p.parse_args()
    if args.wide_features and args.level != "recording":
        p.error("--wide-features requires --level recording")
    if args.wide_features and args.architecture != "cnn_bigru":
        p.error("--wide-features currently requires --architecture cnn_bigru")
    if args.teacher_predictions_csv is not None and args.level != "recording":
        p.error("--teacher-predictions-csv requires --level recording")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    dev = device(args.device)
    torch.manual_seed(args.seed)
    log.info("device: %s", dev)
    window_samples, hop_samples = window_hop_samples(args.window_seconds, args.hop_seconds)
    hop_seconds = hop_samples / SAMPLE_RATE
    log.info("window: %.3fs (%d samples) hop %.3fs (%d samples)",
             args.window_seconds, window_samples, hop_seconds, hop_samples)

    if args.level == "recording":
        teacher_targets = None
        if args.teacher_predictions_csv is not None:
            teacher_targets = load_recording_teacher_targets(
                args.teacher_predictions_csv,
                prob_column=args.teacher_prob_column,
                aggregation=args.teacher_aggregation,
                topk=args.teacher_topk,
            )
        ds = CirCorRecordingMurmurDataset(
            args.data,
            apply_cardiac=not args.no_cardiac,
            feature_mode=args.feature_mode,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            include_wide_features=args.wide_features,
            feature_cache_dir=args.feature_cache_dir,
            teacher_targets=teacher_targets,
        )
        log.info("dataset: %d recordings | balance: %s", len(ds), ds.class_balance())
        if teacher_targets is not None:
            log.info("teacher target coverage: %s", ds.teacher_target_coverage())
        train_ds, val_ds = recording_patient_split(ds, args.val_fraction, args.seed)
        if args.wide_features:
            ds.fit_wide_normalization(list(train_ds.indices))
        log.info(
            "split:  train=%d  val=%d  aggregation=%s topk=%d",
            len(train_ds),
            len(val_ds),
            args.aggregation,
            args.topk,
        )
    else:
        ds = CirCorMurmurDataset(
            args.data,
            apply_cardiac=not args.no_cardiac,
            feature_mode=args.feature_mode,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
        )
        log.info("dataset: %d windows | balance: %s", len(ds), ds.class_balance())
        train_ds, val_ds = patient_split(ds, args.val_fraction, args.seed)
        log.info("split:  train=%d  val=%d", len(train_ds), len(val_ds))

    augmentation = augmentation_config_from_args(args)
    train_ds = apply_train_augmentation(train_ds, augmentation, args.seed)
    if augmentation.enabled:
        log.info("train-only augmentation: %s", augmentation)

    pin = dev.type in ("cuda",)
    sampler = positive_weighted_sampler(train_ds, args.positive_sample_weight, args.seed)
    shuffle_train = sampler is None
    if args.level == "recording":
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=shuffle_train,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=pin,
            drop_last=True,
            collate_fn=recording_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin,
            collate_fn=recording_collate,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=shuffle_train, sampler=sampler,
            num_workers=args.workers, pin_memory=pin, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=pin,
        )

    channels = feature_channels(args.feature_mode)
    wide_feature_dim = ds.wide_feature_dim if args.level == "recording" and args.wide_features else 0
    model = build_model(args.architecture, in_channels=channels, wide_feature_dim=wide_feature_dim).to(dev)
    log.info("features: %s channels=%d | params: %d", args.feature_mode, channels, count_parameters(model))
    # Class-balanced loss: pos_weight = N_neg / N_pos so the minority
    # (murmur-present) class gets a proportional gradient pull.
    bal = ds.class_balance()
    pos_weight_val = max(1.0, bal["absent"] / max(bal["present"], 1)) * args.pos_weight_multiplier
    log.info(
        "loss=%s pos_weight=%.3f positive_sample_weight=%.3f",
        args.loss,
        pos_weight_val,
        args.positive_sample_weight,
    )
    pos_weight = torch.tensor([pos_weight_val], device=dev)
    loss_fn = build_loss_fn(
        args.loss,
        pos_weight,
        focal_gamma=args.focal_gamma,
        asymmetric_gamma_pos=args.asymmetric_gamma_pos,
        asymmetric_gamma_neg=args.asymmetric_gamma_neg,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
        )

    best_auc = float("-inf")
    best_select = float("-inf")
    best_select_row: dict[str, float] = {}
    saved_best = False
    epochs_since_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        if args.level == "recording":
            tr_loss, tr_auc = run_recording_epoch(
                model,
                train_loader,
                loss_fn,
                opt,
                dev,
                args.aggregation,
                args.topk,
                args.grad_clip_norm,
                teacher_distill_weight=args.teacher_distill_weight,
            )
            va_loss, va_auc = run_recording_epoch(
                model, val_loader, loss_fn, None, dev, args.aggregation, args.topk
            )
        else:
            tr_loss, tr_auc = run_epoch(model, train_loader, loss_fn, opt, dev, args.grad_clip_norm)
            va_loss, va_auc = run_epoch(model, val_loader, loss_fn, None, dev)
        va_select = va_auc
        va_select_row: dict[str, float] = {}
        if args.select_metric != "auc":
            if args.level == "recording":
                va_labels, va_probs = predict_recordings(
                    model,
                    val_loader,
                    dev,
                    args.aggregation,
                    args.topk,
                )
            else:
                va_labels, va_probs = predict_windows(model, val_loader, dev)
            va_select, va_select_row = validation_selection(
                va_labels,
                va_probs,
                args.select_metric,
                args.select_specificity_target,
            )
        lr_now = float(opt.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "tr_loss": tr_loss,
            "tr_auc": tr_auc,
            "va_loss": va_loss,
            "va_auc": va_auc,
            "va_select": va_select,
            "va_select_threshold": va_select_row.get("threshold"),
            "va_select_sensitivity": va_select_row.get("sensitivity"),
            "va_select_specificity": va_select_row.get("specificity"),
            "va_select_f1": va_select_row.get("f1"),
            "va_select_constraint_met": va_select_row.get("constraint_met"),
            "lr": lr_now,
        })
        log.info(
            "ep %02d | tr loss %.4f auc %.3f | va loss %.4f auc %.3f select %.3f",
            epoch,
            tr_loss,
            tr_auc,
            va_loss,
            va_auc,
            va_select,
        )
        torch.save(model.state_dict(), args.out / "last.pt")
        improved = bool(
            np.isfinite(va_select)
            and va_select > best_select + args.early_stopping_min_delta
        )
        if improved or not saved_best:
            if np.isfinite(va_auc):
                best_auc = va_auc
            if np.isfinite(va_select):
                best_select = va_select
                best_select_row = dict(va_select_row)
            if improved:
                epochs_since_improvement = 0
            saved_best = True
            torch.save(model.state_dict(), args.out / "best.pt")
            (args.out / "best_meta.json").write_text(json.dumps({
                "epoch": epoch,
                "val_auc": va_auc,
                "select_metric": args.select_metric,
                "select_specificity_target": args.select_specificity_target,
                "select_score": va_select,
                "select_threshold": va_select_row,
                "level": args.level,
                "architecture": args.architecture,
                "feature_mode": args.feature_mode,
                "input_channels": channels,
                "wide_features": args.wide_features,
                "wide_feature_dim": wide_feature_dim,
                "wide_feature_names": list(ds.wide_feature_names)
                if args.level == "recording" and args.wide_features
                else [],
                "feature_cache_dir": str(args.feature_cache_dir)
                if args.feature_cache_dir is not None
                else None,
                "window_seconds": args.window_seconds,
                "hop_seconds": hop_seconds,
                "lr_scheduler": args.lr_scheduler,
                "early_stopping_patience": args.early_stopping_patience,
                "loss": args.loss,
                "pos_weight": pos_weight_val,
                "pos_weight_multiplier": args.pos_weight_multiplier,
                "focal_gamma": args.focal_gamma,
                "asymmetric_gamma_pos": args.asymmetric_gamma_pos,
                "asymmetric_gamma_neg": args.asymmetric_gamma_neg,
                "positive_sample_weight": args.positive_sample_weight,
                "grad_clip_norm": args.grad_clip_norm,
                "teacher_predictions_csv": str(args.teacher_predictions_csv)
                if args.teacher_predictions_csv is not None
                else None,
                "teacher_prob_column": args.teacher_prob_column,
                "teacher_aggregation": args.teacher_aggregation,
                "teacher_topk": args.teacher_topk,
                "teacher_distill_weight": args.teacher_distill_weight,
                "augmentation": augmentation.__dict__,
                "aggregation": args.aggregation if args.level == "recording" else None,
                "topk": args.topk if args.level == "recording" else None,
            }))
        else:
            epochs_since_improvement += 1

        if scheduler is not None:
            scheduler.step(va_select if np.isfinite(va_select) else -va_loss)

        if (
            args.early_stopping_patience > 0
            and epochs_since_improvement >= args.early_stopping_patience
        ):
            log.info(
                "early stopping after %d epochs without validation-%s improvement",
                epochs_since_improvement,
                args.select_metric,
            )
            break

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    best_auc_msg = f"{best_auc:.3f}" if np.isfinite(best_auc) else "nan"
    best_select_msg = f"{best_select:.3f}" if np.isfinite(best_select) else "nan"
    log.info(
        "best val AUC: %s | best %s: %s %s → %s/best.pt",
        best_auc_msg,
        args.select_metric,
        best_select_msg,
        best_select_row,
        args.out,
    )


if __name__ == "__main__":
    main()
