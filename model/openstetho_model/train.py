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
from torch.utils.data import DataLoader, Subset

from .dataset import CirCorMurmurDataset, CirCorRecordingMurmurDataset, window_hop_samples
from .model import MurmurCNN, MurmurCNNBiGRU, MurmurScatteringCNN1D, count_parameters
from .preprocess import FEATURE_MODES, FEATURE_MODE_LOGMEL, SAMPLE_RATE, feature_channels

log = logging.getLogger("train")


def build_model(architecture: str, in_channels: int = 1) -> nn.Module:
    if architecture == "cnn":
        return MurmurCNN(in_channels=in_channels)
    if architecture == "cnn_bigru":
        return MurmurCNNBiGRU(in_channels=in_channels)
    if architecture == "scattering_cnn1d":
        return MurmurScatteringCNN1D()
    raise ValueError(f"unknown architecture {architecture}")


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


def recording_collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[list[torch.Tensor], torch.Tensor]:
    recordings = [mel for mel, _ in batch]
    labels = torch.tensor([label for _, label in batch], dtype=torch.float32)
    return recordings, labels


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


def run_epoch(
    model: nn.Module,
    loader: Iterable,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer | None,
    dev: torch.device,
) -> tuple[float, float]:
    model.train(opt is not None)
    losses: list[float] = []
    probs: list[float] = []
    labels: list[float] = []
    for mel, label in loader:
        mel = mel.to(dev, non_blocking=True)
        label = label.to(dev, non_blocking=True).float()
        with torch.set_grad_enabled(opt is not None):
            logit = model(mel)
            loss = loss_fn(logit, label)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
        losses.append(loss.item())
        probs.extend(torch.sigmoid(logit).detach().cpu().numpy().tolist())
        labels.extend(label.detach().cpu().numpy().tolist())
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
) -> tuple[float, float]:
    model.train(opt is not None)
    losses: list[float] = []
    probs: list[float] = []
    labels_out: list[float] = []
    for recordings, labels in loader:
        labels = labels.to(dev, non_blocking=True)
        agg_logits: list[torch.Tensor] = []
        with torch.set_grad_enabled(opt is not None):
            for mel in recordings:
                mel = mel.to(dev, non_blocking=True)
                logits = model(mel)
                agg_logits.append(aggregate_logits(logits, aggregation, topk))
            batch_logits = torch.stack(agg_logits)
            loss = loss_fn(batch_logits, labels)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
        losses.append(loss.item())
        probs.extend(torch.sigmoid(batch_logits).detach().cpu().numpy().tolist())
        labels_out.extend(labels.detach().cpu().numpy().tolist())
    auc = float("nan")
    if len(set(labels_out)) == 2:
        auc = roc_auc_score(labels_out, probs)
    return float(np.mean(losses)), auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="CirCor root (training_data.csv lives here)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
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
    p.add_argument("--architecture", choices=["cnn", "cnn_bigru", "scattering_cnn1d"], default="cnn")
    p.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default=FEATURE_MODE_LOGMEL,
        help="input representation: logmel production baseline or multi-channel research stack",
    )
    p.add_argument("--aggregation", choices=["mean", "topk_mean", "max"], default="mean")
    p.add_argument("--topk", type=int, default=3, help="k for --aggregation topk_mean")
    p.add_argument(
        "--lr-scheduler",
        choices=["none", "plateau"],
        default="none",
        help="optional validation-AUC ReduceLROnPlateau scheduler",
    )
    p.add_argument("--plateau-patience", type=int, default=2)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="stop after N epochs without validation-AUC improvement; 0 disables",
    )
    p.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    p.add_argument(
        "--no-cardiac",
        action="store_true",
        help="skip legacy cardiac filter; matches current stetho-ui preprocessing",
    )
    args = p.parse_args()

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
        ds = CirCorRecordingMurmurDataset(
            args.data,
            apply_cardiac=not args.no_cardiac,
            feature_mode=args.feature_mode,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
        )
        log.info("dataset: %d recordings | balance: %s", len(ds), ds.class_balance())
        train_ds, val_ds = recording_patient_split(ds, args.val_fraction, args.seed)
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

    pin = dev.type in ("cuda",)
    if args.level == "recording":
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
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
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=pin, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=pin,
        )

    channels = feature_channels(args.feature_mode)
    model = build_model(args.architecture, in_channels=channels).to(dev)
    log.info("features: %s channels=%d | params: %d", args.feature_mode, channels, count_parameters(model))
    # Class-balanced loss: pos_weight = N_neg / N_pos so the minority
    # (murmur-present) class gets a proportional gradient pull.
    bal = ds.class_balance()
    pos_weight_val = max(1.0, bal["absent"] / max(bal["present"], 1))
    log.info("pos_weight: %.3f", pos_weight_val)
    pos_weight = torch.tensor([pos_weight_val], device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
    saved_best = False
    epochs_since_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        if args.level == "recording":
            tr_loss, tr_auc = run_recording_epoch(
                model, train_loader, loss_fn, opt, dev, args.aggregation, args.topk
            )
            va_loss, va_auc = run_recording_epoch(
                model, val_loader, loss_fn, None, dev, args.aggregation, args.topk
            )
        else:
            tr_loss, tr_auc = run_epoch(model, train_loader, loss_fn, opt, dev)
            va_loss, va_auc = run_epoch(model, val_loader, loss_fn, None, dev)
        lr_now = float(opt.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "tr_loss": tr_loss,
            "tr_auc": tr_auc,
            "va_loss": va_loss,
            "va_auc": va_auc,
            "lr": lr_now,
        })
        log.info("ep %02d | tr loss %.4f auc %.3f | va loss %.4f auc %.3f", epoch, tr_loss, tr_auc, va_loss, va_auc)
        torch.save(model.state_dict(), args.out / "last.pt")
        improved = bool(np.isfinite(va_auc) and va_auc > best_auc + args.early_stopping_min_delta)
        if improved or not saved_best:
            if np.isfinite(va_auc):
                best_auc = va_auc
            if improved:
                epochs_since_improvement = 0
            saved_best = True
            torch.save(model.state_dict(), args.out / "best.pt")
            (args.out / "best_meta.json").write_text(json.dumps({
                "epoch": epoch,
                "val_auc": va_auc,
                "level": args.level,
                "architecture": args.architecture,
                "feature_mode": args.feature_mode,
                "input_channels": channels,
                "window_seconds": args.window_seconds,
                "hop_seconds": hop_seconds,
                "lr_scheduler": args.lr_scheduler,
                "early_stopping_patience": args.early_stopping_patience,
                "aggregation": args.aggregation if args.level == "recording" else None,
                "topk": args.topk if args.level == "recording" else None,
            }))
        else:
            epochs_since_improvement += 1

        if scheduler is not None:
            scheduler.step(va_auc if np.isfinite(va_auc) else -va_loss)

        if (
            args.early_stopping_patience > 0
            and epochs_since_improvement >= args.early_stopping_patience
        ):
            log.info(
                "early stopping after %d epochs without validation-AUC improvement",
                epochs_since_improvement,
            )
            break

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    best_auc_msg = f"{best_auc:.3f}" if np.isfinite(best_auc) else "nan"
    log.info("best val AUC: %s → %s/best.pt", best_auc_msg, args.out)


if __name__ == "__main__":
    main()
