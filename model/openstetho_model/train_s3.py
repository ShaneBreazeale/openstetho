"""Train the S3 detector.

CLI:
    uv run python -m openstetho_model.train_s3 \
        --wavs data/circor/training_data \
        --pattern '*.wav' \
        --epochs 20 \
        --batch-size 64 \
        --out runs/s3_v1

Reuses MurmurCNN as the backbone (variable-T via AdaptiveAvgPool2d). Labels
come from `S3CycleDataset` — synthetic S3 injection at training time, with
real-S3 contamination of negatives treated as label noise. See
[[s3-annotation-protocol]] for the road to real labels.

Metrics per epoch: BCE loss, AUROC, AUPRC, F1 at 0.5, calibration ECE (10
bins). Saves the best-AUPRC checkpoint plus a `metrics.csv` history.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .model import MurmurCNN, S3CNN, S3CNN_v2, S3CNN_v3, count_parameters
from .s3_dataset import S3CycleDataset


_BACKBONES = {
    "murmur": MurmurCNN,
    "s3cnn": S3CNN,
    "s3cnn_v2": S3CNN_v2,
    "s3cnn_v3": S3CNN_v3,
}

log = logging.getLogger(__name__)


# ─── metrics ────────────────────────────────────────────────────────────────

@dataclass
class EpochMetrics:
    loss: float
    auroc: float
    auprc: float
    f1_at_0_5: float
    ece: float


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    n_pos = float(y_true.sum())
    n_neg = float(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y_true.sum(), 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision, recall))


def _f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    pred = (y_score >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def _ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            continue
        conf = float(y_score[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


def _write_live_heartbeat(path: Path, **fields) -> None:
    """Atomically overwrite a tiny JSON status file. The dashboard CLI tails
    this; replacing via rename avoids torn reads.

    Throttle to one write every 250 ms — heartbeat writes per-batch on every
    optimizer step would otherwise be a measurable share of the train loop on
    small batches.
    """
    now = time.perf_counter()
    last = getattr(_write_live_heartbeat, "_last_t", 0.0)
    if now - last < 0.25 and fields.get("phase") == "train":
        return
    _write_live_heartbeat._last_t = now  # type: ignore[attr-defined]
    fields["timestamp"] = time.time()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(fields))
    tmp.replace(path)


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    multiclass: bool = False,
) -> EpochMetrics:
    model.train(False)
    total_loss = 0.0
    total_n = 0
    if multiclass:
        # Treat softmax[1] (P(S3)) as the score for the S3-vs-rest AUPRC,
        # and accumulate a 3x3 confusion matrix for diagnostics.
        all_logits: list[np.ndarray] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for mel, label in loader:
                mel = mel.to(device)
                label_t = label.to(device).long()
                logits = model(mel)
                loss = loss_fn(logits, label_t)
                total_loss += float(loss.item()) * mel.size(0)
                total_n += mel.size(0)
                all_logits.append(logits.detach().cpu().numpy())
                all_labels.extend(label.cpu().int().tolist())
        logits_arr = np.concatenate(all_logits, axis=0)
        # Softmax stable.
        m = logits_arr.max(axis=1, keepdims=True)
        e = np.exp(logits_arr - m)
        probs = e / e.sum(axis=1, keepdims=True)
        y_true = np.asarray(all_labels, dtype=np.int64)
        # Binary AUPRC/AUROC: S3 (class 1) vs anything-not-S3.
        s3_binary = (y_true == 1).astype(np.int64)
        s3_score = probs[:, 1]
        # Confusion matrix logged via logger; not stored in EpochMetrics for
        # backward-compatible CSV schema.
        pred = probs.argmax(axis=1)
        cm = np.zeros((3, 3), dtype=np.int64)
        for t, p in zip(y_true, pred):
            cm[int(t), int(p)] += 1
        log.info("confusion (rows=true, cols=pred, classes=[clean,S3,S4]):\n%s", cm.tolist())
        return EpochMetrics(
            loss=total_loss / max(total_n, 1),
            auroc=_auroc(s3_binary, s3_score),
            auprc=_auprc(s3_binary, s3_score),
            f1_at_0_5=_f1_at_threshold(s3_binary, s3_score, 0.5),
            ece=_ece(s3_binary, s3_score),
        )

    # Binary path (BCEWithLogitsLoss).
    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for mel, label in loader:
            mel = mel.to(device)
            label_t = label.to(device).float()
            logits = model(mel)
            loss = loss_fn(logits, label_t)
            prob = torch.sigmoid(logits)
            total_loss += float(loss.item()) * mel.size(0)
            total_n += mel.size(0)
            scores.extend(prob.cpu().tolist())
            labels.extend(label.cpu().int().tolist())
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    return EpochMetrics(
        loss=total_loss / max(total_n, 1),
        auroc=_auroc(y_true, y_score),
        auprc=_auprc(y_true, y_score),
        f1_at_0_5=_f1_at_threshold(y_true, y_score, 0.5),
        ece=_ece(y_true, y_score),
    )


# ─── training loop ──────────────────────────────────────────────────────────

import re

_CIRCOR_PATTERN = re.compile(r"^(\d+)_[A-Za-z]+$")


def _patient_id(wav: Path) -> str:
    """Compute a corpus-aware patient identifier for the train/val split.

    - CirCor 2022 filenames `<patient>_<AV|MV|PV|TV|Phc>.wav` collapse all
      four locations to one patient so recordings from the same child cannot
      leak between train and val.
    - PhysioNet 2016 / PASCAL 2011 do not expose per-patient metadata, so we
      fall back to recording-level identity. This is looser than true patient
      separation but still prevents cycle-level leakage.

    Corpus tag prefix avoids ID collisions across datasets.
    """
    stem = wav.stem
    parent = wav.parent.name or "root"
    m = _CIRCOR_PATTERN.match(stem)
    if m:
        return f"circor:{m.group(1)}"
    return f"{parent}:{stem}"


def _patient_split(wavs: list[Path], val_frac: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Patient-disjoint train/val split. Cycles from the same recording
    cannot leak across the boundary."""
    rng = np.random.default_rng(seed)
    patients = sorted({_patient_id(w) for w in wavs})
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_frac))
    val_set = set(patients[:n_val])
    train_wavs = [w for w in wavs if _patient_id(w) not in val_set]
    val_wavs = [w for w in wavs if _patient_id(w) in val_set]
    return train_wavs, val_wavs


def train(
    wavs: list[Path],
    out_dir: Path,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    positive_rate: float = 0.5,
    snr_db_range: tuple[float, float] = (0.0, 12.0),
    num_workers: int = 0,
    seed: int = 0,
    lr_schedule: str = "cosine",
    patience: int = 0,
    prob_multi: float = 0.0,
    prob_s4: float = 0.0,
    freq_mask_max_width: int = 0,
    time_mask_max_width: int = 0,
    backbone: str = "murmur",
    mixup_alpha: float = 0.0,
    crop_anchor: str = "s2",
    num_classes: int = 1,
    segmenter: str = "heuristic",
    init_from: Path | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    train_wavs, val_wavs = _patient_split(wavs, val_frac, seed)
    log.info("patient split: %d train wavs, %d val wavs", len(train_wavs), len(val_wavs))

    multiclass = num_classes >= 3
    train_ds = S3CycleDataset(
        train_wavs,
        positive_rate=positive_rate,
        snr_db_range=snr_db_range,
        seed=seed,
        prob_multi=prob_multi,
        prob_s4=prob_s4,
        freq_mask_max_width=freq_mask_max_width,
        time_mask_max_width=time_mask_max_width,
        apply_spec_masks=True,
        crop_anchor=crop_anchor,
        emit_multiclass=multiclass,
        segmenter=segmenter,
    )
    val_ds = S3CycleDataset(
        val_wavs,
        positive_rate=positive_rate,
        snr_db_range=snr_db_range,
        seed=seed + 1,
        prob_multi=0.0,
        prob_s4=prob_s4,
        apply_spec_masks=False,
        crop_anchor=crop_anchor,
        emit_multiclass=multiclass,
        segmenter=segmenter,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("empty train or val split — check segmenter confidence floor")

    log.info(
        "train: %d cycles (balance=%s); val: %d cycles (balance=%s)",
        len(train_ds), train_ds.class_balance(),
        len(val_ds), val_ds.class_balance(),
    )

    # `persistent_workers` keeps the per-worker LRU cache alive across epochs;
    # without it workers respawn and re-load every WAV on each epoch.
    loader_kwargs: dict = {"batch_size": batch_size, "num_workers": num_workers}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if backbone not in _BACKBONES:
        raise ValueError(f"unknown backbone {backbone}; choose from {list(_BACKBONES)}")
    model = _BACKBONES[backbone](n_classes=num_classes).to(device)
    log.info("backbone: %s | classes: %d | params: %d", backbone, num_classes, count_parameters(model))
    if init_from is not None:
        log.info("initializing encoder from %s", init_from)
        ckpt = torch.load(init_from, map_location=device)
        if not isinstance(ckpt, dict) or "encoder_state_dict" not in ckpt:
            raise ValueError(f"{init_from} is not an SSL encoder checkpoint")
        if ckpt.get("backbone") and ckpt["backbone"] != backbone:
            log.warning(
                "checkpoint backbone (%s) != current backbone (%s); proceeding with strict=False",
                ckpt["backbone"], backbone,
            )
        # The SSL checkpoint's encoder state dict omits the head's `Linear`
        # layers (those didn't exist during pretraining). Load with
        # strict=False so the new head's randomly-initialised Linear weights
        # stay untouched.
        missing, unexpected = model.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        log.info("loaded encoder weights | missing keys: %d | unexpected: %d", len(missing), len(unexpected))
        if unexpected:
            log.warning("unexpected keys (will be ignored): %s", unexpected[:8])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss() if multiclass else nn.BCEWithLogitsLoss()

    if lr_schedule == "cosine":
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        )
    elif lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    elif lr_schedule == "none":
        scheduler = None
    else:
        raise ValueError(f"unknown lr_schedule: {lr_schedule}")

    metrics_csv = out_dir / "metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "auroc", "auprc", "f1_at_0_5", "ece"]
        )

    best_auprc = -math.inf
    live_path = out_dir / "live.json"
    total_batches = len(train_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        ema_loss = None
        epoch_start = time.perf_counter()
        for batch_no, (mel, label) in enumerate(train_loader, start=1):
            mel = mel.to(device)
            if multiclass:
                label_t = label.to(device).long()
            else:
                label_t = label.to(device).float()

            # Mixup: convex-combine batch with a shuffled copy of itself.
            # Soft labels feed into BCEWithLogitsLoss unchanged. Skip for
            # multiclass CE (would need soft-target CE; revisit if needed).
            if mixup_alpha > 0.0 and mel.size(0) > 1 and not multiclass:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(mel.size(0), device=device)
                mel = lam * mel + (1.0 - lam) * mel[perm]
                label_t = lam * label_t + (1.0 - lam) * label_t[perm]

            optimizer.zero_grad()
            logits = model(mel)
            loss = loss_fn(logits, label_t)
            loss.backward()
            optimizer.step()
            loss_val = float(loss.item())
            running += loss_val * mel.size(0)
            seen += mel.size(0)
            ema_loss = loss_val if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_val

            elapsed = time.perf_counter() - epoch_start
            items_per_sec = seen / max(elapsed, 1e-6)
            batches_left = total_batches - batch_no
            eta_s = batches_left * (elapsed / max(batch_no, 1))
            _write_live_heartbeat(
                live_path,
                epoch=epoch,
                epochs=epochs,
                batch_no=batch_no,
                total_batches=total_batches,
                seen=seen,
                items_per_sec=items_per_sec,
                eta_s=eta_s,
                running_loss=running / max(seen, 1),
                ema_loss=ema_loss,
                best_auprc=best_auprc if math.isfinite(best_auprc) else None,
                phase="train",
            )
        train_loss = running / max(seen, 1)

        _write_live_heartbeat(
            live_path,
            epoch=epoch,
            epochs=epochs,
            batch_no=total_batches,
            total_batches=total_batches,
            seen=seen,
            items_per_sec=seen / max(time.perf_counter() - epoch_start, 1e-6),
            eta_s=0.0,
            running_loss=train_loss,
            ema_loss=ema_loss,
            best_auprc=best_auprc if math.isfinite(best_auprc) else None,
            phase="validate",
        )
        m = validate_epoch(model, val_loader, device, loss_fn, multiclass=multiclass)
        log.info(
            "epoch %d: train_loss=%.4f val_loss=%.4f auroc=%.3f auprc=%.3f f1@0.5=%.3f ece=%.3f",
            epoch, train_loss, m.loss, m.auroc, m.auprc, m.f1_at_0_5, m.ece,
        )
        with metrics_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, m.loss, m.auroc, m.auprc, m.f1_at_0_5, m.ece])

        if not math.isnan(m.auprc) and m.auprc > best_auprc:
            best_auprc = m.auprc
            torch.save(model.state_dict(), out_dir / "best.pt")
            log.info("saved best.pt (AUPRC=%.3f)", m.auprc)
            epochs_since_best = 0
        else:
            epochs_since_best = locals().get("epochs_since_best", 0) + 1
            if patience > 0 and epochs_since_best >= patience:
                log.info("early stop: %d epochs since AUPRC improved", epochs_since_best)
                break

        if scheduler is not None:
            scheduler.step()
            log.info("lr -> %.2e", optimizer.param_groups[0]["lr"])

    log.info("training complete; best AUPRC=%.3f -> %s", best_auprc, out_dir / "best.pt")


# ─── CLI ────────────────────────────────────────────────────────────────────

def _resolve_wavs(roots: list[str], pattern: str) -> list[Path]:
    out: list[Path] = []
    for r in roots:
        rp = Path(r)
        if rp.is_file():
            out.append(rp)
        else:
            out.extend(sorted(rp.rglob(pattern)))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--wavs", required=True, nargs="+", help="one or more directories containing PCG WAVs")
    p.add_argument("--pattern", default="*.wav")
    p.add_argument("--out", default="runs/s3_v1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--positive-rate", type=float, default=0.5)
    p.add_argument("--snr-min", type=float, default=0.0)
    p.add_argument("--snr-max", type=float, default=12.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr-schedule", choices=["cosine", "step", "none"], default="cosine")
    p.add_argument("--patience", type=int, default=0, help="early-stop after N epochs without AUPRC improvement; 0 disables")
    p.add_argument("--prob-multi", type=float, default=0.0, help="prob of injecting a second S3 in a positive cycle")
    p.add_argument("--prob-s4", type=float, default=0.0, help="prob of injecting a synthetic S4 into a negative cycle (negative mining)")
    p.add_argument("--freq-mask-max-width", type=int, default=0, help="SpecAugment frequency-mask max width (mel bins)")
    p.add_argument("--time-mask-max-width", type=int, default=0, help="SpecAugment time-mask max width (frames)")
    p.add_argument("--backbone", choices=list(_BACKBONES), default="murmur", help="CNN backbone selection")
    p.add_argument("--mixup-alpha", type=float, default=0.0, help="mixup Beta(α,α) parameter; 0 disables (try 0.2)")
    p.add_argument("--crop-anchor", choices=["s1", "s2"], default="s2", help="cycle crop anchor; s2 keeps S3/S4 in fixed frame ranges")
    p.add_argument("--num-classes", type=int, default=1, help="1 (binary BCE) or 3 (multiclass: clean/S3/S4 cross-entropy)")
    p.add_argument("--segmenter", choices=["heuristic", "hsmm"], default="heuristic", help="cycle segmentation algorithm")
    p.add_argument("--init-from", type=Path, default=None, help="load encoder weights from an SSL pretrain checkpoint (encoder.pt)")
    args = p.parse_args()

    wavs = _resolve_wavs(args.wavs, args.pattern)
    if not wavs:
        raise SystemExit(f"no WAV files matched roots={args.wavs} pattern={args.pattern}")
    log.info("resolved %d WAVs across %d root(s)", len(wavs), len(args.wavs))

    train(
        wavs=wavs,
        out_dir=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        positive_rate=args.positive_rate,
        snr_db_range=(args.snr_min, args.snr_max),
        num_workers=args.num_workers,
        seed=args.seed,
        lr_schedule=args.lr_schedule,
        patience=args.patience,
        prob_multi=args.prob_multi,
        prob_s4=args.prob_s4,
        freq_mask_max_width=args.freq_mask_max_width,
        time_mask_max_width=args.time_mask_max_width,
        backbone=args.backbone,
        mixup_alpha=args.mixup_alpha,
        crop_anchor=args.crop_anchor,
        num_classes=args.num_classes,
        segmenter=args.segmenter,
        init_from=args.init_from,
    )


if __name__ == "__main__":
    main()
