"""Patient-level cross-validation for CirCor murmur experiments."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from .bench_murmur import best_by, binary_metrics, json_safe, sweep_thresholds
from .dataset import CirCorRecordingMurmurDataset, window_hop_samples
from .model import count_parameters
from .preprocess import FEATURE_MODES, FEATURE_MODE_LOGMEL, SAMPLE_RATE, feature_channels
from .train import (
    _patient_labels,
    aggregate_logits,
    build_model,
    device,
    recording_collate,
    run_recording_epoch,
)

log = logging.getLogger("cv_murmur")
PROB_EPS = 1e-6
CALIBRATION_METHODS = ("none", "platt", "isotonic")
THRESHOLD_KEYS = ("best_f1", "best_youden_j")
THRESHOLD_METRICS = {"best_f1": "f1", "best_youden_j": "youden_j"}


def stratified_patient_folds(
    patient_labels: dict[int, int],
    n_folds: int,
    seed: int,
) -> list[set[int]]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    rng = np.random.default_rng(seed)
    folds: list[set[int]] = [set() for _ in range(n_folds)]
    for label in (0, 1):
        ids = sorted(pid for pid, y in patient_labels.items() if y == label)
        if len(ids) < n_folds:
            raise ValueError(f"label {label} has only {len(ids)} patients for {n_folds} folds")
        rng.shuffle(ids)
        for i, pid in enumerate(ids):
            folds[i % n_folds].add(pid)
    return folds


def recording_indices_for_patients(ds: CirCorRecordingMurmurDataset, patient_ids: set[int]) -> list[int]:
    return [i for i, (pid, *_rest) in enumerate(ds._records) if pid in patient_ids]


def subset_balance(ds: CirCorRecordingMurmurDataset, indices: list[int]) -> dict[str, int]:
    absent = sum(1 for i in indices if ds._records[i][3] == 0)
    present = len(indices) - absent
    return {"absent": absent, "present": present}


def train_fold(
    args: argparse.Namespace,
    ds: CirCorRecordingMurmurDataset,
    fold: int,
    train_idx: list[int],
    val_idx: list[int],
    dev: torch.device,
) -> dict[str, object]:
    fold_dir = args.out / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed + fold)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    pin = dev.type in ("cuda",)
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

    channels = feature_channels(args.feature_mode)
    model = build_model(args.architecture, in_channels=channels).to(dev)
    bal = subset_balance(ds, train_idx)
    pos_weight_val = max(1.0, bal["absent"] / max(bal["present"], 1))
    pos_weight = torch.tensor([pos_weight_val], device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    best_epoch = 0
    epochs_since_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_auc = run_recording_epoch(
            model, train_loader, loss_fn, opt, dev, args.aggregation, args.topk
        )
        va_loss, va_auc = run_recording_epoch(
            model, val_loader, loss_fn, None, dev, args.aggregation, args.topk
        )
        lr_now = float(opt.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "tr_loss": tr_loss,
            "tr_auc": tr_auc,
            "va_loss": va_loss,
            "va_auc": va_auc,
            "lr": lr_now,
        })
        log.info(
            "fold %02d ep %02d | tr loss %.4f auc %.3f | va loss %.4f auc %.3f",
            fold,
            epoch,
            tr_loss,
            tr_auc,
            va_loss,
            va_auc,
        )
        torch.save(model.state_dict(), fold_dir / "last.pt")
        improved = bool(np.isfinite(va_auc) and va_auc > best_auc + args.early_stopping_min_delta)
        if improved or not saved_best:
            if np.isfinite(va_auc):
                best_auc = va_auc
            if improved:
                epochs_since_improvement = 0
            saved_best = True
            best_epoch = epoch
            torch.save(model.state_dict(), fold_dir / "best.pt")
        else:
            epochs_since_improvement += 1

        if scheduler is not None:
            scheduler.step(va_auc if np.isfinite(va_auc) else -va_loss)

        if (
            args.early_stopping_patience > 0
            and epochs_since_improvement >= args.early_stopping_patience
        ):
            log.info(
                "fold %02d early stopping after %d epochs without validation-AUC improvement",
                fold,
                epochs_since_improvement,
            )
            break

    (fold_dir / "history.json").write_text(json.dumps(json_safe(history), indent=2))
    meta = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "train_recordings": len(train_idx),
        "val_recordings": len(val_idx),
        "train_balance": bal,
        "val_balance": subset_balance(ds, val_idx),
        "feature_mode": args.feature_mode,
        "input_channels": channels,
        "architecture": args.architecture,
        "aggregation": args.aggregation,
        "topk": args.topk,
        "window_seconds": args.window_seconds,
        "hop_seconds": args.hop_seconds_effective,
        "lr_scheduler": args.lr_scheduler,
        "early_stopping_patience": args.early_stopping_patience,
        "params": count_parameters(model),
    }
    (fold_dir / "best_meta.json").write_text(json.dumps(json_safe(meta), indent=2))

    model.load_state_dict(torch.load(fold_dir / "best.pt", map_location=dev, weights_only=True))
    model.eval()
    predictions = predict_recordings(model, val_loader, dev, args.aggregation, args.topk)
    return {**meta, "predictions": predictions}


def predict_recordings(
    model: torch.nn.Module,
    loader: DataLoader,
    dev: torch.device,
    aggregation: str,
    topk: int,
) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    with torch.no_grad():
        for recordings, labels in loader:
            for mel, label in zip(recordings, labels.tolist()):
                mel = mel.to(dev, non_blocking=True)
                logits = model(mel)
                agg = aggregate_logits(logits, aggregation, topk)
                prob = float(torch.sigmoid(agg).detach().cpu().item())
                out.append((int(label), prob))
    return out


def clipped_probs(probs: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probs, dtype=np.float64), PROB_EPS, 1.0 - PROB_EPS)


def prob_logits(probs: np.ndarray) -> np.ndarray:
    probs = clipped_probs(probs)
    return np.log(probs / (1.0 - probs))


def expected_calibration_error(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probs = clipped_probs(probs)
    if labels.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(probs[mask].mean()) - float(labels[mask].mean()))
    return ece


def probability_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probs = clipped_probs(probs)
    return {
        "auroc": float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) == 2 else float("nan"),
        "brier": float(np.mean((probs - labels) ** 2)) if labels.size else float("nan"),
        "ece_10": expected_calibration_error(labels, probs, n_bins=10),
    }


def binary_metrics_from_predictions(
    labels: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    out = probability_metrics(labels, scores)
    out.update({
        "n": int(labels.size),
        "positives": int(labels.sum()),
        "accuracy": (tp + tn) / max(labels.size, 1),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1),
        "f1": 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn),
        "youden_j": (tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1)) - 1.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    })
    return out


def calibrate_probs(
    train_labels: np.ndarray,
    train_probs: np.ndarray,
    val_probs: np.ndarray,
    method: str,
) -> np.ndarray:
    train_labels = np.asarray(train_labels, dtype=np.int64)
    train_probs = clipped_probs(train_probs)
    val_probs = clipped_probs(val_probs)
    if method == "none" or len(set(train_labels.tolist())) < 2:
        return val_probs
    if method == "platt":
        model = LogisticRegression(random_state=0, solver="liblinear")
        model.fit(prob_logits(train_probs).reshape(-1, 1), train_labels)
        return model.predict_proba(prob_logits(val_probs).reshape(-1, 1))[:, 1].astype(np.float64)
    if method == "isotonic":
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(train_probs, train_labels)
        return np.asarray(model.transform(val_probs), dtype=np.float64)
    raise ValueError(f"unknown calibration method {method!r}; valid: {CALIBRATION_METHODS}")


def calibration_curve_points(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    labels = np.asarray(labels, dtype=np.int64)
    probs = clipped_probs(probs)
    if labels.size == 0:
        return []
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy="uniform")
    return [
        {"mean_predicted_probability": float(x), "fraction_positive": float(y)}
        for x, y in zip(prob_pred, prob_true)
    ]


def summarize(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, object]:
    rows = sweep_thresholds(labels, probs)
    return {
        "threshold": binary_metrics(labels, probs, threshold),
        "best_f1": best_by(rows, "f1"),
        "best_youden_j": best_by(rows, "youden_j"),
        **probability_metrics(labels, probs),
        "calibration_curve_10": calibration_curve_points(labels, probs, n_bins=10),
    }


def cross_fold_calibration_report(
    folds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
) -> dict[str, object]:
    """Fit calibration/threshold choices on other folds, apply to each fold."""
    folds = np.asarray(folds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    probs = clipped_probs(probs)
    unique_folds = sorted(set(folds.tolist()))
    report: dict[str, object] = {}
    for method in CALIBRATION_METHODS:
        calibrated = np.zeros_like(probs, dtype=np.float64)
        threshold_predictions = {
            key: np.zeros_like(labels, dtype=np.int64)
            for key in THRESHOLD_KEYS
        }
        fold_rows: list[dict[str, object]] = []
        for fold in unique_folds:
            val_mask = folds == fold
            train_mask = ~val_mask
            train_labels = labels[train_mask]
            train_probs = probs[train_mask]
            val_probs = probs[val_mask]
            train_calibrated = calibrate_probs(train_labels, train_probs, train_probs, method)
            val_calibrated = calibrate_probs(train_labels, train_probs, val_probs, method)
            calibrated[val_mask] = val_calibrated

            train_sweep = sweep_thresholds(train_labels, train_calibrated)
            row: dict[str, object] = {"fold": int(fold)}
            for key in THRESHOLD_KEYS:
                threshold = float(best_by(train_sweep, THRESHOLD_METRICS[key])["threshold"])
                threshold_predictions[key][val_mask] = (val_calibrated >= threshold).astype(np.int64)
                row[f"{key}_threshold"] = threshold
            row.update(probability_metrics(labels[val_mask], val_calibrated))
            fold_rows.append(row)

        method_report: dict[str, object] = {
            "probability": {
                **probability_metrics(labels, calibrated),
                "calibration_curve_10": calibration_curve_points(labels, calibrated, n_bins=10),
            },
            "threshold_transfer": {},
            "folds": fold_rows,
        }
        for key, pred in threshold_predictions.items():
            metrics = binary_metrics_from_predictions(labels, pred, calibrated)
            thresholds = np.asarray([float(row[f"{key}_threshold"]) for row in fold_rows], dtype=np.float64)
            metrics.update({
                "threshold_policy": key,
                "threshold_mean": float(thresholds.mean()),
                "threshold_std": float(thresholds.std()),
                "threshold_min": float(thresholds.min()),
                "threshold_max": float(thresholds.max()),
            })
            method_report["threshold_transfer"][key] = metrics  # type: ignore[index]
        report[method] = method_report
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    args.out.mkdir(parents=True, exist_ok=True)
    window_samples, hop_samples = window_hop_samples(args.window_seconds, args.hop_seconds)
    args.hop_seconds_effective = hop_samples / SAMPLE_RATE
    dev = device(args.device)
    log.info("device: %s", dev)
    log.info(
        "window: %.3fs (%d samples) hop %.3fs (%d samples)",
        args.window_seconds,
        window_samples,
        args.hop_seconds_effective,
        hop_samples,
    )

    ds = CirCorRecordingMurmurDataset(
        args.data,
        apply_cardiac=not args.no_cardiac,
        feature_mode=args.feature_mode,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        cache_features=True,
    )
    patient_labels = _patient_labels([(pid, label) for pid, _loc, _wav, label in ds._records])
    folds = stratified_patient_folds(patient_labels, args.folds, args.seed)
    log.info("dataset: %d recordings | balance: %s", len(ds), ds.class_balance())

    fold_reports: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for fold_num, val_ids in enumerate(folds, start=1):
        train_ids = set(patient_labels) - val_ids
        train_idx = recording_indices_for_patients(ds, train_ids)
        val_idx = recording_indices_for_patients(ds, val_ids)
        log.info(
            "fold %02d/%02d train=%d val=%d",
            fold_num,
            args.folds,
            len(train_idx),
            len(val_idx),
        )
        fold_report = train_fold(args, ds, fold_num, train_idx, val_idx, dev)
        preds = fold_report.pop("predictions")
        assert isinstance(preds, list)
        for idx, (label, prob) in zip(val_idx, preds):
            patient_id, loc, wav, _label = ds._records[idx]
            prediction_rows.append({
                "fold": fold_num,
                "patient_id": int(patient_id),
                "location": str(loc),
                "recording": str(wav),
                "label": int(label),
                "prob": float(prob),
            })
        fold_labels = np.asarray([row["label"] for row in prediction_rows if row["fold"] == fold_num])
        fold_probs = np.asarray([row["prob"] for row in prediction_rows if row["fold"] == fold_num])
        fold_report["metrics"] = summarize(fold_labels, fold_probs, args.threshold)
        fold_reports.append(fold_report)

    labels = np.asarray([row["label"] for row in prediction_rows], dtype=np.int64)
    prediction_folds = np.asarray([row["fold"] for row in prediction_rows], dtype=np.int64)
    probs = np.asarray([row["prob"] for row in prediction_rows], dtype=np.float64)
    calibration_report = cross_fold_calibration_report(prediction_folds, labels, probs)
    report = {
        "data": str(args.data),
        "feature_mode": args.feature_mode,
        "architecture": args.architecture,
        "level": "recording",
        "aggregation": args.aggregation,
        "topk": args.topk,
        "folds": args.folds,
        "seed": args.seed,
        "apply_cardiac": not args.no_cardiac,
        "window_seconds": args.window_seconds,
        "hop_seconds": args.hop_seconds_effective,
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "early_stopping_patience": args.early_stopping_patience,
        "n_recordings": int(labels.size),
        "positives": int(labels.sum()),
        "fold_reports": fold_reports,
        "oof": summarize(labels, probs, args.threshold),
        "cross_fold_calibration": calibration_report,
        "fold_val_auc_mean": float(np.mean([r["best_val_auc"] for r in fold_reports])),
        "fold_val_auc_std": float(np.std([r["best_val_auc"] for r in fold_reports])),
    }

    predictions_path = args.out / "oof_predictions.csv"
    with predictions_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "patient_id", "location", "recording", "label", "prob"],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)
    report["oof_predictions_csv"] = str(predictions_path)
    (args.out / "cv_report.json").write_text(json.dumps(json_safe(report), indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="CirCor root")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument(
        "--architecture",
        choices=["cnn", "cnn_bigru", "scattering_cnn1d"],
        default="cnn_bigru",
    )
    p.add_argument("--feature-mode", choices=FEATURE_MODES, default=FEATURE_MODE_LOGMEL)
    p.add_argument("--window-seconds", type=float, default=5.0)
    p.add_argument("--hop-seconds", type=float, default=None)
    p.add_argument("--aggregation", choices=["mean", "topk_mean", "max"], default="mean")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--lr-scheduler", choices=["none", "plateau"], default="plateau")
    p.add_argument("--plateau-patience", type=int, default=1)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--early-stopping-patience", type=int, default=2)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    p.add_argument(
        "--no-cardiac",
        action="store_true",
        help="skip legacy cardiac filter; matches current stetho-ui preprocessing",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(json_safe(run(args)), indent=2))


if __name__ == "__main__":
    main()
