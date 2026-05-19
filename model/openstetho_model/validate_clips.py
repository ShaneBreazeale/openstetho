"""Score a trained S3 detector on labeled teaching-library clips.

Reads one or more `labels.csv` files (columns: `filename,label_s3`) next to
their WAV clips. For each clip, segments into cycles, runs the model on
every cycle, and aggregates to a single recording-level score (max-pool
over cycles; matches "is S3 audible *anywhere* in this short clip?").

Outputs sens / spec / AUROC / AUPRC / F1@0.5 and writes per-clip
predictions to `predictions.csv` next to the first labels file.

CLI:
    uv run python -m openstetho_model.validate_clips \\
        --checkpoint runs/s3_v1/best.pt \\
        --labels data/s3_validation/michigan/labels.csv \\
                 data/s3_validation/texas/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch

from .model import MurmurCNN, S3CNN, S3CNN_v2, S3CNN_v3
from .preprocess import apply_s3_preset, load_audio, log_mel
from .s3_dataset import (
    CYCLE_WINDOW_FRAMES,
    CYCLE_WINDOW_SAMPLES,
    N_MELS,
    _cycle_crop,
    _cycle_crop_s2,
)
from .segment import segment
from .train_s3 import _auprc, _auroc, _f1_at_threshold

_BACKBONES = {
    "murmur": MurmurCNN,
    "s3cnn": S3CNN,
    "s3cnn_v2": S3CNN_v2,
    "s3cnn_v3": S3CNN_v3,
}

log = logging.getLogger(__name__)


def _set_inference_mode(model: torch.nn.Module) -> None:
    model.train(False)


def score_clip(
    model: torch.nn.Module,
    device: torch.device,
    wav: Path,
    crop_anchor: str = "s2",
    multiclass: bool = False,
) -> tuple[float, int]:
    """Return `(recording_score, n_cycles_scored)`.

    Score is max sigmoid (or softmax[1]) across all cycles. If segmentation
    finds zero cycles, falls back to scoring the first CYCLE_WINDOW_SAMPLES
    starting at offset 0.
    """
    audio = load_audio(str(wav))
    seg = segment(audio)
    cycles = seg.cycles or []

    mels: list[np.ndarray] = []
    if cycles:
        for cyc in cycles:
            if crop_anchor == "s2":
                chunk = _cycle_crop_s2(audio, cyc.s2_idx)
            else:
                chunk = _cycle_crop(audio, cyc.s1_idx)
            chunk = apply_s3_preset(chunk)
            mel = log_mel(chunk)
            if mel.shape[0] != CYCLE_WINDOW_FRAMES:
                padded = np.zeros((CYCLE_WINDOW_FRAMES, N_MELS), dtype=np.float32)
                keep = min(mel.shape[0], CYCLE_WINDOW_FRAMES)
                padded[:keep] = mel[:keep]
                mel = padded
            mels.append(mel)
        n_cycles = len(cycles)
    else:
        # Fallback: score the first window starting at sample 0.
        chunk = audio[:CYCLE_WINDOW_SAMPLES]
        if len(chunk) < CYCLE_WINDOW_SAMPLES:
            pad = np.zeros(CYCLE_WINDOW_SAMPLES, dtype=np.float32)
            pad[: len(chunk)] = chunk
            chunk = pad
        chunk = apply_s3_preset(chunk)
        mel = log_mel(chunk)
        if mel.shape[0] != CYCLE_WINDOW_FRAMES:
            padded = np.zeros((CYCLE_WINDOW_FRAMES, N_MELS), dtype=np.float32)
            keep = min(mel.shape[0], CYCLE_WINDOW_FRAMES)
            padded[:keep] = mel[:keep]
            mel = padded
        mels.append(mel)
        n_cycles = 1

    batch = torch.from_numpy(np.stack(mels, axis=0)).to(device)
    with torch.no_grad():
        logits = model(batch)
        if multiclass:
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(S3)
        else:
            probs = torch.sigmoid(logits)
    return float(probs.max().item()), n_cycles


def _read_labels(path: Path) -> list[tuple[Path, int]]:
    out: list[tuple[Path, int]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav = (path.parent / row["filename"]).resolve()
            if not wav.exists():
                log.warning("missing wav %s (skipping)", wav)
                continue
            out.append((wav, int(row["label_s3"])))
    return out


def run(
    checkpoint: Path,
    label_files: list[Path],
    out_csv: Path,
    backbone: str = "murmur",
    crop_anchor: str = "s2",
    num_classes: int = 1,
) -> None:
    if backbone not in _BACKBONES:
        raise ValueError(f"unknown backbone {backbone}; choose from {list(_BACKBONES)}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = _BACKBONES[backbone](n_classes=num_classes).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    _set_inference_mode(model)
    multiclass = num_classes >= 3
    log.info("loaded %s (backbone=%s, crop=%s, classes=%d)", checkpoint, backbone, crop_anchor, num_classes)

    rows: list[tuple[str, int, float, int]] = []
    for lf in label_files:
        for wav, label in _read_labels(lf):
            score, n = score_clip(model, device, wav, crop_anchor=crop_anchor, multiclass=multiclass)
            rows.append((str(wav), label, score, n))
            log.info("%s label=%d score=%.3f cycles=%d", wav.name, label, score, n)

    if not rows:
        raise SystemExit("no labeled clips found")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label_s3", "score", "n_cycles"])
        for row in rows:
            w.writerow(row)
    log.info("wrote predictions -> %s", out_csv)

    y_true = np.array([r[1] for r in rows], dtype=np.int64)
    y_score = np.array([r[2] for r in rows], dtype=np.float64)
    pred = (y_score >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    print(f"n            : {len(rows)}")
    print(f"positives    : {int(y_true.sum())}")
    print(f"AUROC        : {_auroc(y_true, y_score):.3f}")
    print(f"AUPRC        : {_auprc(y_true, y_score):.3f}")
    print(f"F1 @ 0.5     : {_f1_at_threshold(y_true, y_score, 0.5):.3f}")
    print(f"sensitivity  : {sens:.3f}  ({tp}/{tp + fn})")
    print(f"specificity  : {spec:.3f}  ({tn}/{tn + fp})")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--labels", nargs="+", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None, help="predictions CSV path")
    p.add_argument("--backbone", choices=list(_BACKBONES), default="murmur")
    p.add_argument("--crop-anchor", choices=["s1", "s2"], default="s2")
    p.add_argument("--num-classes", type=int, default=1)
    args = p.parse_args()
    out = args.out or args.labels[0].parent / "predictions.csv"
    run(
        args.checkpoint,
        args.labels,
        out,
        backbone=args.backbone,
        crop_anchor=args.crop_anchor,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()
