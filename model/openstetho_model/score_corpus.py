"""Score every cycle in every WAV with a trained S3 detector.

Output is a per-cycle CSV: one row per (wav, cycle_index) with the model's
sigmoid (or softmax[1] for multiclass) probability, plus the cycle's S1/S2
sample indices so the annotation viewer can replay the exact audio segment.

Used as the feed for the cardiologist annotation pipeline — see
`select_for_annotation.py` and `export_cycle_clips.py`.

CLI:
    uv run python -m openstetho_model.score_corpus \\
        --checkpoint runs/s3_circor_v6/best.pt \\
        --wavs /Users/shane/repos/eko/data/circor \\
        --pattern '*.wav' \\
        --out runs/s3_circor_v6/cycle_scores.csv \\
        --backbone s3cnn_v2 \\
        --crop-anchor s2
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
from .s3_dataset import CYCLE_WINDOW_FRAMES, _cycle_crop, _cycle_crop_s2
from .segment import segment_unified

log = logging.getLogger(__name__)

_BACKBONES = {
    "murmur": MurmurCNN,
    "s3cnn": S3CNN,
    "s3cnn_v2": S3CNN_v2,
    "s3cnn_v3": S3CNN_v3,
}


def _to_fixed_frames(mel: np.ndarray) -> np.ndarray:
    out = np.zeros((CYCLE_WINDOW_FRAMES, mel.shape[1]), dtype=np.float32)
    keep = min(mel.shape[0], CYCLE_WINDOW_FRAMES)
    out[:keep] = mel[:keep]
    return out


def score_wav(
    model: torch.nn.Module,
    device: torch.device,
    wav: Path,
    crop_anchor: str,
    multiclass: bool,
    segmenter: str = "heuristic",
) -> list[dict]:
    audio = load_audio(str(wav))
    seg = segment_unified(audio, method=segmenter)
    rows: list[dict] = []
    if not seg.cycles:
        return rows
    mels: list[np.ndarray] = []
    for cyc in seg.cycles:
        if crop_anchor == "s2":
            chunk = _cycle_crop_s2(audio, cyc.s2_idx)
        else:
            chunk = _cycle_crop(audio, cyc.s1_idx)
        chunk = apply_s3_preset(chunk)
        mel = log_mel(chunk)
        mels.append(_to_fixed_frames(mel))

    batch = torch.from_numpy(np.stack(mels, axis=0)).to(device)
    with torch.no_grad():
        logits = model(batch)
        if multiclass:
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        else:
            probs = torch.sigmoid(logits).cpu().numpy()

    for k, (cyc, p) in enumerate(zip(seg.cycles, probs)):
        rows.append(
            {
                "wav": str(wav),
                "cycle_no": k,
                "s1_idx": cyc.s1_idx,
                "s2_idx": cyc.s2_idx,
                "next_s1_idx": cyc.next_s1_idx,
                "score": float(p),
                "segmenter_confidence": seg.confidence,
            }
        )
    return rows


def run(
    checkpoint: Path,
    wav_paths: list[Path],
    out_csv: Path,
    backbone: str,
    crop_anchor: str,
    num_classes: int,
    segmenter: str,
) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = _BACKBONES[backbone](n_classes=num_classes).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.train(False)
    multiclass = num_classes >= 3
    log.info("loaded %s (backbone=%s, multiclass=%s)", checkpoint, backbone, multiclass)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "wav",
                "cycle_no",
                "s1_idx",
                "s2_idx",
                "next_s1_idx",
                "score",
                "segmenter_confidence",
            ],
        )
        writer.writeheader()
        for i, wav in enumerate(wav_paths):
            try:
                rows = score_wav(model, device, wav, crop_anchor, multiclass, segmenter)
            except Exception as e:  # noqa: BLE001
                log.warning("skip %s: %s", wav, e)
                continue
            for row in rows:
                writer.writerow(row)
            if (i + 1) % 100 == 0:
                log.info("scored %d/%d wavs", i + 1, len(wav_paths))
    log.info("wrote %s", out_csv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--wavs", required=True, nargs="+", help="one or more roots containing WAV files")
    p.add_argument("--pattern", default="*.wav")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--backbone", choices=list(_BACKBONES), default="s3cnn_v2")
    p.add_argument("--crop-anchor", choices=["s1", "s2"], default="s2")
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--segmenter", choices=["heuristic", "hsmm"], default="heuristic")
    args = p.parse_args()

    wavs: list[Path] = []
    for root in args.wavs:
        rp = Path(root)
        if rp.is_file():
            wavs.append(rp)
        else:
            wavs.extend(sorted(rp.rglob(args.pattern)))
    if not wavs:
        raise SystemExit("no WAVs matched")
    log.info("scoring %d wavs", len(wavs))
    run(
        checkpoint=args.checkpoint,
        wav_paths=wavs,
        out_csv=args.out,
        backbone=args.backbone,
        crop_anchor=args.crop_anchor,
        num_classes=args.num_classes,
        segmenter=args.segmenter,
    )


if __name__ == "__main__":
    main()
