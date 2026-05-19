"""Export a trained S3 detector to a Core ML `.mlpackage`.

Targets the Apple Neural Engine. All ops used in `S3CNN_v2` (Conv2D /
Conv1D / BatchNorm{1,2}d / ReLU / MaxPool / AdaptiveAvgPool / Linear /
Dropout) are ANE-supported, and the runtime can route to ANE whenever
the input shape is fully static.

Usage:
    cd model
    uv run python -m openstetho_model.export_s3 \\
        --checkpoint runs/s3_circor_v10/best.pt \\
        --out runs/s3_circor_v10/S3CNN_v2.mlpackage \\
        --backbone s3cnn_v2 --verify
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

from .model import MurmurCNN, S3CNN, S3CNN_v2, S3CNN_v3
from .preprocess import N_MELS
from .s3_dataset import CYCLE_WINDOW_FRAMES

log = logging.getLogger("export_s3")

INPUT_NAME = "log_mel"
OUTPUT_NAME = "s3_logit"

_BACKBONES = {
    "murmur": MurmurCNN,
    "s3cnn": S3CNN,
    "s3cnn_v2": S3CNN_v2,
    "s3cnn_v3": S3CNN_v3,
}

TARGETS = {
    "iOS16": ct.target.iOS16,
    "iOS17": ct.target.iOS17,
    "iOS18": ct.target.iOS18,
    "macOS13": ct.target.macOS13,
    "macOS14": ct.target.macOS14,
    "macOS15": ct.target.macOS15,
}


def load_model(checkpoint: Path, backbone: str, n_classes: int = 1) -> torch.nn.Module:
    model = _BACKBONES[backbone](n_classes=n_classes)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.train(False)
    return model


def export(
    checkpoint: Path,
    out_path: Path,
    backbone: str = "s3cnn_v2",
    n_frames: int = CYCLE_WINDOW_FRAMES,
    n_classes: int = 1,
    target_key: str = "iOS17",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if target_key not in TARGETS:
        raise ValueError(f"unknown target {target_key}; valid: {sorted(TARGETS)}")
    if backbone not in _BACKBONES:
        raise ValueError(f"unknown backbone {backbone}; valid: {sorted(_BACKBONES)}")

    log.info("loading checkpoint %s (backbone=%s, n_classes=%d)", checkpoint, backbone, n_classes)
    model = load_model(checkpoint, backbone, n_classes)
    example = torch.zeros(1, 1, n_frames, N_MELS, dtype=torch.float32)
    log.info("tracing on shape (1, 1, %d, %d)", n_frames, N_MELS)
    traced = torch.jit.trace(model, example)

    log.info("converting to Core ML (mlprogram, compute_units=ALL)")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=INPUT_NAME, shape=(1, 1, n_frames, N_MELS), dtype=np.float32)],
        outputs=[ct.TensorType(name=OUTPUT_NAME)],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=TARGETS[target_key],
    )

    mlmodel.short_description = (
        f"{backbone} - synthetic-S3-trained third-heart-sound detector "
        "(binary cycle-level)"
    )
    mlmodel.author = "Shane Breazeale"
    mlmodel.license = (
        "Apache-2.0 code; model trained on PhysioNet CirCor 2022 (ODC-By 1.0), "
        "PhysioNet/CinC 2016, and PASCAL 2011 — all public PCG datasets."
    )
    mlmodel.version = "0.1.0"
    if hasattr(mlmodel, "input_description"):
        mlmodel.input_description[INPUT_NAME] = (
            f"Log-mel spectrogram, shape (1, 1, {n_frames}, {N_MELS}). "
            "S2-anchored 1.5-s window @ 4 kHz, HP15→LP120 biquad chain, "
            "Hann/256-FFT/no-overlap, 32 Slaney mel bands (20-1000 Hz), "
            "log10×10, per-frame z-score, -80 dB clip."
        )
        mlmodel.output_description[OUTPUT_NAME] = (
            "Raw logit for synthetic-S3 presence in the cycle. Apply sigmoid "
            "for probability. NOTE: calibrated against synthetic injection; "
            "real-audio operating threshold is ~0.93-0.99 (see "
            "docs/real_validation_results.md), not 0.5."
        )

    mlmodel.save(str(out_path))
    log.info("wrote %s", out_path)
    return out_path


def verify(checkpoint: Path, mlpackage: Path, backbone: str, n_frames: int, n_classes: int) -> dict:
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((1, 1, n_frames, N_MELS)).astype(np.float32)

    model = load_model(checkpoint, backbone, n_classes)
    with torch.no_grad():
        y_torch = model(torch.from_numpy(x_np)).numpy().reshape(-1)

    mlmodel = ct.models.MLModel(str(mlpackage))
    t0 = time.perf_counter()
    out = mlmodel.predict({INPUT_NAME: x_np})
    cm_latency_ms = (time.perf_counter() - t0) * 1000
    y_coreml = np.asarray(out[OUTPUT_NAME]).reshape(-1)

    diff = np.abs(y_torch - y_coreml)
    return {
        "torch_logit": [float(v) for v in y_torch],
        "coreml_logit": [float(v) for v in y_coreml],
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "coreml_latency_ms": cm_latency_ms,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backbone", choices=list(_BACKBONES), default="s3cnn_v2")
    p.add_argument("--n-classes", type=int, default=1)
    p.add_argument("--n-frames", type=int, default=CYCLE_WINDOW_FRAMES,
                   help=f"input mel frames (default {CYCLE_WINDOW_FRAMES} for the 1.5-s S2-anchored crop)")
    p.add_argument("--target", default="iOS17")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export(args.checkpoint, args.out, args.backbone, args.n_frames, args.n_classes, args.target)

    if args.verify:
        log.info("running torch-vs-coreml parity check")
        report = verify(args.checkpoint, args.out, args.backbone, args.n_frames, args.n_classes)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
