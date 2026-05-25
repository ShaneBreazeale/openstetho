"""Export a trained murmur checkpoint to a Core ML `.mlpackage`.

The package targets the Apple Neural Engine: all ops used in MurmurCNN
(Conv2D / BatchNorm / ReLU / AvgPool / Linear / Dropout) are ANE-supported,
and we ask Core ML to compile to the `mlprogram` flavor with
`compute_units=ALL` so the runtime can route to ANE whenever the input
shape is known and static.

Usage:
    cd model
    uv run python -m openstetho_model.export \\
        --checkpoint runs/v1/best.pt \\
        --out runs/v1/MurmurCNN.mlpackage \\
        --verify
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

from .model import MurmurCNN, MurmurCNNBiGRU
from .preprocess import N_MELS, SAMPLE_RATE, STFT_HOP

log = logging.getLogger("export")

# 4-second window @ 4 kHz / 256-sample STFT hop = 62 frames.
DEFAULT_WINDOW_SECONDS = 4.0
INPUT_NAME = "log_mel"
OUTPUT_NAME = "murmur_logit"

# Map CLI strings to coremltools deployment target constants without
# using getattr (which trips overly-eager static analysers).
TARGETS = {
    "iOS16": ct.target.iOS16,
    "iOS17": ct.target.iOS17,
    "iOS18": ct.target.iOS18,
    "macOS13": ct.target.macOS13,
    "macOS14": ct.target.macOS14,
    "macOS15": ct.target.macOS15,
}


def load_model(checkpoint: Path, architecture: str) -> torch.nn.Module:
    if architecture == "cnn":
        model = MurmurCNN()
    elif architecture == "cnn_bigru":
        model = MurmurCNNBiGRU()
    else:
        raise ValueError(f"unknown architecture {architecture}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def n_frames_for_window(window_seconds: float) -> int:
    return int(round(window_seconds * SAMPLE_RATE)) // STFT_HOP


def trace_model(model: torch.nn.Module, n_frames: int) -> torch.jit.ScriptModule:
    example = torch.zeros(1, 1, n_frames, N_MELS, dtype=torch.float32)
    return torch.jit.trace(model, example)


def export(
    checkpoint: Path,
    out_path: Path,
    architecture: str = "cnn",
    target_key: str = "iOS17",
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    murmur_aggregation: str | None = None,
    murmur_threshold: float | None = None,
    murmur_topk: int | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if target_key not in TARGETS:
        raise ValueError(f"unknown target {target_key}; valid: {sorted(TARGETS)}")

    n_frames = n_frames_for_window(window_seconds)
    log.info("loading checkpoint %s", checkpoint)
    model = load_model(checkpoint, architecture)
    log.info("tracing %.3fs input (%d frames)", window_seconds, n_frames)
    traced = trace_model(model, n_frames)

    log.info("converting to Core ML (mlprogram, compute_units=ALL)")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=INPUT_NAME, shape=(1, 1, n_frames, N_MELS), dtype=np.float32)],
        outputs=[ct.TensorType(name=OUTPUT_NAME)],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=TARGETS[target_key],
    )

    mlmodel.short_description = f"{architecture} heart-sound murmur classifier (binary)"
    mlmodel.author = "Shane Breazeale"
    mlmodel.license = "Apache-2.0 code; model trained on PhysioNet CirCor 2022 (ODC-By 1.0)"
    mlmodel.version = "0.2.0"
    mlmodel.user_defined_metadata["architecture"] = architecture
    mlmodel.user_defined_metadata["window_seconds"] = f"{window_seconds:g}"
    mlmodel.user_defined_metadata["n_frames"] = str(n_frames)
    if hasattr(mlmodel, "input_description"):
        mlmodel.input_description[INPUT_NAME] = (
            f"Log-mel spectrogram, shape (1, 1, {n_frames}, {N_MELS}). "
            f"{window_seconds:g}-s window @ 4 kHz, Hann/256-FFT/no-overlap, 32 Slaney mel bands, "
            "log10x10, per-frame z-score, -80 dB clip."
        )
        mlmodel.output_description[OUTPUT_NAME] = (
            "Raw logit for murmur-present. Apply sigmoid for probability."
        )

    mlmodel.save(str(out_path))
    metadata_path = sidecar_metadata_path(out_path)
    metadata = {
        "kind": "openstetho_model_metadata",
        "model": out_path.name,
        "architecture": architecture,
        "window_seconds": window_seconds,
        "n_frames": n_frames,
        "n_mels": N_MELS,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
    }
    if murmur_aggregation is not None:
        metadata["murmur_aggregation"] = murmur_aggregation
    if murmur_topk is not None:
        metadata["murmur_topk"] = murmur_topk
    if murmur_threshold is not None:
        metadata["murmur_threshold"] = murmur_threshold
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    log.info("wrote %s", out_path)
    return out_path


def sidecar_metadata_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}.openstetho.json")


def verify(
    checkpoint: Path,
    mlpackage: Path,
    architecture: str = "cnn",
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> dict:
    """Run both PyTorch and Core ML on the same random input. Reports the
    max absolute and mean absolute difference of the logit output."""
    rng = np.random.default_rng(0)
    n_frames = n_frames_for_window(window_seconds)
    x_np = rng.standard_normal((1, 1, n_frames, N_MELS)).astype(np.float32)

    model = load_model(checkpoint, architecture)
    with torch.no_grad():
        y_torch = model(torch.from_numpy(x_np)).numpy()

    mlmodel = ct.models.MLModel(str(mlpackage))
    t0 = time.perf_counter()
    out = mlmodel.predict({INPUT_NAME: x_np})
    cm_latency_ms = (time.perf_counter() - t0) * 1000
    y_coreml = np.asarray(out[OUTPUT_NAME]).reshape(y_torch.shape)

    diff = np.abs(y_torch - y_coreml)
    return {
        "torch_logit": float(y_torch.ravel()[0]),
        "coreml_logit": float(y_coreml.ravel()[0]),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "coreml_latency_ms": cm_latency_ms,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--architecture", choices=["cnn", "cnn_bigru"], default="cnn")
    p.add_argument("--target", default="iOS17", help=f"deployment target; one of {sorted(TARGETS)}")
    p.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    p.add_argument(
        "--murmur-aggregation",
        choices=["mean", "top3_mean", "top4_mean", "top5_mean", "topk_mean"],
    )
    p.add_argument("--murmur-topk", type=int)
    p.add_argument("--murmur-threshold", type=float)
    p.add_argument("--verify", action="store_true", help="torch vs coreml output check")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export(
        args.checkpoint,
        args.out,
        args.architecture,
        args.target,
        args.window_seconds,
        args.murmur_aggregation,
        args.murmur_threshold,
        args.murmur_topk,
    )

    if args.verify:
        log.info("running parity check")
        report = verify(args.checkpoint, args.out, args.architecture, args.window_seconds)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
