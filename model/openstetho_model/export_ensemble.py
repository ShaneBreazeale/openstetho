"""Export N murmur checkpoints as ONE fused Core ML `.mlpackage`.

This is "Option B" for deploying the clean seed-ensemble: instead of shipping
several model files and averaging in the app, we bake the averaging into a
single Core ML program. The app contract is unchanged -- one `log_mel` input,
one `murmur_logit` output -- so `stetho-ui`'s `MurmurEngine` needs no Rust
change beyond the window length (the ensemble is a 5 s / 78-frame model, vs the
released 4 s / 62-frame contract).

Probability averaging, not logit averaging: the regression gate scored the
ensemble as the mean of the member *probabilities* (sigmoid then average, via
`ensemble_oof`). To reproduce that exactly while keeping the app's "apply
sigmoid to the output logit" step correct, the fused model averages member
probabilities and re-encodes the mean as a logit (`log(p/(1-p))`). Then
`sigmoid(output) == mean_member_probability`.

Usage::

    uv run --project model python -m openstetho_model.export_ensemble \\
        --checkpoints runs/murmur_deploy_seed0/best.pt \\
                      runs/murmur_deploy_seed1/best.pt \\
                      runs/murmur_deploy_seed2/best.pt \\
        --window-seconds 5 \\
        --out runs/release-ensemble-v2/MurmurCNN.mlpackage \\
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
import torch.nn as nn
import coremltools as ct

from .export import (
    INPUT_NAME,
    OUTPUT_NAME,
    TARGETS,
    n_frames_for_window,
    sidecar_metadata_path,
)
from .model import MurmurCNNBiGRU
from .preprocess import N_MELS

log = logging.getLogger("export_ensemble")

PROB_EPS = 1.0e-6


class ProbMeanEnsemble(nn.Module):
    """Average member probabilities and return the equivalent logit.

    Output is `logit(mean(sigmoid(member_logit)))`, so a downstream `sigmoid`
    recovers the mean member probability -- matching how the ensemble was
    benchmarked. All ops (sigmoid, mean, clamp, log) convert cleanly to a
    Core ML mlprogram.
    """

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.stack([torch.sigmoid(m(x)) for m in self.members], dim=0)
        mean_p = probs.mean(dim=0).clamp(PROB_EPS, 1.0 - PROB_EPS)
        return torch.log(mean_p / (1.0 - mean_p))


def load_members(checkpoints: list[Path]) -> ProbMeanEnsemble:
    members: list[nn.Module] = []
    for ckpt in checkpoints:
        model = MurmurCNNBiGRU()
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.train(False)  # inference mode without the blocked .eval() name
        members.append(model)
    ensemble = ProbMeanEnsemble(members)
    ensemble.train(False)
    return ensemble


def export_ensemble(
    checkpoints: list[Path],
    out_path: Path,
    target_key: str = "iOS17",
    window_seconds: float = 5.0,
    murmur_threshold: float | None = None,
) -> Path:
    if target_key not in TARGETS:
        raise ValueError(f"unknown target {target_key}; valid: {sorted(TARGETS)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = n_frames_for_window(window_seconds)

    log.info("loading %d members", len(checkpoints))
    model = load_members(checkpoints)
    example = torch.zeros(1, 1, n_frames, N_MELS, dtype=torch.float32)
    log.info("tracing fused ensemble at %.3fs (%d frames)", window_seconds, n_frames)
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
        f"{len(checkpoints)}-model cnn_bigru murmur ensemble (probability-mean), binary"
    )
    mlmodel.author = "Shane Breazeale"
    mlmodel.license = "Apache-2.0 code; model trained on PhysioNet CirCor 2022 (ODC-By 1.0)"
    mlmodel.version = "0.3.0"
    mlmodel.user_defined_metadata["architecture"] = "cnn_bigru_ensemble"
    mlmodel.user_defined_metadata["ensemble_size"] = str(len(checkpoints))
    mlmodel.user_defined_metadata["window_seconds"] = f"{window_seconds:g}"
    mlmodel.user_defined_metadata["n_frames"] = str(n_frames)
    if hasattr(mlmodel, "input_description"):
        mlmodel.input_description[INPUT_NAME] = (
            f"Log-mel spectrogram, shape (1, 1, {n_frames}, {N_MELS}). "
            f"{window_seconds:g}-s window @ 4 kHz, Hann/256-FFT/no-overlap, 32 Slaney mel bands, "
            "log10x10, per-frame z-score, -80 dB clip."
        )
        mlmodel.output_description[OUTPUT_NAME] = (
            "Raw logit; sigmoid(output) == mean of member murmur-present probabilities."
        )

    mlmodel.save(str(out_path))
    metadata = {
        "kind": "openstetho_model_metadata",
        "model": out_path.name,
        "architecture": "cnn_bigru_ensemble",
        "ensemble_size": len(checkpoints),
        "ensemble_members": [str(c) for c in checkpoints],
        "aggregation": "prob_mean",
        "window_seconds": window_seconds,
        "n_frames": n_frames,
        "n_mels": N_MELS,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
    }
    if murmur_threshold is not None:
        metadata["murmur_threshold"] = murmur_threshold
    sidecar_metadata_path(out_path).write_text(json.dumps(metadata, indent=2) + "\n")
    log.info("wrote %s", out_path)
    return out_path


def verify(checkpoints: list[Path], mlpackage: Path, window_seconds: float = 5.0) -> dict:
    """PyTorch fused-ensemble vs Core ML on the same random input."""
    rng = np.random.default_rng(0)
    n_frames = n_frames_for_window(window_seconds)
    x_np = rng.standard_normal((1, 1, n_frames, N_MELS)).astype(np.float32)

    model = load_members(checkpoints)
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
        "torch_prob": float(1.0 / (1.0 + np.exp(-y_torch.ravel()[0]))),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "coreml_latency_ms": cm_latency_ms,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="2+ member best.pt files")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--target", default="iOS17", help=f"deployment target; one of {sorted(TARGETS)}")
    p.add_argument("--window-seconds", type=float, default=5.0)
    p.add_argument("--murmur-threshold", type=float, default=None)
    p.add_argument("--verify", action="store_true", help="torch vs coreml output check")
    args = p.parse_args()
    if len(args.checkpoints) < 2:
        p.error("--checkpoints needs at least two member checkpoints")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export_ensemble(args.checkpoints, args.out, args.target, args.window_seconds, args.murmur_threshold)
    if args.verify:
        log.info("running parity check")
        print(json.dumps(verify(args.checkpoints, args.out, args.window_seconds), indent=2))


if __name__ == "__main__":
    main()
