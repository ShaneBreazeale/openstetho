"""Offline single-WAV scorer — load a trained MurmurCNN checkpoint and a
WAV file through the Python preprocess pipeline, then print per-window
murmur probabilities.

This tool exists to triage two distinct failure classes that produce
identical user-visible symptoms (suspicious live murmur %):

  * If this script agrees with the live stetho-ui reading on the same
    audio, the trained model itself is the bug. Fix at the training /
    labels layer.
  * If they disagree, the bug lives in the Rust / CoreML deployment
    path — likely mel-spec or filter drift.

Usage (from `model/`):

    uv run python -m openstetho_model.predict_wav \\
        --checkpoint runs/combined_v4/best.pt \\
        --wav /tmp/normal_test.wav
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .model import MurmurCNN
from .preprocess import (
    N_MELS,
    SAMPLE_RATE,
    apply_cardiac,
    load_audio,
    log_mel,
    split_windows,
)


def _set_inference_mode(model: torch.nn.Module) -> None:
    """Disable training-mode behaviour (dropout, batchnorm running stats
    swap)."""
    model.train(False)


def score_wav(
    checkpoint: Path,
    wav: Path,
    apply_cardiac_filter: bool = False,
) -> dict:
    audio = load_audio(str(wav))
    if apply_cardiac_filter:
        audio = apply_cardiac(audio)
    audio_seconds = len(audio) / SAMPLE_RATE

    windows = split_windows(audio)
    if len(windows) == 0:
        return {
            "wav": str(wav),
            "seconds_of_audio": audio_seconds,
            "n_windows": 0,
            "message": "audio too short for one 4-s window",
        }

    model = MurmurCNN()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    _set_inference_mode(model)

    mels = np.stack([log_mel(w) for w in windows], axis=0)
    x = torch.from_numpy(mels).unsqueeze(1).float()

    with torch.no_grad():
        logits = model(x).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))

    # Zero-input baseline: mean-centred mel-spec is roughly zero, so a
    # zero tensor approximates what the trained head outputs from no
    # signal at all. Indicates how far the final bias drifted.
    with torch.no_grad():
        zero_logit = float(
            model(torch.zeros(1, 1, mels.shape[1], N_MELS)).item()
        )
    zero_prob = 1.0 / (1.0 + float(np.exp(-zero_logit)))

    return {
        "wav": str(wav),
        "checkpoint": str(checkpoint),
        "apply_cardiac": apply_cardiac_filter,
        "seconds_of_audio": round(audio_seconds, 2),
        "n_windows": int(len(windows)),
        "per_window_prob": [round(float(p), 3) for p in probs.tolist()],
        "per_window_logit": [round(float(L), 3) for L in logits.tolist()],
        "mean_prob": round(float(probs.mean()), 3),
        "std_prob": round(float(probs.std()), 3),
        "median_prob": round(float(np.median(probs)), 3),
        "zero_input_prob": round(zero_prob, 3),
        "zero_input_logit": round(zero_logit, 3),
        "interpretation_hint": (
            "Compare mean_prob against the live stetho-ui reading on the same "
            "audio. zero_input_prob is the model's response to a featureless "
            "input — if it is already > 0.5 the trained head's bias is "
            "leaning positive."
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--apply-cardiac", action="store_true",
                   help="Run the Python cardiac biquad chain before mel-spec. "
                        "Default off — matches the decoupled training and "
                        "the Rust deployment path.")
    p.add_argument("--json-only", action="store_true",
                   help="Print only the JSON report.")
    args = p.parse_args()

    if not args.checkpoint.exists():
        print(f"no such checkpoint: {args.checkpoint}", file=sys.stderr)
        sys.exit(2)
    if not args.wav.exists():
        print(f"no such wav: {args.wav}", file=sys.stderr)
        sys.exit(2)

    report = score_wav(args.checkpoint, args.wav, args.apply_cardiac)

    if args.json_only:
        print(json.dumps(report, indent=2))
        return

    print(f"WAV           {report['wav']}")
    print(f"checkpoint    {report['checkpoint']}")
    print(f"apply_cardiac {report['apply_cardiac']}")
    print(f"audio         {report['seconds_of_audio']} s -> {report['n_windows']} windows")
    if report["n_windows"] == 0:
        print(report.get("message", ""))
        return
    print(f"mean prob     {report['mean_prob']}")
    print(f"median prob   {report['median_prob']}")
    print(f"std  prob     {report['std_prob']}")
    print(f"per window    {report['per_window_prob']}")
    print(f"zero-input    prob={report['zero_input_prob']}  logit={report['zero_input_logit']}")
    print()
    print(report["interpretation_hint"])


if __name__ == "__main__":
    main()
