"""Average out-of-fold predictions across clean CV runs into one report.

Seed/recipe bagging: each `cv_murmur` run writes one out-of-fold (OoF)
probability per recording -- always from a model that did not train on that
recording. Averaging those per-recording OoF probabilities across several runs
(different seeds, or seed + augmented sibling) reduces variance and almost
always improves ranking and probability calibration, which is exactly what the
regression gate's calibrated transferred-F1 primary metric rewards.

This is a clean, public-data-only technique: it consumes only committed-style
OoF CSVs, no teacher signal. The averaged predictions are scored with the same
`summarize` / `cross_fold_calibration_report` functions the CV runner uses, so
the resulting `cv_report.json` is drop-in for `scorecard` and the gate.

Fold partitions can differ across runs (fold split is seeded). Each recording
still has exactly one OoF prob per run, so averaging is valid; the canonical
fold assignment for the fold-held-out calibration is taken from the first
(reference) run.

Usage::

    uv run --project model python -m openstetho_model.ensemble_oof \\
        --runs runs/murmur_cv_logmel_5s_bce_select_f1_v1 \\
               runs/murmur_cv_logmel_5s_seed1 \\
               runs/murmur_cv_logmel_5s_seed2 \\
        --out runs/murmur_cv_logmel_5s_ensemble_v1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .cv_murmur import cross_fold_calibration_report, summarize
from .bench_murmur import json_safe


def _load_oof(run_dir: Path) -> dict[str, dict]:
    """Map recording -> row from a run's oof_predictions.csv.

    Carries patient_id/location so the ensemble OoF can double as a
    distillation teacher CSV (the teacher loader groups by patient_id,location).
    """
    rows: dict[str, dict] = {}
    with (run_dir / "oof_predictions.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            rows[row["recording"]] = {
                "fold": int(row["fold"]),
                "patient_id": row.get("patient_id", ""),
                "location": row.get("location", ""),
                "label": int(row["label"]),
                "prob": float(row["prob"]),
            }
    return rows


def ensemble(run_dirs: list[Path], threshold: float = 0.5) -> tuple[dict, list[dict]]:
    """Average per-recording OoF probs and rebuild a CV-style report dict."""
    members = [_load_oof(d) for d in run_dirs]
    reference = members[0]
    # Only score recordings present in every member, so each averaged prob is a
    # true mean over the same number of OoF models.
    shared = set.intersection(*(set(m) for m in members))
    recordings = [r for r in reference if r in shared]
    if len(recordings) != len(reference):
        dropped = len(reference) - len(recordings)
        print(f"warning: {dropped} recordings not shared across all runs; dropped from ensemble")

    pred_rows: list[dict] = []
    for rec in recordings:
        label = reference[rec]["label"]
        if any(m[rec]["label"] != label for m in members):
            raise ValueError(f"label mismatch for recording {rec} across runs")
        prob = float(np.mean([m[rec]["prob"] for m in members]))
        pred_rows.append({
            "fold": reference[rec]["fold"],
            "patient_id": reference[rec]["patient_id"],
            "location": reference[rec]["location"],
            "recording": rec,
            "label": label,
            "prob": prob,
        })

    folds = np.asarray([r["fold"] for r in pred_rows], dtype=np.int64)
    labels = np.asarray([r["label"] for r in pred_rows], dtype=np.int64)
    probs = np.asarray([r["prob"] for r in pred_rows], dtype=np.float64)

    report = {
        "ensemble_members": [str(d) for d in run_dirs],
        "ensemble_size": len(members),
        "feature_mode": "logmel",
        "architecture": "cnn_bigru",
        "aggregation": "mean",
        "n_recordings": int(labels.size),
        "positives": int(labels.sum()),
        "oof": summarize(labels, probs, threshold),
        "cross_fold_calibration": cross_fold_calibration_report(folds, labels, probs),
    }
    return report, pred_rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="2+ CV run dirs to average")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    if len(args.runs) < 2:
        p.error("--runs needs at least two run directories to ensemble")

    report, pred_rows = ensemble(args.runs, args.threshold)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cv_report.json").write_text(json.dumps(json_safe(report), indent=2) + "\n")
    with (args.out / "oof_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "patient_id", "location", "recording", "label", "prob"])
        w.writeheader()
        w.writerows(pred_rows)
    print(f"wrote {args.out}/cv_report.json ({report['ensemble_size']}-member ensemble, "
          f"{report['n_recordings']} recordings)")


if __name__ == "__main__":
    main()
