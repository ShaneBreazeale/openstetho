"""Fit the deployed ensemble's calibration from its out-of-fold predictions.

The fused ensemble (`export_ensemble`) outputs `logit(mean member prob)`. Its
probabilities are only as honest as the members' training objective allows:
pos_weight, synthetic mixing and seed averaging all bend them. This module
fits one affine map `a*z + b` on that logit from the ensemble's OoF CSV
(`ensemble_oof` output: every probability came from models that never saw the
recording), choosing identity, temperature or Platt by patient-grouped
cross-fitting (`calibration.cross_fit_recalibration`). `export_ensemble
--calibration` bakes the map into the Core ML graph.

The operating threshold moves with it: the raw OoF best-Youden threshold `t`
becomes `sigmoid(a*logit(t) + b)`, which gives the same decisions on these
recording-level OoF probabilities. stetho-ui instead averages per-window
probabilities, and averaging after a non-linear map shifts the mean, so check
the threshold on that scale before release (see the benchmark doc).

Usage::

    uv run --project model python -m openstetho_model.calibrate_ensemble \\
        --oof runs/murmur_cv_ensemble/oof_predictions.csv \\
        --out runs/murmur_cv_ensemble/calibration.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from .bench_murmur import json_safe
from .calibration import (
    calibration_line,
    cross_fit_recalibration,
    logit,
    nll,
    sigmoid,
)
from .thresholds import sweep_thresholds, threshold_policy_row

log = logging.getLogger("calibrate_ensemble")


def load_oof(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(labels, probs, patient ids) from an OoF predictions CSV."""
    labels, probs, groups = [], [], []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["label"]))
            probs.append(float(row["prob"]))
            groups.append(int(row["patient_id"]))
    return np.asarray(labels), np.asarray(probs, dtype=np.float64), np.asarray(groups)


def fit_calibration(
    labels: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 0,
    threshold_policy: str = "best_youden_j",
) -> dict:
    z = np.asarray(logit(np.asarray(probs, dtype=np.float64)), dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    rec = cross_fit_recalibration(z, y, groups, n_folds=n_folds, seed=seed)
    raw_threshold = float(threshold_policy_row(sweep_thresholds(labels, probs), threshold_policy)["threshold"])
    mapped = float(sigmoid(rec.a * logit(raw_threshold) + rec.b))
    slope_before, icpt_before = calibration_line(probs, y)
    slope_after, icpt_after = calibration_line(sigmoid(rec.oof_logits), y)
    return {
        "kind": "openstetho_ensemble_calibration",
        "method": rec.method,
        "a": rec.a,
        "b": rec.b,
        "n": int(y.size),
        "prior": float(y.mean()),
        "n_folds": n_folds,
        "seed": seed,
        "heldout_nll": rec.heldout_nll,
        "oof_nll_before": nll(z, y),
        "oof_nll_after": nll(rec.oof_logits, y),
        "cal_slope_before": slope_before,
        "cal_intercept_before": icpt_before,
        "cal_slope_after": slope_after,
        "cal_intercept_after": icpt_after,
        "threshold_policy": threshold_policy,
        "raw_threshold": raw_threshold,
        "murmur_threshold": mapped,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", type=Path, required=True, help="ensemble oof_predictions.csv")
    p.add_argument("--out", type=Path, required=True, help="calibration JSON for export_ensemble")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold-policy", default="best_youden_j")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    labels, probs, groups = load_oof(args.oof)
    record = fit_calibration(
        labels, probs, groups,
        n_folds=args.n_folds, seed=args.seed, threshold_policy=args.threshold_policy,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(record), indent=2) + "\n")
    log.info("held-out NLL by candidate: %s", {k: round(v, 5) for k, v in record["heldout_nll"].items()})
    log.info(
        "chose %s: logit -> %.4f*z %+.4f | OoF NLL %.4f -> %.4f | cal line %.2f/%+.2f -> %.2f/%+.2f",
        record["method"], record["a"], record["b"], record["oof_nll_before"], record["oof_nll_after"],
        record["cal_slope_before"], record["cal_intercept_before"],
        record["cal_slope_after"], record["cal_intercept_after"],
    )
    log.info(
        "%s threshold %.4f -> %.4f after calibration; wrote %s",
        args.threshold_policy, record["raw_threshold"], record["murmur_threshold"], args.out,
    )


if __name__ == "__main__":
    main()
