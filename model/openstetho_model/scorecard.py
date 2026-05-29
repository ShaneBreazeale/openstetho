"""Extract a compact, committable scorecard from a murmur ``cv_report.json``.

The CV runner (``cv_murmur``) writes a large per-run ``cv_report.json`` into a
gitignored ``runs/`` directory. That file is the source of truth for an
experiment, but it is too large and too volatile to commit, and its schema has
grown over time (older runs predate the ``cross_fold_calibration`` section).

A *scorecard* is the small, stable, version-controllable projection of a CV
report: a flat dict of the canonical decision metrics plus enough provenance to
know what produced it. Scorecards are what the regression gate
(:mod:`openstetho_model.regression_gate`) compares, and what we commit under
``model/benchmarks/`` so every "this model is better" claim is reproducible.

Usage::

    uv run --project model python -m openstetho_model.scorecard \\
        model/runs/murmur_cv_logmel_5s_bce_select_f1_v1/cv_report.json \\
        --name logmel_5s_bce_f1 \\
        --out model/benchmarks/scorecards/logmel_5s_bce_f1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# Canonical metric name -> dotted path into a ``cv_report.json``.
#
# All scorecard metrics are oriented "higher is better" EXCEPT the two
# calibration error metrics (``platt_ece_10``, ``platt_brier``); those are
# carried for reporting only and are never used as gate guards. Missing paths
# (e.g. ``cross_fold_calibration`` absent in a pre-calibration run) extract as
# ``None`` rather than raising, so old reports still produce a partial card.
METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "fold_auc_mean": ("fold_val_auc_mean",),
    "fold_auc_std": ("fold_val_auc_std",),
    "oof_auroc": ("oof", "auroc"),
    "oof_bestf1_f1": ("oof", "best_f1", "f1"),
    "oof_bestf1_sensitivity": ("oof", "best_f1", "sensitivity"),
    "oof_bestf1_specificity": ("oof", "best_f1", "specificity"),
    "platt_auroc": ("cross_fold_calibration", "platt", "probability", "auroc"),
    "platt_ece_10": ("cross_fold_calibration", "platt", "probability", "ece_10"),
    "platt_brier": ("cross_fold_calibration", "platt", "probability", "brier"),
    "platt_tt_bestf1_f1": (
        "cross_fold_calibration", "platt", "threshold_transfer", "best_f1", "f1",
    ),
    "platt_tt_bestf1_sensitivity": (
        "cross_fold_calibration", "platt", "threshold_transfer", "best_f1", "sensitivity",
    ),
    "platt_tt_bestf1_specificity": (
        "cross_fold_calibration", "platt", "threshold_transfer", "best_f1", "specificity",
    ),
    "platt_tt_spec90_sensitivity": (
        "cross_fold_calibration", "platt", "threshold_transfer",
        "specificity_ge_0_90", "sensitivity",
    ),
    "platt_tt_spec90_specificity": (
        "cross_fold_calibration", "platt", "threshold_transfer",
        "specificity_ge_0_90", "specificity",
    ),
    "platt_tt_sens80_specificity": (
        "cross_fold_calibration", "platt", "threshold_transfer",
        "sensitivity_ge_0_80", "specificity",
    ),
}

# Config fields copied verbatim from the report for provenance/diagnosis. These
# never gate anything; they answer "what training recipe produced this card".
CONFIG_FIELDS = (
    "feature_mode",
    "architecture",
    "window_seconds",
    "hop_seconds",
    "aggregation",
    "loss",
    "select_metric",
    "epochs",
    "wide_features",
    "no_cardiac",
    "n_recordings",
    "positives",
)


def _dig(report: dict[str, Any], path: tuple[str, ...]) -> float | None:
    """Follow a dotted path; return ``None`` if any hop is missing/non-dict."""
    node: Any = report
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    if isinstance(node, (int, float)):
        return float(node)
    return None


def extract_scorecard(report: dict[str, Any], name: str, source: str = "") -> dict[str, Any]:
    """Project a ``cv_report.json`` dict into a flat, committable scorecard."""
    metrics = {metric: _dig(report, path) for metric, path in METRIC_PATHS.items()}
    config = {field: report.get(field) for field in CONFIG_FIELDS if field in report}
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "source": source,
        "config": config,
        "metrics": metrics,
        "provenance": {
            # Older runs lack calibration; teacher distillation marks a card as
            # not-clean-public (see PROVENANCE.md / benchmark notes).
            "has_cross_fold_calibration": "cross_fold_calibration" in report,
            "teacher_distillation": report.get("teacher_predictions_csv") is not None,
        },
    }


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path, help="path to a cv_report.json")
    p.add_argument("--name", required=True, help="short scorecard identity, e.g. logmel_5s_bce_f1")
    p.add_argument("--out", type=Path, default=None, help="write scorecard JSON here; default stdout")
    args = p.parse_args()

    report = load_report(args.report)
    card = extract_scorecard(report, name=args.name, source=str(args.report))
    text = json.dumps(card, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
