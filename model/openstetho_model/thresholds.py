from __future__ import annotations

import numpy as np


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return {
        "n": int(labels.size),
        "positives": int(labels.sum()),
        "threshold": float(threshold),
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
    }


def sweep_thresholds(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    return [binary_metrics(labels, scores, float(t)) for t in thresholds]


def best_by(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    if not rows:
        return {}
    return max(rows, key=lambda r: (r[key], r["specificity"], r["threshold"]))


THRESHOLD_METRICS = {"best_f1": "f1", "best_youden_j": "youden_j"}
SPECIFICITY_POLICY_TARGETS = {
    "specificity_ge_0_90": 0.90,
    "specificity_ge_0_93": 0.93,
    "specificity_ge_0_94": 0.94,
    "specificity_ge_0_95": 0.95,
}
THRESHOLD_POLICY_NAMES = (
    "best_f1",
    "best_youden_j",
    "sensitivity_ge_0_80",
    *SPECIFICITY_POLICY_TARGETS.keys(),
)


def threshold_policy_row(rows: list[dict[str, float]], policy: str) -> dict[str, float]:
    if policy in THRESHOLD_METRICS:
        out = dict(best_by(rows, THRESHOLD_METRICS[policy]))
        out["constraint_met"] = 1.0
        return out
    if policy == "sensitivity_ge_0_80":
        candidates = [row for row in rows if row["sensitivity"] >= 0.80]
        if candidates:
            out = dict(max(candidates, key=lambda r: (r["specificity"], r["f1"], r["threshold"])))
            out["constraint_met"] = 1.0
            return out
        out = dict(max(rows, key=lambda r: (r["sensitivity"], r["specificity"], r["f1"])))
        out["constraint_met"] = 0.0
        return out
    if policy in SPECIFICITY_POLICY_TARGETS:
        return specificity_constrained_row(rows, SPECIFICITY_POLICY_TARGETS[policy])
    raise ValueError(f"unknown threshold policy {policy!r}; valid: {THRESHOLD_POLICY_NAMES}")


def specificity_constrained_row(
    rows: list[dict[str, float]],
    min_specificity: float,
) -> dict[str, float]:
    candidates = [row for row in rows if row["specificity"] >= min_specificity]
    if candidates:
        out = dict(max(candidates, key=lambda r: (r["sensitivity"], r["f1"], r["threshold"])))
        out["constraint_met"] = 1.0
        return out
    out = dict(max(rows, key=lambda r: (r["specificity"], r["sensitivity"], r["f1"])))
    out["constraint_met"] = 0.0
    return out
