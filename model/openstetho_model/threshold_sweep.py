"""Threshold sweep + ROC analysis on combined real-labeled validation sets.

Combines multiple `predictions.csv` files (the per-clip output of
`validate_clips`), walks the decision threshold from 0 to 1, and prints
sensitivity / specificity / F1 / Youden's J at every step. Writes a CSV of
the curve points for later plotting.

CLI:
    uv run python -m openstetho_model.threshold_sweep \\
        --preds data/s3_validation/uw/predictions.csv \\
                data/s3_validation/umich/predictions.csv \\
        --out  data/s3_validation/threshold_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load(paths: list[Path]) -> list[tuple[str, int, float]]:
    rows: list[tuple[str, int, float]] = []
    for p in paths:
        with p.open() as f:
            for row in csv.DictReader(f):
                rows.append((row["filename"], int(row["label_s3"]), float(row["score"])))
    return rows


def sweep(rows: list[tuple[str, int, float]]) -> list[dict]:
    pos = [r for r in rows if r[1] == 1]
    neg = [r for r in rows if r[1] == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    out: list[dict] = []
    if n_pos == 0 or n_neg == 0:
        return out
    thresholds = sorted({round(r[2], 3) for r in rows} | {0.0, 1.0})
    for thr in thresholds:
        tp = sum(1 for _, _, s in pos if s >= thr)
        fn = n_pos - tp
        fp = sum(1 for _, _, s in neg if s >= thr)
        tn = n_neg - fp
        sens = tp / max(n_pos, 1)
        spec = tn / max(n_neg, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 0.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn)
        youden = sens + spec - 1.0
        out.append({
            "threshold": thr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": sens, "specificity": spec,
            "precision": prec, "f1": f1, "youden_j": youden,
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preds", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    rows = _load(args.preds)
    print(f"n total: {len(rows)} | positives: {sum(1 for r in rows if r[1]==1)}")
    curve = sweep(rows)
    if not curve:
        raise SystemExit("no positives or no negatives in combined set")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader()
        for row in curve:
            w.writerow(row)
    print(f"wrote {args.out}")

    # Identify operating points.
    by_youden = max(curve, key=lambda r: r["youden_j"])
    by_f1     = max(curve, key=lambda r: r["f1"])
    perfect_spec = [r for r in curve if r["sensitivity"] == 1.0 and r["specificity"] == 1.0]
    print()
    print(f"best Youden J  : thr={by_youden['threshold']:.3f}  sens={by_youden['sensitivity']:.3f}  spec={by_youden['specificity']:.3f}  J={by_youden['youden_j']:.3f}")
    print(f"best F1        : thr={by_f1['threshold']:.3f}  F1={by_f1['f1']:.3f}  sens={by_f1['sensitivity']:.3f}  spec={by_f1['specificity']:.3f}")
    if perfect_spec:
        # Highest threshold that still keeps perfect sens=1, spec=1.
        perfect_low_thr = min(perfect_spec, key=lambda r: r["threshold"])
        perfect_high_thr = max(perfect_spec, key=lambda r: r["threshold"])
        print(f"perfect window : thr in [{perfect_low_thr['threshold']:.3f}, {perfect_high_thr['threshold']:.3f}]  sens=1.0 spec=1.0")
    else:
        print("no threshold achieves perfect sens=spec=1.0 on this set")


if __name__ == "__main__":
    main()
