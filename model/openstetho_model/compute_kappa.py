"""Cohen's kappa for two cardiologist raters on the same cycle set.

Inputs are two CSVs with identical `slug` columns and per-rater `label_s3`
columns (0 / 1, with 9 treated as missing). Returns kappa, the 2×2 binary
confusion matrix on the agreed subset, and a list of slugs where raters
disagreed (for senior-reviewer adjudication).

CLI:
    uv run python -m openstetho_model.compute_kappa \\
        --rater-a labels_alice.csv --rater-b labels_bob.csv \\
        --out adjudication_needed.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _read_labels(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            slug = row["slug"]
            try:
                lbl = int(row["label_s3"])
            except (ValueError, KeyError):
                continue
            out[slug] = lbl
    return out


def cohens_kappa(pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return float("nan")
    n = len(pairs)
    classes = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    cm = {(a, b): 0 for a in classes for b in classes}
    for a, b in pairs:
        cm[(a, b)] += 1
    p_o = sum(cm[(c, c)] for c in classes) / n
    row_marg = {c: sum(cm[(c, b)] for b in classes) / n for c in classes}
    col_marg = {c: sum(cm[(a, c)] for a in classes) / n for c in classes}
    p_e = sum(row_marg[c] * col_marg[c] for c in classes)
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if p_o > 0.999 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--rater-a", required=True, type=Path)
    p.add_argument("--rater-b", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None, help="adjudication CSV for disagreements")
    args = p.parse_args()

    a = _read_labels(args.rater_a)
    b = _read_labels(args.rater_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("rater CSVs have no overlapping slugs")

    valid = [(a[s], b[s]) for s in shared if a[s] in (0, 1) and b[s] in (0, 1)]
    kappa = cohens_kappa(valid)
    print(f"n shared       : {len(shared)}")
    print(f"n usable (0/1) : {len(valid)}")
    print(f"Cohen's kappa  : {kappa:.3f}")

    # Confusion matrix (binary).
    cm = [[0, 0], [0, 0]]
    for ra, rb in valid:
        cm[ra][rb] += 1
    print("confusion (rows=rater_a, cols=rater_b, classes=[0=no_S3, 1=S3]):")
    for row in cm:
        print(" ", row)

    disagreements = [s for s in shared if a[s] in (0, 1) and b[s] in (0, 1) and a[s] != b[s]]
    print(f"disagreements  : {len(disagreements)}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["slug", "rater_a", "rater_b"])
            for s in disagreements:
                writer.writerow([s, a[s], b[s]])
        print(f"wrote adjudication list -> {args.out}")


if __name__ == "__main__":
    main()
