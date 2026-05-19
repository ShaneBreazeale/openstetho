"""Pick a stratified subset of cycles for cardiologist annotation.

Inputs:
    --scores       per-cycle CSV produced by `score_corpus.py`
    --murmur-csv   optional CirCor `training_data.csv` for patient-level
                   murmur stratification

Sampling strategy (Springer-inspired, see [[s3-annotation-protocol]]):
    * top-K confident positives (score >= q90)
    * top-K confident negatives (score <= q10)
    * top-K uncertain cycles around the decision boundary (0.4 <= s <= 0.6)
    * balance across strata so the annotator sees variety

The output CSV has one row per selected cycle; columns match the input
plus a `stratum` tag.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _load_scores(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["score"] = float(r["score"])
        r["s1_idx"] = int(r["s1_idx"])
        r["s2_idx"] = int(r["s2_idx"])
        r["next_s1_idx"] = int(r["next_s1_idx"])
        r["cycle_no"] = int(r["cycle_no"])
        r["segmenter_confidence"] = float(r["segmenter_confidence"])
    return rows


def _patient_from_wav(wav_path: str) -> str:
    stem = Path(wav_path).stem
    return stem.split("_", 1)[0] if "_" in stem else stem


def _maybe_load_murmur(murmur_csv: Path | None) -> dict[str, str]:
    """Patient → murmur status from CirCor training_data.csv.

    Returns empty dict if no file provided; downstream code stratifies on
    `unknown` for unmatched patients.
    """
    if murmur_csv is None:
        return {}
    out: dict[str, str] = {}
    with murmur_csv.open() as f:
        for row in csv.DictReader(f):
            pid = str(row.get("Patient ID", "")).strip()
            murmur = row.get("Murmur", "Unknown")
            if pid:
                out[pid] = murmur
    return out


def sample(
    rows: list[dict],
    n_per_stratum: int,
    murmur_status: dict[str, str],
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    scores = np.array([r["score"] for r in rows])
    if len(rows) == 0:
        return []

    q10 = float(np.quantile(scores, 0.10))
    q90 = float(np.quantile(scores, 0.90))
    log.info("score quantiles: q10=%.3f q50=%.3f q90=%.3f", q10, float(np.quantile(scores, 0.5)), q90)

    strata: dict[str, list[dict]] = {
        "high_positive": [r for r in rows if r["score"] >= q90],
        "uncertain":     [r for r in rows if 0.4 <= r["score"] <= 0.6],
        "high_negative": [r for r in rows if r["score"] <= q10],
    }

    out: list[dict] = []
    for name, pool in strata.items():
        if not pool:
            continue
        # Within each stratum, prefer balance across patient-level murmur
        # status so we don't oversample a single subgroup.
        groups: dict[str, list[dict]] = {}
        for r in pool:
            mstat = murmur_status.get(_patient_from_wav(r["wav"]), "Unknown")
            groups.setdefault(mstat, []).append(r)
        # Round-robin draw without replacement.
        keys = list(groups.keys())
        for k in keys:
            rng.shuffle(groups[k])
        picked: list[dict] = []
        while len(picked) < n_per_stratum and any(groups.values()):
            for k in keys:
                if groups[k] and len(picked) < n_per_stratum:
                    chosen = groups[k].pop()
                    chosen = dict(chosen, stratum=name)
                    picked.append(chosen)
        out.extend(picked)
        log.info("stratum %s: drew %d", name, len(picked))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-per-stratum", type=int, default=200)
    p.add_argument("--murmur-csv", type=Path, default=None, help="CirCor training_data.csv (optional)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = _load_scores(args.scores)
    murmur = _maybe_load_murmur(args.murmur_csv)
    selected = sample(rows, args.n_per_stratum, murmur, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "wav",
                "cycle_no",
                "s1_idx",
                "s2_idx",
                "next_s1_idx",
                "score",
                "segmenter_confidence",
                "stratum",
            ],
        )
        writer.writeheader()
        for r in selected:
            writer.writerow(r)
    log.info("wrote %d cycles -> %s", len(selected), args.out)


if __name__ == "__main__":
    main()
