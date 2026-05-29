"""Regression gate for murmur-detector scorecards.

Given a frozen *baseline* scorecard and a *candidate* scorecard (see
:mod:`openstetho_model.scorecard`), decide one of three verdicts:

* ``IMPROVED``  - the primary metric beat the baseline by at least
  ``improvement_margin`` and no guard metric regressed beyond ``guard_tolerance``.
* ``PASS``      - within the noise band: primary did not regress and no guard
  regressed, but the primary gain is below the improvement margin.
* ``REGRESSED`` - the primary dropped beyond tolerance, a guard metric dropped
  beyond tolerance, or a required metric is missing from the candidate.

This is the machine-checkable form of "prove the new model is better". The gate
config (primary metric, margins, guards) lives inside the committed baseline
file so the rule travels with the number it protects.

CLI (exits non-zero on ``REGRESSED``)::

    uv run --project model python -m openstetho_model.regression_gate \\
        --baseline model/benchmarks/murmur_baseline.json \\
        --candidate model/benchmarks/scorecards/logmel_5s_teacher_w020.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Default gate policy. Mirrors the design decision recorded in the benchmark
# notes: the primary metric is the calibrated, fold-held-out (transferred)
# best-F1, which is the least-optimistic deployable operating point. Guards are
# the ranking metric and the screening operating point, so a candidate cannot
# trade away ranking or recall-at-specificity to win on F1.
DEFAULT_GATE: dict[str, Any] = {
    "primary_metric": "platt_tt_bestf1_f1",
    "improvement_margin": 0.005,
    "guard_tolerance": 0.01,
    "guard_metrics": [
        "oof_auroc",
        "platt_auroc",
        "platt_tt_spec90_sensitivity",
    ],
}

IMPROVED = "IMPROVED"
PASS = "PASS"
REGRESSED = "REGRESSED"


@dataclass
class MetricDelta:
    name: str
    role: str  # "primary" | "guard"
    baseline: float | None
    candidate: float | None
    delta: float | None
    status: str  # "improved" | "ok" | "regressed" | "missing"


@dataclass
class GateResult:
    verdict: str
    primary: MetricDelta
    guards: list[MetricDelta] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict != REGRESSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "ok": self.ok,
            "primary": self.primary.__dict__,
            "guards": [g.__dict__ for g in self.guards],
            "reasons": self.reasons,
        }


def _metric(card: dict[str, Any], name: str) -> float | None:
    val = card.get("metrics", {}).get(name)
    return float(val) if isinstance(val, (int, float)) else None


def compute_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    gate: dict[str, Any] | None = None,
) -> GateResult:
    """Compare a candidate scorecard against a baseline under a gate policy.

    All gate metrics are treated as higher-is-better. A guard or primary metric
    that the baseline reports but the candidate lacks counts as a regression:
    you cannot drop coverage of a metric the baseline proved.
    """
    gate = {**DEFAULT_GATE, **(gate or {})}
    margin = float(gate["improvement_margin"])
    tol = float(gate["guard_tolerance"])
    primary_name = gate["primary_metric"]

    reasons: list[str] = []

    def make_delta(name: str, role: str) -> MetricDelta:
        b = _metric(baseline, name)
        c = _metric(candidate, name)
        if b is None:
            # Baseline never claimed this metric; nothing to protect.
            return MetricDelta(name, role, b, c, None, "ok")
        if c is None:
            reasons.append(f"{role} metric '{name}' missing from candidate (baseline={b:.4f})")
            return MetricDelta(name, role, b, c, None, "missing")
        d = c - b
        if role == "primary":
            status = "improved" if d >= margin else ("regressed" if d < -tol else "ok")
        else:
            status = "regressed" if d < -tol else ("improved" if d > tol else "ok")
        if status == "regressed":
            reasons.append(f"{role} metric '{name}' regressed: {b:.4f} -> {c:.4f} (Δ{d:+.4f}, tol {tol})")
        return MetricDelta(name, role, b, c, d, status)

    primary = make_delta(primary_name, "primary")
    guards = [make_delta(name, "guard") for name in gate["guard_metrics"]]

    any_regressed = (
        primary.status in ("regressed", "missing")
        or any(g.status in ("regressed", "missing") for g in guards)
    )
    if any_regressed:
        verdict = REGRESSED
    elif primary.status == "improved":
        verdict = IMPROVED
        reasons.append(
            f"primary '{primary_name}' improved by {primary.delta:+.4f} "
            f"(>= margin {margin}) with no guard regression"
        )
    else:
        verdict = PASS
        reasons.append(
            f"primary '{primary_name}' within band (Δ{primary.delta:+.4f}); no regression"
        )
    return GateResult(verdict=verdict, primary=primary, guards=guards, reasons=reasons)


def _fmt(x: float | None) -> str:
    return "  n/a  " if x is None else f"{x:7.4f}"


def render(result: GateResult) -> str:
    lines = [f"verdict: {result.verdict}"]
    rows = [result.primary, *result.guards]
    lines.append(f"  {'metric':32s} {'role':8s} {'baseline':>8s} {'candidate':>9s} {'delta':>8s}  status")
    for r in rows:
        d = "" if r.delta is None else f"{r.delta:+.4f}"
        lines.append(
            f"  {r.name:32s} {r.role:8s} {_fmt(r.baseline)} {_fmt(r.candidate)} {d:>8s}  {r.status}"
        )
    for reason in result.reasons:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def load_baseline(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(scorecard, gate_config)`` from a baseline file.

    A baseline file bundles the frozen scorecard with the gate policy that
    protects it, so the comparison rule cannot drift away from the number.
    """
    doc = json.loads(path.read_text())
    return doc["scorecard"], doc.get("gate", {})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True, help="frozen baseline file (scorecard + gate)")
    p.add_argument("--candidate", type=Path, required=True, help="candidate scorecard JSON")
    p.add_argument("--json", type=Path, default=None, help="optional path to write the gate result JSON")
    args = p.parse_args()

    baseline_card, gate = load_baseline(args.baseline)
    candidate_card = json.loads(args.candidate.read_text())
    result = compute_gate(baseline_card, candidate_card, gate)

    print(render(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_dict(), indent=2) + "\n")

    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
