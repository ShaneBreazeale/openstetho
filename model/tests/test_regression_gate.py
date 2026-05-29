"""Regression gate tests for the murmur detector.

These tests are the CI-enforceable form of "the model did not get worse":

* the committed baseline is internally consistent and self-comparison PASSes;
* a synthetic regression is caught;
* the documented clean-public improvement (BCE/F1 selection) beats the prior
  AUROC-selected lead;
* scorecard extraction is faithful and degrades gracefully on old reports.

They run offline against committed scorecards under ``model/benchmarks`` -- no
dataset, checkpoints, or GPU required -- so the gate is cheap enough to keep in
the normal ``uv run pytest`` path.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from openstetho_model.regression_gate import (
    BLOCKED,
    IMPROVED,
    PASS,
    REGRESSED,
    compute_gate,
    load_baseline,
)
from openstetho_model.scorecard import extract_scorecard

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
BASELINE_FILE = BENCH / "murmur_baseline.json"
SCORECARDS = BENCH / "scorecards"


def _card(name: str) -> dict:
    return json.loads((SCORECARDS / f"{name}.json").read_text())


def test_baseline_file_is_well_formed() -> None:
    card, gate = load_baseline(BASELINE_FILE)
    assert gate["primary_metric"] == "platt_tt_bestf1_f1"
    # The frozen baseline must actually report its own primary + guard metrics,
    # otherwise the gate can never detect a regression against it.
    assert card["metrics"][gate["primary_metric"]] is not None
    for guard in gate["guard_metrics"]:
        assert card["metrics"][guard] is not None
    # Public-posture invariant: the frozen baseline is not teacher-distilled.
    assert card["provenance"]["teacher_distillation"] is False


def test_self_comparison_passes_without_improving() -> None:
    card, gate = load_baseline(BASELINE_FILE)
    result = compute_gate(card, card, gate)
    assert result.verdict == PASS
    assert result.ok
    assert result.primary.delta == 0.0


def test_synthetic_regression_is_caught() -> None:
    card, gate = load_baseline(BASELINE_FILE)
    worse = copy.deepcopy(card)
    # Drop the primary metric well beyond tolerance.
    worse["metrics"]["platt_tt_bestf1_f1"] -= 0.05
    result = compute_gate(card, worse, gate)
    assert result.verdict == REGRESSED
    assert not result.ok


def test_guard_regression_is_caught_even_if_primary_improves() -> None:
    card, gate = load_baseline(BASELINE_FILE)
    candidate = copy.deepcopy(card)
    candidate["metrics"]["platt_tt_bestf1_f1"] += 0.05  # primary up
    candidate["metrics"]["oof_auroc"] -= 0.05  # but ranking collapses
    result = compute_gate(card, candidate, gate)
    assert result.verdict == REGRESSED


def test_missing_primary_metric_is_a_regression() -> None:
    """A candidate from an old schema that lacks the primary metric cannot pass."""
    card, gate = load_baseline(BASELINE_FILE)
    candidate = copy.deepcopy(card)
    candidate["metrics"]["platt_tt_bestf1_f1"] = None
    result = compute_gate(card, candidate, gate)
    assert result.verdict == REGRESSED
    assert any("missing" in r for r in result.reasons)


def test_shipped_ensemble_beats_prior_baseline() -> None:
    """Committed proof of the shipped clean win.

    The frozen baseline (v2) is a 3-seed bagged ensemble. This asserts it is a
    real, guard-clean improvement over the prior single-model baseline
    (BCE/F1 selection): calibrated transferred F1 0.597 -> 0.644 with ranking
    and the screening operating point also up. Public data only, no teacher.
    """
    _, gate = load_baseline(BASELINE_FILE)
    prior = _card("logmel_5s_bce_f1")
    shipped = _card("logmel_5s_ensemble3")
    result = compute_gate(prior, shipped, gate)
    assert result.verdict == IMPROVED
    assert result.primary.delta is not None and result.primary.delta >= gate["improvement_margin"]
    assert all(g.status != "regressed" for g in result.guards)
    assert shipped["provenance"]["teacher_distillation"] is False


def test_weaker_teacher_lead_no_longer_clears_raised_bar() -> None:
    """The bar rose: the teacher-distilled w=0.20 lead (transferred F1 0.626)
    no longer beats the clean ensemble baseline (0.644). It is also teacher-
    distilled, so the gate BLOCKs it on policy regardless.
    """
    baseline, gate = load_baseline(BASELINE_FILE)
    teacher = _card("logmel_5s_teacher_w020")
    result = compute_gate(baseline, teacher, gate)
    assert result.verdict != IMPROVED


def test_teacher_distilled_candidate_is_blocked_even_if_metrics_pass() -> None:
    """A teacher-distilled candidate is never promotable, even when its metrics
    beat the bar. Use the prior single-model baseline so the teacher w=0.20 card
    (transferred F1 0.626 > 0.597) WOULD pass on metrics — yet must BLOCK.
    """
    prior = _card("logmel_5s_bce_f1")
    teacher = _card("logmel_5s_teacher_w020")
    assert teacher["provenance"]["teacher_distillation"] is True

    _, gate = load_baseline(BASELINE_FILE)
    result = compute_gate(prior, teacher, gate)
    assert result.verdict == BLOCKED
    assert not result.ok
    assert any("teacher-distilled" in r for r in result.reasons)


def test_allow_teacher_distillation_unblocks_for_research_evaluation() -> None:
    """With the opt-in flag, a teacher card is evaluated on metrics (so it can be
    read as a research lead) instead of being policy-blocked."""
    prior = _card("logmel_5s_bce_f1")
    teacher = _card("logmel_5s_teacher_w020")
    _, gate = load_baseline(BASELINE_FILE)

    result = compute_gate(prior, teacher, {**gate, "allow_teacher_distillation": True})
    assert result.verdict != BLOCKED
    assert result.verdict == IMPROVED


def test_clean_candidate_is_unaffected_by_teacher_policy() -> None:
    """A non-teacher card never trips the policy block."""
    prior = _card("logmel_5s_bce_f1")
    shipped = _card("logmel_5s_ensemble3")
    assert shipped["provenance"]["teacher_distillation"] is False
    _, gate = load_baseline(BASELINE_FILE)
    result = compute_gate(prior, shipped, gate)
    assert result.verdict == IMPROVED


def test_extract_scorecard_is_faithful_and_graceful() -> None:
    full_report = {
        "fold_val_auc_mean": 0.83,
        "fold_val_auc_std": 0.02,
        "oof": {"auroc": 0.81, "best_f1": {"f1": 0.6, "sensitivity": 0.55, "specificity": 0.94}},
        "cross_fold_calibration": {
            "platt": {
                "probability": {"auroc": 0.80, "ece_10": 0.04, "brier": 0.11},
                "threshold_transfer": {
                    "best_f1": {"f1": 0.59, "sensitivity": 0.52, "specificity": 0.94},
                    "specificity_ge_0_90": {"sensitivity": 0.58, "specificity": 0.90},
                    "sensitivity_ge_0_80": {"specificity": 0.64},
                },
            }
        },
        "feature_mode": "logmel",
        "teacher_predictions_csv": None,
    }
    card = extract_scorecard(full_report, name="probe", source="x")
    assert card["metrics"]["oof_auroc"] == pytest.approx(0.81)
    assert card["metrics"]["platt_tt_bestf1_f1"] == pytest.approx(0.59)
    assert card["provenance"]["has_cross_fold_calibration"] is True
    assert card["provenance"]["teacher_distillation"] is False

    # An old, pre-calibration report yields None for the platt metrics rather
    # than raising -- so the extractor never breaks on legacy runs.
    old_report = {"fold_val_auc_mean": 0.82, "oof": {"auroc": 0.80}}
    old_card = extract_scorecard(old_report, name="old")
    assert old_card["metrics"]["oof_auroc"] == pytest.approx(0.80)
    assert old_card["metrics"]["platt_tt_bestf1_f1"] is None
    assert old_card["provenance"]["has_cross_fold_calibration"] is False
