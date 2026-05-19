"""Tests for Cohen's kappa helper."""
from __future__ import annotations

from openstetho_model.compute_kappa import cohens_kappa


def test_perfect_agreement_returns_one():
    pairs = [(0, 0), (1, 1), (1, 1), (0, 0)]
    assert cohens_kappa(pairs) == 1.0


def test_perfect_disagreement_is_negative():
    pairs = [(0, 1), (1, 0), (0, 1), (1, 0)]
    k = cohens_kappa(pairs)
    assert k < 0.0


def test_partial_agreement_in_zero_one_range():
    pairs = [(0, 0)] * 7 + [(1, 1)] * 2 + [(0, 1)] * 1
    k = cohens_kappa(pairs)
    assert 0.0 < k < 1.0


def test_empty_returns_nan():
    import math

    assert math.isnan(cohens_kappa([]))


def test_all_same_class_returns_one_or_zero():
    # If every pair is (0, 0), kappa is degenerate. We return 1.0 when
    # everyone agreed and the class collapsed; this is the expected behavior
    # of the implementation (and is acceptable downstream).
    pairs = [(0, 0)] * 10
    assert cohens_kappa(pairs) == 1.0
