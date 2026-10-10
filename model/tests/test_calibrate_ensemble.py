from __future__ import annotations

import numpy as np
import pytest

from openstetho_model.calibrate_ensemble import fit_calibration, load_oof
from openstetho_model.calibration import logit, sigmoid


def _overconfident_oof(n_patients: int = 400, seed: int = 0):
    """OoF probabilities from a true posterior pushed through logit*2.5:
    a temperature of 2.5 should come back out."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_patients), 3)
    true_z = rng.normal(-1.2, 1.5, groups.size)
    labels = (rng.random(groups.size) < sigmoid(true_z)).astype(int)
    probs = sigmoid(2.5 * true_z)
    return labels, probs, groups


def test_fit_recovers_temperature_and_lowers_heldout_nll():
    labels, probs, groups = _overconfident_oof()
    record = fit_calibration(labels, probs, groups, n_folds=5)
    assert record["method"] in ("temperature", "platt")
    assert record["a"] == pytest.approx(1 / 2.5, rel=0.15)
    assert record["oof_nll_after"] < record["oof_nll_before"]
    assert record["cal_slope_after"] == pytest.approx(1.0, abs=0.15)


def test_threshold_maps_through_the_same_affine():
    labels, probs, groups = _overconfident_oof(seed=1)
    record = fit_calibration(labels, probs, groups, n_folds=5)
    expected = sigmoid(record["a"] * logit(record["raw_threshold"]) + record["b"])
    assert record["murmur_threshold"] == pytest.approx(float(expected))
    # Same decisions before and after: the map is monotone increasing.
    raw_pos = probs >= record["raw_threshold"]
    cal = sigmoid(record["a"] * logit(probs) + record["b"])
    assert np.array_equal(raw_pos, cal >= record["murmur_threshold"] - 1e-12)


def test_load_oof_reads_ensemble_csv(tmp_path):
    path = tmp_path / "oof.csv"
    path.write_text(
        "fold,patient_id,location,recording,label,prob\n"
        "1,7,AV,a.wav,0,0.2\n2,8,MV,b.wav,1,0.9\n"
    )
    labels, probs, groups = load_oof(path)
    assert labels.tolist() == [0, 1] and probs.tolist() == [0.2, 0.9] and groups.tolist() == [7, 8]
