"""Tests for openstetho_model.calibration — every helper is checked against a
synthetic problem whose answer is known in closed form."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from openstetho_model.calibration import (
    bake_affine,
    brier,
    calibration_line,
    captures_needed,
    choose_recalibration,
    classification_metrics,
    cross_fit_recalibration,
    deploy_affine,
    effective_positive_prior,
    fit_deploy_affine,
    fit_learning_curve,
    fit_platt,
    logit,
    nll,
    prior_shift_offset,
    sigmoid,
    temperature_scale,
)
from openstetho_model.model import MurmurCNN


# ─── prior_shift_offset ─────────────────────────────────────────────────────

@pytest.mark.parametrize("p,w", [(0.3, 2.5), (0.1, 9.0), (0.6, 0.5)])
def test_weighted_bce_minimiser_is_logit_plus_log_w(p, w):
    """The derivation in prior_shift_offset's docstring, checked numerically."""
    z = np.linspace(-10, 10, 400_001)
    loss = w * p * np.logaddexp(0, -z) + (1 - p) * np.logaddexp(0, z)
    assert z[loss.argmin()] == pytest.approx(logit(p) + math.log(w), abs=1e-3)


def test_prior_shift_offset_recovers_true_bias_after_weighted_training():
    """End to end: train a 1-D logistic model with pos_weight on Gaussian
    classes whose log-likelihood ratio is known, apply the offset, and
    land on the Bayes-optimal bias for the deployment prior."""
    rng = np.random.default_rng(0)
    mu0, mu1 = 0.0, 1.5
    pi_train, pi_deploy = 0.25, 0.10
    w = (1 - pi_train) / pi_train  # train.py's "balanced" pos_weight

    n = 40_000
    y = (rng.random(n) < pi_train).astype(np.float64)
    x = rng.normal(np.where(y == 1, mu1, mu0), 1.0)

    def loss_and_grad(theta):
        a, b = theta
        z = a * x + b
        loss = np.mean(w * y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z))
        r = -w * y * sigmoid(-z) + (1 - y) * sigmoid(z)
        return loss, np.array([np.mean(r * x), np.mean(r)])

    a_hat, b_hat = minimize(loss_and_grad, x0=[0.0, 0.0], jac=True, method="BFGS").x

    # LLR(x) = (mu1 - mu0) x - (mu1² - mu0²) / 2
    true_deploy_bias = -(mu1**2 - mu0**2) / 2 + logit(pi_deploy)
    corrected = b_hat + prior_shift_offset(w, pi_train, pi_deploy)
    assert a_hat == pytest.approx(mu1 - mu0, abs=0.05)
    assert corrected == pytest.approx(true_deploy_bias, abs=0.1)


def test_prior_shift_offset_balanced_sampler_case():
    # Balanced sampler: w=1, effective train prior 0.5 → offset = logit(π_d).
    assert prior_shift_offset(1.0, 0.5, 0.3) == pytest.approx(logit(0.3))


def test_prior_shift_offset_rejects_nonpositive_weight():
    with pytest.raises(ValueError):
        prior_shift_offset(0.0, 0.5, 0.5)


# ─── calibration_line ───────────────────────────────────────────────────────

def _bernoulli(rng, p):
    return (rng.random(p.shape) < p).astype(np.float64)


def test_calibration_line_identity_for_calibrated_model():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, 50_000)
    p = sigmoid(z)
    slope, intercept = calibration_line(p, _bernoulli(rng, p))
    assert slope == pytest.approx(1.0, abs=0.05)
    assert intercept == pytest.approx(0.0, abs=0.05)


def test_calibration_line_detects_overconfidence():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 2, 50_000)
    y = _bernoulli(rng, sigmoid(z / 2))  # truth is half as sharp as reported
    slope, intercept = calibration_line(sigmoid(z), y)
    assert slope == pytest.approx(0.5, abs=0.05)
    assert intercept == pytest.approx(0.0, abs=0.05)


def test_calibration_line_detects_pos_weight_bias():
    """CirCor pos_weight ≈ 3.88 pushes every logit up by ln 3.88 ≈ 1.36.
    The calibration intercept must report that shift with opposite sign."""
    rng = np.random.default_rng(3)
    z = rng.normal(-1, 1.5, 50_000)
    y = _bernoulli(rng, sigmoid(z))
    shift = math.log(3.88)
    slope, intercept = calibration_line(sigmoid(z + shift), y)
    assert slope == pytest.approx(1.0, abs=0.05)
    assert intercept == pytest.approx(-shift, abs=0.05)



@pytest.mark.parametrize("scale,slope,intercept", [(25, 0.1, 1.5), (15, 0.2, -1.0), (8, 0.3, 2.0)])
def test_calibration_line_converges_for_badly_overconfident_logits(scale, slope, intercept):
    """Undamped Newton from (1, 0) overshoots when logits are far too
    spread out — exactly the overconfident checkpoints we need to catch.
    Seen live: circor_v2 epoch 7 logged a NaN calibration line."""
    rng = np.random.default_rng(12)
    z = rng.normal(0, scale, 20_000)
    y = _bernoulli(rng, sigmoid(slope * z + intercept))
    # Raw logits, as the trainers pass them: going through probabilities
    # would clip |z| > 16 and bias the slope regardless of the optimiser.
    m = classification_metrics(z, y)
    assert m["cal_slope"] == pytest.approx(slope, abs=0.03)
    assert m["cal_intercept"] == pytest.approx(intercept, abs=0.15)
    assert fit_platt(z, y) == pytest.approx((m["cal_slope"], m["cal_intercept"]))

# ─── learning curve ─────────────────────────────────────────────────────────

N_GRID = np.array([25, 50, 100, 200, 400, 800], dtype=np.float64)


def _synthetic_curve(a_inf, b, c, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return a_inf - b * N_GRID ** (-c) + rng.normal(0, noise, N_GRID.shape)


def test_learning_curve_recovers_ceiling_below_one():
    """Label noise caps real AUC well below 1.0, so a fit that pins
    a_inf = 1 must fail this test."""
    a_inf, b, c = fit_learning_curve(N_GRID, _synthetic_curve(0.92, 0.6, 0.5))
    assert a_inf == pytest.approx(0.92, abs=0.01)
    assert b == pytest.approx(0.6, rel=0.1)
    assert c == pytest.approx(0.5, abs=0.05)


def test_learning_curve_tolerates_measurement_noise():
    auc = _synthetic_curve(0.90, 0.8, 0.6, noise=0.003, seed=4)
    a_inf, b, c = fit_learning_curve(N_GRID, auc)
    assert a_inf == pytest.approx(0.90, abs=0.03)
    assert 0.3 < c < 1.0
    assert b > 0


def test_learning_curve_ceiling_never_exceeds_one():
    a_inf, _, _ = fit_learning_curve(N_GRID, _synthetic_curve(0.995, 0.5, 0.7))
    assert a_inf <= 1.0


def test_captures_needed_inverts_curve():
    a_inf, b, c = 0.92, 0.6, 0.5
    n = captures_needed(0.88, a_inf, b, c)
    assert a_inf - b * n ** (-c) == pytest.approx(0.88, abs=1e-9)


def test_captures_needed_unreachable_target_is_infinite():
    assert captures_needed(0.95, 0.92, 0.6, 0.5) == float("inf")


def test_learning_curve_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_learning_curve(np.array([10.0, 20.0]), np.array([0.7, 0.8]))


def test_learning_curve_rejects_auc_at_ceiling():
    with pytest.raises(ValueError):
        fit_learning_curve(N_GRID, np.full(N_GRID.shape, 1.0))



def test_learning_curve_rejects_auc_that_falls_with_n():
    with pytest.raises(ValueError):
        fit_learning_curve(N_GRID, 0.9 - 0.01 * np.log(N_GRID))

# ─── metric helpers ─────────────────────────────────────────────────────────

def test_nll_matches_hand_computed_bce():
    z = np.array([0.0, 2.0, -1.0])
    y = np.array([1.0, 0.0, 0.0])
    expected = np.mean([math.log(2), math.log1p(math.exp(2)), math.log1p(math.exp(-1))])
    assert nll(z, y) == pytest.approx(expected, rel=1e-12)


def test_nll_is_stable_for_extreme_logits():
    assert np.isfinite(nll(np.array([800.0, -800.0]), np.array([0.0, 1.0])))
    assert nll(np.array([800.0, -800.0]), np.array([1.0, 0.0])) == pytest.approx(0.0, abs=1e-12)


def test_brier_matches_hand_computed():
    p = np.array([0.9, 0.2, 0.5])
    y = np.array([1.0, 0.0, 1.0])
    assert brier(p, y) == pytest.approx((0.01 + 0.04 + 0.25) / 3)


def test_classification_metrics_reports_all_keys_and_agrees_with_parts():
    rng = np.random.default_rng(5)
    z = rng.normal(0, 2, 5_000)
    y = _bernoulli(rng, sigmoid(z))
    m = classification_metrics(z, y)
    assert set(m) == {"nll", "brier", "auc", "cal_slope", "cal_intercept", "mean_prob", "prior"}
    assert m["nll"] == pytest.approx(nll(z, y))
    assert m["brier"] == pytest.approx(brier(sigmoid(z), y))
    assert m["auc"] == pytest.approx(roc_auc_score(y, z))
    assert m["cal_slope"] == pytest.approx(1.0, abs=0.1)
    assert m["prior"] == pytest.approx(y.mean())


def test_classification_metrics_single_class_gives_nan_for_rank_and_line():
    m = classification_metrics(np.array([0.1, -0.3, 0.7]), np.zeros(3))
    assert math.isnan(m["auc"])
    assert math.isnan(m["cal_slope"]) and math.isnan(m["cal_intercept"])
    assert np.isfinite(m["nll"]) and np.isfinite(m["brier"])


def test_effective_positive_prior_reads_sampler_mass():
    labels = np.array([0, 0, 0, 1])
    # Balanced sampler: 1/6 per negative, 1/2 for the lone positive.
    weights = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 2])
    assert effective_positive_prior(weights, labels) == pytest.approx(0.5)
    assert effective_positive_prior(np.ones(4), labels) == pytest.approx(0.25)


# ─── recalibration fits ─────────────────────────────────────────────────────

def test_temperature_scale_recovers_known_temperature():
    rng = np.random.default_rng(6)
    z = rng.normal(0, 3, 50_000)
    y = _bernoulli(rng, sigmoid(z / 2.5))
    assert temperature_scale(z, y) == pytest.approx(2.5, rel=0.05)


def test_temperature_scale_is_one_for_calibrated_logits():
    rng = np.random.default_rng(7)
    z = rng.normal(0, 2, 50_000)
    y = _bernoulli(rng, sigmoid(z))
    assert temperature_scale(z, y) == pytest.approx(1.0, abs=0.05)


def test_fit_platt_recovers_affine_map():
    rng = np.random.default_rng(8)
    z = rng.normal(0, 2, 50_000)
    y = _bernoulli(rng, sigmoid(0.6 * z - 0.8))
    a, b = fit_platt(z, y)
    assert a == pytest.approx(0.6, abs=0.05)
    assert b == pytest.approx(-0.8, abs=0.05)


def test_choose_recalibration_prefers_platt_when_an_intercept_is_needed():
    rng = np.random.default_rng(9)
    z = rng.normal(0, 2, 40_000)
    y = _bernoulli(rng, sigmoid(z - 1.36))  # pos_weight-style shift
    name, a, b = choose_recalibration(z[:20_000], y[:20_000], z[20_000:], y[20_000:])
    assert name == "platt"
    assert nll(a * z[20_000:] + b, y[20_000:]) < nll(z[20_000:], y[20_000:])


def test_choose_recalibration_returns_affine_params_for_temperature():
    rng = np.random.default_rng(10)
    z = rng.normal(0, 3, 40_000)
    y = _bernoulli(rng, sigmoid(z / 2.0))
    name, a, b = choose_recalibration(z[:20_000], y[:20_000], z[20_000:], y[20_000:])
    assert name in {"temperature", "platt"}
    assert a == pytest.approx(0.5, abs=0.05)
    assert b == pytest.approx(0.0, abs=0.05)


def test_choose_recalibration_keeps_identity_for_calibrated_logits():
    # Already calibrated: any fitted map only chases noise, so its
    # held-out NLL gain stays below the default min_gain margin.
    rng = np.random.default_rng(11)
    z = rng.normal(0, 2, 40_000)
    y = _bernoulli(rng, sigmoid(z))
    name, a, b = choose_recalibration(z[:20_000], y[:20_000], z[20_000:], y[20_000:])
    assert (name, a, b) == ("identity", 1.0, 0.0)


# ─── cross-fitted recalibration (grouped by patient) ────────────────────────

def _grouped(rng, n=30_000, n_groups=300):
    return rng.integers(0, n_groups, n)


def test_cross_fit_recovers_underconfident_scale():
    """Logits squeezed by 1/1.3 (calibration slope 1.3, like v3) need a ≈ 1.3."""
    rng = np.random.default_rng(20)
    true_z = rng.normal(0, 2, 30_000)
    y = _bernoulli(rng, sigmoid(true_z))
    rec = cross_fit_recalibration(true_z / 1.3, y, _grouped(rng), seed=0)
    assert rec.method in {"temperature", "platt"}
    assert rec.a == pytest.approx(1.3, abs=0.05)
    assert rec.b == pytest.approx(0.0, abs=0.05)
    assert rec.heldout_nll[rec.method] < rec.heldout_nll["identity"]


def test_cross_fit_out_of_fold_logits_are_calibrated():
    rng = np.random.default_rng(21)
    true_z = rng.normal(0, 2, 30_000)
    y = _bernoulli(rng, sigmoid(true_z))
    rec = cross_fit_recalibration(true_z / 1.3 + 0.3, y, _grouped(rng), seed=0)
    slope, intercept = calibration_line(sigmoid(rec.oof_logits), y)
    assert slope == pytest.approx(1.0, abs=0.06)
    assert intercept == pytest.approx(0.0, abs=0.06)


def test_cross_fit_keeps_identity_for_calibrated_logits():
    rng = np.random.default_rng(22)
    z = rng.normal(0, 2, 30_000)
    y = _bernoulli(rng, sigmoid(z))
    rec = cross_fit_recalibration(z, y, _grouped(rng), seed=0)
    assert (rec.method, rec.a, rec.b) == ("identity", 1.0, 0.0)
    assert np.array_equal(rec.oof_logits, z)


def test_cross_fit_never_splits_a_patient_across_folds():
    rng = np.random.default_rng(23)
    groups = _grouped(rng, n=5_000, n_groups=60)
    z = rng.normal(0, 2, 5_000)
    rec = cross_fit_recalibration(z, _bernoulli(rng, sigmoid(z)), groups, n_folds=3, seed=1)
    for g in np.unique(groups):
        assert len(np.unique(rec.folds[groups == g])) == 1
    assert set(np.unique(rec.folds)) == {0, 1, 2}


def test_cross_fit_needs_at_least_two_groups():
    z = np.zeros(10)
    with pytest.raises(ValueError):
        cross_fit_recalibration(z, np.r_[np.zeros(5), np.ones(5)], np.zeros(10), seed=0)


def test_deploy_affine_composes_recalibration_with_prior_shift():
    """Fit at the val prior, then move to the deploy prior:
    deployed = a·(z + off_val) + b + (off_deploy − off_val)."""
    z = np.linspace(-6, 6, 25)
    a, b, off_val, off_dep = 1.3, 0.2, -0.5, -1.2
    A, B = deploy_affine(a, b, off_val, off_dep)
    assert np.allclose(A * z + B, a * (z + off_val) + b + (off_dep - off_val))
    assert deploy_affine(1.0, 0.0, off_val, off_dep) == pytest.approx((1.0, off_dep))


def test_fit_deploy_affine_record_on_synthetic_val_set():
    """End to end: raw logits from a pos_weight-trained, underconfident
    model; the record's (a, b) must make deployed probabilities calibrated
    at the deploy prior."""
    rng = np.random.default_rng(24)
    n, w, train_prior, deploy_prior = 40_000, 4.0, 0.2, 0.3
    true_val_logit = rng.normal(-1.0, 2.0, n)  # calibrated at the val prior
    y = _bernoulli(rng, sigmoid(true_val_logit))
    val_prior = float(y.mean())
    off_val = prior_shift_offset(w, train_prior, val_prior)
    raw = true_val_logit / 1.3 - off_val  # model output before any correction
    rec = fit_deploy_affine(raw, y, _grouped(rng, n), pos_weight=w, train_prior=train_prior,
                            deploy_prior=deploy_prior, seed=0)
    assert rec["method"] in {"temperature", "platt"}
    assert rec["fit_a"] == pytest.approx(1.3, abs=0.05)
    deployed = rec["a"] * raw + rec["b"]
    expected = true_val_logit + logit(deploy_prior) - logit(val_prior)
    assert np.allclose(sigmoid(deployed), sigmoid(expected), atol=0.02)
    for key in ("val_prior", "heldout_nll", "oof_cal_slope", "oof_cal_intercept", "n_folds"):
        assert key in rec
    assert rec["oof_cal_slope"] == pytest.approx(1.0, abs=0.06)


# ─── bake_affine ────────────────────────────────────────────────────────────

def test_bake_affine_folds_exactly_into_final_linear():
    torch.manual_seed(0)
    model = MurmurCNN()
    model.train(False)
    x = torch.randn(8, 62, 32)
    with torch.no_grad():
        before = model(x).clone()
    bake_affine(model, a=0.7, b=-1.36)
    with torch.no_grad():
        after = model(x)
    assert torch.allclose(after, 0.7 * before - 1.36, atol=1e-5)


def test_bake_affine_rejects_non_linear_head():
    model = MurmurCNN()
    model.head[-1] = nn.Identity()
    with pytest.raises(TypeError):
        bake_affine(model, 1.0, 0.0)
