"""Closed-form calibration and curve-fitting helpers for the murmur model.

Every constant produced here is *derived* (Bayes, calculus, least squares)
rather than hand-tuned, and each function has a test in
`tests/test_calibration.py` that checks it against the underlying math on
synthetic data with a known answer.

Why this module exists: AUC is invariant to any monotone transform of the
logit, so a large logit bias drift leaves it untouched. Calibration
slope/intercept, NLL, Brier and the prior-shift offset are the quantities
that actually move when a healthy heart reads 80 % murmur.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.metrics import roc_auc_score

_EPS = 1e-7


def logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return expit(z)


# ─── Scores ─────────────────────────────────────────────────────────────────

def nll(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean unweighted binary cross-entropy, computed from logits.

    BCE = y·softplus(-z) + (1-y)·softplus(z) = softplus(z) - y·z, which
    stays finite for any z (no log(0) from a saturated sigmoid)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Everything the trainers log per epoch, from raw logits.

    `auc` and the calibration line need both classes; with a single class
    they come back NaN while `nll` / `brier` stay finite."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    p = sigmoid(z)
    both = len(np.unique(y)) == 2
    slope, intercept = float("nan"), float("nan")
    if both:
        beta = _logistic_irls(z, y)
        if beta is not None:
            slope, intercept = float(beta[0]), float(beta[1])
    return {
        "nll": nll(z, y),
        "brier": brier(p, y),
        "auc": float(roc_auc_score(y, z)) if both else float("nan"),
        "cal_slope": slope,
        "cal_intercept": intercept,
        "mean_prob": float(p.mean()) if p.size else float("nan"),
        "prior": float(y.mean()) if y.size else float("nan"),
    }


# ─── Prior shift ────────────────────────────────────────────────────────────

def prior_shift_offset(pos_weight: float, train_prior: float, deploy_prior: float) -> float:
    """Additive logit correction for a model trained with
    `BCEWithLogitsLoss(pos_weight=w)` on class prior `train_prior`, so its
    outputs become calibrated under `deploy_prior`.

    Derivation. Pointwise weighted BCE at true posterior p:
        L(z) = w·p·softplus(-z) + (1-p)·softplus(z)
        dL/dz = -w·p·σ(-z) + (1-p)·σ(z) = 0   ⇒   e^z = w·p / (1-p)
        ⇒ z* = logit(p) + ln w
    Bayes moves the posterior between priors by adding
    logit(π_d) - logit(π_t). Combine and undo the ln w term.

    A WeightedRandomSampler that balances classes is the same thing with
    w = 1 and train_prior = 0.5 (see `effective_positive_prior`).
    """
    if pos_weight <= 0:
        raise ValueError(f"pos_weight must be positive, got {pos_weight}")
    return float(logit(deploy_prior) - logit(train_prior) - np.log(pos_weight))


def effective_positive_prior(weights: np.ndarray, labels: np.ndarray) -> float:
    """Class prior the model actually trains on when windows are drawn in
    proportion to `weights` (e.g. WeightedRandomSampler)."""
    w = np.asarray(weights, dtype=np.float64)
    y = np.asarray(labels)
    total = w.sum()
    if total <= 0:
        raise ValueError("weights sum to zero")
    return float(w[y == 1].sum() / total)


# ─── Logistic fits on the logit (Newton / IRLS) ─────────────────────────────

def _logistic_irls(
    x: np.ndarray, y: np.ndarray, fit_intercept: bool = True, max_iter: int = 50,
) -> np.ndarray | None:
    """Maximum-likelihood fit of P(y=1) = σ(β₀·x [+ β₁]) by damped
    Newton–Raphson.

    Gradient of the log-likelihood is Xᵀ(y - p); the negative Hessian is
    Xᵀ diag(p(1-p)) X. Starts at β = (1, 0), i.e. "already calibrated".
    Pure Newton overshoots from there when the logits are far too spread
    out (slope ≪ 1, the overconfident case), so each step is halved until
    the log-likelihood rises. The log-likelihood is concave, so this always
    makes progress. Returns None when the Hessian is singular or the fit
    does not converge (perfectly separable data has no finite maximiser)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = np.column_stack([x, np.ones_like(x)]) if fit_intercept else x[:, None]

    def loglik(b: np.ndarray) -> float:
        z = X @ b
        return float(np.sum(y * z - np.logaddexp(0.0, z)))

    beta = np.zeros(X.shape[1])
    beta[0] = 1.0
    ll = loglik(beta)
    for _ in range(max_iter):
        p = sigmoid(X @ beta)
        grad = X.T @ (y - p)
        hess = (X * (p * (1.0 - p))[:, None]).T @ X
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return None
        t = 1.0
        while t > 1e-8:
            ll_new = loglik(beta + t * step)
            if ll_new >= ll:
                break
            t *= 0.5
        else:
            return None
        beta = beta + t * step
        ll = ll_new
        if np.abs(t * step).max() < 1e-10:
            return beta
    return None


def calibration_line(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Cox (1958) calibration line — logistic regression of the true label
    on the model's logit:

        P(y=1) = σ(slope · logit(p) + intercept)

    Perfect calibration is (slope, intercept) = (1, 0). slope < 1 means
    over-confident (probabilities too extreme); intercept < 0 means the
    model over-predicts positives — the healthy-heart-reads-murmur failure.
    Returns NaNs when the fit has no finite solution.
    """
    beta = _logistic_irls(logit(np.asarray(probs, dtype=np.float64)), labels)
    if beta is None:
        return float("nan"), float("nan")
    return float(beta[0]), float(beta[1])


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Platt scaling: (a, b) so that σ(a·z + b) is calibrated. Same fit as
    `calibration_line`, taken directly on logits."""
    beta = _logistic_irls(logits, labels)
    if beta is None:
        raise ValueError("Platt fit did not converge (separable data?)")
    return float(beta[0]), float(beta[1])


def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature T so that σ(z / T) is calibrated. One-parameter logistic
    fit with no intercept; NLL is convex in 1/T, so Newton converges."""
    beta = _logistic_irls(logits, labels, fit_intercept=False)
    if beta is None or beta[0] <= 0:
        raise ValueError("temperature fit did not converge to a positive value")
    return float(1.0 / beta[0])


def choose_recalibration(
    fit_logits: np.ndarray,
    fit_labels: np.ndarray,
    held_logits: np.ndarray,
    held_labels: np.ndarray,
    min_gain: float = 1e-3,
) -> tuple[str, float, float]:
    """Pick identity, temperature or Platt by held-out NLL.

    Candidates are tried in order of parameter count; a richer map only
    replaces the current choice when it lowers held-out NLL by more than
    `min_gain` nats. Returns (name, a, b) for `bake_affine`."""
    best = ("identity", 1.0, 0.0)
    best_nll = nll(held_logits, held_labels)
    held = np.asarray(held_logits, dtype=np.float64)
    for name, (a, b) in _candidate_maps(fit_logits, fit_labels).items():
        if name == "identity":
            continue
        score = nll(a * held + b, held_labels)
        if best_nll - score > min_gain:
            best, best_nll = (name, a, b), score
    return best


_RECALIBRATIONS = ("identity", "temperature", "platt")


def _candidate_maps(logits: np.ndarray, labels: np.ndarray) -> dict[str, tuple[float, float]]:
    """Every recalibration that fits, as `a·z + b`, in order of parameter
    count. A fit that does not converge is left out."""
    maps = {"identity": (1.0, 0.0)}
    try:
        maps["temperature"] = (1.0 / temperature_scale(logits, labels), 0.0)
    except ValueError:
        pass
    try:
        maps["platt"] = fit_platt(logits, labels)
    except ValueError:
        pass
    return maps


@dataclass(frozen=True)
class Recalibration:
    method: str
    a: float
    b: float
    heldout_nll: dict[str, float]  # out-of-fold NLL of every candidate
    oof_logits: np.ndarray  # chosen map, each fold scored by a fit on the others
    folds: np.ndarray  # fold index per sample


def cross_fit_recalibration(
    logits: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_folds: int = 2,
    seed: int = 0,
    min_gain: float = 1e-3,
) -> Recalibration:
    """Choose identity, temperature or Platt by out-of-fold NLL, folding
    by group (patient) so no patient is both fitted and scored.

    Each candidate is fitted on all folds but one and scored on the one
    left out; the out-of-fold NLLs are summed over folds. As in
    `choose_recalibration`, a richer map replaces the current choice only
    when it lowers held-out NLL by more than `min_gain` nats. The chosen
    map is then refitted on everything. `oof_logits` are an honest
    preview of calibration on new patients."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    g = np.asarray(groups)
    unique = np.unique(g)
    if len(unique) < max(2, n_folds):
        raise ValueError(f"need at least {max(2, n_folds)} groups, got {len(unique)}")
    order = np.random.default_rng(seed).permutation(unique)
    fold_of = {grp: i % n_folds for i, grp in enumerate(order.tolist())}
    folds = np.array([fold_of[v] for v in g.tolist()])

    oof = {c: np.full_like(z, np.nan) for c in _RECALIBRATIONS}
    for k in range(n_folds):
        fit, held = folds != k, folds == k
        for name, (a, b) in _candidate_maps(z[fit], y[fit]).items():
            oof[name][held] = a * z[held] + b
    heldout = {c: nll(oof[c], y) if np.isfinite(oof[c]).all() else float("inf") for c in _RECALIBRATIONS}

    method = "identity"
    for c in _RECALIBRATIONS[1:]:
        if heldout[method] - heldout[c] > min_gain:
            method = c
    a, b = _candidate_maps(z, y).get(method, (1.0, 0.0))
    return Recalibration(method, float(a), float(b), heldout, oof[method], folds)


def deploy_affine(a: float, b: float, val_offset: float, deploy_offset: float) -> tuple[float, float]:
    """Compose a recalibration fitted at the validation prior with the
    move to the deployment prior, as one map on the raw model logit z:

        deployed = a·(z + val_offset) + b + (deploy_offset − val_offset)

    The last term is the Bayes prior shift between the two priors
    (pos_weight and training-prior terms cancel)."""
    return float(a), float(a * val_offset + b + deploy_offset - val_offset)


def fit_deploy_affine(
    raw_logits: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    pos_weight: float,
    train_prior: float,
    deploy_prior: float,
    n_folds: int = 2,
    seed: int = 0,
    min_gain: float = 1e-3,
) -> dict:
    """Validation logits in, `best_meta.json` record out: `a`, `b` such
    that `σ(a·z + b)` is calibrated at `deploy_prior`, plus the evidence
    (held-out NLL per candidate, out-of-fold calibration line)."""
    z = np.asarray(raw_logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    val_prior = float(y.mean())
    off_val = prior_shift_offset(pos_weight, train_prior, val_prior)
    off_dep = prior_shift_offset(pos_weight, train_prior, deploy_prior)
    rec = cross_fit_recalibration(z + off_val, y, groups, n_folds=n_folds, seed=seed, min_gain=min_gain)
    a, b = deploy_affine(rec.a, rec.b, off_val, off_dep)
    before = classification_metrics(z + off_val, y)
    after = classification_metrics(rec.oof_logits, y)
    return {
        "method": rec.method,
        "a": a,
        "b": b,
        "fit_a": rec.a,
        "fit_b": rec.b,
        "val_prior": val_prior,
        "deploy_prior": deploy_prior,
        "n_folds": n_folds,
        "seed": seed,
        "heldout_nll": rec.heldout_nll,
        "val_nll_before": before["nll"],
        "oof_nll": after["nll"],
        "val_brier_before": before["brier"],
        "oof_brier": after["brier"],
        "val_cal_slope_before": before["cal_slope"],
        "val_cal_intercept_before": before["cal_intercept"],
        "oof_cal_slope": after["cal_slope"],
        "oof_cal_intercept": after["cal_intercept"],
    }


def bake_affine(model: nn.Module, a: float, b: float) -> None:
    """Fold `a·logit + b` into the model's final Linear layer in place:

        weight ← a·weight,   bias ← a·bias + b

    Exact because the logit is an affine function of that layer's
    parameters. Deployment (Rust / Core ML) needs no change."""
    final = model.head[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError(f"expected final head layer to be nn.Linear, got {type(final).__name__}")
    with torch.no_grad():
        final.weight.mul_(a)
        final.bias.mul_(a).add_(b)


# ─── Learning curve ─────────────────────────────────────────────────────────

def fit_learning_curve(n: np.ndarray, auc: np.ndarray) -> tuple[float, float, float]:
    """Fit the inverse power law  AUC(n) = a_inf - b · n^(-c)  to validation
    AUC measured at several training-set sizes `n`.

    Returns (a_inf, b, c) with b > 0, c > 0, max(auc) < a_inf <= 1.

    For a fixed a_inf the model is a straight line,
        log(a_inf - AUC) = log b - c · log n,
    so (b, c) come from weighted least squares. a_inf is profiled: scan
    candidates in (max(auc), 1], fit the line for each, keep the one with
    the lowest n-weighted squared error on the AUC scale.

    Weights: Var[AUC] ∝ 1/n, and by the delta method
    Var[log(a_inf - AUC)] ≈ Var[AUC] / (a_inf - AUC)², so each point's
    log-space weight is n·(a_inf - AUC)². np.polyfit squares `w`.
    """
    n = np.asarray(n, dtype=np.float64)
    auc = np.asarray(auc, dtype=np.float64)
    if n.shape != auc.shape or n.size < 3:
        raise ValueError("need at least 3 matching (n, auc) points")
    top = float(auc.max())
    if top >= 1.0:
        raise ValueError("AUC already at 1.0; no gap left to fit")

    log_n = np.log(n)
    best: tuple[float, float, float, float] | None = None
    for a_inf in np.linspace(top + 1e-6, 1.0, 2000):
        gap = a_inf - auc
        slope, icpt = np.polyfit(log_n, np.log(gap), 1, w=np.sqrt(n) * gap)
        b, c = float(np.exp(icpt)), float(-slope)
        if c <= 0:
            continue
        sse = float(np.sum(n * (auc - (a_inf - b * n ** (-c))) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(a_inf), b, c)
    if best is None:
        raise ValueError("AUC does not rise with n; no power-law fit")
    return best[1], best[2], best[3]


def captures_needed(target_auc: float, a_inf: float, b: float, c: float) -> float:
    """Invert the learning curve: smallest n with AUC(n) >= target_auc.
    Returns +inf when the target sits at or above the asymptote."""
    gap = a_inf - target_auc
    if gap <= 0:
        return float("inf")
    return float((b / gap) ** (1.0 / c))
