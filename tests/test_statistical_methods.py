"""
test_statistical_methods.py
=============================
Tests verifying the correctness of the statistical validation utilities
(bootstrap CI, Diebold-Mariano test). These are general-purpose
correctness checks independent of the specific dataset.

Run with: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.validate import bootstrap_r2_ci, diebold_mariano  # noqa: E402


def test_bootstrap_ci_contains_point_estimate():
    """The bootstrap CI must always contain the point estimate computed on
    the full (non-resampled) sample -- this was a defect identified and
    fixed during manuscript preparation (a previous bootstrap
    implementation resampled actual and predicted values independently,
    breaking the pairing and producing CIs that did not contain the
    reported point estimate)."""
    rng = np.random.RandomState(0)
    n = 500
    y_true = rng.normal(0, 20, n)
    noise = rng.normal(0, 15, n)
    y_pred = 0.5 * y_true + noise

    from sklearn.metrics import r2_score

    point_estimate = r2_score(y_true, y_pred)
    ci = bootstrap_r2_ci(y_true, y_pred, n_iter=1000, seed=42)

    assert ci["ci_lower"] <= point_estimate <= ci["ci_upper"], (
        f"Point estimate {point_estimate:.4f} falls outside bootstrap CI "
        f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
    )


def test_bootstrap_ci_preserves_pairing():
    """Bootstrap resampling must resample (actual, predicted) as matched
    pairs, not independently. Independent resampling destroys the
    correlation structure and produces a degenerate (near-zero, wide) CI
    even when the underlying model has genuine predictive skill."""
    rng = np.random.RandomState(1)
    n = 300
    y_true = rng.normal(0, 10, n)
    y_pred = y_true + rng.normal(0, 1, n)  # near-perfect predictor

    ci = bootstrap_r2_ci(y_true, y_pred, n_iter=500, seed=1)
    assert ci["mean"] > 0.8, (
        "Bootstrap CI mean R² is implausibly low for a near-perfect "
        "predictor -- check that resampling uses paired indices, not "
        "independent resampling of y_true and y_pred."
    )


def test_diebold_mariano_symmetry():
    """DM statistic comparing model A to model B should have the opposite
    sign of the statistic comparing B to A."""
    rng = np.random.RandomState(2)
    n = 400
    e1 = rng.normal(0, 5, n)  # smaller errors (better model)
    e2 = rng.normal(0, 10, n)  # larger errors (worse model)

    dm_ab = diebold_mariano(e1, e2)
    dm_ba = diebold_mariano(e2, e1)

    assert np.isclose(dm_ab["dm_statistic"], -dm_ba["dm_statistic"], atol=1e-6)


def test_diebold_mariano_detects_better_model():
    """When model 1 has systematically smaller errors, the DM statistic
    should be positive and the test should reject the null of equal
    accuracy at a conventional significance level."""
    rng = np.random.RandomState(3)
    n = 1000
    e1 = rng.normal(0, 5, n)
    e2 = rng.normal(0, 15, n)

    result = diebold_mariano(e1, e2)
    assert result["dm_statistic"] > 0
    assert result["p_value"] < 0.05


def test_diebold_mariano_no_difference_when_equal():
    """When two models have errors drawn from the same distribution, the
    DM test should generally fail to reject the null (not statistically
    significant) -- checked probabilistically, not deterministically, so
    this test uses a fixed seed and a generous threshold."""
    rng = np.random.RandomState(4)
    n = 1000
    e1 = rng.normal(0, 10, n)
    e2 = rng.normal(0, 10, n)

    result = diebold_mariano(e1, e2)
    assert abs(result["dm_statistic"]) < 3.0  # not an extreme statistic
