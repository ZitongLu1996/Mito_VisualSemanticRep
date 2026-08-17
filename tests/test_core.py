from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from vsvariance.analysis import predictive_r2
from vsvariance.ridge import fit_design, predict_per_target_alpha
from vsvariance.stacking import apply_stacking, fit_convex_stacking


def test_svd_ridge_matches_sklearn() -> None:
    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(80, 12))
    X_test = rng.normal(size=(13, 12))
    Y_train = rng.normal(size=(80, 4))
    alpha = 3.5
    expected = Ridge(alpha=alpha, fit_intercept=True).fit(X_train, Y_train).predict(X_test)
    actual = predict_per_target_alpha(
        fit_design(X_train), Y_train, X_test, np.full(Y_train.shape[1], alpha)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_convex_stacking_constraints_and_preference() -> None:
    rng = np.random.default_rng(22)
    target = rng.normal(size=(200, 3)).astype(np.float32)
    good = target + rng.normal(scale=0.02, size=target.shape)
    bad = rng.normal(size=target.shape)
    predictions = np.stack([good, bad], axis=1).astype(np.float32)
    weights, intercept = fit_convex_stacking(predictions, target)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-6)
    assert np.all(weights[0] > weights[1])
    fitted = apply_stacking(predictions, weights, intercept)
    assert np.nanmean(predictive_r2(target, fitted)) > 0.99


def test_variance_partition_identity() -> None:
    r2_visual = np.array([0.30, 0.20])
    r2_semantic = np.array([0.25, 0.35])
    r2_joint = np.array([0.40, 0.45])
    unique_visual = r2_joint - r2_semantic
    unique_semantic = r2_joint - r2_visual
    shared = r2_visual + r2_semantic - r2_joint
    np.testing.assert_allclose(unique_visual + unique_semantic + shared, r2_joint)


def test_kfold_validation_rows_appear_once() -> None:
    splitter = KFold(n_splits=5, shuffle=True, random_state=7)
    counts = np.zeros(103, dtype=int)
    for _, validation in splitter.split(np.arange(103)):
        counts[validation] += 1
    np.testing.assert_array_equal(counts, np.ones_like(counts))
