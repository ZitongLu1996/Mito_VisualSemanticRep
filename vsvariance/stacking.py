from __future__ import annotations

import numpy as np


def _project_simplex(columns: np.ndarray) -> np.ndarray:
    """Project each column of [models, targets] onto the probability simplex."""
    sorted_values = np.sort(columns, axis=0)[::-1]
    cumulative = np.cumsum(sorted_values, axis=0) - 1.0
    ranks = np.arange(1, columns.shape[0] + 1, dtype=np.float64)[:, None]
    support = sorted_values - cumulative / ranks > 0
    rho = support.sum(axis=0) - 1
    theta = cumulative[rho, np.arange(columns.shape[1])] / (rho + 1)
    return np.maximum(columns - theta[None, :], 0.0)


def fit_convex_stacking(
    predictions: np.ndarray,
    targets: np.ndarray,
    max_iter: int = 500,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit target-specific non-negative weights that sum to one.

    predictions is [samples, models, targets]; targets is [samples, targets].
    Returns weights [models, targets] and intercepts [targets].
    """
    if predictions.ndim != 3 or targets.ndim != 2:
        raise ValueError("Unexpected stacking array dimensions")
    if predictions.shape[0] != targets.shape[0] or predictions.shape[2] != targets.shape[1]:
        raise ValueError("Stacking predictions and targets are misaligned")
    Z = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    valid = np.all(np.isfinite(Z), axis=(0, 1)) & np.all(np.isfinite(y), axis=0)
    if not np.all(valid):
        weights = np.full((Z.shape[1], Z.shape[2]), np.nan, dtype=np.float32)
        intercept = np.full(Z.shape[2], np.nan, dtype=np.float32)
        if np.any(valid):
            valid_weights, valid_intercept = fit_convex_stacking(
                Z[:, :, valid], y[:, valid], max_iter=max_iter, tolerance=tolerance
            )
            weights[:, valid] = valid_weights
            intercept[valid] = valid_intercept
        return weights, intercept
    z_mean = Z.mean(axis=0)
    y_mean = y.mean(axis=0)
    Z -= z_mean[None, :, :]
    y -= y_mean[None, :]
    gram = np.einsum("nmp,nkp->mkp", Z, Z) / Z.shape[0]
    cross = np.einsum("nmp,np->mp", Z, y) / Z.shape[0]
    eigenvalues = np.linalg.eigvalsh(np.moveaxis(gram, 2, 0))
    step = 1.0 / np.maximum(eigenvalues[:, -1], 1e-12)
    weights = np.full((Z.shape[1], Z.shape[2]), 1.0 / Z.shape[1], dtype=np.float64)
    for _ in range(max_iter):
        gradient = np.einsum("mkp,kp->mp", gram, weights) - cross
        updated = _project_simplex(weights - gradient * step[None, :])
        if np.max(np.abs(updated - weights)) < tolerance:
            weights = updated
            break
        weights = updated
    intercept = y_mean - np.einsum("mp,mp->p", weights, z_mean)
    return weights.astype(np.float32), intercept.astype(np.float32)


def apply_stacking(
    predictions: np.ndarray, weights: np.ndarray, intercept: np.ndarray
) -> np.ndarray:
    return (
        np.einsum("nmp,mp->np", predictions, weights) + intercept[None, :]
    ).astype(np.float32, copy=False)
