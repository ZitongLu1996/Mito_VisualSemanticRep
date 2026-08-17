from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class RidgeDesign:
    x_mean: np.ndarray
    U: np.ndarray
    singular_values: np.ndarray
    Vt: np.ndarray


@dataclass(frozen=True)
class InnerDesign:
    train_index: np.ndarray
    validation_index: np.ndarray
    design: RidgeDesign


def fit_design(X: np.ndarray) -> RidgeDesign:
    """Factorize a feature matrix once; the result can be reused for all vertices."""
    x_mean = np.mean(X, axis=0, dtype=np.float64)
    Xc = np.asarray(X - x_mean, dtype=np.float64)
    U, singular_values, Vt = np.linalg.svd(Xc, full_matrices=False)
    return RidgeDesign(x_mean=x_mean, U=U, singular_values=singular_values, Vt=Vt)


def make_inner_designs(X: np.ndarray, n_splits: int, random_state: int) -> list[InnerDesign]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [
        InnerDesign(train_index, validation_index, fit_design(X[train_index]))
        for train_index, validation_index in splitter.split(X)
    ]


def _fit_targets(design: RidgeDesign, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_mean = np.mean(Y, axis=0, dtype=np.float64)
    Uy = design.U.T @ np.asarray(Y - y_mean, dtype=np.float64)
    return y_mean, Uy


def _predict(
    design: RidgeDesign,
    target_fit: tuple[np.ndarray, np.ndarray],
    X: np.ndarray,
    alpha: float,
) -> np.ndarray:
    y_mean, Uy = target_fit
    projection = (np.asarray(X, dtype=np.float64) - design.x_mean) @ design.Vt.T
    shrinkage = design.singular_values / (design.singular_values**2 + alpha)
    return (projection * shrinkage) @ Uy + y_mean


def select_alphas_from_designs(
    X: np.ndarray,
    Y: np.ndarray,
    alphas: Sequence[float],
    inner_designs: Sequence[InnerDesign],
) -> np.ndarray:
    """Choose one ridge alpha per target from pre-factorized inner folds."""
    candidates = np.asarray(alphas, dtype=np.float64)
    if np.any(candidates <= 0):
        raise ValueError("All ridge alphas must be positive")
    losses = np.zeros((len(candidates), Y.shape[1]), dtype=np.float64)
    for inner in inner_designs:
        target_fit = _fit_targets(inner.design, Y[inner.train_index])
        for alpha_index, alpha in enumerate(candidates):
            prediction = _predict(inner.design, target_fit, X[inner.validation_index], float(alpha))
            residual = np.asarray(Y[inner.validation_index], dtype=np.float64) - prediction
            losses[alpha_index] += np.einsum("ij,ij->j", residual, residual)
    return candidates[np.argmin(losses, axis=0)]


def predict_per_target_alpha(
    design: RidgeDesign,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    selected_alphas: np.ndarray,
) -> np.ndarray:
    target_fit = _fit_targets(design, Y_train)
    output = np.empty((X_test.shape[0], Y_train.shape[1]), dtype=np.float32)
    for alpha in np.unique(selected_alphas):
        columns = np.flatnonzero(selected_alphas == alpha)
        prediction = _predict(design, target_fit, X_test, float(alpha))
        output[:, columns] = prediction[:, columns].astype(np.float32, copy=False)
    return output

