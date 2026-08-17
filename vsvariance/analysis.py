from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from .ridge import (
    fit_design,
    make_inner_designs,
    predict_per_target_alpha,
    select_alphas_from_designs,
)
from .stacking import apply_stacking, fit_convex_stacking


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    visual_groups: tuple[str, ...]
    semantic_group: str


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _fit_pca(
    X_train: np.ndarray, X_other: np.ndarray, n_components: int, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    max_components = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    if max_components < 1:
        raise ValueError("Not enough samples/features for PCA")
    pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        whiten=True,
        random_state=random_state,
    )
    train_scores = pca.fit_transform(np.asarray(X_train, dtype=np.float32))
    other_scores = pca.transform(np.asarray(X_other, dtype=np.float32))
    return train_scores.astype(np.float32), other_scores.astype(np.float32)


def prepare_nested_pca_cache(
    raw_train: dict[str, Path],
    raw_test: dict[str, Path],
    groups: Sequence[str],
    cache_dir: Path,
    n_components: int,
    outer_splits: int,
    random_state: int,
) -> Path:
    """Fit PCA only on each outer-training partition; test PCA is fit on full training."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays_train = {name: np.load(raw_train[name], mmap_mode="r") for name in groups}
    arrays_test = {name: np.load(raw_test[name], mmap_mode="r") for name in groups}
    n_train = next(iter(arrays_train.values())).shape[0]
    n_test = next(iter(arrays_test.values())).shape[0]
    if any(array.shape[0] != n_train for array in arrays_train.values()):
        raise ValueError("Training feature groups have inconsistent row counts")
    if any(array.shape[0] != n_test for array in arrays_test.values()):
        raise ValueError("Test feature groups have inconsistent row counts")
    payload = {
        "groups": list(groups),
        "raw_train": {name: _file_signature(raw_train[name]) for name in groups},
        "raw_test": {name: _file_signature(raw_test[name]) for name in groups},
        "shapes_train": {name: arrays_train[name].shape for name in groups},
        "shapes_test": {name: arrays_test[name].shape for name in groups},
        "n_components": n_components,
        "outer_splits": outer_splits,
        "random_state": random_state,
    }
    fingerprint = _fingerprint(payload)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint:
            return manifest_path
        raise RuntimeError(f"PCA cache configuration changed; use a new/empty directory: {cache_dir}")

    splitter = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    folds = []
    for fold_index, (train_index, validation_index) in enumerate(splitter.split(np.arange(n_train))):
        fold_dir = cache_dir / f"outer_{fold_index:02d}"
        fold_dir.mkdir()
        np.save(fold_dir / "train_index.npy", train_index)
        np.save(fold_dir / "validation_index.npy", validation_index)
        for group_index, group in enumerate(groups):
            print(
                f"PCA_PROGRESS fold={fold_index + 1}/{outer_splits} group={group}",
                flush=True,
            )
            train_scores, validation_scores = _fit_pca(
                arrays_train[group][train_index],
                arrays_train[group][validation_index],
                n_components,
                random_state + 1000 * fold_index + group_index,
            )
            np.save(fold_dir / f"{group}_train.npy", train_scores)
            np.save(fold_dir / f"{group}_validation.npy", validation_scores)
        folds.append(fold_dir.name)

    final_dir = cache_dir / "full_training"
    final_dir.mkdir()
    for group_index, group in enumerate(groups):
        print(f"PCA_PROGRESS fold=full_training group={group}", flush=True)
        train_scores, test_scores = _fit_pca(
            arrays_train[group], arrays_test[group], n_components, random_state + 90000 + group_index
        )
        np.save(final_dir / f"{group}_train.npy", train_scores)
        np.save(final_dir / f"{group}_test.npy", test_scores)

    manifest = payload | {
        "fingerprint": fingerprint,
        "n_train": n_train,
        "n_test": n_test,
        "folds": folds,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _generate_base_predictions(
    pca_cache: Path,
    group: str,
    Y_train_path: Path,
    temporary_dir: Path,
    alpha_output: np.memmap,
    alphas: Sequence[float],
    inner_splits: int,
    vertex_chunk_size: int,
    random_state: int,
) -> tuple[np.memmap, np.memmap]:
    """Generate OOF/test predictions while factorizing every X fold only once."""
    manifest = json.loads((pca_cache / "manifest.json").read_text(encoding="utf-8"))
    Y_train_all = np.load(Y_train_path, mmap_mode="r")
    n_vertices = Y_train_all.shape[1]
    oof_path = temporary_dir / f"{group}_oof.npy"
    test_path = temporary_dir / f"{group}_test.npy"
    # Keep prediction precision: unique/shared components can be much smaller than total R2.
    oof = np.lib.format.open_memmap(
        oof_path, "w+", np.float32, (manifest["n_train"], n_vertices)
    )
    test_prediction = np.lib.format.open_memmap(
        test_path, "w+", np.float32, (manifest["n_test"], n_vertices)
    )
    for fold_number, fold_name in enumerate(manifest["folds"]):
        print(
            f"RIDGE_PROGRESS group={group} outer_fold={fold_number + 1}/{len(manifest['folds'])}",
            flush=True,
        )
        fold_dir = pca_cache / fold_name
        train_index = np.load(fold_dir / "train_index.npy")
        validation_index = np.load(fold_dir / "validation_index.npy")
        X_train = np.load(fold_dir / f"{group}_train.npy", mmap_mode="r")
        X_validation = np.load(fold_dir / f"{group}_validation.npy", mmap_mode="r")
        inner_designs = make_inner_designs(
            X_train, inner_splits, random_state + 100 * fold_number
        )
        outer_design = fit_design(X_train)
        for start in range(0, n_vertices, vertex_chunk_size):
            stop = min(start + vertex_chunk_size, n_vertices)
            Y_outer = np.asarray(Y_train_all[train_index, start:stop], dtype=np.float32)
            selected = select_alphas_from_designs(X_train, Y_outer, alphas, inner_designs)
            prediction = predict_per_target_alpha(
                outer_design, Y_outer, X_validation, selected
            )
            oof[validation_index, start:stop] = prediction
            if start == 0 or stop == n_vertices or start % (vertex_chunk_size * 16) == 0:
                print(
                    f"RIDGE_VERTEX_PROGRESS group={group} fold={fold_number + 1} vertices={stop}/{n_vertices}",
                    flush=True,
                )

    final_dir = pca_cache / "full_training"
    X_train_full = np.load(final_dir / f"{group}_train.npy", mmap_mode="r")
    X_test = np.load(final_dir / f"{group}_test.npy", mmap_mode="r")
    inner_designs = make_inner_designs(
        X_train_full, inner_splits, random_state + 80000
    )
    full_design = fit_design(X_train_full)
    print(f"RIDGE_PROGRESS group={group} full_training", flush=True)
    for start in range(0, n_vertices, vertex_chunk_size):
        stop = min(start + vertex_chunk_size, n_vertices)
        Y_full = np.asarray(Y_train_all[:, start:stop], dtype=np.float32)
        selected = select_alphas_from_designs(X_train_full, Y_full, alphas, inner_designs)
        prediction = predict_per_target_alpha(full_design, Y_full, X_test, selected)
        test_prediction[:, start:stop] = prediction
        alpha_output[start:stop] = selected.astype(np.float32)
        if start == 0 or stop == n_vertices or start % (vertex_chunk_size * 16) == 0:
            print(
                f"RIDGE_VERTEX_PROGRESS group={group} fold=full vertices={stop}/{n_vertices}",
                flush=True,
            )
    oof.flush()
    test_prediction.flush()
    alpha_output.flush()
    return oof, test_prediction


def predictive_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    residual = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    numerator = np.einsum("ij,ij->j", residual, residual)
    centered = np.asarray(y_true, dtype=np.float64) - np.mean(y_true, axis=0, dtype=np.float64)
    denominator = np.einsum("ij,ij->j", centered, centered)
    result = np.full_like(numerator, np.nan)
    valid = denominator > 0
    result[valid] = 1.0 - numerator[valid] / denominator[valid]
    return result.astype(np.float32)


def predictive_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation on held-out observations."""
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    true -= np.mean(true, axis=0, keepdims=True)
    pred -= np.mean(pred, axis=0, keepdims=True)
    numerator = np.einsum("ij,ij->j", true, pred)
    denominator = np.sqrt(
        np.einsum("ij,ij->j", true, true) * np.einsum("ij,ij->j", pred, pred)
    )
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result.astype(np.float32)


def run_hemisphere_analysis(
    Y_train_path: Path,
    Y_test_path: Path,
    pca_cache: Path,
    spec: AnalysisSpec,
    output_dir: Path,
    alphas: Sequence[float],
    inner_splits: int,
    vertex_chunk_size: int,
    random_state: int,
    save_joint_test_predictions: bool = False,
) -> None:
    Y_train_all = np.load(Y_train_path, mmap_mode="r")
    Y_test_all = np.load(Y_test_path, mmap_mode="r")
    if Y_train_all.shape[1] != Y_test_all.shape[1]:
        raise ValueError("Training/test vertex counts do not match")
    output_dir.mkdir(parents=True, exist_ok=True)
    n_vertices = Y_train_all.shape[1]
    map_names = (
        "r_visual",
        "r_semantic",
        "r_joint",
        "r2_visual",
        "r2_semantic",
        "r2_joint",
        "unique_visual",
        "unique_semantic",
        "shared",
    )
    maps = {
        name: np.lib.format.open_memmap(output_dir / f"{name}.npy", "w+", np.float32, (n_vertices,))
        for name in map_names
    }
    weights_map = np.lib.format.open_memmap(
        output_dir / "visual_layer_weights.npy",
        "w+",
        np.float32,
        (len(spec.visual_groups), n_vertices),
    )
    joint_test_predictions = None
    if save_joint_test_predictions:
        joint_test_predictions = np.lib.format.open_memmap(
            output_dir / "test_prediction_joint.npy",
            "w+",
            np.float32,
            (Y_test_all.shape[0], n_vertices),
        )
    alpha_maps = {
        group: np.lib.format.open_memmap(
            output_dir / f"selected_alpha_{group}.npy", "w+", np.float32, (n_vertices,)
        )
        for group in (*spec.visual_groups, spec.semantic_group)
    }

    all_groups = (*spec.visual_groups, spec.semantic_group)
    with tempfile.TemporaryDirectory(prefix="base_predictions_", dir=output_dir) as temp_name:
        temporary_dir = Path(temp_name)
        oof_paths: dict[str, Path] = {}
        test_paths: dict[str, Path] = {}
        for group_index, group in enumerate(all_groups):
            print(f"BASE_MODEL_START group={group}", flush=True)
            oof, test_prediction = _generate_base_predictions(
                pca_cache,
                group,
                Y_train_path,
                temporary_dir,
                alpha_maps[group],
                alphas,
                inner_splits,
                vertex_chunk_size,
                random_state,
            )
            oof.flush()
            test_prediction.flush()
            oof_paths[group] = temporary_dir / f"{group}_oof.npy"
            test_paths[group] = temporary_dir / f"{group}_test.npy"
            oof._mmap.close()
            test_prediction._mmap.close()
            print(f"BASE_MODEL_DONE group={group}", flush=True)

        oof_groups = {group: np.load(path, mmap_mode="r") for group, path in oof_paths.items()}
        test_groups = {group: np.load(path, mmap_mode="r") for group, path in test_paths.items()}

        for start in range(0, n_vertices, vertex_chunk_size):
            stop = min(start + vertex_chunk_size, n_vertices)
            Y_train = np.asarray(Y_train_all[:, start:stop], dtype=np.float32)
            Y_test = np.asarray(Y_test_all[:, start:stop], dtype=np.float32)
            visual_oof = np.stack(
                [np.asarray(oof_groups[group][:, start:stop], dtype=np.float32) for group in spec.visual_groups],
                axis=1,
            )
            visual_test = np.stack(
                [np.asarray(test_groups[group][:, start:stop], dtype=np.float32) for group in spec.visual_groups],
                axis=1,
            )
            visual_weights, visual_intercept = fit_convex_stacking(visual_oof, Y_train)
            prediction_visual = apply_stacking(visual_test, visual_weights, visual_intercept)
            weights_map[:, start:stop] = visual_weights

            semantic_oof = np.asarray(
                oof_groups[spec.semantic_group][:, start:stop], dtype=np.float32
            )[:, None, :]
            semantic_base_test = np.asarray(
                test_groups[spec.semantic_group][:, start:stop], dtype=np.float32
            )[:, None, :]
            # One-model stacking learns the same OOF calibration/intercept used by nested joint stacking.
            semantic_weights, semantic_intercept = fit_convex_stacking(semantic_oof, Y_train)
            prediction_semantic = apply_stacking(
                semantic_base_test, semantic_weights, semantic_intercept
            )
            joint_oof = np.concatenate([visual_oof, semantic_oof], axis=1)
            joint_test = np.concatenate([visual_test, semantic_base_test], axis=1)
            joint_weights, joint_intercept = fit_convex_stacking(joint_oof, Y_train)
            prediction_joint = apply_stacking(joint_test, joint_weights, joint_intercept)
            if joint_test_predictions is not None:
                joint_test_predictions[:, start:stop] = prediction_joint

            r2_visual = predictive_r2(Y_test, prediction_visual)
            r2_semantic = predictive_r2(Y_test, prediction_semantic)
            r2_joint = predictive_r2(Y_test, prediction_joint)
            maps["r_visual"][start:stop] = predictive_correlation(Y_test, prediction_visual)
            maps["r_semantic"][start:stop] = predictive_correlation(Y_test, prediction_semantic)
            maps["r_joint"][start:stop] = predictive_correlation(Y_test, prediction_joint)
            maps["r2_visual"][start:stop] = r2_visual
            maps["r2_semantic"][start:stop] = r2_semantic
            maps["r2_joint"][start:stop] = r2_joint
            maps["unique_visual"][start:stop] = r2_joint - r2_semantic
            maps["unique_semantic"][start:stop] = r2_joint - r2_visual
            maps["shared"][start:stop] = r2_visual + r2_semantic - r2_joint
            if start == 0 or stop == n_vertices or start % (vertex_chunk_size * 16) == 0:
                print(f"STACK_PROGRESS vertices={stop}/{n_vertices}", flush=True)

        # Release Windows memmap handles before TemporaryDirectory removes its files.
        for array in (*oof_groups.values(), *test_groups.values()):
            array.flush()
            array._mmap.close()
        oof_groups.clear()
        test_groups.clear()
        del oof, test_prediction

    arrays_to_flush = [*maps.values(), weights_map, *alpha_maps.values()]
    if joint_test_predictions is not None:
        arrays_to_flush.append(joint_test_predictions)
    for array in arrays_to_flush:
        array.flush()
    metadata = {
        "analysis": spec.name,
        "visual_groups": list(spec.visual_groups),
        "semantic_group": spec.semantic_group,
        "n_vertices": n_vertices,
        "r2_definition": "1 - SSE_test / sum((y_test - mean(y_test))^2)",
        "held_out_metrics": (
            "Pearson correlation and predictive R2; the encoding-significance "
            "mask is defined from joint-model predictive R2"
        ),
        "joint_test_predictions_saved": save_joint_test_predictions,
        "variance_partition": {
            "unique_visual": "r2_joint - r2_semantic",
            "unique_semantic": "r2_joint - r2_visual",
            "shared": "r2_visual + r2_semantic - r2_joint",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
