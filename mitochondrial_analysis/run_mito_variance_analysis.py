"""Utilities used by the final NSD mitochondrial-map analyses.

Only the retained MitoD/MRC surface preparation and shared numerical helpers
live here; discarded legacy analyses are intentionally absent.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "mitochondrial_analysis"
SOURCE_ROOT = ANALYSIS_ROOT / "source_maps" / "mosharov2025"
NEUROMAPS_ROOT = ANALYSIS_ROOT / "neuromaps_data"
PROCESSED_ROOT = ANALYSIS_ROOT / "processed_maps"
WORKBENCH_BIN = ANALYSIS_ROOT / "tools" / "workbench" / "bin_windows64"

os.environ.setdefault("NEUROMAPS_DATA", str(NEUROMAPS_ROOT))
if WORKBENCH_BIN.exists():
    os.environ["PATH"] = (
        str(WORKBENCH_BIN) + os.pathsep + os.environ.get("PATH", "")
    )

HEMIS = ("lh", "rh")
MITO_MAPS = ("MitoD", "MRC")
SEED = 20260717


def _gii_data(image: nib.GiftiImage) -> np.ndarray:
    return np.asarray(image.agg_data(), dtype=np.float64).squeeze()


def _save_gifti_pair(images, stem: str) -> tuple[Path, Path]:
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    paths = []
    for hemi, image in zip(HEMIS, images):
        path = PROCESSED_ROOT / f"{stem}_{hemi}.func.gii"
        nib.save(image, path)
        paths.append(path)
    return tuple(paths)


def prepare_surface_maps(force: bool = False) -> None:
    """Project PG1, MitoD, and MRC to fsaverage10k."""
    from neuromaps.datasets import fetch_annotation
    from neuromaps.transforms import fslr_to_fsaverage, mni152_to_fsaverage

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    NEUROMAPS_ROOT.mkdir(parents=True, exist_ok=True)
    hierarchy_paths = [
        PROCESSED_ROOT / f"margulies2016_pg1_{hemi}.func.gii"
        for hemi in HEMIS
    ]
    if force or not all(path.exists() for path in hierarchy_paths):
        pg1_fslr = fetch_annotation(
            source="margulies2016",
            desc="fcgradient01",
            space="fsLR",
            den="32k",
            data_dir=NEUROMAPS_ROOT,
        )
        _save_gifti_pair(
            fslr_to_fsaverage(
                pg1_fslr, target_density="10k", method="linear"
            ),
            "margulies2016_pg1",
        )

    for name in MITO_MAPS:
        data_paths = [
            PROCESSED_ROOT / f"mosharov2025_{name}_{hemi}.func.gii"
            for hemi in HEMIS
        ]
        support_paths = [
            PROCESSED_ROOT
            / f"mosharov2025_{name}_support_{hemi}.func.gii"
            for hemi in HEMIS
        ]
        if not force and all(
            path.exists() for path in data_paths + support_paths
        ):
            continue
        image = nib.load(SOURCE_ROOT / f"{name}.nii.gz")
        array = np.asarray(image.dataobj)
        support_image = nib.Nifti1Image(
            (array != 0).astype(np.float32), image.affine, image.header
        )
        _save_gifti_pair(
            mni152_to_fsaverage(
                image, fsavg_density="10k", method="linear"
            ),
            f"mosharov2025_{name}",
        )
        _save_gifti_pair(
            mni152_to_fsaverage(
                support_image, fsavg_density="10k", method="linear"
            ),
            f"mosharov2025_{name}_support",
        )


def load_external_surfaces(
    support_threshold: float = 0.5,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
]:
    """Load fsaverage10k PG1 and support-corrected mitochondrial maps."""
    hierarchy: dict[str, np.ndarray] = {}
    mito = {name: {} for name in MITO_MAPS}
    valid = {name: {} for name in MITO_MAPS}
    for hemi in HEMIS:
        hierarchy[hemi] = _gii_data(
            nib.load(
                PROCESSED_ROOT
                / f"margulies2016_pg1_{hemi}.func.gii"
            )
        )
        for name in MITO_MAPS:
            projected = _gii_data(
                nib.load(
                    PROCESSED_ROOT
                    / f"mosharov2025_{name}_{hemi}.func.gii"
                )
            )
            weight = _gii_data(
                nib.load(
                    PROCESSED_ROOT
                    / f"mosharov2025_{name}_support_{hemi}.func.gii"
                )
            )
            keep = (
                np.isfinite(projected)
                & np.isfinite(weight)
                & (weight > support_threshold)
            )
            values = np.full(projected.shape, np.nan, dtype=np.float64)
            values[keep] = projected[keep] / weight[keep]
            mito[name][hemi] = values
            valid[name][hemi] = keep
    return hierarchy, mito, valid


def partial_r(
    x: np.ndarray, y: np.ndarray, covariate: np.ndarray
) -> float:
    """Pearson correlation after linear residualization of one covariate."""
    design = np.column_stack([np.ones(len(covariate)), covariate])
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def row_correlations(
    matrix: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    """Pearson correlations between every matrix row and one vector."""
    centered_matrix = matrix - np.mean(matrix, axis=1, keepdims=True)
    centered_vector = vector - np.mean(vector)
    numerator = centered_matrix @ centered_vector
    denominator = np.linalg.norm(centered_matrix, axis=1) * np.linalg.norm(
        centered_vector
    )
    return numerator / denominator


def row_partial_correlations(
    matrix: np.ndarray,
    vector: np.ndarray,
    covariate: np.ndarray,
) -> np.ndarray:
    """Row-wise partial correlations controlling one covariate."""
    design = np.column_stack([np.ones(len(covariate)), covariate])
    residual_vector = vector - design @ np.linalg.lstsq(
        design, vector, rcond=None
    )[0]
    coefficients = np.linalg.lstsq(design, matrix.T, rcond=None)[0]
    residual_matrix = matrix - (design @ coefficients).T
    return row_correlations(residual_matrix, residual_vector)


def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values."""
    pvalues = np.asarray(pvalues, dtype=float)
    output = np.full(pvalues.shape, np.nan)
    valid = np.isfinite(pvalues)
    p = pvalues[valid]
    order = np.argsort(p)
    ranked = p[order]
    q = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(q)
    restored[order] = np.clip(q, 0.0, 1.0)
    output[valid] = restored
    return output


def group_t_stat(
    zvalues: np.ndarray, axis: int = -1
) -> np.ndarray:
    """One-sample t statistic against zero."""
    zvalues = np.asarray(zvalues, dtype=float)
    mean = np.mean(zvalues, axis=axis)
    sem = np.std(zvalues, axis=axis, ddof=1) / np.sqrt(
        zvalues.shape[axis]
    )
    return np.divide(
        mean, sem, out=np.zeros_like(mean), where=sem > 0
    )


def bootstrap_fisher_summary(
    rvalues: np.ndarray,
    seed: int = SEED,
    n_bootstrap: int = 100_000,
) -> tuple[float, float, float]:
    """Fisher-mean effect and subject-bootstrap percentile interval."""
    rvalues = np.asarray(rvalues, dtype=float)
    zvalues = np.arctanh(np.clip(rvalues, -0.999999, 0.999999))
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(zvalues), size=(n_bootstrap, len(zvalues))
    )
    distribution = np.tanh(zvalues[indices].mean(axis=1))
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(np.tanh(zvalues.mean())), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-surfaces",
        action="store_true",
        help="Project PG1, MitoD, and MRC to fsaverage10k.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.prepare_surfaces:
        parser.error("--prepare-surfaces is required")
    prepare_surface_maps(force=args.force)


if __name__ == "__main__":
    main()
