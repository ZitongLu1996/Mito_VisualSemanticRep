"""Mitochondrial-map inference using subject/model-specific encoding masks.

The functional maps remain unnormalized by noise ceiling. Each subject/model
uses its own held-out encoding-significance mask. Spatial nulls are constructed
separately on those masks and combined only at the subject-effect level.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from brainspace.null_models import MoranRandomization
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import pearsonr, ttest_1samp


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
sys.path.insert(0, str(WORKSPACE))

NEUROMAPS_DATA = ROOT / "neuromaps_data"
WORKBENCH = WORKSPACE / "mitochondrial_analysis" / "tools" / "workbench" / "bin_windows64"

# Windows desktop sessions can contain duplicate PATH keys. Normalize them
# before neuromaps launches wb_command.
old_path = ";".join(value for key, value in os.environ.items() if key.lower() == "path")
for key in [key for key in os.environ if key.lower() == "path"]:
    del os.environ[key]
os.environ["PATH"] = f"{WORKBENCH};{old_path}"
os.environ["NEUROMAPS_DATA"] = str(NEUROMAPS_DATA)

from neuromaps.transforms import fsaverage_to_fsaverage  # noqa: E402
from nilearn import surface  # noqa: E402
from mitochondrial_analysis.run_mito_variance_analysis import (  # noqa: E402
    bootstrap_fisher_summary,
    fdr_bh,
    group_t_stat,
    load_external_surfaces,
    partial_r,
    row_correlations,
    row_partial_correlations,
)


BASE = ROOT / "derivatives" / "encoding_significance"
RESULTS = BASE / "results"
RESPONSES = BASE / "responses"
MASKS = BASE / "masks"
OUTPUT = ROOT / "derivatives" / "mitochondrial_analysis"
CACHE = OUTPUT / "cache"

SUBJECTS = ("subj01", "subj02", "subj05", "subj07")
ANALYSES = ("dinov2_minilm", "cornet_s_mpnet")
VARIANCE_MAPS = ("unique_visual", "unique_semantic", "shared")
MITO_MAPS = ("MitoD", "MRC")
HEMIS = ("lh", "rh")
SEED = 20260722


def gifti(values: np.ndarray) -> nib.GiftiImage:
    array = np.asarray(values, dtype=np.float32)
    return nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(array)])


def gii_data(image: nib.GiftiImage) -> np.ndarray:
    return np.asarray(image.agg_data(), dtype=np.float64).squeeze()


def resample_pair(
    left: np.ndarray, right: np.ndarray, method: str = "linear"
) -> tuple[np.ndarray, np.ndarray]:
    output = fsaverage_to_fsaverage(
        (gifti(left), gifti(right)), target_density="10k", method=method
    )
    return gii_data(output[0]), gii_data(output[1])


def candidate_full(subject: str, hemi: str, values: np.ndarray | None = None) -> np.ndarray:
    indices = np.load(RESPONSES / subject / hemi / "vertex_indices.npy").astype(int)
    full = np.zeros(163842, dtype=np.float64)
    if values is None:
        full[indices] = 1.0
    else:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != indices.shape:
            raise ValueError(f"Unexpected value shape for {subject} {hemi}: {values.shape}")
        full[indices] = values
    return full


def build_vectors(force: bool = False, support_threshold: float = 0.5) -> dict[str, np.ndarray]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / "subject_masked_vectors_fsaverage10k.npz"
    if cache_file.exists() and not force:
        saved = np.load(cache_file)
        return {key: saved[key] for key in saved.files}

    hierarchy, mito, mito_valid = load_external_surfaces(support_threshold=support_threshold)
    external_valid = {}
    for hemi in HEMIS:
        valid = np.isfinite(hierarchy[hemi])
        for name in MITO_MAPS:
            valid &= mito_valid[name][hemi] & np.isfinite(mito[name][hemi])
        external_valid[hemi] = valid

    vectors: dict[str, np.ndarray] = {}
    vertex_rows: list[dict] = []

    for subject in SUBJECTS:
        candidate = {
            hemi: candidate_full(subject, hemi)
            for hemi in HEMIS
        }
        candidate10 = dict(zip(HEMIS, resample_pair(candidate["lh"], candidate["rh"], method="linear")))

        for analysis in ANALYSES:
            mask164 = np.load(MASKS / f"{subject}_{analysis}_encoding_mask.npz")
            sig10_pair = resample_pair(
                mask164["mask_lh"].astype(float),
                mask164["mask_rh"].astype(float),
                method="nearest",
            )
            analysis_masks = {}
            for hemi, sig10 in zip(HEMIS, sig10_pair):
                analysis_masks[hemi] = (
                    (sig10 > 0.5)
                    & (candidate10[hemi] > support_threshold)
                    & external_valid[hemi]
                )
                key = f"{subject}_{analysis}"
                vectors[f"mask_{key}_{hemi}"] = analysis_masks[hemi]
                vectors[f"vertex_{key}_{hemi}"] = np.flatnonzero(analysis_masks[hemi])
                for vertex in np.flatnonzero(analysis_masks[hemi]):
                    vertex_rows.append(
                        {
                            "subject": subject,
                            "analysis": analysis,
                            "hemisphere": hemi,
                            "fsaverage10k_vertex": int(vertex),
                        }
                    )

            key = f"{subject}_{analysis}"
            vectors[f"hierarchy_{key}"] = np.concatenate(
                [hierarchy[h][analysis_masks[h]] for h in HEMIS]
            )
            for mito_name in MITO_MAPS:
                vectors[f"mito_{mito_name}_{key}"] = np.concatenate(
                    [mito[mito_name][h][analysis_masks[h]] for h in HEMIS]
                )

            for map_name in VARIANCE_MAPS:
                numerator = {}
                for hemi in HEMIS:
                    values = np.load(RESULTS / subject / analysis / hemi / f"{map_name}.npy")
                    numerator[hemi] = candidate_full(subject, hemi, values)
                num_lh, num_rh = resample_pair(numerator["lh"], numerator["rh"], method="linear")
                resampled = {
                    "lh": num_lh / np.maximum(candidate10["lh"], 1e-12),
                    "rh": num_rh / np.maximum(candidate10["rh"], 1e-12),
                }
                vector = np.concatenate(
                    [resampled[h][analysis_masks[h]] for h in HEMIS]
                )
                if not np.isfinite(vector).all():
                    raise ValueError(f"Non-finite map: {subject} {analysis} {map_name}")
                vectors[f"func_{map_name}_{key}"] = vector

            lengths = {name: len(vectors[f"func_{name}_{key}"]) for name in VARIANCE_MAPS}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Inconsistent functional vector lengths: {key} {lengths}")
            if lengths["unique_visual"] < 100:
                raise ValueError(f"Too few aligned vertices: {key} {lengths}")

    np.savez_compressed(cache_file, **vectors)
    pd.DataFrame(vertex_rows).to_csv(OUTPUT / "subject_masked_vertex_index.csv", index=False)

    qc_rows = []
    for subject in SUBJECTS:
        for analysis in ANALYSES:
            key = f"{subject}_{analysis}"
            for hemi in HEMIS:
                qc_rows.append(
                    {
                        "subject": subject,
                        "analysis": analysis,
                        "hemisphere": hemi,
                        "n_encoding_significant_164k": int(
                            np.load(MASKS / f"{subject}_{analysis}_encoding_mask.npz")[f"mask_{hemi}"].sum()
                        ),
                        "n_aligned_10k": int(vectors[f"mask_{key}_{hemi}"].sum()),
                    }
                )
    pd.DataFrame(qc_rows).to_csv(OUTPUT / "alignment_qc.csv", index=False)
    return vectors


def surface_graph(hemi: str):
    letter = "L" if hemi == "lh" else "R"
    atlas = NEUROMAPS_DATA / "atlases" / "fsaverage"
    coords, faces = surface.load_surf_mesh(
        atlas / f"tpl-fsaverage_den-10k_hemi-{letter}_pial.surf.gii"
    )
    coords = np.asarray(coords, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.unique(
        np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
        axis=0,
    )
    lengths = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    return coo_matrix(
        (
            np.r_[lengths, lengths],
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(coords.shape[0], coords.shape[0]),
    ).tocsr()


def moran_model(vectors: dict[str, np.ndarray], subject: str, analysis: str) -> MoranRandomization:
    key = f"{subject}_{analysis}"
    mem_file = CACHE / f"moran_mem_{key}.npy"
    model = MoranRandomization(
        procedure="singleton",
        spectrum="nonzero",
        joint=True,
        n_rep=1,
        tol=1e-6,
        random_state=SEED,
    )
    if mem_file.exists():
        model.mem_ = np.load(mem_file, mmap_mode="r")
        return model

    graphs = {hemi: surface_graph(hemi) for hemi in HEMIS}
    weights_blocks = []
    for hemi in HEMIS:
        vertices = vectors[f"vertex_{key}_{hemi}"].astype(int)
        distance = np.asarray(
            dijkstra(graphs[hemi], directed=False, indices=vertices)[:, vertices],
            dtype=np.float64,
        )
        weights = np.zeros_like(distance)
        finite = np.isfinite(distance) & (distance > 0)
        weights[finite] = 1.0 / distance[finite]
        weights_blocks.append(weights)

    n_lh, n_rh = (block.shape[0] for block in weights_blocks)
    weights = np.zeros((n_lh + n_rh, n_lh + n_rh), dtype=np.float64)
    weights[:n_lh, :n_lh] = weights_blocks[0]
    weights[n_lh:, n_lh:] = weights_blocks[1]
    model.fit(weights)
    np.save(mem_file, np.asarray(model.mem_, dtype=np.float32))
    model.mem_ = np.load(mem_file, mmap_mode="r")
    return model


def spatial_p_from_count(count: int, n_perm: int) -> float:
    return float((count + 1) / (n_perm + 1))


def spectral_null_correlations(
    model: MoranRandomization,
    mito_matrix: np.ndarray,
    functions: dict[str, np.ndarray],
    hierarchy: np.ndarray,
    n_rep: int,
    random_state: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Evaluate singleton Moran null correlations without rebuilding maps.

    BrainSpace's singleton procedure flips the signs of the Moran spectral
    coefficients. Because the Moran eigenvectors are orthonormal and centered,
    correlation with a fixed cortical map is exactly the dot product between
    the sign-flipped coefficients and that map's spectral projection. The same
    signs are used for all mitochondrial maps (``joint=True``), matching
    ``MoranRandomization.randomize`` while avoiding an unnecessary dense
    vertex-space reconstruction for every permutation.
    """
    mem = np.asarray(model.mem_)
    coefficients = 1.0 - cdist(mito_matrix.T, mem.T, "correlation").T
    rng = np.random.RandomState(random_state)
    signs = rng.choice((-1.0, 1.0), size=(n_rep, mem.shape[1]))

    projections = {}
    norms = {}
    for name, values in {**functions, "hierarchy": hierarchy}.items():
        centered = np.asarray(values, dtype=np.float64) - np.mean(values)
        projections[name] = mem.T @ centered
        norms[name] = np.linalg.norm(centered)

    raw_by_map: dict[str, np.ndarray] = {}
    partial_by_map: dict[str, np.ndarray] = {}
    r_function_hierarchy = {
        name: float(pearsonr(values, hierarchy).statistic)
        for name, values in functions.items()
    }
    hierarchy_projection = projections["hierarchy"]
    hierarchy_norm = norms["hierarchy"]

    # Coefficient norms are invariant to sign flips. MEM columns are
    # orthonormal, so this is also the norm of each centered surrogate.
    coefficient_norms = np.linalg.norm(coefficients, axis=0)
    map_names = tuple(functions)
    for mito_index, mito_name in enumerate(MITO_MAPS):
        weighted_signs = signs * coefficients[:, mito_index]
        r_mito_hierarchy = (
            weighted_signs @ hierarchy_projection
            / (coefficient_norms[mito_index] * hierarchy_norm)
        )
        for map_name in map_names:
            r_mito_function = (
                weighted_signs @ projections[map_name]
                / (coefficient_norms[mito_index] * norms[map_name])
            )
            r_function_h = r_function_hierarchy[map_name]
            denominator = np.sqrt(
                np.maximum(1e-15, (1.0 - r_mito_hierarchy**2) * (1.0 - r_function_h**2))
            )
            key = f"{mito_name}:{map_name}"
            raw_by_map[key] = r_mito_function
            partial_by_map[key] = (
                r_mito_function - r_mito_hierarchy * r_function_h
            ) / denominator
    return raw_by_map, partial_by_map


def validate_spectral_implementation(vectors: dict[str, np.ndarray]) -> None:
    """Numerically compare spectral evaluation with explicit reconstruction."""
    max_raw_error = 0.0
    max_partial_error = 0.0
    n_rep = 12
    for analysis_index, analysis in enumerate(ANALYSES):
        for subject_index, subject in enumerate(SUBJECTS):
            key = f"{subject}_{analysis}"
            model = moran_model(vectors, subject, analysis)
            seed = SEED + 700001 + analysis_index * 1009 + subject_index * 101
            mito_matrix = np.column_stack([
                vectors[f"mito_{name}_{key}"] for name in MITO_MAPS
            ])
            hierarchy = vectors[f"hierarchy_{key}"]
            functions = {
                map_name: vectors[f"func_{map_name}_{key}"]
                for map_name in VARIANCE_MAPS
            }
            spectral_raw, spectral_partial = spectral_null_correlations(
                model, mito_matrix, functions, hierarchy, n_rep, seed
            )
            model.n_rep = n_rep
            model.random_state = seed
            explicit = model.randomize(mito_matrix)
            for mito_index, mito_name in enumerate(MITO_MAPS):
                null_mito = np.asarray(explicit[:, :, mito_index], dtype=np.float64)
                for map_name in VARIANCE_MAPS:
                    comparison_key = f"{mito_name}:{map_name}"
                    raw_explicit = row_correlations(null_mito, functions[map_name])
                    partial_explicit = row_partial_correlations(
                        null_mito, functions[map_name], hierarchy
                    )
                    max_raw_error = max(
                        max_raw_error,
                        float(np.max(np.abs(raw_explicit - spectral_raw[comparison_key]))),
                    )
                    max_partial_error = max(
                        max_partial_error,
                        float(np.max(np.abs(partial_explicit - spectral_partial[comparison_key]))),
                    )
            print(f"SPECTRAL_VALIDATION_DONE {subject} {analysis}", flush=True)
    summary = {
        "max_abs_error_raw_r": max_raw_error,
        "max_abs_error_partial_r": max_partial_error,
        "tolerance": 2e-5,
        "passed": bool(max(max_raw_error, max_partial_error) < 2e-5),
    }
    (OUTPUT / "spectral_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passed"]:
        raise RuntimeError("Spectral Moran evaluation did not match explicit reconstruction")


def observed_statistics(vectors: dict[str, np.ndarray]):
    raw: dict[tuple, float] = {}
    partial: dict[tuple, float] = {}
    for subject in SUBJECTS:
        for analysis in ANALYSES:
            key = f"{subject}_{analysis}"
            hierarchy = vectors[f"hierarchy_{key}"]
            for mito_name in MITO_MAPS:
                mito = vectors[f"mito_{mito_name}_{key}"]
                for map_name in VARIANCE_MAPS:
                    func = vectors[f"func_{map_name}_{key}"]
                    test = (subject, analysis, mito_name, map_name)
                    raw[test] = float(pearsonr(func, mito).statistic)
                    partial[test] = partial_r(func, mito, hierarchy)
    return raw, partial


def run_inference(vectors: dict[str, np.ndarray], n_perm: int, batch_size: int) -> None:
    if n_perm % batch_size:
        raise ValueError("n_perm must be divisible by batch_size")

    raw, partial = observed_statistics(vectors)
    models = {
        (subject, analysis): moran_model(vectors, subject, analysis)
        for subject in SUBJECTS
        for analysis in ANALYSES
    }

    metrics = ("raw", "partial")
    observed = {"raw": raw, "partial": partial}
    individual_counts = {
        metric: {key: 0 for key in observed[metric]}
        for metric in metrics
    }
    group_counts = {
        metric: {
            (analysis, mito, map_name): 0
            for analysis in ANALYSES
            for mito in MITO_MAPS
            for map_name in VARIANCE_MAPS
        }
        for metric in metrics
    }
    observed_group = {metric: {} for metric in metrics}
    for metric in metrics:
        for analysis in ANALYSES:
            for mito_name in MITO_MAPS:
                for map_name in VARIANCE_MAPS:
                    z = np.asarray([
                        np.arctanh(np.clip(observed[metric][(s, analysis, mito_name, map_name)], -.999999, .999999))
                        for s in SUBJECTS
                    ])
                    observed_group[metric][(analysis, mito_name, map_name)] = float(group_t_stat(z))

    for batch in range(n_perm // batch_size):
        null_store = {
            metric: {
                (analysis, mito, map_name): [None] * len(SUBJECTS)
                for analysis in ANALYSES
                for mito in MITO_MAPS
                for map_name in VARIANCE_MAPS
            }
            for metric in metrics
        }
        for analysis_index, analysis in enumerate(ANALYSES):
            for subject_index, subject in enumerate(SUBJECTS):
                key = f"{subject}_{analysis}"
                model = models[(subject, analysis)]
                random_state = SEED + (batch + 1) * 100003 + analysis_index * 1009 + subject_index * 101
                mito_matrix = np.column_stack([
                    vectors[f"mito_{name}_{key}"] for name in MITO_MAPS
                ])
                hierarchy = vectors[f"hierarchy_{key}"]
                functions = {
                    map_name: vectors[f"func_{map_name}_{key}"]
                    for map_name in VARIANCE_MAPS
                }
                spectral_raw, spectral_partial = spectral_null_correlations(
                    model,
                    mito_matrix,
                    functions,
                    hierarchy,
                    batch_size,
                    random_state,
                )

                for mito_index, mito_name in enumerate(MITO_MAPS):
                    for map_name in VARIANCE_MAPS:
                        spectral_key = f"{mito_name}:{map_name}"
                        null_raw = spectral_raw[spectral_key]
                        null_partial = spectral_partial[spectral_key]
                        for metric, null in (("raw", null_raw), ("partial", null_partial)):
                            test = (subject, analysis, mito_name, map_name)
                            individual_counts[metric][test] += int(
                                np.count_nonzero(np.abs(null) >= abs(observed[metric][test]))
                            )
                            null_store[metric][(analysis, mito_name, map_name)][subject_index] = null

                del spectral_raw, spectral_partial

        for metric in metrics:
            for analysis in ANALYSES:
                for mito_name in MITO_MAPS:
                    for map_name in VARIANCE_MAPS:
                        null_r = np.column_stack(null_store[metric][(analysis, mito_name, map_name)])
                        null_t = group_t_stat(
                            np.arctanh(np.clip(null_r, -.999999, .999999)), axis=1
                        )
                        gkey = (analysis, mito_name, map_name)
                        group_counts[metric][gkey] += int(
                            np.count_nonzero(np.abs(null_t) >= abs(observed_group[metric][gkey]))
                        )
        print(
            f"PERMUTATION_BATCH_DONE {batch + 1}/{n_perm // batch_size} "
            f"({(batch + 1) * batch_size:,}/{n_perm:,})",
            flush=True,
        )

    subject_rows, group_rows = [], []
    for subject in SUBJECTS:
        for analysis in ANALYSES:
            key = f"{subject}_{analysis}"
            n_vertices = len(vectors[f"hierarchy_{key}"])
            for mito_name in MITO_MAPS:
                for map_name in VARIANCE_MAPS:
                    test = (subject, analysis, mito_name, map_name)
                    subject_rows.append(
                        {
                            "subject": subject,
                            "analysis": analysis,
                            "variance_map": map_name,
                            "mitochondrial_map": mito_name,
                            "n_vertices": n_vertices,
                            "raw_r": raw[test],
                            "raw_spatial_p_100k": spatial_p_from_count(individual_counts["raw"][test], n_perm),
                            "partial_r_pg1": partial[test],
                            "partial_spatial_p_100k": spatial_p_from_count(individual_counts["partial"][test], n_perm),
                        }
                    )
    for analysis in ANALYSES:
        for mito_name in MITO_MAPS:
            for map_name in VARIANCE_MAPS:
                raw_r_values = np.asarray([raw[(s, analysis, mito_name, map_name)] for s in SUBJECTS])
                partial_r_values = np.asarray([partial[(s, analysis, mito_name, map_name)] for s in SUBJECTS])
                raw_z = np.arctanh(np.clip(raw_r_values, -.999999, .999999))
                partial_z = np.arctanh(np.clip(partial_r_values, -.999999, .999999))
                raw_est, raw_low, raw_high = bootstrap_fisher_summary(raw_r_values, seed=SEED)
                part_est, part_low, part_high = bootstrap_fisher_summary(partial_r_values, seed=SEED)
                gkey = (analysis, mito_name, map_name)
                group_rows.append(
                    {
                        "analysis": analysis,
                        "variance_map": map_name,
                        "mitochondrial_map": mito_name,
                        "n_subjects": len(SUBJECTS),
                        "min_subject_vertices": int(min(
                            len(vectors[f"hierarchy_{s}_{analysis}"]) for s in SUBJECTS
                        )),
                        "max_subject_vertices": int(max(
                            len(vectors[f"hierarchy_{s}_{analysis}"]) for s in SUBJECTS
                        )),
                        "fisher_mean_raw_r": raw_est,
                        "raw_bootstrap_ci_low": raw_low,
                        "raw_bootstrap_ci_high": raw_high,
                        "raw_group_t": float(group_t_stat(raw_z)),
                        "raw_group_t_p": float(ttest_1samp(raw_z, 0).pvalue),
                        "raw_group_spatial_p_100k": spatial_p_from_count(group_counts["raw"][gkey], n_perm),
                        "raw_n_positive": int((raw_r_values > 0).sum()),
                        "fisher_mean_partial_r": part_est,
                        "partial_bootstrap_ci_low": part_low,
                        "partial_bootstrap_ci_high": part_high,
                        "partial_group_t": float(group_t_stat(partial_z)),
                        "partial_group_t_p": float(ttest_1samp(partial_z, 0).pvalue),
                        "partial_group_spatial_p_100k": spatial_p_from_count(group_counts["partial"][gkey], n_perm),
                        "partial_n_positive": int((partial_r_values > 0).sum()),
                    }
                )

    subjects = pd.DataFrame(subject_rows)
    groups = pd.DataFrame(group_rows)

    for metric in metrics:
        subjects[f"{metric}_spatial_q_6tests"] = subjects.groupby(
            ["subject", "analysis"]
        )[f"{metric}_spatial_p_100k"].transform(lambda x: fdr_bh(x.to_numpy()))
        groups[f"{metric}_group_spatial_q_6tests"] = groups.groupby("analysis")[
            f"{metric}_group_spatial_p_100k"
        ].transform(lambda x: fdr_bh(x.to_numpy()))
    subjects.to_csv(OUTPUT / "subject_results_100k.csv", index=False)
    groups.to_csv(OUTPUT / "group_results_100k.csv", index=False)

    settings = {
        "status": "complete",
        "subjects": list(SUBJECTS),
        "analyses": list(ANALYSES),
        "variance_maps": list(VARIANCE_MAPS),
        "mitochondrial_maps": list(MITO_MAPS),
        "noise_ceiling_normalization": False,
        "functional_masks": "subject/model-specific held-out encoding significance masks",
        "surface_space": "fsaverage10k",
        "covariate": "Margulies PG1 cortical hierarchy",
        "n_permutations": n_perm,
        "batch_size": batch_size,
        "spatial_null": "mask-specific Moran spectral surrogates using fsaverage pial geodesic distances",
        "group_null": "group t across four subject effects; each subject uses an independently randomized mask-specific mitochondrial surrogate at the same permutation index",
        "primary_family": "6 tests (UV, US, shared x MitoD, MRC), separately within visual model and correlation type",
        "primary_inference": "PG1-adjusted partial correlation",
        "group_average_maps": "not computed because subject-specific masks differ",
        "p_value": "two-sided add-one empirical spatial p",
    }
    (OUTPUT / "analysis_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    print(json.dumps(settings, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--force-align", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-spectral", action="store_true")
    args = parser.parse_args()
    vectors = build_vectors(force=args.force_align)
    qc = pd.read_csv(OUTPUT / "alignment_qc.csv")
    print(qc.to_string(index=False), flush=True)
    if args.prepare_only:
        return
    if args.validate_spectral:
        validate_spectral_implementation(vectors)
        return
    if args.n_perm % args.batch_size:
        raise ValueError("n-perm must be divisible by batch-size")
    run_inference(vectors, args.n_perm, args.batch_size)


if __name__ == "__main__":
    main()
