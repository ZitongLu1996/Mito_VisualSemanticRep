"""Genome-wide AHBA association analysis for NSD variance maps.

The primary analysis uses every gene that passes a target-independent AHBA
quality-control workflow. Each fMRI subject and encoding model retains its own
held-out encoding-significance mask. Gene associations control AHBA donor
fixed effects and the Margulies PG1 cortical hierarchy. Statistical inference
uses mask-specific, surface-geodesic Moran spatial surrogates and group-level
statistics across the four fMRI subjects.
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import sys
from pathlib import Path

import abagen
import nibabel as nib
import numpy as np
import pandas as pd
from abagen import correct, datasets, io, probes_, samples_
from brainspace.null_models import MoranRandomization
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.stats import ttest_1samp

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from mitochondrial_analysis.run_mito_variance_analysis import load_external_surfaces
from run_subject_masked_mito_analysis import (
    MASKS,
    RESULTS as ENCODING_RESULTS,
    candidate_full,
    resample_pair,
)

AHBA = PROJECT / "ahba_analysis"
OUTPUT = ROOT / "derivatives" / "ahba_genomewide_gene_analysis"
CACHE = OUTPUT / "cache"
FIGURES = OUTPUT / "figures"

REGFUSION = (
    AHBA
    / "atlas_data"
    / "atlases"
    / "regfusion"
    / "tpl-MNI152_space-fsaverage_den-10k_hemi-L_regfusion.txt"
)
SURFACE = (
    ROOT
    / "neuromaps_data"
    / "atlases"
    / "fsaverage"
    / "tpl-fsaverage_den-10k_hemi-L_pial.surf.gii"
)

DONORS = ("9861", "10021", "12876", "14380", "15496", "15697")
SUBJECTS = ("01", "02", "05", "07")
ANALYSES = ("dinov2_minilm", "cornet_s_mpnet")
VARIANCE_MAPS = ("unique_visual", "unique_semantic", "shared")
SEED = 20260723


def enable_abagen_pandas_compatibility() -> None:
    """Restore the pandas ``set_axis(..., inplace=)`` API expected by abagen."""
    if "inplace" in inspect.signature(pd.DataFrame.set_axis).parameters:
        return
    original = pd.DataFrame.set_axis

    def set_axis_compat(self, labels, *, axis=0, inplace=None, copy=None):
        result = original(self, labels, axis=axis, copy=copy)
        if inplace:
            self._mgr = result._mgr
            return None
        return result

    pd.DataFrame.set_axis = set_axis_compat


def fdr_bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    output = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    p = values[valid]
    order = np.argsort(p)
    ranked = p[order]
    q = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(q)
    restored[order] = np.clip(q, 0.0, 1.0)
    output[valid] = restored
    return output


def group_t(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, float)
    mean = np.mean(z, axis=axis)
    sem = np.std(z, axis=axis, ddof=1) / np.sqrt(z.shape[axis])
    return np.divide(mean, sem, out=np.zeros_like(mean), where=sem > 0)


def exact_bootstrap_fisher_ci(r_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact four-subject percentile bootstrap CI over all 4**4 resamples."""
    z = np.arctanh(np.clip(np.asarray(r_values, float), -0.999999, 0.999999))
    indices = np.asarray(list(itertools.product(range(z.shape[0]), repeat=z.shape[0])))
    distribution = np.tanh(z[indices].mean(axis=1))
    low, high = np.quantile(distribution, [0.025, 0.975], axis=0)
    return low, high


def build_gene_vectors(force: bool = False) -> dict[str, np.ndarray]:
    """Align raw variance maps using encoding masks and PG1 validity only."""
    CACHE.mkdir(parents=True, exist_ok=True)
    output = CACHE / "gene_analysis_vectors_fsaverage10k.npz"
    if output.exists() and not force:
        saved = np.load(output)
        return {key: saved[key] for key in saved.files}

    hierarchy, _, _ = load_external_surfaces()
    vectors: dict[str, np.ndarray] = {}
    qc_rows = []
    for subject in SUBJECTS:
        candidate = {
            hemi: candidate_full(f"subj{subject}", hemi)
            for hemi in ("lh", "rh")
        }
        candidate10 = dict(
            zip(
                ("lh", "rh"),
                resample_pair(candidate["lh"], candidate["rh"], method="linear"),
            )
        )
        for analysis in ANALYSES:
            mask164 = np.load(MASKS / f"subj{subject}_{analysis}_encoding_mask.npz")
            sig_lh, sig_rh = resample_pair(
                mask164["mask_lh"].astype(float),
                mask164["mask_rh"].astype(float),
                method="nearest",
            )
            sig10 = {"lh": sig_lh, "rh": sig_rh}
            masks = {}
            for hemi in ("lh", "rh"):
                masks[hemi] = (
                    (sig10[hemi] > 0.5)
                    & (candidate10[hemi] > 0.5)
                    & np.isfinite(hierarchy[hemi])
                )
                qc_rows.append(
                    {
                        "subject": int(subject),
                        "analysis": analysis,
                        "hemisphere": hemi,
                        "n_encoding_significant_164k": int(mask164[f"mask_{hemi}"].sum()),
                        "n_gene_analysis_vertices_10k": int(masks[hemi].sum()),
                    }
                )

            key = f"subj{subject}_{analysis}"
            vectors[f"mask_{key}_lh"] = masks["lh"]
            vectors[f"vertex_{key}_lh"] = np.flatnonzero(masks["lh"])
            vectors[f"hierarchy_{key}_lh"] = hierarchy["lh"][masks["lh"]]

            for map_name in VARIANCE_MAPS:
                numerator = {}
                for hemi in ("lh", "rh"):
                    values = np.load(
                        ENCODING_RESULTS
                        / f"subj{subject}"
                        / analysis
                        / hemi
                        / f"{map_name}.npy"
                    )
                    numerator[hemi] = candidate_full(f"subj{subject}", hemi, values)
                num_lh, num_rh = resample_pair(
                    numerator["lh"], numerator["rh"], method="linear"
                )
                resampled_lh = num_lh / np.maximum(candidate10["lh"], 1e-12)
                vector = resampled_lh[masks["lh"]]
                if not np.isfinite(vector).all():
                    raise ValueError(f"Non-finite functional map: {key} {map_name}")
                vectors[f"func_{map_name}_{key}_lh"] = vector

    np.savez_compressed(output, **vectors)
    pd.DataFrame(qc_rows).to_csv(OUTPUT / "surface_alignment_qc.csv", index=False)
    return vectors


def prepare_cortical_expression(
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess all left cortical AHBA samples before selecting analysis sites."""
    CACHE.mkdir(parents=True, exist_ok=True)
    expression_file = CACHE / "left_cortical_gene_expression_srs.csv.gz"
    metadata_file = CACHE / "left_cortical_sample_metadata.csv"
    universe_file = OUTPUT / "gene_universe.csv"
    if expression_file.exists() and metadata_file.exists() and universe_file.exists() and not force:
        expression = pd.read_csv(expression_file, index_col=[0, 1])
        expression.index = pd.MultiIndex.from_arrays(
            [
                expression.index.get_level_values(0).astype(str),
                expression.index.get_level_values(1).astype(int),
            ],
            names=["donor", "well_id"],
        )
        metadata = pd.read_csv(metadata_file, dtype={"donor": str})
        universe = pd.read_csv(universe_file)
        return expression, metadata, universe

    enable_abagen_pandas_compatibility()
    files = datasets.fetch_microarray(
        data_dir=AHBA / "abagen_data",
        donors="all",
        verbose=1,
        n_proc=1,
    )
    first = files[DONORS[0]]
    probe_info = io.read_probes(first["probes"])
    n_raw_probes = len(probe_info)
    probe_info = probes_.reannotate_probes(probe_info)
    probe_info = probe_info.dropna(subset=["entrez_id"])
    n_reannotated_probes = len(probe_info)

    annotations: dict[str, pd.DataFrame] = {}
    for donor in DONORS:
        annotation = samples_.update_coords(
            files[donor]["annotation"], corrected_mni=True, native_space=None
        )
        annotation = samples_.drop_mismatch_samples(
            annotation, files[donor]["ontology"]
        )
        annotation = annotation.query(
            'structure == "cortex" and hemisphere == "L"'
        ).copy()
        if annotation.empty:
            raise RuntimeError(f"No left cortical samples for donor {donor}")
        annotations[donor] = annotation

    probe_info = probes_.filter_probes(
        {donor: files[donor]["pacall"] for donor in DONORS},
        annotations,
        probe_info,
        threshold=0.5,
    )
    n_background_pass_probes = len(probe_info)
    microarray = probes_.collapse_probes(
        {donor: files[donor]["microarray"] for donor in DONORS},
        annotations,
        probe_info,
        method="diff_stability",
        donor_probes="aggregate",
    )

    frames = []
    metadata_rows = []
    for donor in DONORS:
        frame = microarray[donor].copy()
        frame.index = frame.index.astype(int)
        cortical_samples = annotations[donor].index.astype(int).to_numpy()
        missing = sorted(set(cortical_samples) - set(frame.index))
        if missing:
            raise RuntimeError(
                f"Donor {donor} is missing left cortical samples: {missing}"
            )
        frame = frame.loc[cortical_samples]
        frame = correct.normalize_expression(
            frame.T, norm="srs", ignore_warn=True
        ).T
        frame = correct.normalize_expression(frame, norm="srs", ignore_warn=True)
        sample_annotation = annotations[donor].loc[frame.index].copy()
        well_ids = sample_annotation["well_id"].astype(int).to_numpy()
        frame.index = well_ids
        frame.index = pd.MultiIndex.from_product(
            [[donor], frame.index], names=["donor", "well_id"]
        )
        frames.append(frame)

        sample_annotation["well_id"] = well_ids
        for _, row in sample_annotation.iterrows():
            metadata_rows.append(
                {
                    "donor": donor,
                    "well_id": int(row["well_id"]),
                    "mni_x": float(row["mni_x"]),
                    "mni_y": float(row["mni_y"]),
                    "mni_z": float(row["mni_z"]),
                }
            )

    expression = pd.concat(frames)
    expression.columns = expression.columns.astype(str).str.upper()
    expression = expression.loc[:, ~expression.columns.duplicated()]
    donor_std = expression.groupby(level="donor").std(ddof=1)
    valid = (
        np.isfinite(expression.to_numpy()).all(axis=0)
        & (donor_std.to_numpy() > 1e-12).all(axis=0)
    )
    expression = expression.loc[:, valid]

    symbol_to_entrez = (
        probe_info.assign(
            gene_symbol=probe_info["gene_symbol"].astype(str).str.upper()
        )
        .drop_duplicates("gene_symbol")
        .set_index("gene_symbol")["entrez_id"]
    )
    universe = pd.DataFrame({"gene": expression.columns})
    universe["entrez_id"] = universe["gene"].map(symbol_to_entrez)
    universe["passed_reannotation"] = True
    universe["passed_background_50pct"] = True
    universe["probe_selection"] = "highest differential stability across donors"
    universe["gene_ds_threshold_applied"] = False
    universe["all_six_donors_nonzero_variance"] = True

    metadata = pd.DataFrame(metadata_rows).sort_values(["donor", "well_id"])
    expression = expression.loc[
        pd.MultiIndex.from_frame(metadata[["donor", "well_id"]])
    ]
    expression.to_csv(expression_file, compression="gzip")
    metadata.to_csv(metadata_file, index=False)
    universe.to_csv(universe_file, index=False)
    qc = {
        "raw_probes": int(n_raw_probes),
        "reannotated_probes_with_entrez": int(n_reannotated_probes),
        "probes_passing_50pct_background": int(n_background_pass_probes),
        "genes_after_diff_stability_probe_selection_and_variance_qc": int(
            expression.shape[1]
        ),
        "left_cortical_samples": int(expression.shape[0]),
        "donors": len(DONORS),
    }
    (OUTPUT / "gene_qc_summary.json").write_text(
        json.dumps(qc, indent=2), encoding="utf-8"
    )
    return expression, metadata, universe


def build_site_metadata(
    vectors: dict[str, np.ndarray],
    cortical_metadata: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:
    output = CACHE / "left_cortical_sites_to_subject_masks.csv"
    if output.exists() and not force:
        metadata = pd.read_csv(
            output, dtype={"donor": str, "subject": str}
        )
        metadata["subject"] = metadata["subject"].str.zfill(2)
        return metadata

    surface_mni = np.loadtxt(REGFUSION)
    tree = cKDTree(surface_mni)
    distance, vertex = tree.query(
        cortical_metadata[["mni_x", "mni_y", "mni_z"]].to_numpy()
    )
    base = cortical_metadata.copy()
    base["nearest_vertex"] = vertex.astype(int)
    base["surface_distance_mm"] = distance
    base = base[base.surface_distance_mm <= 3.0].copy()

    rows = []
    for subject in SUBJECTS:
        for analysis in ANALYSES:
            key = f"subj{subject}_{analysis}"
            mask = vectors[f"mask_{key}_lh"].astype(bool)
            selected = base[mask[base.nearest_vertex.to_numpy(int)]].copy()
            selected["subject"] = subject
            selected["analysis"] = analysis
            rows.append(selected)
    metadata = pd.concat(rows, ignore_index=True)
    metadata = metadata.sort_values(
        ["subject", "analysis", "donor", "well_id"]
    )
    metadata.to_csv(output, index=False)
    counts = (
        metadata.groupby(["subject", "analysis"])
        .agg(n_sites=("well_id", "size"), n_donors=("donor", "nunique"))
        .reset_index()
    )
    counts.to_csv(OUTPUT / "site_counts_by_subject_model.csv", index=False)
    return metadata


def surface_graph() -> coo_matrix:
    surface = nib.load(SURFACE)
    xyz = surface.darrays[0].data.astype(float)
    faces = surface.darrays[1].data.astype(int)
    edges = np.vstack(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    )
    edges = np.vstack([edges, edges[:, ::-1]])
    length = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
    return coo_matrix(
        (length, (edges[:, 0], edges[:, 1])),
        shape=(len(xyz), len(xyz)),
    ).tocsr()


def residual_basis(
    metadata: pd.DataFrame, hierarchy: np.ndarray, include_hierarchy: bool
) -> np.ndarray:
    donor = pd.get_dummies(
        metadata["donor"], drop_first=True, dtype=float
    ).to_numpy()
    columns = [np.ones((len(metadata), 1)), donor]
    if include_hierarchy:
        columns.append(np.asarray(hierarchy, float)[:, None])
    return np.linalg.qr(np.column_stack(columns), mode="reduced")[0]


def normalized_residual(values: np.ndarray, q: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    residual = values - q @ (q.T @ values)
    norm = np.sqrt(np.sum(residual * residual, axis=0, keepdims=True))
    return np.divide(
        residual,
        norm,
        out=np.zeros_like(residual),
        where=norm > 0,
    )


def fit_moran(
    vertices: np.ndarray,
    graph,
    subject: str,
    analysis: str,
) -> tuple[MoranRandomization, np.ndarray, np.ndarray]:
    unique, first, inverse = np.unique(
        np.asarray(vertices, int), return_index=True, return_inverse=True
    )
    mem_file = CACHE / f"moran_mem_subj{subject}_{analysis}.npy"
    model = MoranRandomization(
        procedure="singleton",
        spectrum="nonzero",
        joint=True,
        n_rep=1,
        tol=1e-6,
        random_state=SEED,
    )
    if mem_file.exists():
        mem = np.load(mem_file, mmap_mode="r")
        if mem.shape[0] == len(unique):
            model.mem_ = mem
            return model, first, inverse
        mem_file.unlink()

    distance = dijkstra(graph, directed=False, indices=unique)[:, unique]
    weights = np.zeros_like(distance)
    valid = np.isfinite(distance) & (distance > 0)
    weights[valid] = 1.0 / distance[valid]
    model.fit(weights)
    np.save(mem_file, np.asarray(model.mem_, dtype=np.float32))
    model.mem_ = np.load(mem_file, mmap_mode="r")
    return model, first, inverse


def extract_subject_data(
    vectors: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    expression: pd.DataFrame,
    subject: str,
    analysis: str,
):
    selected = metadata[
        metadata.subject.astype(str).str.zfill(2).eq(subject)
        & metadata.analysis.eq(analysis)
    ].copy()
    selected = selected.sort_values(["donor", "well_id"]).reset_index(drop=True)
    index = pd.MultiIndex.from_frame(selected[["donor", "well_id"]])
    genes = expression.loc[index].to_numpy(float)

    key = f"subj{subject}_{analysis}"
    vertices = vectors[f"vertex_{key}_lh"].astype(int)
    lookup = {int(vertex): i for i, vertex in enumerate(vertices)}
    positions = np.asarray(
        [lookup[int(vertex)] for vertex in selected.nearest_vertex], int
    )
    hierarchy = vectors[f"hierarchy_{key}_lh"][positions]
    functional = {
        map_name: vectors[f"func_{map_name}_{key}_lh"][positions]
        for map_name in VARIANCE_MAPS
    }
    return selected, genes, hierarchy, functional


def run_statistics(
    vectors: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    expression: pd.DataFrame,
    n_perm: int,
    perm_batch: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_perm % perm_batch:
        raise ValueError("n_perm must be divisible by perm_batch")
    genes = expression.columns.to_numpy()
    graph = surface_graph()

    prepared = {}
    for subject in SUBJECTS:
        for analysis in ANALYSES:
            meta, gene_values, hierarchy, functional = extract_subject_data(
                vectors, metadata, expression, subject, analysis
            )
            q_raw = residual_basis(meta, hierarchy, include_hierarchy=False)
            q_partial = residual_basis(meta, hierarchy, include_hierarchy=True)
            gene_raw = normalized_residual(gene_values, q_raw)
            gene_partial = normalized_residual(gene_values, q_partial)
            moran, unique_first, site_inverse = fit_moran(
                meta.nearest_vertex.to_numpy(int),
                graph,
                subject,
                analysis,
            )
            prepared[(subject, analysis)] = {
                "meta": meta,
                "hierarchy": hierarchy,
                "functional": functional,
                "q_raw": q_raw,
                "q_partial": q_partial,
                "gene_raw": gene_raw,
                "gene_partial": gene_partial,
                "moran": moran,
                "unique_first": unique_first,
                "site_inverse": site_inverse,
            }

    subject_rows = []
    group_rows = []
    for analysis_index, analysis in enumerate(ANALYSES):
        for map_index, map_name in enumerate(VARIANCE_MAPS):
            observed_raw = []
            observed_partial = []
            subject_exceedance = []
            for subject in SUBJECTS:
                context = prepared[(subject, analysis)]
                x = context["functional"][map_name][:, None]
                observed_raw.append(
                    (
                        normalized_residual(x, context["q_raw"]).T
                        @ context["gene_raw"]
                    ).ravel()
                )
                observed_partial.append(
                    (
                        normalized_residual(x, context["q_partial"]).T
                        @ context["gene_partial"]
                    ).ravel()
                )
                subject_exceedance.append(
                    np.zeros(len(genes), dtype=np.int64)
                )

            raw_r = np.stack(observed_raw)
            partial_r = np.stack(observed_partial)
            observed_z = np.arctanh(
                np.clip(partial_r, -0.999999, 0.999999)
            )
            observed_group_t = group_t(observed_z, axis=0)
            group_exceedance = np.zeros(len(genes), dtype=np.int64)

            n_batches = n_perm // perm_batch
            for batch in range(n_batches):
                null_subject_z = []
                for subject_index, subject in enumerate(SUBJECTS):
                    context = prepared[(subject, analysis)]
                    x = context["functional"][map_name]
                    model = context["moran"]
                    model.n_rep = perm_batch
                    model.random_state = (
                        SEED
                        + analysis_index * 10_000_000
                        + map_index * 1_000_000
                        + subject_index * 100_000
                        + batch
                    )
                    null = np.asarray(
                        model.randomize(x[context["unique_first"]])
                    )
                    if null.ndim == 3:
                        null = null[..., 0]
                    if null.shape == (
                        len(context["unique_first"]),
                        perm_batch,
                    ):
                        null = null.T
                    null = null[:, context["site_inverse"]].T
                    null_residual = normalized_residual(
                        null, context["q_partial"]
                    )
                    null_r = null_residual.T @ context["gene_partial"]
                    subject_exceedance[subject_index] += np.count_nonzero(
                        np.abs(null_r)
                        >= np.abs(partial_r[subject_index])[None, :],
                        axis=0,
                    )
                    null_subject_z.append(
                        np.arctanh(
                            np.clip(null_r, -0.999999, 0.999999)
                        )
                    )

                null_group_t = group_t(
                    np.stack(null_subject_z, axis=1), axis=1
                )
                group_exceedance += np.count_nonzero(
                    np.abs(null_group_t)
                    >= np.abs(observed_group_t)[None, :],
                    axis=0,
                )
                completed = (batch + 1) * perm_batch
                if completed % 1_000 == 0 or completed == n_perm:
                    print(
                        f"{analysis} | {map_name}: "
                        f"{completed:,}/{n_perm:,} spatial permutations",
                        flush=True,
                    )

            subject_p = [
                (1 + counts) / (n_perm + 1)
                for counts in subject_exceedance
            ]
            group_p = (1 + group_exceedance) / (n_perm + 1)
            fisher_mean = np.tanh(observed_z.mean(axis=0))
            ci_low, ci_high = exact_bootstrap_fisher_ci(partial_r)
            parametric_p = ttest_1samp(
                observed_z, 0.0, axis=0
            ).pvalue

            for gene_index, gene in enumerate(genes):
                group_rows.append(
                    {
                        "analysis": analysis,
                        "variance_map": map_name,
                        "gene": gene,
                        "n_subjects": len(SUBJECTS),
                        "fisher_mean_partial_r": float(
                            fisher_mean[gene_index]
                        ),
                        "bootstrap_ci_low": float(ci_low[gene_index]),
                        "bootstrap_ci_high": float(ci_high[gene_index]),
                        "group_t": float(observed_group_t[gene_index]),
                        "group_df": len(SUBJECTS) - 1,
                        "group_t_p": float(parametric_p[gene_index]),
                        "group_spatial_p": float(group_p[gene_index]),
                        "n_positive_subjects": int(
                            np.count_nonzero(
                                partial_r[:, gene_index] > 0
                            )
                        ),
                    }
                )
                for subject_index, subject in enumerate(SUBJECTS):
                    context = prepared[(subject, analysis)]
                    subject_rows.append(
                        {
                            "analysis": analysis,
                            "variance_map": map_name,
                            "gene": gene,
                            "subject": int(subject),
                            "n_sites": len(context["meta"]),
                            "n_donors": context["meta"].donor.nunique(),
                            "donor_adjusted_raw_r": float(
                                raw_r[subject_index, gene_index]
                            ),
                            "donor_hierarchy_partial_r": float(
                                partial_r[subject_index, gene_index]
                            ),
                            "subject_spatial_p": float(
                                subject_p[subject_index][gene_index]
                            ),
                        }
                    )

    groups = pd.DataFrame(group_rows)
    subjects = pd.DataFrame(subject_rows)
    groups["group_spatial_q_bh_genomewide"] = groups.groupby(
        ["analysis", "variance_map"], group_keys=False
    )["group_spatial_p"].transform(lambda p: fdr_bh(p.to_numpy()))
    groups["significant_q05"] = (
        groups.group_spatial_q_bh_genomewide < 0.05
    )
    subjects["subject_spatial_q_bh_genomewide"] = subjects.groupby(
        ["subject", "analysis", "variance_map"], group_keys=False
    )["subject_spatial_p"].transform(lambda p: fdr_bh(p.to_numpy()))
    subjects["significant_q05"] = (
        subjects.subject_spatial_q_bh_genomewide < 0.05
    )
    groups = groups.sort_values(
        [
            "analysis",
            "variance_map",
            "group_spatial_q_bh_genomewide",
            "group_spatial_p",
            "gene",
        ]
    )
    subjects = subjects.sort_values(
        [
            "subject",
            "analysis",
            "variance_map",
            "subject_spatial_q_bh_genomewide",
            "subject_spatial_p",
            "gene",
        ]
    )
    groups.to_csv(
        OUTPUT / "gene_group_results_100k.csv.gz",
        index=False,
        compression="gzip",
    )
    subjects.to_csv(
        OUTPUT / "gene_subject_results_100k.csv.gz",
        index=False,
        compression="gzip",
    )
    top = groups.groupby(
        ["analysis", "variance_map"], group_keys=False
    ).head(100)
    top.to_csv(OUTPUT / "top_100_genes_per_panel.csv", index=False)
    return groups, subjects


def write_settings(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    n_perm: int,
    perm_batch: int,
) -> None:
    settings = {
        "status": "prepared",
        "hemisphere": "left",
        "surface_distance_mm": 3.0,
        "fmri_subjects": [int(subject) for subject in SUBJECTS],
        "encoding_models": list(ANALYSES),
        "variance_maps": list(VARIANCE_MAPS),
        "ahba_donors": list(DONORS),
        "n_valid_genes": int(expression.shape[1]),
        "gene_universe": "all genes passing target-independent technical QC",
        "probe_reannotation": True,
        "intensity_filter": "probe exceeds AHBA background in >=50% of all left cortical samples",
        "probe_selection": "highest differential stability across donors",
        "gene_level_differential_stability_cutoff": None,
        "sample_normalization": "scaled robust sigmoid within donor across genes",
        "gene_normalization": "scaled robust sigmoid within donor across left cortical samples",
        "covariates": "AHBA donor fixed effects + Margulies PG1",
        "spatial_null": "Moran spectral surrogates at unique AHBA-linked fsaverage10k vertices using surface-geodesic inverse-distance weights",
        "group_statistic": "one-sample t across four subject Fisher-z partial correlations",
        "n_spatial_permutations": n_perm,
        "permutation_batch": perm_batch,
        "p_value": "two-sided add-one empirical spatial p",
        "fdr": "BH separately across all valid genes for each encoding model and variance map",
        "n_site_rows_across_subject_models": int(len(metadata)),
    }
    (OUTPUT / "analysis_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=100000)
    parser.add_argument("--perm-batch", type=int, default=250)
    parser.add_argument("--force-vectors", action="store_true")
    parser.add_argument("--force-expression", action="store_true")
    parser.add_argument("--force-sites", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    vectors = build_gene_vectors(force=args.force_vectors)
    expression, cortical_metadata, _ = prepare_cortical_expression(
        force=args.force_expression
    )
    metadata = build_site_metadata(
        vectors, cortical_metadata, force=args.force_sites
    )
    write_settings(expression, metadata, args.n_perm, args.perm_batch)
    print(f"Valid genome-wide genes: {expression.shape[1]:,}")
    print(f"All left cortical AHBA samples: {expression.shape[0]:,}")
    print(
        metadata.groupby(["subject", "analysis"])
        .agg(n_sites=("well_id", "size"), n_donors=("donor", "nunique"))
        .to_string()
    )
    if args.prepare_only:
        return

    groups, subjects = run_statistics(
        vectors,
        metadata,
        expression,
        n_perm=args.n_perm,
        perm_batch=args.perm_batch,
    )
    settings_file = OUTPUT / "analysis_settings.json"
    settings = json.loads(settings_file.read_text(encoding="utf-8"))
    settings["status"] = "complete"
    settings["n_group_tests"] = int(len(groups))
    settings["n_subject_tests"] = int(len(subjects))
    settings["n_group_significant_q05"] = int(
        groups.significant_q05.sum()
    )
    settings["n_unique_group_significant_genes_q05"] = int(
        groups.loc[groups.significant_q05, "gene"].nunique()
    )
    settings_file.write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    print(
        f"Group tests: {len(groups):,}; "
        f"q<.05: {int(groups.significant_q05.sum()):,}; "
        f"unique significant genes: "
        f"{groups.loc[groups.significant_q05, 'gene'].nunique():,}"
    )


if __name__ == "__main__":
    main()
