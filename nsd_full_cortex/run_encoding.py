"""Materialize subject features and prepare nested PCA caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vsvariance.analysis import AnalysisSpec, prepare_nested_pca_cache


ROOT = Path(__file__).resolve().parent
DESIGN = ROOT / "derivatives" / "design"
FEATURES = ROOT / "derivatives" / "features"
OUTPUT = ROOT / "derivatives" / "encoding"
SUBJECTS = ("subj01", "subj02", "subj05", "subj07")
SPECS = {
    "dinov2_minilm": AnalysisSpec(
        "dinov2_minilm", ("block03", "block06", "block09", "block12"), "minilm"
    ),
    "cornet_s_mpnet": AnalysisSpec("cornet_s_mpnet", ("V1", "V2", "V4", "IT"), "mpnet"),
}
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
RANDOM_STATE = 20260718


def source_path(group: str) -> Path:
    if group.startswith("block"):
        return FEATURES / "dinov2" / f"{group}.npy"
    if group in ("V1", "V2", "V4", "IT"):
        return FEATURES / "cornet_s" / f"{group}.npy"
    if group in ("mpnet", "minilm"):
        return FEATURES / f"{group}.npy"
    raise KeyError(group)


def subset_features(subject: str) -> None:
    marker = FEATURES / "subjects" / subject / "complete"
    required_groups = (
        "block03", "block06", "block09", "block12", "V1", "V2", "V4", "IT", "mpnet", "minilm"
    )
    required_paths = [
        FEATURES / "subjects" / subject / split_name / f"{group}.npy"
        for split_name in ("train", "test")
        for group in required_groups
    ]
    if marker.exists() and all(path.exists() for path in required_paths):
        print(f"FEATURE_SUBSET_SKIP {subject}", flush=True)
        return
    union_ids = np.load(FEATURES / "image_ids.npy")
    split = np.load(DESIGN / f"{subject}_image_split.npz")
    for split_name, key in (("train", "train_nsd_id"), ("test", "test_nsd_id")):
        ids = split[key]
        rows = np.searchsorted(union_ids, ids)
        if np.any(rows == len(union_ids)) or not np.array_equal(union_ids[rows], ids):
            raise ValueError(f"Feature union does not contain every {subject} {split_name} image")
        destination_dir = FEATURES / "subjects" / subject / split_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for group in required_groups:
            destination_path = destination_dir / f"{group}.npy"
            if destination_path.exists():
                print(f"FEATURE_SUBSET_REUSE {subject} {split_name} {group}", flush=True)
                continue
            source = np.load(source_path(group), mmap_mode="r")
            destination = np.lib.format.open_memmap(
                destination_path, mode="w+", dtype=source.dtype, shape=(len(rows), source.shape[1])
            )
            for start in range(0, len(rows), 256):
                stop = min(len(rows), start + 256)
                destination[start:stop] = source[rows[start:stop]]
            destination.flush()
            destination._mmap.close()
            print(f"FEATURE_SUBSET {subject} {split_name} {group}", flush=True)
    marker.write_text("complete\n", encoding="utf-8")


def prepare_pca(subject: str, analysis: str) -> None:
    spec = SPECS[analysis]
    groups = (*spec.visual_groups, spec.semantic_group)
    raw_train = {group: FEATURES / "subjects" / subject / "train" / f"{group}.npy" for group in groups}
    raw_test = {group: FEATURES / "subjects" / subject / "test" / f"{group}.npy" for group in groups}
    pca_cache = OUTPUT / "pca_cache" / subject / analysis
    prepare_nested_pca_cache(
        raw_train,
        raw_test,
        groups,
        pca_cache,
        n_components=512,
        outer_splits=5,
        random_state=RANDOM_STATE,
    )
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("subset", "pca"), required=True)
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--analysis", choices=tuple(SPECS))
    args = parser.parse_args()
    subjects = (args.subject,) if args.subject else SUBJECTS
    if args.stage == "subset":
        for subject in subjects:
            subset_features(subject)
        return
    analyses = (args.analysis,) if args.analysis else tuple(SPECS)
    for subject in subjects:
        for analysis in analyses:
            prepare_pca(subject, analysis)


if __name__ == "__main__":
    main()
