"""Numerical integrity QC for subject/model-specific masked NSD variance maps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "derivatives" / "encoding_significance"
RESULTS = BASE / "results"
MASKS = BASE / "masks"
RESPONSES = BASE / "responses"
OUTPUT = ROOT / "derivatives" / "variance_qc"
SUBJECTS = ("subj01", "subj02", "subj05", "subj07")
MODELS = ("dinov2_minilm", "cornet_s_mpnet")
HEMISPHERES = ("lh", "rh")
MAPS = ("r2_visual", "r2_semantic", "r2_joint", "unique_visual", "unique_semantic", "shared")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    identity_rows = []
    for model in MODELS:
        for subject in SUBJECTS:
            combined = {}
            mask_file = np.load(MASKS / f"{subject}_{model}_encoding_mask.npz")
            for name in MAPS:
                parts = []
                for hemi, expected in (("lh", 149955), ("rh", 149926)):
                    path = RESULTS / subject / model / hemi / f"{name}.npy"
                    data = np.load(path)
                    if data.shape != (expected,):
                        raise ValueError(f"Invalid map: {path} shape={data.shape}")
                    vertex_index = np.load(RESPONSES / subject / hemi / "vertex_indices.npy")
                    selected = np.asarray(mask_file[f"mask_{hemi}"][vertex_index], dtype=bool)
                    masked = np.asarray(data[selected])
                    if not np.isfinite(masked).all():
                        raise ValueError(f"Non-finite masked values: {path}")
                    parts.append(masked)
                combined[name] = np.concatenate(parts)
                data = combined[name]
                summary_rows.append(
                    {
                        "model": model,
                        "subject": subject,
                        "map": name,
                        "mean": float(data.mean()),
                        "median": float(np.median(data)),
                        "positive_fraction": float((data > 0).mean()),
                        "minimum": float(data.min()),
                        "maximum": float(data.max()),
                    }
                )
            errors = {
                "unique_visual": np.max(np.abs(combined["unique_visual"] - (combined["r2_joint"] - combined["r2_semantic"]))),
                "unique_semantic": np.max(np.abs(combined["unique_semantic"] - (combined["r2_joint"] - combined["r2_visual"]))),
                "shared": np.max(np.abs(combined["shared"] - (combined["r2_visual"] + combined["r2_semantic"] - combined["r2_joint"]))),
            }
            for name, error in errors.items():
                identity_rows.append({"model": model, "subject": subject, "identity": name, "max_abs_error": float(error)})
                if error > 1e-6:
                    raise ValueError(f"Variance identity failed: {model} {subject} {name} {error}")

    pd.DataFrame(summary_rows).to_csv(OUTPUT / "map_distribution_qc.csv", index=False)
    pd.DataFrame(identity_rows).to_csv(OUTPUT / "variance_identity_qc.csv", index=False)
    payload = {
        "status": "passed",
        "n_subject_model_results": 8,
        "n_primary_variance_maps": 24,
        "all_masked_values_finite": True,
        "all_shapes_expected": True,
        "variance_partition_identities_passed": True,
        "cross_subject_map_comparisons": "not performed because masks are subject/model-specific",
    }
    (OUTPUT / "complete.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
