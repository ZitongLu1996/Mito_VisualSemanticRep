"""Create subject- and model-specific held-out encoding significance masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "derivatives" / "encoding_significance"
SUBJECTS = ("subj01", "subj02", "subj05", "subj07")
ANALYSES = ("dinov2_minilm", "cornet_s_mpnet")
HEMIS = ("lh", "rh")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--analysis", choices=ANALYSES)
    args = parser.parse_args()
    selected_subjects = (args.subject,) if args.subject else SUBJECTS
    selected_analyses = (args.analysis,) if args.analysis else ANALYSES
    output = BASE / "masks"
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for subject in selected_subjects:
        for analysis in selected_analyses:
            p_values = []
            observed_values = []
            lengths = []
            for hemi in HEMIS:
                result = BASE / "results" / subject / analysis / hemi
                p_hemi = np.load(result / "joint_performance_bootstrap_p.npy")
                observed_hemi = np.load(result / "joint_performance_observed_r2.npy")
                p_values.append(np.asarray(p_hemi, dtype=np.float64))
                observed_values.append(np.asarray(observed_hemi, dtype=np.float64))
                lengths.append(len(p_hemi))
            p = np.concatenate(p_values)
            observed = np.concatenate(observed_values)
            reject, q, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
            reject &= np.isfinite(observed) & (observed > 0)
            start = 0
            payload = {}
            for hemi, length in zip(HEMIS, lengths):
                stop = start + length
                vertex_index = np.load(BASE / "responses" / subject / hemi / "vertex_indices.npy")
                full_mask = np.zeros(163842, dtype=bool)
                full_q = np.full(163842, np.nan, dtype=np.float32)
                full_performance = np.full(163842, np.nan, dtype=np.float32)
                full_mask[vertex_index] = reject[start:stop]
                full_q[vertex_index] = q[start:stop]
                full_performance[vertex_index] = observed[start:stop]
                payload[f"mask_{hemi}"] = full_mask
                payload[f"q_{hemi}"] = full_q
                payload[f"joint_r2_{hemi}"] = full_performance
                summary.append({
                    "subject": subject, "analysis": analysis, "hemisphere": hemi,
                    "n_candidate": int(length), "n_significant": int(reject[start:stop].sum()),
                    "fraction_significant": float(reject[start:stop].mean()),
                })
                start = stop
            np.savez_compressed(output / f"{subject}_{analysis}_encoding_mask.npz", **payload)
    summary_frame = pd.DataFrame(summary)
    summary_path = output / "mask_summary.csv"
    if (args.subject or args.analysis) and summary_path.exists():
        previous = pd.read_csv(summary_path)
        replace_pairs = set(zip(summary_frame["subject"], summary_frame["analysis"]))
        keep = [
            (subject, analysis) not in replace_pairs
            for subject, analysis in zip(previous["subject"], previous["analysis"])
        ]
        summary_frame = pd.concat([previous.loc[keep], summary_frame], ignore_index=True)
    summary_frame.to_csv(summary_path, index=False)
    (output / "metadata.json").write_text(json.dumps({
        "performance": "joint-model predictive R2 on the shared-1000 held-out test set",
        "test": "10,000 paired test-image bootstrap samples; one-sided p from bootstrap R2 <= 0",
        "multiple_comparisons": "Benjamini-Hochberg FDR q < .05 across both hemispheres, separately per subject and visual-model analysis",
        "candidate_cortex": "fsaverage164k non-medial-wall vertices; no noise-ceiling threshold",
        "mask_count": int(summary_frame[["subject", "analysis"]].drop_duplicates().shape[0]),
    }, indent=2), encoding="utf-8")
    print("SUBJECT_ENCODING_MASKS_DONE", flush=True)


if __name__ == "__main__":
    main()
