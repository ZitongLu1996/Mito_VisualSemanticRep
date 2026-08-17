"""Freeze the image split used by the final four-subject NSD analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DERIVATIVES = ROOT / "derivatives" / "design"
SUBJECTS = (1, 2, 5, 7)


def make_trial_table(subject: int, design: dict[str, np.ndarray], info: pd.DataFrame) -> pd.DataFrame:
    subject_images_1based = design["subjectim"][subject - 1].astype(np.int64)
    order_0based = design["masterordering"].ravel().astype(np.int64) - 1
    nsd_ids = subject_images_1based[order_0based] - 1
    if nsd_ids.size != 30000:
        raise ValueError(f"Unexpected trial count for subject {subject}: {nsd_ids.size}")
    counts = pd.Series(nsd_ids).value_counts()
    if len(counts) != 10000 or not np.all(counts.to_numpy() == 3):
        raise ValueError(f"Subject {subject} does not have exactly three trials per image")
    lookup = info.set_index("nsdId")
    table = pd.DataFrame(
        {
            "subject": f"subj{subject:02d}",
            "trial_index": np.arange(nsd_ids.size),
            "session": np.arange(nsd_ids.size) // 750 + 1,
            "within_session_trial": np.arange(nsd_ids.size) % 750,
            "nsd_id": nsd_ids,
        }
    )
    table["shared1000"] = lookup.loc[nsd_ids, "shared1000"].to_numpy(dtype=bool)
    table["flagged"] = lookup.loc[nsd_ids, "flagged"].to_numpy(dtype=bool)
    return table


def main() -> None:
    DERIVATIVES.mkdir(parents=True, exist_ok=True)
    info = pd.read_csv(DATA / "nsddata/experiments/nsd/nsd_stim_info_merged.csv")
    design = loadmat(DATA / "nsddata/experiments/nsd/nsd_expdesign.mat")
    shared_ids = np.sort(info.loc[info["shared1000"], "nsdId"].to_numpy(dtype=np.int64))
    if shared_ids.size != 1000 or info.loc[info["shared1000"], "flagged"].any():
        raise ValueError("The official shared1000 test set failed validation")

    trial_tables = []
    image_rows = []
    for subject in SUBJECTS:
        trials = make_trial_table(subject, design, info)
        trials.to_csv(DERIVATIVES / f"subj{subject:02d}_trials.csv.gz", index=False)
        trial_tables.append(trials)
        unique = trials.drop_duplicates("nsd_id")
        train = unique.loc[~unique["shared1000"] & ~unique["flagged"], "nsd_id"].to_numpy()
        test = unique.loc[unique["shared1000"], "nsd_id"].to_numpy()
        if not np.array_equal(np.sort(test), shared_ids):
            raise ValueError(f"Subject {subject} test set is not shared1000")
        np.savez_compressed(
            DERIVATIVES / f"subj{subject:02d}_image_split.npz",
            train_nsd_id=np.sort(train),
            test_nsd_id=np.sort(test),
        )
        image_rows.append(
            {
                "subject": f"subj{subject:02d}",
                "n_trials": len(trials),
                "n_train_images": len(train),
                "n_test_images": len(test),
                "n_flagged_train_images_excluded": int(unique.loc[~unique["shared1000"], "flagged"].sum()),
            }
        )
    pd.DataFrame(image_rows).to_csv(DERIVATIVES / "image_split_summary.csv", index=False)

    settings = {
        "subjects": [f"subj{s:02d}" for s in SUBJECTS],
        "beta_version": "betas_fithrf_GLMdenoise_RR",
        "surface_space": "fsaverage (163842 vertices per hemisphere)",
        "test_definition": "official NSD shared1000; all three trials averaged per image",
        "training_definition": "subject-specific non-shared images; flagged images excluded; all three trials averaged per image",
        "candidate_cortex": "fsaverage non-medial-wall cortex; no noise-ceiling threshold",
        "final_mask": "subject- and model-specific held-out encoding-significance mask",
    }
    (DERIVATIVES / "analysis_design.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
