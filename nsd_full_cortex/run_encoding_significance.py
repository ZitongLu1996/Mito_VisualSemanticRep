"""Run both variance analyses on the full cortical candidate set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vsvariance.analysis import run_hemisphere_analysis
from run_encoding import ALPHAS, RANDOM_STATE, SPECS


ROOT = Path(__file__).resolve().parent
RESPONSES = ROOT / "derivatives" / "encoding_significance" / "responses"
PCA = ROOT / "derivatives" / "encoding" / "pca_cache"
OUTPUT = ROOT / "derivatives" / "encoding_significance" / "results"
SUBJECTS = ("subj01", "subj02", "subj05", "subj07")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--analysis", choices=tuple(SPECS))
    parser.add_argument("--hemisphere", choices=("lh", "rh"))
    args = parser.parse_args()
    for subject in ((args.subject,) if args.subject else SUBJECTS):
        for analysis in ((args.analysis,) if args.analysis else tuple(SPECS)):
            for hemi in ((args.hemisphere,) if args.hemisphere else ("lh", "rh")):
                out = OUTPUT / subject / analysis / hemi
                if (out / "metadata.json").exists() and (out / "r_joint.npy").exists():
                    print(f"FULL_CORTEX_ENCODING_SKIP {subject} {analysis} {hemi}", flush=True)
                    continue
                print(f"FULL_CORTEX_ENCODING_START {subject} {analysis} {hemi}", flush=True)
                run_hemisphere_analysis(
                    RESPONSES / subject / hemi / "train_responses.npy",
                    RESPONSES / subject / hemi / "test_responses.npy",
                    PCA / subject / analysis,
                    SPECS[analysis],
                    out,
                    alphas=ALPHAS,
                    inner_splits=4,
                    vertex_chunk_size=256,
                    random_state=RANDOM_STATE,
                    save_joint_test_predictions=True,
                )
                print(f"FULL_CORTEX_ENCODING_DONE {subject} {analysis} {hemi}", flush=True)


if __name__ == "__main__":
    main()
