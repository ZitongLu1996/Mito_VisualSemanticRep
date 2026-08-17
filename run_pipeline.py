"""Command-line entry point for the manuscript analysis workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "nsd_full_cortex"


def run(*arguments: str, cwd: Path = ANALYSIS) -> None:
    command = [sys.executable, *arguments]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def data_stage() -> None:
    run("download_nsd.py", "--stage", "all")
    run("prepare_design.py")
    run("prepare_encoding_significance_betas.py")


def feature_stage(device: str) -> None:
    for model in ("dinov2", "cornet_s", "minilm", "mpnet"):
        run("extract_features.py", "--model", model, "--device", device)
    run("run_encoding.py", "--stage", "subset")


def encoding_stage() -> None:
    run("run_encoding.py", "--stage", "pca")
    run("run_encoding_significance.py")
    run("bootstrap_joint_encoding.py")
    run("make_subject_encoding_masks.py")
    run("qc_variance_maps.py")


def mitochondrial_stage() -> None:
    run(
        "run_mito_variance_analysis.py",
        "--prepare-surfaces",
        cwd=ROOT / "mitochondrial_analysis",
    )
    run(
        "run_subject_masked_mito_analysis.py",
        "--n-perm",
        "100000",
        "--batch-size",
        "5000",
        "--force-align",
    )


def transcriptomic_stage() -> None:
    run(
        "run_ahba_genomewide_gene_analysis.py",
        "--n-perm",
        "100000",
        "--perm-batch",
        "250",
        "--force-vectors",
        "--force-sites",
    )


def enrichment_stage() -> None:
    run("run_ahba_mannwhitney_enrichment.py")
    run("reduce_go_bp_redundancy.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "data",
            "features",
            "encoding",
            "mitochondrial",
            "transcriptomic",
            "enrichment",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    stages = {
        "data": data_stage,
        "features": lambda: feature_stage(args.device),
        "encoding": encoding_stage,
        "mitochondrial": mitochondrial_stage,
        "transcriptomic": transcriptomic_stage,
        "enrichment": enrichment_stage,
    }
    if args.stage == "all":
        for stage in stages.values():
            stage()
    else:
        stages[args.stage]()


if __name__ == "__main__":
    main()

