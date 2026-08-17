"""Paired stimulus bootstrap of held-out joint-model predictive R2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "derivatives" / "encoding_significance"
SUBJECTS = ("subj01", "subj02", "subj05", "subj07")
ANALYSES = ("dinov2_minilm", "cornet_s_mpnet")
HEMIS = ("lh", "rh")
N_BOOT = 10_000
SEED = 20260719
CHUNK = 512


def bootstrap(subject: str, analysis: str, hemi: str) -> None:
    result_dir = BASE / "results" / subject / analysis / hemi
    output = result_dir / "joint_performance_bootstrap_p.npy"
    if output.exists():
        print(f"BOOTSTRAP_SKIP {subject} {analysis} {hemi}", flush=True)
        return
    y = np.load(BASE / "responses" / subject / hemi / "test_responses.npy", mmap_mode="r")
    prediction = np.load(result_dir / "test_prediction_joint.npy", mmap_mode="r")
    if y.shape != prediction.shape or y.shape[0] != 1000:
        raise ValueError(f"Unexpected bootstrap input shapes: {y.shape}, {prediction.shape}")
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(y.shape[0], np.full(y.shape[0], 1 / y.shape[0]), size=N_BOOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count_tensor = torch.as_tensor(counts, dtype=torch.float32, device=device)
    del counts
    p = np.empty(y.shape[1], dtype=np.float32)
    observed = np.empty(y.shape[1], dtype=np.float32)
    for start in range(0, y.shape[1], CHUNK):
        stop = min(y.shape[1], start + CHUNK)
        target = np.asarray(y[:, start:stop], dtype=np.float32)
        pred = np.asarray(prediction[:, start:stop], dtype=np.float32)
        residual_sq = (target - pred) ** 2
        stats = np.concatenate((residual_sq, target, target * target), axis=1)
        weighted = count_tensor @ torch.as_tensor(stats, device=device)
        width = stop - start
        sse = weighted[:, :width]
        sum_y = weighted[:, width:2 * width]
        sum_y2 = weighted[:, 2 * width:]
        sst = sum_y2 - sum_y.square() / y.shape[0]
        boot_r2 = 1.0 - sse / torch.clamp(sst, min=1e-20)
        nonpositive = torch.count_nonzero(boot_r2 <= 0, dim=0).cpu().numpy()
        p[start:stop] = (nonpositive + 1.0) / (N_BOOT + 1.0)
        obs_sse = residual_sq.sum(axis=0, dtype=np.float64)
        obs_sst = ((target - target.mean(axis=0, keepdims=True)) ** 2).sum(axis=0, dtype=np.float64)
        observed[start:stop] = 1.0 - obs_sse / obs_sst
        if start == 0 or stop == y.shape[1] or start % (CHUNK * 32) == 0:
            print(f"BOOTSTRAP_PROGRESS {subject} {analysis} {hemi} vertices={stop}/{y.shape[1]}", flush=True)
        del weighted, sse, sum_y, sum_y2, sst, boot_r2
    np.save(output, p)
    np.save(result_dir / "joint_performance_observed_r2.npy", observed)
    (result_dir / "bootstrap_metadata.json").write_text(json.dumps({
        "n_bootstrap": N_BOOT,
        "resampling": "paired sampling of shared-1000 test images with replacement",
        "statistic": "predictive R2 = 1 - SSE/SST",
        "p_value": "one-sided (1 + number of bootstrap R2 <= 0) / (10000 + 1)",
        "seed": SEED,
        "device": str(device),
    }, indent=2), encoding="utf-8")
    print(f"BOOTSTRAP_DONE {subject} {analysis} {hemi}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--analysis", choices=ANALYSES)
    parser.add_argument("--hemisphere", choices=HEMIS)
    args = parser.parse_args()
    for subject in ((args.subject,) if args.subject else SUBJECTS):
        for analysis in ((args.analysis,) if args.analysis else ANALYSES):
            for hemi in ((args.hemisphere,) if args.hemisphere else HEMIS):
                bootstrap(subject, analysis, hemi)


if __name__ == "__main__":
    main()
