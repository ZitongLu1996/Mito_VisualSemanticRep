"""Average three NSD trials on the full fsaverage cortex (medial wall excluded)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DESIGN = ROOT / "derivatives" / "design"
OUTPUT = ROOT / "derivatives" / "encoding_significance" / "responses"
ATLAS = ROOT / "neuromaps_data" / "atlases" / "fsaverage"
BETA_KIND = "betas_fithrf_GLMdenoise_RR"
SUBJECTS = ("subj01", "subj02", "subj05", "subj07")
HEMISPHERES = ("lh", "rh")
SCALE = 300.0


def load_session(path: Path, mask: np.ndarray) -> np.ndarray:
    image = nib.load(path)
    values = np.asanyarray(image.dataobj).squeeze()
    if values.shape == (mask.size, 750):
        values = values[mask].T
    elif values.shape == (750, mask.size):
        values = values[:, mask]
    else:
        raise ValueError(f"Unexpected beta shape {values.shape} in {path}")
    return np.asarray(values, dtype=np.float32) / SCALE


def grouped_add(sums: np.memmap, rows: np.ndarray, values: np.ndarray) -> None:
    """Accumulate repeated-image responses without element-wise indexing."""
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    sorted_values = values[order]
    unique_rows, starts = np.unique(sorted_rows, return_index=True)
    sums[unique_rows] += np.add.reduceat(sorted_values, starts, axis=0)


def cortex_mask(hemi: str) -> np.ndarray:
    side = "L" if hemi == "lh" else "R"
    path = ATLAS / f"tpl-fsaverage_den-164k_hemi-{side}_desc-nomedialwall_dparc.label.gii"
    values = np.asarray(nib.load(path).darrays[0].data).squeeze()
    mask = values != 0
    if mask.shape != (163842,):
        raise ValueError(f"Unexpected fsaverage mask shape: {mask.shape}")
    return mask


def prepare(subject: str, hemi: str) -> None:
    output = OUTPUT / subject / hemi
    marker = output / "complete.json"
    if marker.exists():
        print(f"FULL_CORTEX_BETA_SKIP {subject} {hemi}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    split = np.load(DESIGN / f"{subject}_image_split.npz")
    train_ids = split["train_nsd_id"].astype(np.int64)
    test_ids = split["test_nsd_id"].astype(np.int64)
    image_ids = np.concatenate([train_ids, test_ids])
    lookup = np.full(73000, -1, dtype=np.int32)
    lookup[image_ids] = np.arange(image_ids.size, dtype=np.int32)
    trials = pd.read_csv(DESIGN / f"{subject}_trials.csv.gz")
    mask = cortex_mask(hemi)
    n_vertices = int(mask.sum())
    sums_path = output / "response_sums.temporary.npy"
    sums = np.lib.format.open_memmap(
        sums_path, mode="w+", dtype=np.float32, shape=(len(image_ids), n_vertices)
    )
    sums[:] = 0
    counts = np.zeros(len(image_ids), dtype=np.int16)
    beta_dir = DATA / f"nsddata_betas/ppdata/{subject}/fsaverage/{BETA_KIND}"
    for session in range(1, 41):
        rows_df = trials.loc[trials["session"].eq(session)].sort_values("within_session_trial")
        rows = lookup[rows_df["nsd_id"].to_numpy(dtype=np.int64)]
        keep = rows >= 0
        values = load_session(beta_dir / f"{hemi}.betas_session{session:02d}.mgh", mask)
        grouped_add(sums, rows[keep], values[keep])
        np.add.at(counts, rows[keep], 1)
        sums.flush()
        print(f"FULL_CORTEX_BETA {subject} {hemi} session={session:02d}/40", flush=True)
    if not np.all(counts == 3):
        raise ValueError(f"Not all retained images have three trials: {np.unique(counts, return_counts=True)}")
    sums /= counts[:, None]
    sums.flush()
    n_train = len(train_ids)
    for name, start, stop in (("train", 0, n_train), ("test", n_train, len(image_ids))):
        dest = np.lib.format.open_memmap(
            output / f"{name}_responses.npy", "w+", np.float32, (stop - start, n_vertices)
        )
        for a in range(start, stop, 128):
            b = min(stop, a + 128)
            dest[a - start:b - start] = sums[a:b]
        dest.flush()
        dest._mmap.close()
    sums._mmap.close()
    sums_path.unlink()
    np.save(output / "vertex_indices.npy", np.flatnonzero(mask))
    metadata = {
        "subject": subject,
        "hemisphere": hemi,
        "candidate_mask": "fsaverage164k non-medial-wall cortex; no noise-ceiling threshold",
        "source_scale_divisor": SCALE,
        "n_vertices": n_vertices,
        "n_train_images": len(train_ids),
        "n_test_images": len(test_ids),
        "repetitions_averaged": 3,
    }
    marker.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--hemisphere", choices=HEMISPHERES)
    args = parser.parse_args()
    for subject in ((args.subject,) if args.subject else SUBJECTS):
        for hemi in ((args.hemisphere,) if args.hemisphere else HEMISPHERES):
            prepare(subject, hemi)


if __name__ == "__main__":
    main()
