"""Resumable downloader for the preregistered four-subject NSD analysis.

Only official public NSD S3 objects are used. Existing complete files are skipped;
partial files are resumed with HTTP Range requests and atomically renamed on finish.
"""

from __future__ import annotations

import argparse
import re
import time
import urllib.request
from pathlib import Path


BUCKET = "https://natural-scenes-dataset.s3.us-east-2.amazonaws.com"
ROOT = Path(__file__).resolve().parent / "data"
SUBJECTS = ("subj01", "subj02", "subj05", "subj07")
BETA_KIND = "betas_fithrf_GLMdenoise_RR"

CORE_OBJECTS = (
    "nsddata/experiments/nsd/nsd_expdesign.mat",
    "nsddata/experiments/nsd/nsd_stim_info_merged.csv",
)


def beta_objects(subject: str) -> list[str]:
    prefix = f"nsddata_betas/ppdata/{subject}/fsaverage/{BETA_KIND}"
    return [
        f"{prefix}/{hemi}.betas_session{session:02d}.mgh"
        for hemi in ("lh", "rh")
        for session in range(1, 41)
    ]


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download_object(key: str, retries: int = 8) -> Path:
    target = ROOT / key
    partial = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET}/{key}"
    expected = remote_size(url)
    if target.exists() and target.stat().st_size == expected:
        print(f"SKIP {key}", flush=True)
        return target
    if target.exists():
        target.replace(partial)
    if partial.exists() and partial.stat().st_size == expected:
        partial.replace(target)
        print(f"DONE {key}", flush=True)
        return target

    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("ab" if offset else "wb") as stream:
                disposition = response.headers.get("Content-Range", "")
                if offset and not re.match(rf"bytes {offset}-", disposition):
                    raise RuntimeError(f"Server did not honor range request: {disposition!r}")
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
            if partial.stat().st_size != expected:
                raise RuntimeError(f"Size mismatch: {partial.stat().st_size} != {expected}")
            partial.replace(target)
            print(f"DONE {key}", flush=True)
            return target
        except Exception as error:
            if attempt + 1 == retries:
                raise
            delay = min(60, 2 ** attempt)
            print(f"RETRY {key}: {error}; waiting {delay}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("metadata", "betas", "stimuli", "all"), required=True)
    args = parser.parse_args()
    objects: list[str] = []
    if args.stage in ("metadata", "all"):
        objects.extend(CORE_OBJECTS)
    if args.stage in ("betas", "all"):
        for subject in SUBJECTS:
            objects.extend(beta_objects(subject))
    if args.stage in ("stimuli", "all"):
        objects.append("nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")
    for key in objects:
        download_object(key)


if __name__ == "__main__":
    main()
