from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


IMAGE_RE = re.compile(r"^(?:train|test)-(\d+)_nsd-(\d+)\.png$")


@dataclass(frozen=True)
class SubjectPaths:
    subject: str
    train_images: tuple[Path, ...]
    test_images: tuple[Path, ...]
    train_lh: Path
    train_rh: Path
    test_lh: Path
    test_rh: Path


def _ordered_images(directory: Path, split: str) -> tuple[Path, ...]:
    found: list[tuple[int, Path]] = []
    for path in directory.glob(f"{split}-*_nsd-*.png"):
        match = IMAGE_RE.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    if not found:
        raise FileNotFoundError(f"No {split} images found in {directory}")
    expected = list(range(1, len(found) + 1))
    observed = [index for index, _ in found]
    if observed != expected:
        raise ValueError(f"Non-contiguous {split} image indices in {directory}")
    return tuple(path for _, path in found)


def subject_paths(data_root: Path, subject: str) -> SubjectPaths:
    train_base = data_root / "train_data" / subject / "training_split"
    test_base = data_root / "test_data" / subject / "test_split"
    result = SubjectPaths(
        subject=subject,
        train_images=_ordered_images(train_base / "training_images", "train"),
        test_images=_ordered_images(test_base / "test_images", "test"),
        train_lh=train_base / "training_fmri" / "lh_training_fmri.npy",
        train_rh=train_base / "training_fmri" / "rh_training_fmri.npy",
        test_lh=test_base / "test_fmri" / "lh_test_fmri.npy",
        test_rh=test_base / "test_fmri" / "rh_test_fmri.npy",
    )
    for fmri_path in (result.train_lh, result.train_rh, result.test_lh, result.test_rh):
        if not fmri_path.exists():
            raise FileNotFoundError(f"Missing fMRI file: {fmri_path}")
    return result


def validate_subject_shapes(paths: SubjectPaths) -> None:
    pairs = (
        (paths.train_lh, len(paths.train_images)),
        (paths.train_rh, len(paths.train_images)),
        (paths.test_lh, len(paths.test_images)),
        (paths.test_rh, len(paths.test_images)),
    )
    for fmri_path, expected_rows in pairs:
        array = np.load(fmri_path, mmap_mode="r")
        if array.ndim != 2 or array.shape[0] != expected_rows:
            raise ValueError(
                f"{fmri_path} has shape {array.shape}; expected ({expected_rows}, vertices)"
            )


def nsd_key(image_path: Path) -> str:
    match = IMAGE_RE.match(image_path.name)
    if not match:
        raise ValueError(f"Unexpected Algonauts image name: {image_path.name}")
    return f"{int(match.group(2)):05d}"


def load_caption_mapping(path: Path) -> dict[str, list[str]]:
    """Load {nsd_filename_id: [caption, ...]} with zero-padded or integer-like keys."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Caption JSON must be an object mapping NSD IDs to caption lists")
    normalized: dict[str, list[str]] = {}
    for key, value in raw.items():
        normalized_key = f"{int(str(key).replace('nsd-', '')):05d}"
        if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
            raise ValueError(f"Caption entry {key!r} must be a non-empty list of strings")
        normalized[normalized_key] = [caption.strip() for caption in value if caption.strip()]
    return normalized


def captions_for_images(
    image_paths: Iterable[Path], mapping: dict[str, list[str]], expected_count: int = 5
) -> list[list[str]]:
    result: list[list[str]] = []
    missing: list[str] = []
    wrong_count: list[tuple[str, int]] = []
    for image_path in image_paths:
        key = nsd_key(image_path)
        captions = mapping.get(key)
        if captions is None:
            missing.append(key)
            continue
        if expected_count > 0 and len(captions) != expected_count:
            wrong_count.append((key, len(captions)))
        result.append(captions)
    if missing:
        preview = ", ".join(missing[:10])
        raise KeyError(f"Missing captions for {len(missing)} NSD IDs, including: {preview}")
    if wrong_count:
        preview = ", ".join(f"{key}:{count}" for key, count in wrong_count[:10])
        raise ValueError(f"Expected {expected_count} captions per image; mismatches: {preview}")
    return result

