from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .data import captions_for_images, load_caption_mapping


class _ImageDataset:
    """Top-level map-style dataset so Windows worker processes can pickle it."""

    def __init__(self, image_paths: Sequence[Path], transform):
        self.image_paths = tuple(image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        from PIL import Image

        with Image.open(self.image_paths[index]) as image:
            return self.transform(image.convert("RGB"))


def _lazy_torch():
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    return torch, Image, DataLoader, Dataset, transforms


def spatial_pyramid_pool(feature_map, levels: Sequence[int]):
    import torch
    import torch.nn.functional as functional

    if feature_map.ndim != 4:
        raise ValueError(f"Expected [batch, channels, height, width], got {feature_map.shape}")
    pooled = [functional.adaptive_avg_pool2d(feature_map, (level, level)).flatten(1) for level in levels]
    return torch.cat(pooled, dim=1)


def _image_loader(image_paths: Sequence[Path], transform, batch_size: int, workers: int):
    torch, _, DataLoader, _, _ = _lazy_torch()

    return DataLoader(
        _ImageDataset(image_paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _write_feature_batches(
    batches: Iterator[dict[str, np.ndarray]], output_dir: Path, n_samples: int, dtype: str
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, np.memmap] = {}
    paths: dict[str, Path] = {}
    cursor = 0
    next_report = 0.1
    for batch in batches:
        if not batch:
            raise ValueError("Feature extractor returned an empty batch")
        batch_rows = next(iter(batch.values())).shape[0]
        if any(value.shape[0] != batch_rows for value in batch.values()):
            raise ValueError("Feature groups disagree on batch size")
        if not writers:
            for name, value in batch.items():
                path = output_dir / f"{name}.npy"
                paths[name] = path
                writers[name] = np.lib.format.open_memmap(
                    path, mode="w+", dtype=dtype, shape=(n_samples, value.shape[1])
                )
        for name, value in batch.items():
            writers[name][cursor : cursor + batch_rows] = value.astype(dtype, copy=False)
        cursor += batch_rows
        fraction = cursor / n_samples
        if fraction >= next_report or cursor == n_samples:
            print(f"FEATURE_PROGRESS rows={cursor}/{n_samples} ({fraction:.0%})", flush=True)
            next_report += 0.1
    if cursor != n_samples:
        raise ValueError(f"Wrote {cursor} feature rows, expected {n_samples}")
    for writer in writers.values():
        writer.flush()
    return paths


def extract_dinov2(
    image_paths: Sequence[Path],
    output_dir: Path,
    batch_size: int,
    workers: int,
    levels: Sequence[int],
    device: str,
    dtype: str = "float16",
) -> dict[str, Path]:
    torch, _, _, _, transforms = _lazy_torch()
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    model.eval().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    loader = _image_loader(image_paths, transform, batch_size, workers)
    block_indices = (2, 5, 8, 11)  # human-readable blocks 3, 6, 9, 12
    names = tuple(f"block{index + 1:02d}" for index in block_indices)

    def batches():
        with torch.inference_mode():
            for images in loader:
                images = images.to(device, non_blocking=True)
                outputs = model.get_intermediate_layers(
                    images, n=list(block_indices), reshape=True, return_class_token=False
                )
                yield {
                    name: spatial_pyramid_pool(output, levels).cpu().numpy()
                    for name, output in zip(names, outputs, strict=True)
                }

    return _write_feature_batches(batches(), output_dir, len(image_paths), dtype)


@contextmanager
def _capture_modules(modules: dict[str, object]):
    captured: dict[str, object] = {}
    handles = []
    for name, module in modules.items():
        handles.append(module.register_forward_hook(lambda _m, _i, out, key=name: captured.__setitem__(key, out)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def extract_cornet_s(
    image_paths: Sequence[Path],
    output_dir: Path,
    batch_size: int,
    workers: int,
    levels: Sequence[int],
    device: str,
    dtype: str = "float16",
) -> dict[str, Path]:
    torch, _, _, _, transforms = _lazy_torch()
    try:
        from cornet import cornet_s
    except ImportError as error:
        raise ImportError(
            "Install the official CORnet package listed in requirements-variance.txt"
        ) from error
    model = cornet_s(pretrained=True, map_location=device)
    model.eval().to(device)
    backbone = model.module if hasattr(model, "module") else model
    modules = {name: getattr(backbone, name) for name in ("V1", "V2", "V4", "IT")}
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    loader = _image_loader(image_paths, transform, batch_size, workers)

    def batches():
        with torch.inference_mode(), _capture_modules(modules) as captured:
            for images in loader:
                captured.clear()
                model(images.to(device, non_blocking=True))
                missing = set(modules) - set(captured)
                if missing:
                    raise RuntimeError(f"CORnet-S hooks did not capture: {sorted(missing)}")
                yield {
                    name: spatial_pyramid_pool(captured[name], levels).cpu().numpy()
                    for name in modules
                }

    return _write_feature_batches(batches(), output_dir, len(image_paths), dtype)


def extract_mpnet(
    image_paths: Sequence[Path],
    caption_json: Path,
    output_path: Path,
    batch_size: int,
    device: str,
) -> Path:
    from sentence_transformers import SentenceTransformer

    mapping = load_caption_mapping(caption_json)
    caption_groups = captions_for_images(image_paths, mapping, expected_count=5)
    flat = [caption for group in caption_groups for caption in group]
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    embeddings = model.encode(
        flat,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32, copy=False)
    features = embeddings.reshape(len(caption_groups), 5, -1).mean(axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)
    return output_path


def write_feature_manifest(
    output_path: Path, image_paths: Sequence[Path], groups: dict[str, Path]
) -> None:
    payload = {
        "images": [path.name for path in image_paths],
        "groups": {name: str(path.resolve()) for name, path in groups.items()},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
