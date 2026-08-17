"""Extract DINOv2, CORnet-S, MPNet, and MiniLM features from NSD images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vsvariance.features import _capture_modules, _write_feature_batches, spatial_pyramid_pool


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DESIGN = ROOT / "derivatives" / "design"
OUTPUT = ROOT / "derivatives" / "features"
CAPTIONS = ROOT.parent / "metadata" / "nsd_captions.json"
SUBJECTS = ("subj01", "subj02", "subj05", "subj07")
LEVELS = (1, 2, 4)


def union_image_ids() -> np.ndarray:
    ids = []
    for subject in SUBJECTS:
        split = np.load(DESIGN / f"{subject}_image_split.npz")
        ids.extend((split["train_nsd_id"], split["test_nsd_id"]))
    return np.unique(np.concatenate(ids).astype(np.int64))


class Hdf5ImageDataset:
    """Windows-safe lazy HDF5 image reader; each worker owns its file handle."""

    def __init__(self, path: Path, image_ids: np.ndarray, transform):
        self.path = str(path)
        self.image_ids = image_ids
        self.transform = transform
        self._file = None
        self._images = None

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        from PIL import Image

        if self._file is None:
            self._file = h5py.File(self.path, "r")
            self._images = self._file["imgBrick"]
        image = Image.fromarray(np.asarray(self._images[int(self.image_ids[index])])).convert("RGB")
        return self.transform(image)


def image_loader(image_ids: np.ndarray, transform, batch_size: int, workers: int):
    import torch
    from torch.utils.data import DataLoader

    hdf5_path = DATA / "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    return DataLoader(
        Hdf5ImageDataset(hdf5_path, image_ids, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def extract_dino(image_ids: np.ndarray, batch_size: int, workers: int, device: str) -> None:
    import torch
    from torchvision import transforms

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
    loader = image_loader(image_ids, transform, batch_size, workers)
    indices = (2, 5, 8, 11)
    names = tuple(f"block{i + 1:02d}" for i in indices)

    def batches():
        with torch.inference_mode():
            for images in loader:
                outputs = model.get_intermediate_layers(
                    images.to(device, non_blocking=True), n=list(indices), reshape=True, return_class_token=False
                )
                yield {name: spatial_pyramid_pool(value, LEVELS).cpu().numpy() for name, value in zip(names, outputs, strict=True)}

    _write_feature_batches(batches(), OUTPUT / "dinov2", len(image_ids), "float16")


def extract_cornet(image_ids: np.ndarray, batch_size: int, workers: int, device: str) -> None:
    import torch
    from cornet import cornet_s
    from torchvision import transforms

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
    loader = image_loader(image_ids, transform, batch_size, workers)

    def batches():
        with torch.inference_mode(), _capture_modules(modules) as captured:
            for images in loader:
                captured.clear()
                model(images.to(device, non_blocking=True))
                yield {name: spatial_pyramid_pool(captured[name], LEVELS).cpu().numpy() for name in modules}

    _write_feature_batches(batches(), OUTPUT / "cornet_s", len(image_ids), "float16")


TEXT_MODELS = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


def extract_text(image_ids: np.ndarray, batch_size: int, device: str, model_name: str) -> None:
    from sentence_transformers import SentenceTransformer

    if model_name not in TEXT_MODELS:
        raise KeyError(model_name)
    mapping = json.loads(CAPTIONS.read_text(encoding="utf-8"))
    groups = [mapping[f"{int(image_id):05d}"] for image_id in image_ids]
    if any(len(group) != 5 for group in groups):
        raise ValueError("Every retained NSD image must have five COCO captions")
    flat = [caption for group in groups for caption in group]
    model_id = TEXT_MODELS[model_name]
    model = SentenceTransformer(model_id, device=device)
    embeddings = model.encode(
        flat, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False
    ).astype(np.float32, copy=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    averaged = embeddings.reshape(len(groups), 5, -1).mean(axis=1)
    np.save(OUTPUT / f"{model_name}.npy", averaged)
    (OUTPUT / f"{model_name}_metadata.json").write_text(
        json.dumps(
            {
                "model": model_id,
                "image_order": str((OUTPUT / "image_ids.npy").resolve()),
                "n_images": int(len(groups)),
                "captions_per_image": 5,
                "caption_aggregation": "arithmetic mean of five independently encoded COCO captions",
                "embedding_dimension": int(averaged.shape[1]),
                "normalize_embeddings": False,
                "dtype": str(averaged.dtype),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=("dinov2", "cornet_s", "mpnet", "minilm", "all"), required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    ids = union_image_ids()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT / "image_ids.npy", ids)
    if args.model in ("dinov2", "all"):
        extract_dino(ids, args.image_batch_size, args.workers, args.device)
        (OUTPUT / "dinov2.complete").write_text("complete\n", encoding="utf-8")
    if args.model in ("cornet_s", "all"):
        extract_cornet(ids, args.image_batch_size, args.workers, args.device)
        (OUTPUT / "cornet_s.complete").write_text("complete\n", encoding="utf-8")
    if args.model in ("mpnet", "all"):
        extract_text(ids, args.text_batch_size, args.device, "mpnet")
        (OUTPUT / "mpnet.complete").write_text("complete\n", encoding="utf-8")
    if args.model in ("minilm", "all"):
        extract_text(ids, args.text_batch_size, args.device, "minilm")
        (OUTPUT / "minilm.complete").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
