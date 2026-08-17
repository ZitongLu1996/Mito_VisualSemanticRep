"""Build filename-aligned NSD caption JSON from official NSD and COCO metadata."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_coco(path: Path) -> dict[int, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, list[str]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        result[int(annotation["image_id"])].append(str(annotation["caption"]).strip())
    return dict(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsd-stim-info", type=Path, required=True)
    parser.add_argument("--coco-train-captions", type=Path, required=True)
    parser.add_argument("--coco-val-captions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    coco = {
        "train2017": _load_coco(args.coco_train_captions),
        "val2017": _load_coco(args.coco_val_captions),
    }
    output: dict[str, list[str]] = {}
    caption_count_distribution: Counter[int] = Counter()
    with args.nsd_stim_info.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"nsdId", "cocoId", "cocoSplit"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"NSD metadata requires columns {sorted(required)}")
        for row in reader:
            nsd_id = int(row["nsdId"])
            coco_id = int(row["cocoId"])
            split = row["cocoSplit"].strip()
            if split not in coco:
                raise ValueError(f"Unexpected COCO split {split!r} for NSD ID {nsd_id}")
            captions = coco[split].get(coco_id, [])
            caption_count_distribution[len(captions)] += 1
            if len(captions) < 5:
                raise ValueError(
                    f"Expected at least five captions for NSD {nsd_id}/COCO {coco_id}, found {len(captions)}"
                )
            # COCO has 189 NSD images with six captions and two with seven.
            # Use the first five annotation records deterministically to match the
            # five-caption representation used in the target NSD literature.
            output[f"{nsd_id:05d}"] = captions[:5]
    if len(output) != 73_000:
        raise ValueError(f"Expected 73,000 NSD rows, found {len(output)}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    audit_path = args.output_json.with_suffix(".metadata.json")
    audit = {
        "n_nsd_images": len(output),
        "captions_retained_per_image": 5,
        "source_caption_count_distribution": {
            str(key): value for key, value in sorted(caption_count_distribution.items())
        },
        "selection_rule": "first five COCO annotation records per image",
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
