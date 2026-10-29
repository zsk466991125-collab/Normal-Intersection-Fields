"""Create an Ultralytics OBB view of the already sliced DOTA train/val data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, box


SOURCE_NAMES = ["plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle", "large-vehicle",
                "ship", "tennis-court", "basketball-court", "storage-tank", "soccer-ball-field", "roundabout",
                "harbor", "swimming-pool", "helicopter"]
NAMES = ["plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court",
         "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter",
         "roundabout", "soccer ball field", "swimming pool"]
CLASS_MAP = {name: NAMES.index(name.replace("-", " ")) for name in SOURCE_NAMES}


def clipped_quad(points: list[float], width: int, height: int) -> np.ndarray | None:
    polygon = Polygon(np.asarray(points, dtype=np.float32).reshape(4, 2))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    clipped = polygon.intersection(box(0, 0, width - 1, height - 1))
    if clipped.is_empty:
        return None
    if clipped.geom_type == "MultiPolygon":
        clipped = max(clipped.geoms, key=lambda p: p.area)
    coords = np.asarray(clipped.exterior.coords[:-1], dtype=np.float32)
    if len(coords) < 3 or abs(cv2.contourArea(coords)) < 1.0:
        return None
    rect = cv2.boxPoints(cv2.minAreaRect(coords)).astype(np.float32)
    rect[:, 0] = rect[:, 0].clip(0, width - 1)
    rect[:, 1] = rect[:, 1].clip(0, height - 1)
    return rect


def convert_split(ann_src: Path, image_src: Path, dst: Path, split: str) -> dict:
    image_dir = image_src / split / "images"
    ann_dir = ann_src / split / "annfiles"
    label_dir = dst / "labels" / split
    image_link = dst / "images" / split
    label_dir.mkdir(parents=True, exist_ok=True)
    image_link.parent.mkdir(parents=True, exist_ok=True)
    if not image_link.exists():
        image_link.symlink_to(image_dir.resolve(), target_is_directory=True)
    source_label_dir = image_dir.parent / "labels"
    if source_label_dir.is_symlink() and not source_label_dir.exists():
        source_label_dir.unlink()
    if not source_label_dir.exists():
        source_label_dir.symlink_to(os.path.relpath(label_dir, source_label_dir.parent), target_is_directory=True)
    objects = 0
    images = 0
    for image_path in sorted(image_dir.glob("*.png")):
        images += 1
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"cannot read {image_path}")
        height, width = image.shape[:2]
        output = []
        ann_path = ann_dir / f"{image_path.stem}.txt"
        if ann_path.exists():
            for line in ann_path.read_text().splitlines():
                fields = line.split()
                if len(fields) < 9 or fields[8] not in CLASS_MAP:
                    continue
                quad = clipped_quad([float(x) for x in fields[:8]], width, height)
                if quad is None:
                    continue
                norm = quad.copy()
                norm[:, 0] /= width
                norm[:, 1] /= height
                output.append(f"{CLASS_MAP[fields[8]]} " + " ".join(f"{x:.7f}" for x in norm.reshape(-1)))
                objects += 1
        (label_dir / f"{image_path.stem}.txt").write_text("\n".join(output) + ("\n" if output else ""))
    return {"split": split, "images": images, "objects": objects}


def create_trainval_view(output: Path) -> dict:
    """Create a merged trainval tree with symlinks to existing split files."""
    view = output / "trainval"
    merged_stems = {}
    counts = {}
    for folder, suffix in (("images", ".png"), ("labels", ".txt")):
        destination = view / folder
        destination.mkdir(parents=True, exist_ok=True)
        expected = {}
        for split in ("train", "val"):
            source_dir = output / folder / split
            if not source_dir.is_dir():
                raise FileNotFoundError(source_dir)
            for source in sorted(source_dir.glob(f"*{suffix}")):
                if source.stem in expected:
                    raise ValueError(f"Duplicate {folder} stem across train/val: {source.stem}")
                expected[source.stem] = source
        existing = {path.stem for path in destination.glob(f"*{suffix}")}
        if existing - set(expected):
            raise ValueError(f"Unexpected files in {destination}")
        for stem, source in expected.items():
            target = destination / f"{stem}{suffix}"
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    raise ValueError(f"Conflicting trainval symlink: {target}")
            elif target.exists():
                raise ValueError(f"Refusing to replace non-symlink trainval file: {target}")
            else:
                target.symlink_to(os.path.relpath(source.resolve(), destination.resolve()))
        merged_stems[folder] = set(expected)
        counts[folder] = len(expected)

    if merged_stems["images"] != merged_stems["labels"]:
        raise ValueError("Trainval image and label stems do not match")
    base_yaml = output / "DOTA10_nif.yaml"
    names = json.loads(base_yaml.read_text())["names"] if base_yaml.exists() else {
        index: name for index, name in enumerate(NAMES)
    }
    data = {"path": str(output), "train": "trainval/images",
            "val": "trainval/images", "names": names}
    (output / "DOTA10_nif_trainval.yaml").write_text(json.dumps(data, indent=2) + "\n")
    stats = {
        "split": "trainval", "images": counts["images"], "labels": counts["labels"],
        "objects": sum(len(path.read_text().splitlines())
                        for path in (view / "labels").glob("*.txt")),
        "storage": "symlinks to existing train and val files",
        "val_is_training_only": True,
    }
    (output / "trainval_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/dota")
    parser.add_argument("--images-source", default="data/dota")
    parser.add_argument("--output", default="data/yolo_dota10")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    source, image_source, output = (Path(args.source).resolve(), Path(args.images_source).resolve(),
                                    Path(args.output).resolve())
    if not args.merge_only:
        stats = [convert_split(source, image_source, output, split) for split in ("train", "val")]
        data = {"path": str(output), "train": "images/train", "val": "images/val",
                "names": {i: name for i, name in enumerate(NAMES)}}
        (output / "DOTA10_nif.yaml").write_text(json.dumps(data, indent=2) + "\n")
        (output / "conversion_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(create_trainval_view(output), indent=2))


if __name__ == "__main__":
    main()
