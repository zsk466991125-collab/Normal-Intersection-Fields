"""Run tiled YOLO OBB inference on DOTA test images and export Task1 ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np


NIF_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NIF_ROOT))
ULTRALYTICS_ROOT = Path(os.environ.get(
    "ULTRALYTICS_ROOT", str(NIF_ROOT / "third_party/ultralytics")))
PADDING = (104, 116, 124)
WINDOW_SIZE = 1024
WINDOW_GAP = 200
IMAGE_RATE_THRESHOLD = 0.6
TASK1_CLASSES = (
    "plane", "baseball-diamond", "bridge", "ground-track-field",
    "small-vehicle", "large-vehicle", "ship", "tennis-court",
    "basketball-court", "storage-tank", "soccer-ball-field", "roundabout",
    "harbor", "swimming-pool", "helicopter",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_manifest(images: list[Path], weights: Path, args) -> dict:
    return {
        "weights": str(weights.resolve()),
        "weights_sha256": sha256(weights),
        "implementation_sha256": {
            "inference_script": sha256(Path(__file__)),
            "nif_adapter": sha256(NIF_ROOT / "adapters/ultralytics/nif_obb.py"),
        },
        "images": [
            {"name": image.name, "size": image.stat().st_size,
             "mtime_ns": image.stat().st_mtime_ns}
            for image in images
        ],
        "tiling": {"size": WINDOW_SIZE, "gap": WINDOW_GAP, "rate": 1.0,
                   "image_rate_threshold": IMAGE_RATE_THRESHOLD,
                   "padding_bgr": list(PADDING)},
        "prediction": {"image_size": args.imgsz, "batch_size": args.batch_size,
                       "confidence": args.conf, "patch_nms_iou": args.patch_iou,
                       "merge_nms_iou": args.merge_iou, "max_det": args.max_det,
                       "precision": "ultralytics-default"},
        "device_ids": args.devices,
    }


def load_original_windows(width: int, height: int) -> np.ndarray:
    from data.dota.split.img_split import get_sliding_window

    return get_sliding_window(
        {"width": width, "height": height}, [WINDOW_SIZE], [WINDOW_GAP],
        IMAGE_RATE_THRESHOLD)


def prepare_patch(image: np.ndarray, window: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(int, window)
    crop = image[y1:y2, x1:x2]
    patch = np.empty((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    patch[...] = PADDING
    patch[:crop.shape[0], :crop.shape[1]] = crop
    return patch


def detection_arrays(result, name_to_task1: dict[str, int]) -> list[np.ndarray]:
    output = [[] for _ in TASK1_CLASSES]
    obb = result.obb
    if obb is not None and len(obb):
        boxes = obb.xywhr.detach().float().cpu().numpy()
        scores = obb.conf.detach().float().cpu().numpy()
        labels = obb.cls.detach().long().cpu().numpy()
        names = result.names
        for box, score, label in zip(boxes, scores, labels):
            class_name = names[int(label)]
            task1_index = name_to_task1[class_name]
            output[task1_index].append(
                np.concatenate((box, np.asarray([score], dtype=np.float32))))
    return [np.asarray(values, dtype=np.float32).reshape(-1, 6)
            for values in output]


def save_merged(cache_path: Path, height: int, width: int,
                patch_results: list[list[np.ndarray]], windows: np.ndarray,
                device: str, iou_threshold: float) -> None:
    from mmrotate.core import merge_results

    merged = merge_results(patch_results, windows[:, :2].copy(),
                           (width, height), iou_thr=iou_threshold, device=device)
    payload = {f"cls_{index}": np.asarray(value, dtype=np.float32).reshape(-1, 6)
               for index, value in enumerate(merged)}
    temporary = cache_path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(cache_path)


def worker_main(args) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(0)
    import torch

    torch.set_num_threads(1)
    if torch.cuda.is_available() is False:
        raise RuntimeError("CUDA is unavailable in this worker")
    device_id = int(args.devices[args.worker_index])
    torch.cuda.set_device(device_id)

    if str(ULTRALYTICS_ROOT) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS_ROOT))
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    names = model.names
    name_to_task1 = {
        name.replace("-", " "): index
        for index, name in enumerate(TASK1_CLASSES)
    }
    model_names = {names[index].replace("-", " ") for index in names}
    if model.task != "obb" or model_names != set(name_to_task1):
        raise ValueError(
            f"Expected a 15-class DOTA OBB checkpoint, got task={model.task}, "
            f"classes={sorted(model_names)}")

    image_paths = sorted(args.images.glob("*.png"))
    if args.limit:
        image_paths = image_paths[:args.limit]
    assigned = image_paths[args.worker_index::len(args.devices)]
    cache_dir = args.output / "per_image"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    active = {}
    started = time.monotonic()
    finished = 0

    def run_batch():
        nonlocal pending, finished
        if not pending:
            return
        predictions = model.predict(
            [task[3] for task in pending], imgsz=args.imgsz,
            batch=len(pending), device=device_id, conf=args.conf,
            iou=args.patch_iou, max_det=args.max_det,
            verbose=False, stream=False)
        for task, prediction in zip(pending, predictions):
            image_id, patch_index = task[0], task[1]
            record = active[image_id]
            record["results"][patch_index] = detection_arrays(
                prediction, name_to_task1)
            record["received"] += 1
            if record["received"] == len(record["windows"]):
                save_merged(cache_dir / f"{image_id}.npz", record["height"],
                            record["width"], record["results"],
                            record["windows"], f"cuda:{device_id}", args.merge_iou)
                del active[image_id]
                finished += 1
                if finished % 20 == 0 or finished == len(assigned):
                    print(f"GPU {device_id}: {finished}/{len(assigned)} assigned "
                          f"images merged; {time.monotonic() - started:.1f}s",
                          flush=True)
        pending = []

    for image_path in assigned:
        cache_path = cache_dir / f"{image_path.stem}.npz"
        if cache_path.exists():
            finished += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Cannot decode test image: {image_path}")
        height, width = image.shape[:2]
        windows = load_original_windows(width, height)
        active[image_path.stem] = {
            "height": height, "width": width, "windows": windows,
            "results": [None] * len(windows), "received": 0,
        }
        for patch_index, window in enumerate(windows):
            pending.append((image_path.stem, patch_index, window,
                            prepare_patch(image, window)))
            if len(pending) == args.batch_size:
                run_batch()
        del image
    run_batch()
    if active:
        raise RuntimeError(f"Worker ended with incomplete images: {sorted(active)}")
    print(f"GPU {device_id}: finished {finished}/{len(assigned)} assigned images",
          flush=True)


def write_submission(images: list[Path], output: Path) -> Path:
    from mmrotate.core import obb2poly_np

    pending = [[] for _ in TASK1_CLASSES]
    for image in images:
        cache = output / "per_image" / f"{image.stem}.npz"
        if not cache.exists():
            raise FileNotFoundError(f"Missing merged predictions: {cache}")
        with np.load(cache) as data:
            for class_index in range(len(TASK1_CLASSES)):
                detections = data[f"cls_{class_index}"]
                if not len(detections):
                    continue
                polygons = obb2poly_np(detections, "le90")
                for row in polygons:
                    score = float(row[-1])
                    coords = np.asarray(row[:8], dtype=np.float64)
                    if not np.isfinite(score) or not np.all(np.isfinite(coords)):
                        raise ValueError(f"Non-finite prediction in {cache}")
                    polygon = np.round(coords, 6).astype(np.float32).reshape(4, 2)
                    if abs(float(cv2.contourArea(polygon))) == 0:
                        continue
                    pending[class_index].append(
                        (score, image.stem, coords))

    task1_dir = output / "Task1"
    task1_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for class_index, class_name in enumerate(TASK1_CLASSES):
        name = f"Task1_{class_name}.txt"
        names.append(name)
        rows = sorted(pending[class_index], key=lambda item: item[0], reverse=True)
        with (task1_dir / name).open("w", encoding="utf-8") as stream:
            for score, image_id, coords in rows:
                stream.write(" ".join(
                    [image_id, f"{score:.9f}",
                     *(f"{value:.6f}" for value in coords)]) + "\n")

    archive = output / "test.zip"
    temporary = output / "test.zip.partial"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as zipped:
        for name in names:
            zipped.write(task1_dir / name, arcname=name)
    with zipfile.ZipFile(temporary) as zipped:
        if set(zipped.namelist()) != set(names) or len(zipped.namelist()) != 15:
            raise ValueError("Submission ZIP does not contain exactly 15 Task1 files")
        image_ids = {p.stem for p in images}
        for name in names:
            for line_number, line in enumerate(zipped.read(name).decode().splitlines(), 1):
                fields = line.split()
                if len(fields) != 10 or fields[0] not in image_ids:
                    raise ValueError(f"Invalid DOTA row in {name}:{line_number}")
                values = np.asarray([float(value) for value in fields[1:]],
                                    dtype=np.float64)
                if not np.all(np.isfinite(values)) or not 0 <= values[0] <= 1:
                    raise ValueError(f"Invalid score/coordinates in {name}:{line_number}")
    temporary.replace(archive)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=NIF_ROOT / "yolo26l-obb.pt")
    parser.add_argument("--images", type=Path, default=NIF_ROOT / "data/DOTA/test/images")
    parser.add_argument("--output", type=Path,
                        default=NIF_ROOT / "work_dirs/yolo26l_obb_dota10_test")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--patch-iou", type=float, default=0.7)
    parser.add_argument("--merge-iou", type=float, default=0.1)
    parser.add_argument("--max-det", type=int, default=700)
    parser.add_argument("--expected-images", type=int, default=937)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N images; skips final submission ZIP")
    parser.add_argument("--worker-index", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.weights = args.weights.resolve()
    args.images = args.images.resolve()
    args.output = args.output.resolve()
    args.devices = [int(value) for value in args.devices.split(",") if value.strip()]
    if not args.devices or args.batch_size < 1 or args.imgsz != WINDOW_SIZE:
        parser.error("Provide GPUs, positive batch size, and imgsz=1024")
    if args.limit < 0 or args.expected_images < 1:
        parser.error("limit must be nonnegative and expected-images positive")
    if args.worker_index >= 0:
        if args.worker_count != len(args.devices):
            parser.error("worker-count must match the number of devices")
        worker_main(args)
        return

    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    images = sorted(args.images.glob("*.png"))
    if args.limit:
        images = images[:args.limit]
    elif len(images) != args.expected_images:
        raise ValueError(f"Expected {args.expected_images} images, found {len(images)}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = expected_manifest(images, args.weights, args)
    manifest_path = args.output / "inference_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Output directory belongs to different inference inputs/settings")
    else:
        if any(args.output.iterdir()):
            raise ValueError("Output directory is nonempty without a matching manifest")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if args.limit:
        print(f"Smoke inference: {len(images)} originals; no final ZIP will be written.",
              flush=True)
    start = time.monotonic()
    logs = []
    processes = []
    for worker_index, device_id in enumerate(args.devices):
        log_path = args.output / f"worker_gpu{device_id}.log"
        log = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--weights", str(args.weights), "--images", str(args.images),
            "--output", str(args.output), "--devices",
            ",".join(map(str, args.devices)), "--batch-size", str(args.batch_size),
            "--imgsz", str(args.imgsz), "--conf", str(args.conf),
            "--patch-iou", str(args.patch_iou), "--merge-iou", str(args.merge_iou),
            "--max-det", str(args.max_det), "--expected-images", str(args.expected_images),
            "--worker-index", str(worker_index), "--worker-count", str(len(args.devices)),
        ]
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        environment = os.environ.copy()
        current_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(ULTRALYTICS_ROOT), str(NIF_ROOT), current_pythonpath)
            if value)
        process = subprocess.Popen(command, cwd=NIF_ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        processes.append((device_id, process, log))
        logs.append(log_path)
        print(f"Started GPU {device_id} worker, log: {log_path}", flush=True)

    errors = []
    for device_id, process, log in processes:
        code = process.wait()
        log.close()
        if code:
            errors.append((device_id, code))
    if errors:
        raise RuntimeError(f"Inference workers failed: {errors}; logs: {logs}")

    if args.limit:
        print(f"Smoke inference finished in {time.monotonic() - start:.1f}s; "
              f"review worker logs under {args.output}", flush=True)
        return
    archive = write_submission(images, args.output)
    print(f"Submission: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)",
          flush=True)
    print(f"Inference complete: {len(images)} originals in "
          f"{time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
