"""Export a Task1 ZIP and measure actual inference complexity in the same pass."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import OrderedDict
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import re
import time
import zipfile

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from mmcv.runner.checkpoint import _load_checkpoint, load_state_dict
from mmdet.apis import inference_detector

from mmrotate.core import merge_results, obb2poly_np
from mmrotate.datasets import DOTADataset, DOTADataset15
from mmrotate.models import build_detector
from data.dota.split.img_split import get_sliding_window
from tools.analysis_tools.nif_inference_complexity import (
    InferenceComplexityCounter, atomic_json, cached_complexity,
    parameter_inventory, summarize_inference, update_epoch_summary,
    write_image_csv)

WORKSPACE = Path(__file__).resolve().parents[1]
PADDING = [104, 116, 124]


def load_evaluation_checkpoint(model, filename, rtmdet_ema=False):
    """Load an inference checkpoint while accepting RTMDet EMA buffers.

    ``RTMDetEMAHook`` swaps the EMA values into the regular model keys before
    ``CheckpointHook`` saves a checkpoint, then stores the raw training values
    in buffers named ``ema_<parameter>`` for resume.  The submission model does
    not register that hook, so those auxiliary buffers are not model keys.  We
    remove only the EMA names that can be derived from this model and retain
    strict loading for every real model key and any unknown checkpoint key.
    """
    if not rtmdet_ema:
        return load_checkpoint(model, str(filename), map_location='cpu', strict=True)
    checkpoint = _load_checkpoint(str(filename), map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    state_dict = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f'Checkpoint state_dict is not a mapping: {filename}')
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    state_dict = OrderedDict(
        (re.sub(r'^module\.', '', key), value)
        for key, value in state_dict.items())
    state_dict._metadata = metadata

    model_names = set(dict(model.named_parameters()))
    model_names.update(dict(model.named_buffers()))
    ema_names = {
        'ema_' + name.replace('.', '_')
        for name in model_names
    }
    auxiliary = [key for key in state_dict if key in ema_names]
    for key in auxiliary:
        del state_dict[key]
    load_state_dict(model, state_dict, strict=True)
    if auxiliary:
        print(f'Ignoring {len(auxiliary)} RTMDet EMA resume buffers during '
              'evaluation; regular checkpoint keys remain strictly checked.',
              flush=True)
    return checkpoint


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def padded_patch(image, window):
    """Same pixels, coordinates and BGR padding as img_split.py at rate 1."""
    x1, y1, x2, y2 = window
    crop = image[y1:y2, x1:x2]
    patch = np.empty((y2 - y1, x2 - x1, 3), dtype=image.dtype)
    patch[...] = PADDING
    patch[:crop.shape[0], :crop.shape[1]] = crop
    return patch


def predict_original(model, image, batch_size=1):
    height, width = image.shape[:2]
    windows = get_sliding_window(dict(width=width, height=height),
                                 [1024], [200], 0.6)
    results = []
    for start in range(0, len(windows), batch_size):
        patches = [padded_patch(image, window)
                   for window in windows[start:start + batch_size]]
        results.extend(inference_detector(model, patches))
    return merge_results(results, windows[:, :2], (width, height),
                         iou_thr=0.1, device=next(model.parameters()).device), len(windows)


def infer_cached_original(model, path, cache, batch_size):
    """Cache counts together with predictions; never double count on resume."""
    cached = cache / f'{path.stem}.pkl'
    if cached.exists():
        payload = mmcv.load(str(cached))
        measurement = cached_complexity(payload)
        if measurement is not None:
            return measurement
        # Older caches contain only predictions. Re-run this original once
        # to obtain measured cost; a prediction list cannot recover NIF inputs.
    image = mmcv.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot decode {path}')
    with InferenceComplexityCounter(model) as counter:
        result, count = predict_original(model, image, batch_size)
        measurement = counter.values(count)
    temporary = cache / f'{path.stem}.partial'
    mmcv.dump(dict(results=result, patches=count, complexity=measurement),
              str(temporary), file_format='pkl')
    temporary.replace(cached)
    return measurement


def submission_settings(config, images=None, example=None):
    """Resolve the official class order and test originals from the config."""
    datasets = {'DOTADataset': DOTADataset, 'DOTADataset15': DOTADataset15}
    dataset_type = config.data.test.type
    if dataset_type not in datasets:
        raise ValueError(f'Unsupported DOTA submission dataset: {dataset_type}')
    classes = datasets[dataset_type].CLASSES
    # NIF carriers are two-stage, while the rotated RTMDet port is a
    # single-stage detector.  Both expose the class count on their detection
    # head, but at different config paths.
    if config.model.get('roi_head') is not None:
        head_cfg = config.model.roi_head.bbox_head
    else:
        head_cfg = config.model.bbox_head
    if head_cfg.num_classes != len(classes):
        raise ValueError('Detection head class count differs from the DOTA dataset')
    for split in ('train', 'val', 'test'):
        dataset = config.data.get(split, {})
        if dataset.get('classes') is not None and tuple(dataset.classes) != classes:
            raise ValueError('Custom class order differs from the official DOTA dataset')
    images = Path(images) if images is not None else Path(config.data.test.img_prefix)
    if example is not None:
        example = Path(example)
    elif dataset_type == 'DOTADataset':
        example = None
    # DOTA-v1.5 uses the same Task1 layout with one extra container-crane file.
    # The v1.0 example ZIP has only 15 classes and cannot validate this layout.
    return classes, images, example


def validate_submission(archive, example, image_ids, classes=DOTADataset.CLASSES):
    expected = {'Task1_' + name + '.txt' for name in classes}
    if example is not None:
        with zipfile.ZipFile(example) as source:
            names = source.namelist()
        if set(names) != expected or len(names) != len(expected):
            raise ValueError(f'Example ZIP does not contain the {len(classes)} DOTA Task1 files')
    counts = {}
    with zipfile.ZipFile(archive) as submission:
        names = submission.namelist()
        if set(names) != expected or len(names) != len(expected):
            raise ValueError('Submission ZIP layout differs from the DOTA Task1 class files')
        for name in names:
            count = 0
            with submission.open(name) as stream:
                for line in stream:
                    fields = line.decode('utf-8').split()
                    if len(fields) != 10 or fields[0] not in image_ids:
                        raise ValueError(f'{name}: invalid original image ID or column count')
                    values = [float(x) for x in fields[1:]]
                    if not all(math.isfinite(x) for x in values) or not 0 <= values[0] <= 1:
                        raise ValueError(f'{name}: invalid score/coordinates')
                    points = np.asarray(values[1:]).reshape(4, 2)
                    if abs(cv2.contourArea(points.astype(np.float32))) <= 0:
                        raise ValueError(f'{name}: degenerate polygon')
                    count += 1
            counts[name] = count
    return counts


def pack_results(cache, images, output, example, smoke=False, classes=DOTADataset.CLASSES):
    """Use only original-image predictions; merge is completed before this step."""
    folder = output / 'Task1'
    folder.mkdir(exist_ok=True)
    names = ['Task1_' + name + '.txt' for name in classes]
    # Quality calibration can produce a score slightly above one when the
    # learned factor is greater than its prior.  DOTA Task1 requires scores in
    # [0, 1].  Collect scores first and apply one monotonic normalization to
    # the complete submission so ranking is unchanged (unlike clipping,
    # which would create ties at one).
    pending = []
    filtered_degenerate = []
    maximum_score = 1.0
    scores_above_one = 0
    for image in images:
        result = mmcv.load(str(cache / f'{image.stem}.pkl'))['results']
        if len(result) != len(names):
            raise ValueError('Incorrect detection class count')
        image_rows = []
        for class_index, detections in enumerate(result):
            if not len(detections):
                continue
            for box in obb2poly_np(detections, 'le90'):
                score = float(box[-1])
                if not math.isfinite(score) or score < 0:
                    raise ValueError('invalid score/coordinates')
                coordinates = np.asarray(box[:8], dtype=np.float64)
                if not np.all(np.isfinite(coordinates)):
                    raise ValueError('invalid score/coordinates')
                # Some baseline checkpoints can emit a low-score box with an
                # exactly zero decoded width/height.  Its four polygon
                # vertices contain duplicates and cannot be represented in a
                # DOTA Task1 file.  Validate at the same six-decimal
                # precision used below, then omit only this invalid
                # detection.  Valid predictions and ranking are unchanged.
                rounded = np.round(coordinates, 6).astype(np.float32).reshape(4, 2)
                area = abs(float(cv2.contourArea(rounded)))
                if area <= 0.0:
                    filtered_degenerate.append(dict(
                        image=image.stem, class_name=classes[class_index],
                        score=score, area=area,
                        coordinates=coordinates.tolist()))
                    continue
                maximum_score = max(maximum_score, score)
                if score > 1:
                    scores_above_one += 1
                image_rows.append((class_index, image.stem, score, coordinates))
        pending.extend(image_rows)
    score_divisor = maximum_score
    with ExitStack() as stack:
        streams = [stack.enter_context((folder / name).open('w')) for name in names]
        for class_index, image_id, score, coordinates in pending:
            score = score / score_divisor
            # Six decimals avoid degenerating very small polygons by rounding.
            row = [image_id, f'{score:.9f}'] + [f'{x:.6f}' for x in coordinates]
            streams[class_index].write(' '.join(row) + '\n')
    atomic_json(output / 'score_normalization.json', dict(
        divisor=score_divisor, scores_above_one=scores_above_one,
        method='global_monotonic_division', bounds=[0.0, 1.0]))
    atomic_json(output / 'filtered_detections.json', dict(
        count=len(filtered_degenerate), detections=filtered_degenerate,
        rule='omit polygons with zero area after six-decimal Task1 formatting'))
    filename = 'SMOKE_NOT_FOR_SUBMISSION.zip' if smoke else output.name + '_Task1.zip'
    archive = output / filename
    temporary = output / (filename + '.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for name in names:
            zipped.write(folder / name, arcname=name)
    counts = validate_submission(temporary, example, {p.stem for p in images}, classes)
    temporary.replace(archive)
    return archive, counts


def check_checkpoint(meta, require_trainval, expected_epoch,
                     expected_max_epochs=None, expected_world_size=1):
    if expected_world_size < 1:
        raise ValueError('expected_world_size must be positive')
    if expected_epoch is not None and meta.get('epoch') != expected_epoch:
        raise ValueError(f'Expected epoch {expected_epoch}, found {meta.get("epoch")}')
    if require_trainval:
        trained = Config.fromstring(meta.get('config', ''), '.py')
        ann = Path(trained.data.train.ann_file)
        if ann.parent.name != 'trainval':
            raise ValueError('Checkpoint was not trained on trainval; refusing paper submission')
        if expected_max_epochs is None:
            # Preserve the original paper-protocol gate for callers that do
            # not specify a schedule explicitly.
            expected_max_epochs = 16
        if (trained.runner.max_epochs != expected_max_epochs
                or trained.data.samples_per_gpu != 2):
            raise ValueError('Checkpoint training schedule differs from paper settings')
        gpu_ids = trained.get('gpu_ids', range(1))
        if len(gpu_ids) != expected_world_size:
            if expected_world_size == 1:
                raise ValueError('Paper submission expects single-GPU global batch 2')
            raise ValueError(
                f'Expected {expected_world_size} training GPUs with batch 2 each; '
                f'checkpoint records {len(gpu_ids)} GPUs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--images', type=Path, help='Test originals; defaults to data.test.img_prefix')
    parser.add_argument('--example', type=Path,
                        help='Optional Task1 example ZIP; v1.5 defaults to the canonical 16-class layout')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--expected-images', type=int, default=937)
    parser.add_argument('--expected-epoch', type=int)
    parser.add_argument('--expected-max-epochs', type=int, default=None,
                        help='Expected training length (defaults to the config schedule)')
    parser.add_argument('--require-trainval', action='store_true')
    parser.add_argument('--expected-world-size', type=int, default=1,
                        help='Expected training GPU count, with batch 2 per GPU; '
                             'default preserves the single-GPU paper protocol')
    parser.add_argument('--limit', type=int, default=0, help='Smoke only; never writes a final submission ZIP')
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error('batch-size must be positive and limit nonnegative')
    if args.expected_max_epochs is not None and args.expected_max_epochs < 5:
        parser.error('expected-max-epochs must be at least 5')
    if args.expected_world_size < 1:
        parser.error('expected-world-size must be positive')
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    cfg = Config.fromfile(str(args.config))
    if args.expected_max_epochs is None:
        args.expected_max_epochs = cfg.runner.max_epochs
    classes, args.images, args.example = submission_settings(cfg, args.images, args.example)
    images = sorted(args.images.glob('*.png'))
    if len(images) != args.expected_images:
        raise ValueError(f'Expected {args.expected_images} test images, found {len(images)}')
    if args.limit:
        images = images[:args.limit]
    fingerprint = dict(
        config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
        checkpoint_sha256=file_sha256(args.checkpoint),
        images=str(args.images.resolve()), image_ids=[p.stem for p in images],
        source_files=[(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in images],
        size=1024, gap=200, rate=1.0, padding=PADDING, merge_iou=0.1,
        batch_size=args.batch_size, smoke=bool(args.limit))
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = args.output / 'inference_manifest.json'
    # JSON round-trip normalizes tuples for reproducible resume comparisons.
    fingerprint = json.loads(json.dumps(fingerprint))
    if provenance.exists():
        if json.loads(provenance.read_text()) != fingerprint:
            raise ValueError('Output belongs to different inputs/config/checkpoint')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Output directory is nonempty without a matching manifest')
        provenance.write_text(json.dumps(fingerprint, indent=2) + '\n')
    cache = args.output / 'per_image'
    cache.mkdir(exist_ok=True)
    cfg.model.backbone.init_cfg = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model)
    checkpoint = load_evaluation_checkpoint(
        model, args.checkpoint,
        rtmdet_ema=cfg.model.type in ('RotatedRTMDet', 'RotatedRTMDetNIF'))
    meta = checkpoint.get('meta', {})
    check_checkpoint(meta, args.require_trainval, args.expected_epoch,
                     args.expected_max_epochs, args.expected_world_size)
    model.CLASSES = classes
    model.cfg = cfg
    model.to(args.device).eval()
    if str(args.device).lower().startswith('cpu'):
        # MMCV's CPU BatchNorm implementation cannot execute SyncBN layers.
        # This changes only the module type; checkpoint tensors are unchanged.
        model = revert_sync_batchnorm(model).eval()
    del checkpoint
    parameters = parameter_inventory(model)
    print(f'Model parameters: {parameters["total"]:,}; '
          f'NIF parameters: {parameters["nif"]:,}. '
          'Measuring supported-module FLOPs during test inference.', flush=True)
    start = time.monotonic()
    total_patches = 0
    measurements = {}
    complexity_path = args.output / 'complexity.json'
    for index, path in enumerate(images, 1):
        measurement = infer_cached_original(model, path, cache, args.batch_size)
        measurements[path.stem] = measurement
        total_patches += measurement['patches']
        if index % 10 == 0 or index == len(images):
            complexity = summarize_inference(
                measurements, parameters, fingerprint, args.checkpoint,
                meta.get('epoch'), args.expected_images)
            atomic_json(complexity_path, complexity)
            print(f'{index}/{len(images)} original images; {total_patches} patches; '
                  f'{time.monotonic() - start:.1f}s; '
                  f'mean GFLOPs/patch={complexity["mean_gflops_per_patch"]:.4f}; '
                  f'NIF={complexity["mean_nif_gflops_per_patch"]:.4f}', flush=True)
    archive, counts = pack_results(cache, images, args.output, args.example,
                                   smoke=bool(args.limit), classes=classes)
    filtered = json.loads((args.output / 'filtered_detections.json').read_text())
    complexity = summarize_inference(
        measurements, parameters, fingerprint, args.checkpoint,
        meta.get('epoch'), args.expected_images, complete=not bool(args.limit))
    atomic_json(complexity_path, complexity)
    write_image_csv(args.output / 'complexity_per_image.csv', measurements)
    report = dict(complete=not bool(args.limit), images=len(images), patches=total_patches,
                  dataset=cfg.data.test.type, classes=list(classes),
                  checkpoint=str(args.checkpoint.resolve()), epoch=meta.get('epoch'),
                  trainval_verified=args.require_trainval,
                  zip=str(archive.resolve()), zip_sha256=file_sha256(archive),
                  detections=counts, local_test_mAP=None,
                  filtered_detections=filtered['count'],
                  complexity=dict(
                      report=str(complexity_path.resolve()), parameters=parameters['total'],
                      nif_parameters=parameters['nif'],
                      mean_gflops_per_patch=complexity['mean_gflops_per_patch'],
                      mean_nif_gflops_per_patch=complexity['mean_nif_gflops_per_patch'],
                      mean_refined_rois_per_patch=complexity['mean_refined_rois_per_patch']),
                  note='Official test has no labels. Submit each epoch ZIP to the DOTA server.')
    if args.require_trainval:
        report['training_batch'] = dict(
            world_size=args.expected_world_size, samples_per_gpu=2,
            global_batch=2 * args.expected_world_size)
    atomic_json(args.output / 'report.json', report)
    # A currently running run_nif_paper.py already launches a fresh copy of
    # this script for each epoch. Keep aggregation here so it works without
    # restarting that parent process or changing an ongoing training run.
    epochs = tuple(range(args.expected_max_epochs - 4, args.expected_max_epochs + 1))
    if (not args.limit and meta.get('epoch') in epochs
            and args.output.name == f'epoch_{meta["epoch"]}'):
        aggregate = update_epoch_summary(args.output.parent, epochs)
        print('Complexity epochs available: '
              f'{aggregate["available_epochs"]}; five-checkpoint mean: '
              f'{aggregate["five_checkpoint_mean"]}', flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
