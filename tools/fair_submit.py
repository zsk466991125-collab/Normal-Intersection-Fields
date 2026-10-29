#!/usr/bin/env python3
"""Final-epoch FAIR inference on native FAIR patches; export test/<id>.xml."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector
from mmrotate.core import merge_results, obb2poly_np
from mmrotate.datasets import FairDataset
from mmrotate.models import build_detector
from data.fair.dota_to_fair import dota_to_fair
from tools.analysis_tools.nif_inference_complexity import (
    InferenceComplexityCounter, atomic_json, parameter_inventory, summarize_inference,
    cached_complexity, write_image_csv)

CLASSES = FairDataset.CLASSES


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(cfg):
    folder = Path(cfg.data.test.img_prefix)
    manifest_path = folder.parent / 'test_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest['dataset'] != 'FAIR1M-1.0' or manifest['split'] != 'test'
            or manifest['protocol'] != dict(cfg.fair_preprocessing)):
        raise ValueError('FAIR preprocessing differs from the test manifest')
    entries = manifest['entries']
    if not entries or len(entries) != manifest['images']:
        raise ValueError('Empty or incomplete FAIR test manifest')
    if len({e['id'] for e in entries}) != len(entries):
        raise ValueError('Duplicate FAIR original image IDs')
    groups = {entry['converted_id']: [] for entry in entries}
    for path in folder.glob('*.png'):
        match = re.fullmatch(r'(P\d+)__(\d+)__(\d+)___(\d+)', path.stem)
        if not match or match[1] not in groups:
            raise ValueError(f'Unexpected FAIR patch: {path}')
        groups[match[1]].append((path, int(match[3]), int(match[4])))
    if sum(map(len, groups.values())) != manifest['patches'] or not all(groups.values()):
        raise ValueError('Missing FAIR test patches')
    # Validate exact native window IDs for every original, not only total count.
    import importlib.util
    spec = importlib.util.spec_from_file_location('native_fair_split', REPO / 'data/fair/img_split.py')
    sys.path.insert(0, str(REPO / 'data/fair'))
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    protocol = manifest['protocol']
    sizes = [int(s / rate) for rate in protocol['rates'] for s in protocol['sizes']]
    gaps = [int(g / rate) for rate in protocol['rates'] for g in protocol['gaps']]
    for entry in entries:
        windows = native.get_sliding_window(entry, sizes, gaps, protocol['img_rate_thr'])
        expected = {f"{entry['converted_id']}__{x2-x1}__{x1}___{y1}.png"
                    for x1, y1, x2, y2 in windows}
        group = groups[entry['converted_id']]
        if {p.name for p, _, _ in group} != expected:
            raise ValueError(f'Incorrect native windows for {entry["id"]}')
        group.sort(key=lambda value: value[0].name)
    return manifest_path, manifest, groups


def validate_xml_zip(path, entries):
    metadata = {entry['id']: entry for entry in entries}
    expected = {f'test/{stem}.xml' for stem in metadata}
    count = 0
    official_names = {name.replace('_', ' ') for name in CLASSES}
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
            raise ValueError('Submission must contain exactly one test/<original-id>.xml per image')
        for name in archive.namelist():
            entry = metadata[Path(name).stem]
            root = ET.fromstring(archive.read(name))
            if root.tag != 'annotation' or root.findtext('source/filename') != entry['filename']:
                raise ValueError(f'{name}: wrong original filename')
            for axis in ('width', 'height'):
                if int(root.findtext(f'size/{axis}')) != entry[axis]:
                    raise ValueError(f'{name}: wrong image dimensions')
            if root.find('objects') is None:
                raise ValueError(f'{name}: missing objects element')
            for obj in root.findall('objects/object'):
                if obj.findtext('possibleresult/name') not in official_names:
                    raise ValueError(f'{name}: wrong official class name')
                score = float(obj.findtext('possibleresult/probability'))
                points = [tuple(map(float, p.text.split(','))) for p in obj.findall('points/point')]
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f'{name}: invalid probability')
                if (len(points) != 5 or any(len(p) != 2 for p in points)
                        or points[0] != points[-1] or not np.isfinite(points).all()
                        or abs(cv2.contourArea(np.asarray(points[:4], np.float32))) <= 0):
                    raise ValueError(f'{name}: invalid closed quadrilateral')
                count += 1
    return count


def export_results(output, cache, entries, groups, image_dir, smoke=False):
    task1 = output / 'Task1'
    task1.mkdir(exist_ok=True)
    maximum, filtered, above_one = 1., 0, 0
    for entry in entries:
        results = mmcv.load(str(cache / f'{entry["id"]}.pkl'))['results']
        for values in results:
            if len(values):
                if not np.isfinite(values).all() or np.any(values[:, -1] < 0):
                    raise ValueError('Nonfinite prediction or negative score')
                maximum = max(maximum, float(values[:, -1].max()))
                above_one += int((values[:, -1] > 1).sum())
    streams = [(task1 / f'Task1_{name}.txt').open('w') for name in CLASSES]
    try:
        for entry in entries:
            results = mmcv.load(str(cache / f'{entry["id"]}.pkl'))['results']
            if len(results) != len(CLASSES):
                raise ValueError('Prediction class count must be 37')
            for class_index, values in enumerate(results):
                if not len(values):
                    continue
                for box in obb2poly_np(values, 'le90'):
                    coordinates = np.round(box[:8].astype(np.float64), 6)
                    if abs(cv2.contourArea(coordinates.astype(np.float32).reshape(4, 2))) <= 0:
                        filtered += 1
                        continue
                    row = [entry['converted_id'], f'{float(box[-1]) / maximum:.9f}']
                    row.extend(f'{float(value):.6f}' for value in coordinates)
                    streams[class_index].write(' '.join(row) + '\n')
    finally:
        for stream in streams:
            stream.close()
    # The official converter discovers original IDs from prepared patch names.
    # For smoke runs use an isolated listing, so the result cannot look complete.
    listing = image_dir
    if smoke:
        listing = output / 'smoke_image_listing'
        listing.mkdir(exist_ok=True)
        for old in listing.glob('*.png'):
            old.unlink()
        for entry in entries:
            (listing / groups[entry['converted_id']][0][0].name).touch()
    xml_dir = output / ('smoke_test' if smoke else 'test')
    xml_dir.mkdir(exist_ok=True)
    expected_xmls = {entry['id'] + '.xml' for entry in entries}
    if any(p.name not in expected_xmls for p in xml_dir.iterdir()):
        raise ValueError(f'Unexpected existing files in {xml_dir}')
    dota_to_fair(str(task1), str(xml_dir), str(listing),
                 image_metadata={entry['converted_id']: entry for entry in entries})
    archive = output / ('SMOKE_NOT_FOR_SUBMISSION.zip' if smoke else 'test.zip')
    temporary = archive.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for entry in entries:
            name = entry['id'] + '.xml'
            zipped.write(xml_dir / name, f'test/{name}')
    count = validate_xml_zip(temporary, entries)
    temporary.replace(archive)
    atomic_json(output / 'score_normalization.json', dict(divisor=maximum,
                scores_above_one=above_one, method='global_monotonic_division'))
    return archive, count, filtered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--inference-epoch', type=int,
                        help='checkpoint epoch to export; required for a non-final epoch')
    parser.add_argument('--expected-epoch', type=int, help='must equal the selected inference epoch')
    parser.add_argument('--expected-world-size', type=int)
    parser.add_argument('--limit', type=int, default=0, help='smoke only; never writes test.zip')
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error('batch-size must be positive and limit nonnegative')
    cfg = Config.fromfile(str(args.config))
    final_epoch = cfg.runner.max_epochs
    if final_epoch not in (12, 36):
        raise ValueError('FAIR submission requires a 12- or 36-epoch config')
    inference_epoch = args.inference_epoch if args.inference_epoch is not None else final_epoch
    if inference_epoch < 1 or inference_epoch > final_epoch:
        raise ValueError(f'inference epoch must be between 1 and {final_epoch}')
    if args.inference_epoch is None and inference_epoch != final_epoch:
        raise ValueError('FAIR submission must use the final epoch configured for training')
    if args.expected_epoch is not None and args.expected_epoch != inference_epoch:
        if args.inference_epoch is None:
            raise ValueError('FAIR submission must use the final epoch configured for training')
        raise ValueError('expected epoch does not match inference epoch')
    if (cfg.data.train.type != 'FairDataset' or cfg.data.test.type != 'FairDataset'
            or cfg.model.roi_head.bbox_head.num_classes != 37):
        raise ValueError('Use a FAIR config with all 37 classes')
    manifest_path, manifest, groups = load_manifest(cfg)
    entries = manifest['entries'][:args.limit] if args.limit else manifest['entries']
    fingerprint = dict(checkpoint=sha256(args.checkpoint), config=hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
                       manifest=sha256(manifest_path), batch_size=args.batch_size,
                       converter=sha256(REPO / 'data/fair/dota_to_fair.py'),
                       exporter=sha256(__file__), device=args.device, limit=args.limit,
                       inference_epoch=inference_epoch)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = args.output / 'provenance.json'
        if state_path.exists():
            if json.loads(state_path.read_text()) != fingerprint:
                raise ValueError('Output belongs to different weights/config/data; use a fresh directory')
        elif any(p.name != 'export.lock' for p in args.output.iterdir()):
            raise ValueError('Nonempty output has no FAIR export provenance')
        atomic_json(state_path, fingerprint)
        cfg.model.backbone.init_cfg = None
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        model = build_detector(cfg.model)
        checkpoint = load_checkpoint(model, str(args.checkpoint), map_location='cpu', strict=True)
        meta = checkpoint.get('meta', {})
        if meta.get('epoch') != inference_epoch:
            raise ValueError(f'Expected epoch {inference_epoch}, found {meta.get("epoch")}')
        trained = Config.fromstring(meta.get('config', ''), '.py')
        if (trained.data.train.type != 'FairDataset' or trained.runner.max_epochs != final_epoch
                or trained.data.samples_per_gpu != 2 or trained.model.roi_head.bbox_head.num_classes != 37
                or dict(trained.fair_preprocessing) != dict(cfg.fair_preprocessing)
                or tuple(meta.get('CLASSES', ())) != CLASSES):
            raise ValueError('Checkpoint training protocol or FAIR class order differs')
        if args.expected_world_size is not None and len(trained.gpu_ids) != args.expected_world_size:
            raise ValueError('Checkpoint training GPU count differs')
        model.CLASSES, model.cfg = CLASSES, cfg
        model.to(args.device).eval()
        cache = args.output / 'cache'
        cache.mkdir(exist_ok=True)
        measurements = {}
        for index, entry in enumerate(entries, 1):
            cached = cache / f'{entry["id"]}.pkl'
            if not cached.is_file():
                group = groups[entry['converted_id']]
                results = []
                with torch.no_grad(), InferenceComplexityCounter(model) as counter:
                    for start in range(0, len(group), args.batch_size):
                        paths = [str(p) for p, _, _ in group[start:start + args.batch_size]]
                        results.extend(inference_detector(model, paths))
                    measurement = counter.values(len(group))
                # Native FAIR filenames encode the ORIGINAL-image crop size
                # (2048, 1024, 682 for rates 0.5, 1.0, 1.5), not a resize ratio.
                # inference_detector(..., rescale=True) already returns boxes
                # in each crop's original pixels. Only translate by offsets;
                # another scale division would corrupt multi-scale geometry.
                offsets = np.asarray([(x, y) for _, x, y in group])
                merged = merge_results(results, offsets, (entry['width'], entry['height']),
                                       iou_thr=.1, device=args.device)
                temporary = cached.with_suffix('.partial')
                mmcv.dump(dict(results=merged, complexity=measurement, patches=len(group)),
                          str(temporary), file_format='pkl')
                temporary.replace(cached)
            payload = mmcv.load(str(cached))
            measurement = cached_complexity(payload)
            if measurement is None or payload['patches'] != len(groups[entry['converted_id']]):
                raise ValueError(f'Invalid FAIR inference cache: {cached}')
            measurements[entry['id']] = measurement
            if index % 50 == 0 or index == len(entries):
                print(f'{index}/{len(entries)} original images; '
                      f'{sum(m["patches"] for m in measurements.values())} patches', flush=True)
        archive, detections, filtered = export_results(args.output, cache, entries, groups,
                                                      Path(cfg.data.test.img_prefix), bool(args.limit))
        report = dict(dataset='FAIR1M-1.0', images=len(entries), expected_images=manifest['images'],
                      preprocessing=dict(cfg.fair_preprocessing), patches=manifest['patches'],
                      complete=not bool(args.limit), epoch=inference_epoch, archive=str(archive.resolve()),
                      detections=detections, filtered_degenerate=filtered, classes=list(CLASSES),
                      scored_classes=34, excluded_from_official_map=['other-airplane', 'other-ship', 'other-vehicle'])
        complexity_manifest = dict(image_ids=[entry['id'] for entry in entries], smoke=bool(args.limit),
                                   checkpoint_sha256=fingerprint['checkpoint'], config_sha256=fingerprint['config'],
                                   preprocessing=dict(cfg.fair_preprocessing), dataset='FAIR1M-1.0',
                                   inference_epoch=inference_epoch)
        complexity = summarize_inference(measurements, parameter_inventory(model), complexity_manifest,
                                         args.checkpoint, inference_epoch, manifest['images'], not bool(args.limit))
        atomic_json(args.output / 'complexity.json', complexity)
        write_image_csv(args.output / 'complexity_images.csv', measurements)
        report['complexity'] = str(args.output / 'complexity.json')
        atomic_json(args.output / 'report.json', report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
