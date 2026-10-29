#!/usr/bin/env python3
"""Extract FAIR ZIPs in batches and invoke data/fair/img_split.py unchanged.

All XML conversion, P-prefixed naming, crop geometry, annotation handling and
PNG encoding are performed by the repository FAIR scripts. Temporary extracted
TIFFs and converted full PNGs are removed after each committed batch. This script
does not remove source archives. Completed splits remain reusable after source
archives have been separately removed, provided their inventories still match.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from PIL import Image

WORKSPACE = Path(__file__).resolve().parents[2]
FAIR = WORKSPACE / 'data/fair'


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def members(archive, suffix):
    result = {}
    for entry in archive.infolist():
        if entry.is_dir() or not entry.filename.lower().endswith(suffix):
            continue
        stem = Path(entry.filename).stem
        if stem in result:
            raise ValueError(f'Duplicate archive ID {stem}')
        result[stem] = entry
    return result


def prepare_split(source, output, split, mode, batch_size, nproc, min_free_gb, workers=4):
    template = json.loads((FAIR / f'split_configs/{mode}_{split}.json').read_text())
    target = output / ('trainval' if split == 'train' else split)
    target.mkdir(parents=True, exist_ok=True)
    for directory in ('images', 'annfiles', 'batches'):
        (target / directory).mkdir(exist_ok=True)
    protocol = {k: template[k] for k in ('sizes', 'gaps', 'rates', 'img_rate_thr',
                                       'iof_thr', 'no_padding', 'padding_value', 'save_ext')}
    scripts = {name: hashlib.sha256((FAIR / name).read_bytes()).hexdigest()
               for name in ('fair_to_dota.py', 'img_split.py')}
    identity = dict(protocol=protocol, scripts=scripts, batch_size=batch_size)
    state_file = target / 'preparation_protocol.json'
    if state_file.exists() and json.loads(state_file.read_text()) != identity:
        raise ValueError(f'{target}: existing preparation uses a different protocol')
    atomic_json(state_file, identity)
    pairs = [('test/images.zip', None)] if split == 'test' else [
        ('train/part1/images.zip', 'train/part1/labelXml.zip'),
        ('train/part2/images-1.zip', 'train/part2/labelXmls.zip'),
        ('train/part2/images-2.zip', 'train/part2/labelXmls.zip')]
    manifest_path = target / ('nif_manifest.json' if split == 'train' else 'test_manifest.json')
    source_paths = {name for pair in pairs for name in pair if name is not None}
    if manifest_path.exists() and any(not (source / name).is_file() for name in source_paths):
        # A completed split can outlive its ZIPs. Verify its committed inventory
        # before skipping extraction; a partial split still requires its sources.
        manifest = json.loads(manifest_path.read_text())
        batches = [json.loads(path.read_text()) for path in (target / 'batches').glob('*.json')]
        committed = [name for batch in batches for name in batch['patch_files']]
        entries = sorted((entry for batch in batches for entry in batch['entries']),
                         key=lambda entry: int(entry['id']))
        expected_images = set(committed)
        expected_annotations = {Path(name).stem + '.txt' for name in expected_images}
        if (manifest['dataset'] != 'FAIR1M-1.0' or manifest['split'] != split
                or manifest['mode'] != mode or manifest['protocol'] != protocol
                or manifest['processing_scripts'] != scripts
                or manifest['entries'] != entries or manifest['images'] != len(entries)
                or not entries or len({entry['id'] for entry in entries}) != len(entries)
                or manifest['patches'] != len(committed) or len(committed) != len(expected_images)
                or expected_images != {p.name for p in (target / 'images').glob('*.png')}
                or expected_annotations != {p.name for p in (target / 'annfiles').glob('*.txt')}):
            raise ValueError(f'{target}: completed split inventory differs; source archives are required')
        print(f'{split}: reusing completed {mode} split ({len(entries)} originals, '
              f'{len(committed)} patches); source ZIPs were removed separately', flush=True)
        return manifest
    records, seen = [], set()
    for archive_index, (image_path, label_path) in enumerate(pairs):
        with zipfile.ZipFile(source / image_path) as images:
            labels = zipfile.ZipFile(source / label_path) if label_path else None
            try:
                image_members = members(images, '.tif')
                label_members = members(labels, '.xml') if labels else {}
                if labels and not set(image_members) <= set(label_members):
                    raise ValueError(f'Missing XML labels in {label_path}')
                ids = sorted(image_members, key=int)
                if seen.intersection(ids):
                    raise ValueError('Duplicate original image IDs across archives')
                seen.update(ids)
                def process_batch(start):
                    batch_ids = ids[start:start + batch_size]
                    marker = target / 'batches' / f'{archive_index}_{start:06d}.json'
                    source_id = [(stem, image_members[stem].CRC,
                                  label_members[stem].CRC if labels else None) for stem in batch_ids]
                    source_id = json.loads(json.dumps(source_id))
                    if marker.exists():
                        record = json.loads(marker.read_text())
                        if record['source_crc'] != source_id:
                            raise ValueError(f'Archive changed: {marker}')
                        if not all((target / 'images' / p).is_file() and
                                   (target / 'annfiles' / (Path(p).stem + '.txt')).is_file()
                                   for p in record['patch_files']):
                            raise ValueError(f'Missing committed batch files: {marker}')
                    else:
                        if shutil.disk_usage(output).free < min_free_gb * 2**30:
                            raise OSError(f'Less than {min_free_gb} GiB free; rerun after freeing space')
                        with tempfile.TemporaryDirectory(prefix='.fair_batch_', dir=output) as staging:
                            staging = Path(staging)
                            raw = staging / 'raw'
                            (raw / 'images').mkdir(parents=True)
                            if labels:
                                (raw / 'labelXml').mkdir()
                            entries = []
                            for stem in batch_ids:
                                path = raw / 'images' / f'{stem}.tif'
                                path.write_bytes(images.read(image_members[stem]))
                                with Image.open(path) as image:
                                    width, height = image.size
                                entries.append(dict(id=stem, converted_id='P' + stem.zfill(4),
                                                    filename=path.name, width=width, height=height))
                                if labels:
                                    (raw / 'labelXml' / f'{stem}.xml').write_bytes(labels.read(label_members[stem]))
                            cfg = dict(template, nproc=nproc, img_ann_dirs=str(raw),
                                       converted_dota_dir=str(staging / 'converted'),
                                       save_dir=str(staging / 'split'))
                            cfg_path = staging / 'config.json'
                            atomic_json(cfg_path, cfg)
                            log = target / f'batch_{archive_index}_{start:06d}.log'
                            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                            with log.open('w') as stream:
                                command = [sys.executable, '-u', str(FAIR / 'img_split.py'),
                                           '--base-json', str(cfg_path)]
                                subprocess.run(command, cwd=WORKSPACE, env=env,
                                               stdout=stream, stderr=subprocess.STDOUT, check=True)
                            image_stage = staging / 'split/images'
                            patches = sorted(image_stage.glob('*.png'))
                            if not patches:
                                raise ValueError(f'No patches; see {log}')
                            class_counts = Counter()
                            for label in (staging / 'converted/labelTxt').glob('*.txt'):
                                for line in label.read_text().splitlines():
                                    fields = line.split()
                                    if len(fields) == 10:
                                        class_counts[fields[8]] += 1
                            for patch in patches:
                                # Verify successful writes before committing a batch.
                                with Image.open(patch) as image:
                                    image.verify()
                                ann = staging / 'split/annfiles' / (patch.stem + '.txt')
                                if not ann.is_file():
                                    raise ValueError(f'Missing patch annotation {ann}')
                                patch.replace(target / 'images' / patch.name)
                                ann.replace(target / 'annfiles' / ann.name)
                            record = dict(source_crc=source_id, entries=entries,
                                          patch_files=[p.name for p in patches], class_counts=dict(class_counts))
                            atomic_json(marker, record)
                    return record

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for record in pool.map(process_batch, range(0, len(ids), batch_size)):
                        records.append(record)
                        count = sum(len(r['entries']) for r in records)
                        print(f'{split}: {count} originals, {sum(len(r["patch_files"]) for r in records)} patches (official {mode})', flush=True)
            finally:
                if labels:
                    labels.close()
    entries = sorted((entry for r in records for entry in r['entries']), key=lambda e: int(e['id']))
    patches = sum(len(r['patch_files']) for r in records)
    if len(list((target / 'images').glob('*.png'))) != patches:
        raise ValueError(f'{target}: unexpected image files')
    classes = Counter()
    for record in records:
        classes.update(record['class_counts'])
    manifest = dict(dataset='FAIR1M-1.0', split=split, images=len(entries), patches=patches,
                    entries=entries, protocol=protocol, processing_scripts=scripts,
                    classes=dict(classes), objects=sum(classes.values()), image_dir=str(target / 'images'),
                    ann_dir=str(target / 'annfiles'), mode=mode,
                    test_patching='native_fair_img_split' if split == 'test' else None)
    atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=WORKSPACE / 'data/FAIR1M1.0')
    parser.add_argument('--output', type=Path, help='defaults to data/fair1m_<mode>')
    parser.add_argument('--mode', choices=('ss', 'ms'), default='ms')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nproc', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4, help='concurrent native-script batches')
    parser.add_argument('--min-free-gb', type=int, default=8)
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-test', action='store_true')
    args = parser.parse_args()
    if min(args.batch_size, args.nproc, args.min_free_gb, args.workers) < 1:
        parser.error('batch-size, nproc and min-free-gb must be positive')
    output = (args.output or WORKSPACE / 'data/fair').resolve()
    output.mkdir(parents=True, exist_ok=True)
    for split, skip in [('train', args.skip_train), ('test', args.skip_test)]:
        if not skip:
            print(json.dumps(prepare_split(args.source.resolve(), output, split, args.mode,
                                          args.batch_size, args.nproc, args.min_free_gb, args.workers),
                             default=str, ensure_ascii=False)[:500], flush=True)


if __name__ == '__main__':
    main()
