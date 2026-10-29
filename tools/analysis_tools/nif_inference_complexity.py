"""Measure supported-module FLOPs during submission inference and average epochs.

No extra model forward or architecture-specific FLOPs formula is needed. Per
original-image measurements are cached with predictions for resumable totals.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from tools.analysis_tools.nif_complexity import (
    EXCLUDED, ModuleCounter, group_name, parameter_counts)

SCHEMA_VERSION = 1
EPOCHS = tuple(range(12, 17))
CONVENTION = 'MMCV 1.7 supported-module forward FLOPs; 1 multiply-add = 1 FLOP'


def protocol_hash(manifest):
    protocol = {key: value for key, value in manifest.items() if key != 'checkpoint_sha256'}
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def parameter_inventory(model):
    groups = Counter()
    for name, parameter in model.named_parameters():
        groups[group_name(name)] += parameter.numel()
    return dict(**parameter_counts(model), components=dict(groups),
                nif=sum(value for key, value in groups.items() if key.startswith('nif_')))


class InferenceComplexityCounter:
    """Observe one original image's actual batched patch forwards, unmodified."""

    def __init__(self, model):
        if model.training:
            raise ValueError('Inference complexity requires model.eval()')
        self.model = model
        self.counter = ModuleCounter(model)
        self.handles = []
        self.input_shapes = Counter()
        self.rpn_rois = 0
        self.refined_rois = 0

    def _backbone(self, module, inputs):
        image = inputs[0]
        key = 'x'.join(str(d) for d in image.shape[1:])
        self.input_shapes[key] += int(image.shape[0])

    def _coarse(self, module, inputs):
        self.rpn_rois += len(inputs[1])

    def _nif(self, module, inputs):
        self.refined_rois += len(inputs[1])

    def __enter__(self):
        self.counter.__enter__()
        self.handles.append(self.model.backbone.register_forward_pre_hook(self._backbone))
        # Two-stage NIF models expose proposal and refinement hooks through
        # ``roi_head``.  Single-stage carriers (for example rotated RTMDet)
        # have no RoI path; their base/refinement counts are correctly zero
        # and their dense-head FLOPs are already covered by ModuleCounter.
        roi_head = getattr(self.model, 'roi_head', None)
        roi_extractor = getattr(roi_head, 'bbox_roi_extractor', None)
        if roi_extractor is not None:
            self.handles.append(roi_extractor.register_forward_pre_hook(self._coarse))
        nif_head = getattr(roi_head, 'nif_head', None)
        if nif_head is not None:
            self.handles.append(nif_head.register_forward_pre_hook(self._nif))
        return self

    def values(self, patches):
        if sum(self.input_shapes.values()) != patches:
            raise ValueError('Measured patch forwards differ from the single-scale window count')
        measured = self.counter.values()
        result = dict(schema_version=SCHEMA_VERSION, patches=patches,
                      flops=measured['flops'], components=measured['components'],
                      input_shapes=dict(self.input_shapes), rpn_rois=self.rpn_rois,
                      refined_rois=self.refined_rois)
        validate_measurement(result, patches)
        return result

    def __exit__(self, *error):
        for handle in self.handles:
            handle.remove()
        self.counter.__exit__(*error)


def validate_measurement(value, patches):
    if value.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported inference complexity cache schema')
    numbers = [value['patches'], value['flops'], value['rpn_rois'], value['refined_rois'],
               *value['components'].values(), *value['input_shapes'].values()]
    if any(not isinstance(n, int) or n < 0 for n in numbers):
        raise ValueError('Invalid cached complexity counts')
    if (patches < 1 or value['patches'] != patches
            or sum(value['input_shapes'].values()) != patches
            or sum(value['components'].values()) != value['flops']
            or value['refined_rois'] > value['rpn_rois']):
        raise ValueError('Inconsistent cached complexity totals')


def cached_complexity(cached):
    """Legacy prediction-only caches must be measured, not assigned zero FLOPs."""
    value = cached.get('complexity')
    if value is None or value.get('schema_version') != SCHEMA_VERSION:
        return None
    validate_measurement(value, cached['patches'])
    return value


def summarize_inference(measurements, parameters, manifest, checkpoint, epoch,
                        expected_images, complete=False):
    if not measurements:
        raise ValueError('Cannot summarize empty inference')
    components, shapes = Counter(), Counter()
    patches = rpn_rois = refined_rois = 0
    for value in measurements.values():
        validate_measurement(value, value['patches'])
        components.update(value['components'])
        shapes.update(value['input_shapes'])
        patches += value['patches']
        rpn_rois += value['rpn_rois']
        refined_rois += value['refined_rois']
    if not set(measurements).issubset(manifest['image_ids']):
        raise ValueError('Measurements contain unknown original image IDs')
    if complete and (len(measurements) != expected_images or manifest['smoke']
                     or set(measurements) != set(manifest['image_ids'])):
        raise ValueError('Cannot mark a partial/smoke inference complete')
    total = sum(components.values())
    nif = sum(value for key, value in components.items() if key.startswith('nif_'))
    return dict(schema_version=SCHEMA_VERSION, complete=complete, smoke=manifest['smoke'],
                epoch=epoch, checkpoint=str(Path(checkpoint).resolve()),
                checkpoint_sha256=manifest['checkpoint_sha256'],
                config_sha256=manifest['config_sha256'],
                protocol_sha256=protocol_hash(manifest),
                convention=CONVENTION, excluded=EXCLUDED,
                parameters=parameters['total'], trainable_parameters=parameters['trainable'],
                nif_parameters=parameters['nif'], parameter_components=parameters['components'],
                images=len(measurements), expected_images=expected_images, patches=patches,
                input_shapes=dict(shapes), total_flops=total, component_flops=dict(components),
                mean_gflops_per_patch=total / patches / 1e9,
                mean_gflops_per_original=total / len(measurements) / 1e9,
                mean_nif_gflops_per_patch=nif / patches / 1e9,
                mean_base_gflops_per_patch=(total - nif) / patches / 1e9,
                mean_component_gflops_per_patch={key: value / patches / 1e9
                                                 for key, value in components.items()},
                total_rpn_rois=rpn_rois, total_refined_rois=refined_rois,
                mean_rpn_rois_per_patch=rpn_rois / patches,
                mean_refined_rois_per_patch=refined_rois / patches,
                method='Native submission forward hooks; no extra model forward; includes all zero-refinement patches')


def write_image_csv(path, measurements):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_id', 'patches', 'rpn_rois', 'refined_rois', 'total_gflops',
                         'mean_gflops_per_patch', 'nif_gflops'])
        for image_id, value in sorted(measurements.items()):
            nif = sum(n for key, n in value['components'].items() if key.startswith('nif_'))
            writer.writerow([image_id, value['patches'], value['rpn_rois'], value['refined_rois'],
                             value['flops'] / 1e9, value['flops'] / value['patches'] / 1e9, nif / 1e9])
    temporary.replace(path)


def update_epoch_summary(root, epochs=None):
    """Called by each fresh test subprocess, including already-running launchers.

    Only validated complete epochs qualify (12..16 by default for legacy
    callers). Average all five checkpoint-level means, never a partial set.
    """
    root = Path(root)
    epochs = tuple(EPOCHS if epochs is None else epochs)
    if (len(epochs) != 5 or any(not isinstance(epoch, int) or epoch < 1 for epoch in epochs)
            or epochs != tuple(range(epochs[0], epochs[0] + 5))):
        raise ValueError('epochs must contain five consecutive positive integers')
    summary_tag = f'{epochs[0]}_{epochs[-1]}'
    rows = []
    reference = None
    for epoch in epochs:
        folder = root / f'epoch_{epoch}'
        paths = [folder / name for name in ('complexity.json', 'report.json', 'inference_manifest.json')]
        if not all(path.is_file() for path in paths):
            continue
        row, submission, manifest = [json.loads(path.read_text()) for path in paths]
        if not row['complete'] or not submission['complete'] or row['smoke']:
            continue
        if (row['schema_version'] != SCHEMA_VERSION or row['convention'] != CONVENTION
                or row['epoch'] != epoch or submission['epoch'] != epoch
                or row['images'] != row['expected_images'] or row['images'] != 937
                or row['images'] != submission['images'] or row['patches'] != submission['patches']
                or row['patches'] < 1
                or row['protocol_sha256'] != protocol_hash(manifest)
                or row['config_sha256'] != manifest['config_sha256']
                or row['checkpoint_sha256'] != manifest['checkpoint_sha256']
                or row['checkpoint'] != submission['checkpoint']):
            raise ValueError(f'Invalid complete complexity report: {folder}')
        signature = (row['protocol_sha256'], row['parameters'], row['nif_parameters'],
                     row['patches'], row['images'], row['input_shapes'], row['excluded'])
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError('Cannot average checkpoints with different models, data or inference settings')
        rows.append({key: row[key] for key in (
            'epoch', 'checkpoint', 'checkpoint_sha256', 'parameters', 'nif_parameters',
            'images', 'patches', 'mean_refined_rois_per_patch', 'mean_base_gflops_per_patch',
            'mean_nif_gflops_per_patch', 'mean_gflops_per_patch', 'mean_gflops_per_original')})
    missing = [epoch for epoch in epochs if epoch not in [row['epoch'] for row in rows]]
    means = None
    if not missing:
        fields = ['parameters', 'nif_parameters', 'mean_refined_rois_per_patch',
                  'mean_base_gflops_per_patch', 'mean_nif_gflops_per_patch',
                  'mean_gflops_per_patch', 'mean_gflops_per_original']
        means = {key: sum(row[key] for row in rows) / len(epochs) for key in fields}
    report = dict(schema_version=SCHEMA_VERSION, complete=not missing,
                  expected_epochs=list(epochs), available_epochs=[row['epoch'] for row in rows],
                  missing_epochs=missing, convention=CONVENTION, excluded=EXCLUDED,
                  averaging=(f'For each checkpoint: total FLOPs / total patches; then arithmetic mean '
                            f'across epochs {epochs[0]}..{epochs[-1]}'),
                  epochs=rows, five_checkpoint_mean=means)
    atomic_json(root / f'complexity_epochs_{summary_tag}.json', report)
    csv_path = root / f'complexity_epochs_{summary_tag}.csv'
    temporary = csv_path.with_suffix('.csv.partial')
    columns = ['epoch', 'parameters', 'nif_parameters', 'mean_refined_rois_per_patch',
               'mean_base_gflops_per_patch', 'mean_nif_gflops_per_patch', 'mean_gflops_per_patch']
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
        if means is not None:
            writer.writerow(dict(epoch=f'mean_{summary_tag}', **means))
    temporary.replace(csv_path)
    lines = [f'# 第{epochs[0]}～{epochs[-1]}轮测试：参数量与平均计算量', '',
             f'每轮先按全部切片平均，再对{len(epochs)}轮的均值做算术平均。', '',
             '| 轮次 | 参数 M | NIF参数 M | 平均精炼框数/切片 | NIF GFLOPs/切片 | 总GFLOPs/切片 |',
             '|---|---:|---:|---:|---:|---:|']
    display = rows + ([dict(epoch='五轮平均', **means)] if means is not None else [])
    for row in display:
        lines.append(f"| {row['epoch']} | {row['parameters']/1e6:.6f} | {row['nif_parameters']/1e6:.6f} | "
                     f"{row['mean_refined_rois_per_patch']:.4f} | {row['mean_nif_gflops_per_patch']:.4f} | "
                     f"{row['mean_gflops_per_patch']:.4f} |")
    if missing:
        lines += ['', f'等待轮次：{missing}。五轮未齐全，暂不输出最终平均值。']
    lines += ['', '一次乘加计1 FLOP；包含冻结参数，不含buffer。',
              'GFLOPs来自实际测试前向的支持算子计数；不含grid_sample、旋转RoIAlign、解析几何、NMS及部分函数式运算，不代表耗时。',
              '每轮详情及分模块计算量见epoch_*/complexity.json，逐原图统计见complexity_per_image.csv。', '']
    markdown = root / f'complexity_epochs_{summary_tag}.md'
    temporary = markdown.with_suffix('.md.partial')
    temporary.write_text('\n'.join(lines))
    temporary.replace(markdown)
    return report
