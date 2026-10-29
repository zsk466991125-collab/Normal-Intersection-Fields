"""Evaluate once per epoch and retain independent best-metric checkpoints."""
import copy
import json
import math
from pathlib import Path

from mmdet.core import DistEvalHook, EvalHook


class _MultiMetricMixin:
    """Keep selection state in runner.meta so ordinary checkpoints can resume.

    Register before CheckpointHook: the epoch checkpoint must include the
    current evaluation, not the previous epoch's best scores/history.
    Inference and distributed gathering are inherited from MMDetection.
    """

    def __init__(self, dataloader, save_best, rule='greater', **kwargs):
        if (not isinstance(save_best, (list, tuple)) or not save_best
                or len(set(save_best)) != len(save_best)):
            raise ValueError('save_best must be a nonempty list of distinct metric names')
        if any(not isinstance(key, str) or not key.replace('_', '').isalnum()
               for key in save_best):
            raise ValueError('Best metric names must contain only letters, numbers and underscores')
        if rule != 'greater':
            raise ValueError('Multi-metric evaluation selects the greatest scores')
        if kwargs.get('by_epoch', True) is not True:
            raise ValueError('Multi-metric evaluation requires an epoch-based runner')
        if kwargs.get('out_dir') or kwargs.get('file_client_args'):
            raise ValueError('Multi-metric artifacts are stored in the local runner.work_dir')
        self.best_metrics = tuple(save_best)
        # Disable the parent's single-metric saving; evaluate() saves all winners.
        super().__init__(dataloader, save_best=None, **kwargs)

    def before_run(self, runner):
        super().before_run(runner)
        if runner.meta is None:
            runner.meta = {}
        state = runner.meta.setdefault('hook_msgs', {}).setdefault(
            'multi_metric_eval', dict(best={}, history=[]))
        self._write_reports(runner, state)

    @staticmethod
    def _write_json(path, value):
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    def _write_reports(self, runner, state):
        directory = Path(runner.work_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._write_json(directory / 'eval_history.json', state['history'])
        self._write_json(directory / 'best_metrics.json', dict(
            selection='Highest test score across evaluated epochs; ties keep the earlier epoch',
            selection_metrics=list(self.best_metrics),
            dataset=self.dataloader.dataset.ann_file,
            best=state['best']))

    def evaluate(self, runner, results):
        # Both VOC interpolations consume the SAME predictions, with no second inference.
        values = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        values = {key: float(value) for key, value in values.items()}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError('Evaluation returned a non-finite metric')
        for key in self.best_metrics:
            if key not in values:
                raise KeyError(f'Missing best-checkpoint metric: {key}')
        runner.log_buffer.output.update(values)
        runner.log_buffer.ready = True

        state = runner.meta['hook_msgs']['multi_metric_eval']
        epoch = runner.epoch + 1
        state['history'] = [row for row in state['history'] if row['epoch'] < epoch]
        state['history'].append(dict(epoch=epoch, metrics=values))
        improved = []
        for key in self.best_metrics:
            previous = state['best'].get(key)
            if previous is None or values[key] > previous['score']:
                filename = f'best_{key}_epoch_{epoch}.pth'
                state['best'][key] = dict(
                    epoch=epoch, score=values[key], metrics=copy.deepcopy(values),
                    checkpoint=str(Path(runner.work_dir) / filename))
                improved.append((key, filename, previous))

        # Update all selection metadata before saving either best checkpoint.
        for key, filename, previous in improved:
            runner.save_checkpoint(runner.work_dir, filename_tmpl=filename,
                                   create_symlink=False)
            if previous:
                old = Path(previous['checkpoint'])
                if old.parent == Path(runner.work_dir) and old.name != filename:
                    old.unlink(missing_ok=True)
            runner.logger.info(
                f'Best {key}: {values[key]:.6f} at epoch {epoch}; saved {filename}')
        self._write_reports(runner, state)
        return None

    def after_run(self, runner):
        if runner.rank != 0:
            return
        state = runner.meta['hook_msgs']['multi_metric_eval']
        for key, item in state['best'].items():
            runner.logger.info(
                f'Final best {key}: {item["score"]:.6f}, epoch {item["epoch"]}, '
                f'checkpoint={item["checkpoint"]}')


class MultiMetricEvalHook(_MultiMetricMixin, EvalHook):
    """Single GPU inference with independent best checkpoints."""


class MultiMetricDistEvalHook(_MultiMetricMixin, DistEvalHook):
    """Distributed inference; only rank zero evaluates and saves best checkpoints."""

    def before_run(self, runner):
        # Nonzero ranks participate in inference only and must never write reports.
        if runner.rank == 0:
            super().before_run(runner)
