# Copyright (c) DA-OBB project.
# DA-OBB: Distribution-Annealed training for Oriented Object Detection.
# Components:
#   1. VMFAnnealedLoss  - instance-adaptive, training-annealed von Mises angle supervision
#   2. AnnealingHook    - drives the global annealing progress each iteration
#   3. DARotatedShared2FCBBoxHead - RotatedShared2FCBBoxHead + vmf angle term on decoded boxes
"""Implementation of the DA-OBB training framework (prototype)."""
import math

import torch
import torch.nn as nn

from mmcv.runner import HOOKS, Hook
from mmdet.models.builder import LOSSES, HEADS, build_loss
from mmrotate.models.roi_heads.bbox_heads.convfc_rbbox_head import \
    RotatedShared2FCBBoxHead

# ---------------------------------------------------------------------------
# Global annealing state (single source of truth for all DA components).
# progress: 0.0 (start of training, broad/forgiving supervision)
#        -> 1.0 (end of training, sharp supervision for unambiguous instances)
# ---------------------------------------------------------------------------
ANNEALING = dict(progress=1.0)


def bbox_ambiguity(bboxes, w_shape=0.5, w_density=0.0, w_entropy=0.0,
                   density=None, entropy=None):
    """Per-instance geometric ambiguity score A_i in [0, 1].

    Prototype uses the shape factor only:
        squareness = min(w, h) / max(w, h)  ->  1.0 for a perfect square
    (angle of a square-like box is geometrically unidentifiable).

    Args:
        bboxes (Tensor): (N, >=5) boxes in (cx, cy, w, h, theta) format,
            theta in radians.
        density (Tensor, optional): local instance density, reserved.
        entropy (Tensor, optional): predicted angle-posterior entropy, reserved.

    Returns:
        Tensor: (N,) ambiguity in [0, 1].
    """
    w = bboxes[:, 2].clamp(min=1e-3)
    h = bboxes[:, 3].clamp(min=1e-3)
    squareness = torch.minimum(w, h) / torch.maximum(w, h)
    a = w_shape * squareness
    if w_density > 0 and density is not None:
        a = a + w_density * density
    if w_entropy > 0 and entropy is not None:
        a = a + w_entropy * entropy
    return a.clamp(0.0, 1.0)


@LOSSES.register_module()
class VMFAnnealedLoss(nn.Module):
    """Annealed von Mises angle loss.

    Replaces the point estimate of the angle with a von Mises distribution
    VM(mu_gt; kappa_i(t)) whose concentration is
        kappa_i(t) = kappa_min + (kappa_max - kappa_min) * s(t) * (1 - A_i)
    where s(t) is the global annealing schedule and A_i the per-instance
    ambiguity. The loss is the negative log-likelihood of the target
    distribution evaluated at the predicted angle, up to a constant:
        L_i = -kappa_i(t) * cos(mu_hat_i - mu_gt_i)
    Gradient wrt the predicted angle is kappa_i * sin(...) -- bounded by
    kappa_min at the start of training (no boundary discontinuity), and
    stays soft for ambiguous (square-like) instances.

    Args:
        kappa_min (float): minimum concentration (early training / ambiguous).
        kappa_max (float): maximum concentration (late training / clear).
        schedule (str): 'cosine' | 'linear' (global progress -> schedule value).
        loss_weight (float): overall weight.
    """

    def __init__(self,
                 kappa_min=1.0,
                 kappa_max=50.0,
                 schedule='cosine',
                 loss_weight=1.0,
                 w_shape=0.5,
                 w_density=0.0,
                 w_entropy=0.0):
        super().__init__()
        assert kappa_max > kappa_min > 0
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.schedule = schedule
        self.loss_weight = loss_weight
        self.w_shape = w_shape
        self.w_density = w_density
        self.w_entropy = w_entropy

    def _sched(self, progress):
        if self.schedule == 'cosine':
            return (1.0 - math.cos(math.pi * progress)) / 2.0
        return progress

    def forward(self, pred, target, weight=None, avg_factor=None, **kwargs):
        """pred/target: (N, >=5) decoded boxes (cx, cy, w, h, theta)."""
        theta_p = pred[:, 4]
        theta_t = target[:, 4]
        amb = bbox_ambiguity(
            target,
            w_shape=self.w_shape,
            w_density=self.w_density,
            w_entropy=self.w_entropy)
        progress = self._sched(ANNEALING['progress'])
        kappa = self.kappa_min + (self.kappa_max - self.kappa_min) * \
            progress * (1.0 - amb)
        loss = -kappa * torch.cos(theta_p - theta_t)
        if weight is not None:
            loss = loss * weight
        return self.loss_weight * loss.mean()


@HOOKS.register_module()
class AnnealingHook(Hook):
    """Updates the global annealing progress every iteration.

    Args:
        schedule (str): 'cosine' | 'linear'.
        by_epoch (bool): if True, drive progress by runner.epoch instead of
            iteration (useful if max_iters is unavailable).
    """

    def __init__(self, schedule='cosine', by_epoch=False, interval=50):
        self.schedule = schedule
        self.by_epoch = by_epoch
        self.interval = interval

    def _raw_progress(self, runner):
        if self.by_epoch:
            return runner.epoch / max(runner.max_epochs, 1)
        total = getattr(runner, 'max_iters', None)
        if not total:
            return runner.epoch / max(runner.max_epochs, 1)
        return runner.iter / total

    def after_train_iter(self, runner):
        raw = min(max(self._raw_progress(runner), 0.0), 1.0)
        if self.schedule == 'cosine':
            prog = (1.0 - math.cos(math.pi * raw)) / 2.0
        else:
            prog = raw
        ANNEALING['progress'] = prog
        if runner.every_n_iters(runner, self.interval):
            runner.logger.info(
                f'[DA-OBB] annealing progress={prog:.4f} (raw={raw:.4f})')


@HEADS.register_module()
class DARotatedShared2FCBBoxHead(RotatedShared2FCBBoxHead):
    """RotatedShared2FCBBoxHead + DA-OBB vmf angle loss on decoded boxes.

    The baseline SmoothL1 on encoded deltas is kept unchanged; an additional
    annealed von Mises term is applied to the decoded angle of positive
    samples. Zero inference-time overhead.
    """

    def __init__(self, *args, loss_vmf=None, vmf_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_vmf = build_loss(loss_vmf) if loss_vmf is not None else None
        self.vmf_weight = vmf_weight

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = super().loss(
            cls_score,
            bbox_pred,
            rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            reduction_override=reduction_override)
        if self.loss_vmf is None or bbox_pred is None:
            return losses
        pos_inds = (labels >= 0) & (bbox_weights[:, 0] > 0)
        if pos_inds.any():
            rois_ = rois[pos_inds, 1:]
            if bbox_pred.size(1) > 5:  # class-specific regression
                # use the gt-label channel slice
                cls_per_roi = labels[pos_inds].clamp(max=bbox_pred.size(1) // 5 - 1)
                start = cls_per_roi * 5
                idx = torch.stack(
                    [start + i for i in range(5)], dim=1).to(bbox_pred.device)
                pred_deltas = torch.gather(
                    bbox_pred[pos_inds], 1, idx)
            else:
                pred_deltas = bbox_pred[pos_inds]
            pred_boxes = self.bbox_coder.decode(rois_, pred_deltas)
            gt_boxes = self.bbox_coder.decode(rois_, bbox_targets[pos_inds])
            w = bbox_weights[pos_inds].mean(dim=1)
            loss_vmf = self.loss_vmf(pred_boxes, gt_boxes, weight=w)
            losses['loss_vmf'] = self.vmf_weight * loss_vmf
        else:
            losses['loss_vmf'] = bbox_pred.sum() * 0.0
        return losses
