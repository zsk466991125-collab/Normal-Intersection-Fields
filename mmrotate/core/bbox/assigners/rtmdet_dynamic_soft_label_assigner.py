# Copyright (c) OpenMMLab. All rights reserved.
"""Tensor-based MMDetection 2.x port of RTMDet's rotated soft assignment."""
import torch
import torch.nn.functional as F
from mmdet.core.bbox.assigners import AssignResult, BaseAssigner

from ..builder import ROTATED_BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator


@ROTATED_BBOX_ASSIGNERS.register_module()
class RTMDetDynamicSoftLabelAssigner(BaseAssigner):
    def __init__(self, soft_center_radius=3.0, topk=13, iou_weight=3.0,
                 iou_calculator=dict(type='RBboxOverlaps2D')):
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        self.iou_calculator = build_iou_calculator(iou_calculator)

    @torch.no_grad()
    def assign(self, pred_scores, priors, decoded_bboxes, gt_bboxes,
               gt_labels, gt_bboxes_ignore=None):
        # Like upstream DynamicSoftLabelAssigner, ignored GT are not used.
        count, num_gt = len(priors), len(gt_bboxes)
        gt_inds = priors.new_zeros(count, dtype=torch.long)
        labels = priors.new_full((count,), -1, dtype=torch.long)
        overlaps = priors.new_zeros(count)
        if not count or not num_gt:
            return AssignResult(num_gt, gt_inds, overlaps, labels=labels)

        delta = priors[:, None, :2] - gt_bboxes[None, :, :2]
        angle = gt_bboxes[None, :, 4]
        local_x = delta[..., 0] * angle.cos() + delta[..., 1] * angle.sin()
        local_y = -delta[..., 0] * angle.sin() + delta[..., 1] * angle.cos()
        inside = ((local_x.abs() <= gt_bboxes[None, :, 2] / 2 - 0.01) &
                  (local_y.abs() <= gt_bboxes[None, :, 3] / 2 - 0.01))
        valid = inside.any(dim=1)
        if not valid.any():
            return AssignResult(num_gt, gt_inds, overlaps, labels=labels)

        distance = delta[valid].square().sum(-1).sqrt() / priors[valid, 2:3]
        center_cost = 10 ** (distance - self.soft_center_radius)
        ious = self.iou_calculator(decoded_bboxes[valid], gt_bboxes)
        iou_cost = -(ious + 1e-7).log() * self.iou_weight
        logits = pred_scores[valid, None, :].expand(-1, num_gt, -1)
        soft_labels = F.one_hot(gt_labels, pred_scores.shape[-1]).float()[None]
        soft_labels = soft_labels * ious[..., None]
        cls_cost = (F.binary_cross_entropy_with_logits(
            logits, soft_labels, reduction='none') *
            (soft_labels - logits.sigmoid()).abs().square()).sum(-1)
        cost = cls_cost + iou_cost + center_cost

        matching = torch.zeros_like(cost, dtype=torch.bool)
        top_ious = ious.topk(min(self.topk, len(ious)), dim=0).values
        dynamic_ks = top_ious.sum(0).int().clamp(min=1)
        for gt in range(num_gt):
            matching[cost[:, gt].topk(int(dynamic_ks[gt]), largest=False).indices,
                     gt] = True
        conflicts = matching.sum(1) > 1
        matching[conflicts] = False
        matching[conflicts, cost[conflicts].argmin(1)] = True
        foreground = matching.any(1)
        indices = valid.nonzero(as_tuple=False).squeeze(1)[foreground]
        matched_gt = matching[foreground].long().argmax(1)
        gt_inds[indices] = matched_gt + 1
        labels[indices] = gt_labels[matched_gt]
        overlaps.fill_(-1e8)
        overlaps[indices] = (matching * ious).sum(1)[foreground]
        return AssignResult(num_gt, gt_inds, overlaps, labels=labels)
