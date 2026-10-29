# Copyright (c) OpenMMLab. All rights reserved.
"""Rotated RTMDet SepBN head ported to the MMDetection 2.x tensor API.

The release's convolution sharing, separate BN, exponential distances, raw
radian angles, QFL, rotated IoU and dynamic soft-label assignment are retained.
"""
import torch
from torch import nn
from mmcv import ConfigDict
from mmcv.cnn import (ConvModule, bias_init_with_prob, constant_init, is_norm,
                      normal_init)
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32
from mmdet.core import build_prior_generator, reduce_mean
from mmdet.core.utils import filter_scores_and_topk
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

from mmrotate.core.bbox.builder import build_assigner, build_bbox_coder
from mmrotate.core.bbox.transforms import norm_angle
from ..builder import ROTATED_HEADS, build_loss


@ROTATED_HEADS.register_module()
class RotatedRTMDetSepBNHead(BaseDenseHead):
    """Checkpoint-compatible rotated RTMDet head (pseudo/raw angle coder)."""

    def __init__(self, num_classes, in_channels, feat_channels=256,
                 stacked_convs=2, share_conv=True, exp_on_reg=True,
                 angle_version='le90', with_objectness=False,
                 use_hbbox_loss=False, scale_angle=False, loss_angle=None,
                 pred_kernel_size=1, conv_cfg=None,
                 norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'),
                 anchor_generator=dict(type='MlvlPointGenerator', offset=0,
                                       strides=[8, 16, 32]),
                 bbox_coder=dict(type='DistanceAnglePointCoder',
                                 angle_version='le90'),
                 loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True,
                               beta=2.0, loss_weight=1.0),
                 loss_bbox=dict(type='RotatedIoULoss', mode='linear',
                                loss_weight=2.0),
                 train_cfg=None, test_cfg=None, init_cfg=None):
        super().__init__(init_cfg)
        if with_objectness or use_hbbox_loss or scale_angle or loss_angle:
            raise NotImplementedError('This port implements the DOTA release '
                                      'with raw angles and rotated IoU loss')
        self.num_classes = self.cls_out_channels = num_classes
        self.in_channels, self.feat_channels = in_channels, feat_channels
        self.stacked_convs, self.share_conv = stacked_convs, share_conv
        self.exp_on_reg, self.angle_version = exp_on_reg, angle_version
        self.fp16_enabled = False
        self.prior_generator = build_prior_generator(anchor_generator)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls, self.loss_bbox = build_loss(loss_cls), build_loss(loss_bbox)
        self.train_cfg = ConfigDict(train_cfg or {})
        self.test_cfg = ConfigDict(test_cfg or {})
        if train_cfg is not None:
            self.assigner = build_assigner(train_cfg['assigner'])
            if self.train_cfg.get('allowed_border', -1) != -1:
                raise ValueError('RTMDet point priors require allowed_border=-1')
        self.cls_convs, self.reg_convs = nn.ModuleList(), nn.ModuleList()
        self.rtm_cls, self.rtm_reg, self.rtm_ang = (
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList())
        for level in range(self.prior_generator.num_levels):
            for branch in (self.cls_convs, self.reg_convs):
                branch.append(nn.ModuleList([
                    ConvModule(in_channels if i == 0 else feat_channels,
                               feat_channels, 3, padding=1, conv_cfg=conv_cfg,
                               norm_cfg=norm_cfg, act_cfg=act_cfg)
                    for i in range(stacked_convs)]))
            for branch, channels in ((self.rtm_cls, num_classes),
                                     (self.rtm_reg, 4), (self.rtm_ang, 1)):
                branch.append(nn.Conv2d(feat_channels, channels,
                                        pred_kernel_size,
                                        padding=pred_kernel_size // 2))
        if share_conv:
            for level in range(1, self.prior_generator.num_levels):
                for i in range(stacked_convs):
                    self.cls_convs[level][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[level][i].conv = self.reg_convs[0][i].conv

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                normal_init(module, std=0.01)
            elif is_norm(module):
                constant_init(module, 1)
        for cls in self.rtm_cls:
            normal_init(cls, std=0.01, bias=bias_init_with_prob(0.01))

    def forward(self, feats):
        cls_scores, bbox_preds, angle_preds = [], [], []
        for level, (feature, stride) in enumerate(
                zip(feats, self.prior_generator.strides)):
            cls_feat = reg_feat = feature
            for layer in self.cls_convs[level]:
                cls_feat = layer(cls_feat)
            for layer in self.reg_convs[level]:
                reg_feat = layer(reg_feat)
            distance = self.rtm_reg[level](reg_feat)
            if self.exp_on_reg:
                distance = distance.exp()
            cls_scores.append(self.rtm_cls[level](cls_feat))
            bbox_preds.append(distance * stride[0])
            angle_preds.append(self.rtm_ang[level](reg_feat))
        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds)

    def _flatten(self, cls_scores, bbox_preds, angle_preds):
        batch = cls_scores[0].shape[0]
        sizes = [score.shape[-2:] for score in cls_scores]
        priors = torch.cat(self.prior_generator.grid_priors(
            sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device,
            with_stride=True))
        scores = torch.cat([x.permute(0, 2, 3, 1).reshape(
            batch, -1, self.num_classes) for x in cls_scores], dim=1)
        distances = torch.cat([x.permute(0, 2, 3, 1).reshape(batch, -1, 4)
                               for x in bbox_preds], dim=1)
        angles = torch.cat([x.permute(0, 2, 3, 1).reshape(batch, -1, 1)
                           for x in angle_preds], dim=1)
        points = priors[None, :, :2].expand(batch, -1, -1).reshape(-1, 2)
        boxes = self.bbox_coder.decode(
            points, torch.cat((distances, angles), dim=-1).reshape(-1, 5))
        return scores, boxes.reshape(batch, -1, 5), priors, sizes

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def loss(self, cls_scores, bbox_preds, angle_preds, gt_bboxes, gt_labels,
             img_metas, gt_bboxes_ignore=None):
        # Equivalent to RotatedBoxes.regularize_boxes in the release. Flips
        # and random rotation may leave angles outside the canonical range.
        gt_bboxes = [self._regularize_boxes(boxes) for boxes in gt_bboxes]
        scores, boxes, priors, sizes = self._flatten(
            cls_scores, bbox_preds, angle_preds)
        targets = torch.zeros_like(boxes)
        labels = priors.new_full(scores.shape[:2], self.num_classes,
                                 dtype=torch.long)
        weights = scores.new_zeros(scores.shape[:2])
        metrics = weights.clone()
        for image, meta in enumerate(img_metas):
            valid = torch.cat(self.prior_generator.valid_flags(
                sizes, meta['pad_shape'], device=priors.device))
            assignment = self.assigner.assign(
                scores[image, valid].detach(), priors[valid],
                boxes[image, valid].detach(), gt_bboxes[image], gt_labels[image])
            valid_indices = valid.nonzero(as_tuple=False).squeeze(1)
            positive = assignment.gt_inds > 0
            indices = valid_indices[positive]
            weights[image, valid] = 1
            if len(indices):
                labels[image, indices] = assignment.labels[positive]
                targets[image, indices] = gt_bboxes[image][
                    assignment.gt_inds[positive] - 1]
                metrics[image, indices] = assignment.max_overlaps[positive]
                if self.train_cfg.get('pos_weight', -1) > 0:
                    weights[image, indices] = self.train_cfg.pos_weight
        positive = labels < self.num_classes
        avg_factor = reduce_mean(metrics.sum()).clamp(min=1).item()
        loss_cls = self.loss_cls(
            scores.reshape(-1, self.num_classes),
            (labels.reshape(-1), metrics.reshape(-1)),
            weights.reshape(-1), avg_factor=avg_factor)
        if positive.any():
            loss_bbox = self.loss_bbox(boxes[positive], targets[positive],
                                       weight=metrics[positive],
                                       avg_factor=avg_factor)
        else:
            loss_bbox = boxes.sum() * 0
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def _regularize_boxes(self, boxes):
        if self.angle_version not in ('le90', 'le135'):
            raise NotImplementedError('RTMDet training supports le90/le135')
        center, width, height, angle = boxes.split((2, 1, 1, 1), dim=-1)
        swap = width < height
        return torch.cat((center, torch.where(swap, height, width),
                          torch.where(swap, width, height),
                          norm_angle(angle + swap * (torch.pi / 2),
                                     self.angle_version)), dim=-1)

    @staticmethod
    def _rescale(boxes, scale_factor):
        """Match MMRotate 1.x RotatedBoxes.rescale_, including anisotropy."""
        scale = boxes.new_tensor(scale_factor).flatten()
        sx, sy = 1 / scale[0], 1 / scale[1 if len(scale) > 1 else 0]
        center, width, height, angle = boxes.split((2, 1, 1, 1), dim=-1)
        cosine, sine = angle.cos(), angle.sin()
        return torch.cat((center * torch.stack((sx, sy)),
                          width * ((sx * cosine)**2 + (sy * sine)**2).sqrt(),
                          height * ((sx * sine)**2 + (sy * cosine)**2).sqrt(),
                          torch.atan2(sx * sine, sy * cosine)), dim=-1)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, angle_preds, img_metas,
                   cfg=None, rescale=False, with_nms=True):
        cfg = self.test_cfg if cfg is None else ConfigDict(cfg)
        priors = self.prior_generator.grid_priors(
            [x.shape[-2:] for x in cls_scores], dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)
        results = []
        for image, meta in enumerate(img_metas):
            image_boxes, image_scores, image_labels = [], [], []
            for cls, dist, angle, points in zip(
                    cls_scores, bbox_preds, angle_preds, priors):
                scores = cls[image].permute(1, 2, 0).reshape(
                    -1, self.num_classes).sigmoid()
                distance = dist[image].permute(1, 2, 0).reshape(-1, 4)
                angle = angle[image].permute(1, 2, 0).reshape(-1, 1)
                scores, labels, _, selected = filter_scores_and_topk(
                    scores, cfg.get('score_thr', 0.05), cfg.get('nms_pre', 2000),
                    dict(distance=distance, angle=angle, points=points))
                boxes = self.bbox_coder.decode(selected['points'], torch.cat(
                    (selected['distance'], selected['angle']), dim=-1))
                image_boxes.append(boxes)
                image_scores.append(scores)
                image_labels.append(labels)
            boxes, scores, labels = (torch.cat(x) for x in (
                image_boxes, image_scores, image_labels))
            if rescale:
                boxes = self._rescale(boxes, meta['scale_factor'])
            min_size = cfg.get('min_bbox_size', 0)
            if min_size >= 0:
                valid = (boxes[:, 2:4] > min_size).all(dim=-1)
                boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
            if not with_nms:
                results.append((boxes, scores, labels))
            elif len(boxes):
                dets, keep = batched_nms(boxes, scores, labels, cfg.nms)
                maximum = cfg.get('max_per_img', 2000)
                if maximum > 0:
                    dets, keep = dets[:maximum], keep[:maximum]
                results.append((dets, labels[keep]))
            else:
                results.append((torch.cat((boxes, scores[:, None]), dim=-1),
                                labels))
        return results
