"""Normal-intersection fields for final coarse-box refinement."""
import math

import torch
from mmcv.runner import force_fp32

from mmrotate.core import multiclass_nms_rotated, rbbox2roi
from mmrotate.core.bbox.coder.nif_obb_coder import NIFOBBCoder
from ..builder import ROTATED_HEADS
from ..losses.nif_iou_loss import NIFRotatedIoULoss, exact_aligned_rotated_iou
from .nif_local_head import NIFLocalHead
from .oriented_standard_roi_head import OrientedStandardRoIHead


@ROTATED_HEADS.register_module()
class NIFOrientedStandardRoIHead(OrientedStandardRoIHead):
    """Refine coarse OBBs with 20 observations and a weighted geometric fit."""

    def __init__(self, *args, nif_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.bbox_head.reg_class_agnostic:
            raise ValueError('NIF requires class-agnostic regression')
        if self.bbox_head.bbox_coder.angle_range != 'le90':
            raise ValueError('The two-stage adapter requires le90 boxes')
        cfg = dict(nif_cfg or {})
        self.nif_solver = cfg.pop('solver', 'ols')
        if self.nif_solver != 'ols':
            raise ValueError('NIF uses a weighted least-squares fit')
        self.nif_obs_weight = float(cfg.pop('obs_loss_weight', 1.0))
        self.nif_ref_weight = float(cfg.pop('ref_loss_weight', 1.0))
        self.nif_reliability_power = float(cfg.pop('reliability_power', 1.5))
        self.nif_reliability_loss_weight = float(cfg.pop('reliability_loss_weight', 0.05))
        self.nif_reliability_error_scale = float(cfg.pop('reliability_error_scale', 0.25))
        self.nif_reliability_min = float(cfg.get('reliability_min', 0.05))
        if (not math.isfinite(self.nif_reliability_power)
                or self.nif_reliability_power <= 0
                or self.nif_reliability_error_scale <= 0
                or self.nif_reliability_loss_weight < 0):
            raise ValueError('Invalid reliability settings')
        sampling = dict(cfg.pop('sampling', {}) or {})
        self.nif_sampling_mode = sampling.pop('mode', 'stratified')
        self.nif_max_pos_per_img = int(sampling.pop('max_pos_per_img', 128))
        self.nif_iou_bins = tuple(sampling.pop(
            'iou_bins', (0.3, 0.45, 0.6, 0.75, 0.9, 1.01)))
        self.nif_exclude_gt = bool(sampling.pop('exclude_gt', True))
        self.nif_include_gt_fallback = bool(sampling.pop('include_gt_fallback', True))
        self.nif_strict_iou_range = False
        if sampling:
            raise ValueError(f'Unknown sampling options: {sorted(sampling)}')
        if self.nif_sampling_mode not in ('all', 'stratified'):
            raise ValueError('sampling mode must be all or stratified')
        if (self.nif_max_pos_per_img < 0 or len(self.nif_iou_bins) < 2
                or any(a >= b for a, b in zip(self.nif_iou_bins, self.nif_iou_bins[1:]))):
            raise ValueError('Invalid sampling budget or bins')
        self.nif_coder = NIFOBBCoder(
            num_points=cfg.pop('num_points', 5),
            sample_extent=cfg.pop('sample_extent', 0.75),
            min_size=cfg.pop('min_size', 1e-3))
        cfg.setdefault('projection_channels', 64)
        cfg.setdefault('hidden_channels', 64)
        cfg.setdefault('preserve_normal_position', True)
        cfg.setdefault('reliability_enabled', True)
        if not cfg['reliability_enabled']:
            raise ValueError('NIF requires per-observation reliability')
        self.nif_head = NIFLocalHead(self.nif_coder, **cfg)
        self.nif_box_loss = NIFRotatedIoULoss()
        self.nif_strides = self.bbox_roi_extractor.featmap_strides

    @staticmethod
    def _stratified_indices(ious,
                            eligible,
                            max_count,
                            bins,
                            generator=None):
        """Select proposal indices from IoU strata without changing RCNN sampling.

        The selected proposals are used only by NIF. The normal O-RCNN losses
        still see the complete ``sampling_results`` produced by the sampler.
        """
        candidates = eligible.nonzero(as_tuple=False).flatten()
        if not len(candidates):
            return candidates
        if max_count <= 0 or max_count >= len(candidates):
            return candidates
        selected = []
        quota = max(1, int((max_count + len(bins) - 2) /
                           (len(bins) - 1)))
        for left, right in zip(bins, bins[1:]):
            in_bin = candidates[(ious[candidates] >= left)
                                & (ious[candidates] < right)]
            if len(in_bin):
                order = torch.randperm(len(in_bin), device=in_bin.device,
                                       generator=generator)
                selected.append(in_bin[order[:quota]])
        if selected:
            selected = torch.cat(selected)
            selected = torch.unique(selected, sorted=False)
        else:
            selected = candidates[:0]
        if len(selected) > max_count:
            order = torch.randperm(len(selected), device=selected.device,
                                   generator=generator)
            selected = selected[order[:max_count]]
        if len(selected) < max_count:
            used = torch.zeros_like(eligible)
            used[selected] = True
            remaining = candidates[~used[candidates]]
            order = torch.randperm(len(remaining), device=remaining.device,
                                   generator=generator)
            selected = torch.cat((selected, remaining[order[:max_count -
                                                            len(selected)]]))
        return selected

    def _select_nif_samples(self, sampling_results, bbox_results):
        """Build a train-time NIF batch closer to inference candidates.

        Exact GT-added proposals can make the local field task unrealistically
        easy. The sampler can exclude them and fill a fixed budget across coarse IoU
        strata while leaving the baseline RCNN sampler untouched.
        """
        pred = bbox_results['bbox_pred']
        selected_boxes, selected_gt = [], []
        selected_indices, selected_coarse = [], []
        # ``SamplingResult.bboxes`` is the concatenation of positive and
        # negative boxes (the same order used by ``rbbox2roi`` below).
        # Positive entries therefore form the contiguous prefix of each
        # per-image RoI block.
        start = 0
        with torch.no_grad():
            for res in sampling_results:
                count = len(res.pos_bboxes)
                if not count:
                    selected_boxes.append(res.pos_bboxes)
                    # MMDetection may represent no-GT targets as shape (0,).
                    # Keep rotated-box shape when another image has positives.
                    selected_gt.append(res.pos_bboxes.new_empty((0, 5)))
                    start += len(res.bboxes)
                    continue
                global_indices = torch.arange(
                    start, start + count, device=pred.device)
                if global_indices.numel() and global_indices.max() >= len(pred):
                    raise RuntimeError(
                        f'SamplingResult/prediction misalignment: '
                        f'max index {int(global_indices.max())}, pred size {len(pred)}')
                coarse_all = self.bbox_head.bbox_coder.decode(
                    res.pos_bboxes.float(), pred[global_indices].float())
                ious = exact_aligned_rotated_iou(
                    coarse_all, res.pos_gt_bboxes.float())
                is_gt = getattr(res, 'pos_is_gt', None)
                if is_gt is None:
                    is_gt = torch.zeros(count, dtype=torch.bool,
                                        device=pred.device)
                else:
                    is_gt = is_gt.to(device=pred.device, dtype=torch.bool)
                eligible = ~is_gt if self.nif_exclude_gt else torch.ones_like(
                    is_gt)
                if not eligible.any() and self.nif_include_gt_fallback:
                    eligible = torch.ones_like(is_gt)
                if self.nif_strict_iou_range:
                    # The lower endpoint is inclusive and the upper endpoint
                    # is exclusive, so the strict sampler omits the upper tail.
                    eligible = eligible & (ious >= self.nif_iou_bins[0]) & (
                        ious < self.nif_iou_bins[-1])
                if self.nif_sampling_mode == 'stratified':
                    local = self._stratified_indices(
                        ious, eligible, self.nif_max_pos_per_img,
                        self.nif_iou_bins)
                else:
                    local = eligible.nonzero(as_tuple=False).flatten()
                selected_boxes.append(res.pos_bboxes[local])
                selected_gt.append(res.pos_gt_bboxes[local])
                selected_indices.append(global_indices[local])
                selected_coarse.append(coarse_all[local])
                start += len(res.bboxes)
        if selected_indices:
            return (torch.cat(selected_indices), torch.cat(selected_coarse),
                    selected_boxes, selected_gt)
        empty = pred.new_empty((0, ), dtype=torch.long)
        coarse = pred.new_empty((0, 5))
        return empty, coarse, selected_boxes, selected_gt

    @torch.no_grad()
    def _diagnostics(self, rois, field, target, gt, refined, clipped):
        coarse = rois[:, 1:]
        a, b, ell = self.nif_coder.frame(coarse)
        levels = self.bbox_roi_extractor.map_roi_levels(
            rois, len(self.nif_strides))
        stride = coarse.new_tensor(self.nif_strides)[levels]
        radii = torch.stack((a, a, b, b), -1) * 0.5
        radii = torch.maximum(radii, 2 * stride[:, None])
        distance = target * ell[:, None, None]
        covered = distance.abs() <= radii[:, :, None]
        # The target intersection can lie beyond the *finite* GT edge.
        ga, gb, phi = self.nif_coder.align_target(coarse, gt)
        cos, sin = coarse[:, 4].cos(), coarse[:, 4].sin()
        delta = gt[:, :2] - coarse[:, :2]
        dx = cos * delta[:, 0] + sin * delta[:, 1]
        dy = -sin * delta[:, 0] + cos * delta[:, 1]
        s = self.nif_coder.sample_positions(coarse)[None, :]
        hits = []
        for edge, sigma in ((0, -1), (1, 1), (2, -1), (3, 1)):
            if edge < 2:
                qx = sigma * (a[:, None] + distance[:, edge]) - dx[:, None]
                qy = s * b[:, None] - dy[:, None]
                tangent = -phi.sin()[:, None] * qx + phi.cos()[:, None] * qy
                hits.append(tangent.abs() <= gb[:, None])
            else:
                qx = s * a[:, None] - dx[:, None]
                qy = sigma * (b[:, None] + distance[:, edge]) - dy[:, None]
                tangent = phi.cos()[:, None] * qx + phi.sin()[:, None] * qy
                hits.append(tangent.abs() <= ga[:, None])
        projected = self.nif_coder.field_from_aux(
            coarse, self.nif_coder.project(coarse, field))
        return dict(
            nif_window_coverage=covered.float().mean(),
            nif_full_window_coverage=covered.flatten(1).all(1).float().mean(),
            nif_segment_hit=torch.stack(hits, 1).float().mean(),
            nif_size_clip_rate=clipped.float().mean(),
            nif_short_side_stride=(2 * torch.minimum(a, b) / stride).mean(),
            nif_compat_mse=(field - projected).square().mean(),
            nif_iou_before=exact_aligned_rotated_iou(coarse, gt).mean(),
            nif_iou_after=exact_aligned_rotated_iou(refined, gt).mean())

    @force_fp32(apply_to=('features', 'rois'))
    def _predict_field(self, features, rois):
        with torch.autocast(device_type=rois.device.type, enabled=False):
            rois = rois.float().detach()
            features = [f.float() for f in features[:len(self.nif_strides)]]
            levels = self.bbox_roi_extractor.map_roi_levels(
                rois, len(self.nif_strides))
            return self.nif_head(
                features, rois, levels, self.nif_strides,
                return_reliability=True)

    def _predict_refinement(self, features, rois, cls_score):
        field, reliability = self._predict_field(features, rois)
        boxes = self._decode_field(
            rois[:, 1:].float(), field, reliability=reliability)
        return boxes, None

    def forward_dummy(self, x, proposals):
        rois = rbbox2roi([proposals])
        result = self._bbox_forward(x, rois)
        coarse = self.bbox_head.bbox_coder.decode(
            rois[:, 1:], result['bbox_pred'])
        probe_rois = torch.cat((rois[:, :1], coarse), -1)
        field, reliability = self._predict_field(x, probe_rois)
        return result['cls_score'], self._decode_field(
            coarse, field, reliability=reliability)

    def _nif_loss(self, features, sampling_results, bbox_results,
                  gt_bboxes, gt_bboxes_ignore=None, img_metas=None):
        """Train only on detector positives and reliability-aware geometry."""
        indices, coarse, selected_boxes, selected_gt = self._select_nif_samples(
            sampling_results, bbox_results)
        positive_rois = rbbox2roi(selected_boxes)
        positive_rois = torch.cat((positive_rois[:, :1], coarse), -1).detach()
        field, reliability = self._predict_field(features, positive_rois)
        zero = field.sum() * 0 + reliability.sum() * 0

        if len(positive_rois):
            gt = torch.cat(selected_gt).float()
            obs, ref, refined, target, clipped = self._geometry_loss(
                positive_rois, field, gt, reliability)
            reliability_target = self._reliability_target(field, target)
            reliability_loss = torch.nn.functional.smooth_l1_loss(
                reliability, reliability_target, beta=0.1)
            losses = dict(
                loss_nif_obs=self.nif_obs_weight * obs + zero,
                loss_nif_ref=self.nif_ref_weight * ref,
                loss_nif_reliability=(
                    self.nif_reliability_loss_weight * reliability_loss))
            losses.update(self._diagnostics(
                positive_rois, field, target, gt, refined, clipped))
        else:
            reliability_target = reliability.new_empty(
                (0, 4, self.nif_coder.num_points))
            losses = dict(
                loss_nif_obs=zero,
                loss_nif_ref=zero,
                loss_nif_reliability=zero)
            for key in ('nif_window_coverage', 'nif_full_window_coverage',
                        'nif_segment_hit', 'nif_size_clip_rate',
                        'nif_short_side_stride', 'nif_compat_mse',
                        'nif_iou_before', 'nif_iou_after'):
                losses[key] = zero.detach()

        losses.update(
            nif_positive_count=field.new_tensor(
                len(positive_rois) / len(gt_bboxes)),
            nif_reliability_mean=(
                reliability.detach().mean() if reliability.numel()
                else zero.detach()),
            nif_reliability_target=(
                reliability_target.detach().mean()
                if reliability_target.numel() else zero.detach()),
            nif_reliability_min=(
                reliability.detach().amin(dim=(1, 2)).mean()
                if reliability.numel() else zero.detach()))
        return losses

    def _reliability_target(self, field, target):
        error = (field.detach() - target.detach()).abs()
        return torch.exp(-error / self.nif_reliability_error_scale).clamp(
            self.nif_reliability_min, 1.0)

    def _reliability_weights(self, reliability):
        """Return observation-loss weights with mean one within each RoI.

        Detaching blocks the observation loss from reducing its own weights.
        The fitting path separately keeps gradients to the reliability output.
        """
        if reliability is None:
            raise ValueError('Reliability values are required')
        weights = reliability.detach().float().pow(
            self.nif_reliability_power)
        return weights / weights.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    def _decode_field(self, coarse, field, reliability=None, **kwargs):
        if reliability is None:
            raise ValueError('Decoding requires reliability values')
        weights = reliability.float().pow(self.nif_reliability_power)
        return self.nif_coder.decode(
            coarse, field, self.nif_solver, weights=weights, **kwargs)

    def _geometry_loss(self, rois, field, gt, reliability=None):
        """Use reliability-weighted observations with detached weights."""
        zero = field.sum() * 0
        if not len(rois):
            return zero, zero, rois.new_empty((0, 5)), None, None
        if reliability is None:
            raise ValueError('Geometry loss requires reliability values')
        target = self.nif_coder.encode(rois[:, 1:], gt).detach()
        weights = self._reliability_weights(reliability).to(field.dtype)
        obs = (weights * (field - target).square()).mean()
        refined, clipped = self._decode_field(
            rois[:, 1:], field, reliability=reliability,
            canonicalize=False, return_clipped=True)
        return obs, self.nif_box_loss(refined, gt), refined, target, clipped

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        result = super()._bbox_forward_train(
            x, sampling_results, gt_bboxes, gt_labels, img_metas)
        result['loss_bbox'].update(self._nif_loss(
            x, sampling_results, result, gt_bboxes, img_metas=img_metas))
        return result

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False):
        """Refine score-filtered candidates before the native rotated NMS."""
        if rcnn_test_cfg is not None and not rcnn_test_cfg.get('nif_refine', True):
            return super().simple_test_bboxes(
                x, img_metas, proposals, rcnn_test_cfg, rescale=rescale)
        rois = rbbox2roi(proposals)
        if not len(rois):
            width = 5 if rcnn_test_cfg is None else 6
            boxes = [rois.new_zeros((0, width)) for _ in proposals]
            labels = [rois.new_zeros((0, self.bbox_head.num_classes + 1))
                      if rcnn_test_cfg is None else
                      rois.new_zeros((0,), dtype=torch.long) for _ in proposals]
            return boxes, labels
        result = self._bbox_forward(x, rois)
        counts = tuple(len(p) for p in proposals)
        det_bboxes, det_labels = [], []
        for one_rois, one_cls, one_pred, meta in zip(
                rois.split(counts), result['cls_score'].split(counts),
                result['bbox_pred'].split(counts), img_metas):
            coarse, scores = self.bbox_head.get_bboxes(
                one_rois, one_cls, one_pred, meta['img_shape'],
                meta['scale_factor'], rescale=False, cfg=None)
            if rcnn_test_cfg is not None:
                keep = scores[:, :-1].amax(-1) > rcnn_test_cfg.score_thr
                coarse, scores, one_rois = coarse[keep], scores[keep], one_rois[keep]
                one_cls = one_cls[keep]
            boxes, _ = self._predict_refinement(
                x, torch.cat((one_rois[:, :1], coarse), -1), one_cls)
            if rescale and len(boxes):
                boxes = boxes.clone()
                boxes[:, :4] /= boxes.new_tensor(meta['scale_factor'])
            if rcnn_test_cfg is None:
                det_bboxes.append(boxes)
                det_labels.append(scores)
            else:
                db, dl = multiclass_nms_rotated(
                    boxes, scores, rcnn_test_cfg.score_thr,
                    rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
                det_bboxes.append(db)
                det_labels.append(dl)
        return det_bboxes, det_labels
