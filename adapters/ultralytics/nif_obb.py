"""NIF refinement on a frozen Ultralytics YOLO26 OBB carrier.

The carrier stays on Ultralytics' native OBB path.  NIF consumes the native
FPN maps and the task-aligned positive coarse boxes, so the extension does not
change the detector head or its checkpoint format.
"""

from __future__ import annotations

import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import LOCAL_RANK, RANK
from ultralytics.utils.loss import v8OBBLoss
from ultralytics.utils.metrics import probiou
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import strip_optimizer, torch_distributed_zero_first

_nif_root = Path(__file__).resolve().parents[2]
if str(_nif_root) not in sys.path:
    sys.path.insert(0, str(_nif_root))
from mmrotate.models.losses.nif_iou_loss import exact_aligned_rotated_iou


def _set_periodic_validation(trainer):
    epoch = trainer.epoch + 1
    should_validate = epoch % trainer.val_interval == 0 or epoch >= trainer.epochs
    trainer.args.val = should_validate
    # Keep the schedule observable in DDP logs and avoid stale metrics from a
    # previous validation being mistaken for a current result.
    if RANK in {-1, 0}:
        print(f"NIF validation: epoch {epoch}/{trainer.epochs}, "
              f"interval={trainer.val_interval}, enabled={should_validate}")
    if not should_validate:
        trainer.fitness = None


class NIFOBBCoder:
    """Five-parameter normal-intersection field coder for xywhr boxes."""

    def __init__(self, num_points: int = 5, sample_extent: float = 0.75, min_size: float = 1e-3):
        self.num_points = int(num_points)
        self.sample_extent = float(sample_extent)
        self.min_size = float(min_size)
        if self.num_points < 2 or not 0 < self.sample_extent < 1:
            raise ValueError("NIF requires num_points >= 2 and 0 < sample_extent < 1")

    def sample_positions(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.linspace(
            -self.sample_extent,
            self.sample_extent,
            self.num_points,
            device=reference.device,
            dtype=reference.dtype,
        )

    def frame(self, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b = (boxes[:, 2:4].clamp_min(self.min_size) * 0.5).unbind(-1)
        return a, b, torch.sqrt(a.square() + b.square())

    def align_target(self, coarse: torch.Tensor, target: torch.Tensor):
        delta = (target[:, 4] - coarse[:, 4] + math.pi / 2) % math.pi - math.pi / 2
        swap = (delta >= math.pi / 4) | (delta < -math.pi / 4)
        phi = torch.where(swap, delta - delta.sign() * math.pi / 2, delta)
        ga = torch.where(swap, target[:, 3], target[:, 2]) * 0.5
        gb = torch.where(swap, target[:, 2], target[:, 3]) * 0.5
        return ga, gb, phi

    def encode_aux(self, coarse: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        a, b, ell = self.frame(coarse)
        ga, gb, phi = self.align_target(coarse, target)
        offset = target[:, :2] - coarse[:, :2]
        cos, sin = coarse[:, 4].cos(), coarse[:, 4].sin()
        dx = cos * offset[:, 0] + sin * offset[:, 1]
        dy = -sin * offset[:, 0] + cos * offset[:, 1]
        t = phi.tan()
        kappa = self.sample_positions(coarse).square().mean().sqrt()
        return torch.stack(((dx + t * dy) / ell, (dy - t * dx) / ell,
                            (ga / phi.cos() - a) / ell,
                            (gb / phi.cos() - b) / ell, kappa * t), dim=-1)

    def field_from_aux(self, coarse: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        a, b, ell = self.frame(coarse)
        s = self.sample_positions(coarse)[None, :]
        tx, ty, da, db, zt = (v[:, None] for v in aux.unbind(-1))
        kappa = self.sample_positions(coarse).square().mean().sqrt()
        t = zt / kappa
        xslope = (b / ell)[:, None] * s * t
        yslope = (a / ell)[:, None] * s * t
        return torch.stack((-tx + da + xslope, tx + da - xslope,
                            -ty + db - yslope, ty + db + yslope), dim=1)

    def encode(self, coarse: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.field_from_aux(coarse, self.encode_aux(coarse, target))

    def design_matrix(self, coarse: torch.Tensor) -> torch.Tensor:
        n = coarse.size(0)
        eye = torch.eye(5, device=coarse.device, dtype=coarse.dtype)
        boxes = coarse[:, None, :].expand(n, 5, 5).reshape(-1, 5)
        fields = self.field_from_aux(boxes, eye.repeat(n, 1))
        return fields.reshape(n, 5, 4 * self.num_points).transpose(1, 2)

    def project(self, coarse: torch.Tensor, field: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        """Fit five auxiliary parameters with detached r_i^1.5 weights."""
        # PyTorch 1.13 has no CUDA Half implementation for linalg.solve.  AMP
        # also autocasts the matmuls even when their operands were cast to float,
        # so keep this small weighted least-squares block explicitly in FP32.
        autocast = torch.cuda.amp.autocast(enabled=False) if coarse.device.type == "cuda" else nullcontext()
        with autocast:
            coarse_fp32 = coarse.float()
            weights = reliability.detach().float().clamp_min(1e-4).pow(1.5)
            weights = weights / weights.mean(dim=(1, 2), keepdim=True).clamp_min(1e-4)
            design = self.design_matrix(coarse_fp32)
            obs = field.flatten(1).float()
            diagonal = weights.flatten(1)
            normal = design.transpose(1, 2) @ (design * diagonal[..., None])
            rhs = design.transpose(1, 2) @ (obs * diagonal)[..., None]
            return torch.linalg.solve(normal, rhs).squeeze(-1)

    def decode_aux(self, coarse: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        a, b, ell = self.frame(coarse)
        tx, ty, da, db, zt = aux.unbind(-1)
        kappa = self.sample_positions(coarse).square().mean().sqrt()
        t = zt / kappa
        norm2 = 1 + t.square()
        dx = ell * (tx - t * ty) / norm2
        dy = ell * (t * tx + ty) / norm2
        cos, sin = coarse[:, 4].cos(), coarse[:, 4].sin()
        cx = coarse[:, 0] + cos * dx - sin * dy
        cy = coarse[:, 1] + sin * dx + cos * dy
        w = (2 * (a + ell * da) / norm2.sqrt()).clamp_min(self.min_size)
        h = (2 * (b + ell * db) / norm2.sqrt()).clamp_min(self.min_size)
        angle = coarse[:, 4] + t.atan()
        return torch.stack((cx, cy, w, h, angle), dim=-1)

    def decode(self, coarse: torch.Tensor, field: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        autocast = torch.cuda.amp.autocast(enabled=False) if coarse.device.type == "cuda" else nullcontext()
        with autocast:
            return self.decode_aux(coarse.float(), self.project(coarse, field, reliability))


class NIFLocalHead(nn.Module):
    """Shared 3x9 strip sampler with normal-axis position preservation."""

    def __init__(self, feature_channels: tuple[int, ...], coder: NIFOBBCoder):
        super().__init__()
        self.coder = coder
        self.strip_normal, self.strip_tangent = 9, 3
        self.projections = nn.ModuleList(nn.Conv2d(c, 64, 1) for c in feature_channels)
        self.visual = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.normal_position = nn.Sequential(nn.Linear(64 * self.strip_normal, 64), nn.ReLU(inplace=True))
        self.predictor = nn.Sequential(nn.Linear(64 + 7, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.constant_(self.predictor[-1].bias[1], math.log((0.8 - 0.05) / (1 - 0.8)))

    def _strips(self, boxes: torch.Tensor, stride: float):
        n, m = len(boxes), self.coder.num_points
        a, b, ell = self.coder.frame(boxes)
        s = self.coder.sample_positions(boxes)
        normal_offsets = torch.linspace(-1, 1, self.strip_normal, device=boxes.device, dtype=boxes.dtype)
        tangent_offsets = torch.linspace(-stride, stride, self.strip_tangent, device=boxes.device, dtype=boxes.dtype)
        u = torch.stack((boxes[:, 4].cos(), boxes[:, 4].sin()), -1)
        v = torch.stack((-boxes[:, 4].sin(), boxes[:, 4].cos()), -1)
        strips, geometry = [], []
        for edge, sigma, half, span, normal, tangent in ((0, -1, a, b, -u, v), (0, 1, a, b, u, v),
                                                          (1, -1, b, a, -v, u), (1, 1, b, a, v, u)):
            radius = (0.5 * half).clamp_min(2 * stride)
            center = boxes[:, None, :2] + half[:, None, None] * normal[:, None]
            center = center + s[None, :, None] * span[:, None, None] * tangent[:, None]
            points = (center[:, :, None, None, :] + normal_offsets[None, None, :, None, None] *
                      radius[:, None, None, None, None] * normal[:, None, None, None, :] +
                      tangent_offsets[None, None, None, :, None] * tangent[:, None, None, None, :])
            strips.append(points)
            geometry.append(torch.stack((boxes.new_full((n, m), edge), boxes.new_full((n, m), sigma),
                                          s[None].expand(n, m), (a / ell)[:, None].expand(n, m),
                                          (b / ell)[:, None].expand(n, m), (radius / ell)[:, None].expand(n, m),
                                          (stride / ell)[:, None].expand(n, m)), -1))
        return torch.cat(strips, 1), torch.cat(geometry, 1)

    def _sample(self, feature: torch.Tensor, boxes: torch.Tensor, stride: float):
        points, geometry = self._strips(boxes, stride)
        n, k, rows, cols = points.shape[:4]
        h, w = feature.shape[-2:]
        xy = points / stride - 0.5
        grid = 2 * (xy + 0.5) / points.new_tensor([w, h]) - 1
        grid = grid.reshape(1, n, k * rows * cols, 2)
        sampled = F.grid_sample(feature, grid, align_corners=False, mode="bilinear", padding_mode="zeros")
        sampled = sampled[0].permute(1, 2, 0).reshape(n, k, 64, rows, cols).permute(0, 1, 2, 3, 4)
        return sampled, geometry

    def _predict(self, feature: torch.Tensor, boxes: torch.Tensor, stride: float):
        strips, geometry = self._sample(feature, boxes, stride)
        # Pool along the tangent axis only; the nine normal samples remain ordered.
        visual = self.visual(strips.flatten(0, 1)).mean(dim=3).flatten(1)
        visual = self.normal_position(visual)
        output = self.predictor(torch.cat((visual, geometry.flatten(0, 1)), -1))
        output = output.reshape(len(boxes), 4, self.coder.num_points, 2)
        field = output[..., 0]
        reliability = 0.05 + 0.95 * output[..., 1].sigmoid()
        return field, reliability

    def forward(self, features: list[torch.Tensor], rois: torch.Tensor, levels: torch.Tensor):
        zero = sum(p.sum() * 0 for p in self.parameters())
        field = rois.new_zeros((len(rois), 4, self.coder.num_points)) + zero
        reliability = rois.new_full(field.shape, 0.8) + zero
        strides = (8.0, 16.0, 32.0)
        for level, (feature, stride, projection) in enumerate(zip(features, strides, self.projections)):
            for batch_idx in range(feature.shape[0]):
                idx = ((levels == level) & (rois[:, 0] == batch_idx)).nonzero(as_tuple=False).flatten()
                if len(idx):
                    f, r = self._predict(projection(feature[batch_idx:batch_idx + 1]), rois[idx, 1:], stride)
                    field = field.index_copy(0, idx, f.to(field.dtype))
                    reliability = reliability.index_copy(0, idx, r.to(reliability.dtype))
        return field, reliability


class NIFOBBLoss:
    """Native OBB assignment plus geometry-only NIF supervision."""

    def __init__(self, model: "NIFYOLOOBBModel"):
        self.model = model
        self.det_loss = v8OBBLoss(model, tal_topk=10)
        self.coder = model.nif_coder
        self.device = next(model.parameters()).device
        self.loss_names = (*self.det_loss.loss_names, "nif_obs", "nif_ref", "nif_reliability")

    @staticmethod
    def _levels(coarse: torch.Tensor) -> torch.Tensor:
        scale = coarse[:, 2:4].clamp_min(1).prod(-1).sqrt()
        return torch.log2(scale / 64).floor().clamp(0, 2).long()

    def _select(self, coarse: torch.Tensor, target: torch.Tensor, fg: torch.Tensor):
        rois, gts = [], []
        bins = (0.3, 0.45, 0.6, 0.75, 0.9, 1.01)
        for batch_idx in range(coarse.shape[0]):
            ids = fg[batch_idx].nonzero(as_tuple=False).flatten()
            if not len(ids):
                continue
            values = probiou(coarse[batch_idx, ids], target[batch_idx, ids]).detach().flatten()
            chosen = []
            per_bin = max(1, 128 // (len(bins) - 1))
            for low, high in zip(bins, bins[1:]):
                in_bin = (values >= low) & (values < high)
                inside = ids[in_bin]
                if len(inside):
                    order = values[in_bin].argsort()
                    chosen.append(inside[order[:per_bin]])
            chosen = torch.cat(chosen) if chosen else ids[values.argsort(descending=True)[:128]]
            if len(chosen) > 128:
                chosen = chosen[:128]
            rois.append(torch.cat((coarse.new_full((len(chosen), 1), batch_idx), coarse[batch_idx, chosen]), -1))
            gts.append(target[batch_idx, chosen])
        if not rois:
            return coarse.new_empty((0, 6)), coarse.new_empty((0, 5))
        return torch.cat(rois), torch.cat(gts)

    def _assign_obb(self, raw: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        """Return native rotated assignments and detached carrier diagnostics."""
        pred_dist = raw["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = raw["scores"].permute(0, 2, 1).contiguous()
        pred_angle = raw["angle"].permute(0, 2, 1).contiguous()
        anchors, stride = make_anchors(raw["feats"], self.det_loss.stride, 0.5)
        batch_size = pred_angle.shape[0]
        dtype = pred_scores.dtype
        imgsz = torch.tensor(raw["feats"][0].shape[2:], device=pred_scores.device, dtype=dtype) * self.det_loss.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
        rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
        targets = targets[(rw >= 2) & (rh >= 2)]
        targets = self.det_loss.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 5), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.det_loss.bbox_decode(anchors, pred_dist, pred_angle)
        bboxes_for_assigner = pred_bboxes.detach().clone()
        bboxes_for_assigner[..., :4] *= stride
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.det_loss.assigner(
            pred_scores.detach().sigmoid(), bboxes_for_assigner.type(gt_bboxes.dtype), anchors * stride,
            gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = target_scores.sum().clamp_(min=1)
        det = torch.zeros(4, device=pred_scores.device, dtype=pred_scores.dtype)
        det[1] = self.det_loss.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        if fg_mask.any():
            bbox_target = target_bboxes.clone()
            bbox_target[..., :4] /= stride
            det[0], det[2] = self.det_loss.bbox_loss(
                pred_dist, pred_bboxes, anchors, bbox_target, target_scores, target_scores_sum,
                fg_mask, imgsz, stride)
            weight = target_scores[fg_mask].sum(-1)
            det[3] = self.det_loss.calculate_angle_loss(
                pred_bboxes, bbox_target, fg_mask, weight, target_scores_sum)
        det[0] *= self.det_loss.hyp.box
        det[1] *= self.det_loss.hyp.cls
        det[2] *= self.det_loss.hyp.dfl
        det[3] *= self.det_loss.hyp.angle
        return pred_bboxes, target_bboxes, fg_mask, det.detach()

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]):
        raw = preds[1] if isinstance(preds, tuple) else preds
        raw = raw["one2many"] if "one2many" in raw else raw
        pred_boxes, target, fg, det_vec = self._assign_obb(raw, batch)
        anchors, stride = make_anchors(raw["feats"], self.det_loss.stride, 0.5)
        coarse = pred_boxes.detach().clone()
        coarse[..., :4] *= stride
        target = target.detach()
        rois, gt = self._select(coarse, target, fg)
        if len(rois):
            levels = self._levels(rois[:, 1:])
            field, reliability = self.model.nif_refiner(raw["feats"], rois, levels)
            coarse_sel = rois[:, 1:]
            field_target = self.coder.encode(coarse_sel, gt).detach()
            weights = reliability.detach().float().pow(1.5)
            weights = weights / weights.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            obs = (weights * (field - field_target).square()).mean()
            refined = self.coder.decode(coarse_sel, field, reliability)
            ref = (1 - exact_aligned_rotated_iou(refined, gt)).mean()
            rel_target = torch.exp(-(field - field_target).detach().abs() / 0.25).clamp(0.05, 1)
            rel = F.smooth_l1_loss(reliability, rel_target, beta=0.1)
            zero = field.sum() * 0
        else:
            zero = sum(p.sum() * 0 for p in self.model.nif_refiner.parameters())
            obs = ref = rel = zero
        items = dict(zip(self.det_loss.loss_names, det_vec))
        items.update(nif_obs=obs.detach(), nif_ref=ref.detach(), nif_reliability=rel.detach())
        return (obs + ref + 0.05 * rel) * raw["boxes"].shape[0], items


class NIFYOLOOBBModel(OBBModel):
    """YOLO26 OBB carrier with an inference-time NIF refinement branch."""

    def __init__(self, cfg="yolo26l-obb.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        head = self.model[-1]
        channels = tuple(layer[0].conv.in_channels for layer in head.cv2)
        self.nif_coder = NIFOBBCoder(num_points=5, sample_extent=0.75)
        self.nif_refiner = NIFLocalHead(channels, self.nif_coder)
        self.nif_max_candidates = 300

    def init_criterion(self):
        return NIFOBBLoss(self)

    def predict(self, x, profile=False, augment=False, embed=None):
        output = super().predict(x, profile=profile, augment=augment, embed=embed)
        if self.training or not isinstance(output, tuple):
            return output
        decoded, raw = output
        head = self.model[-1]
        end2end = bool(getattr(head, "end2end", False))
        raw_one = raw.get("one2one" if end2end else "one2many", raw)
        # YOLO26 is end-to-end by default, so postprocess returns
        # [xywh, score, class, angle] rows rather than channel-first anchors.
        if end2end:
            count = min(self.nif_max_candidates, decoded.shape[1])
            for batch_idx in range(decoded.shape[0]):
                coarse = torch.cat((decoded[batch_idx, :count, :4], decoded[batch_idx, :count, -1:]), -1)
                rois = torch.cat((decoded.new_full((count, 1), batch_idx), coarse), -1)
                levels = NIFOBBLoss._levels(coarse)
                field, reliability = self.nif_refiner(raw_one["feats"], rois, levels)
                refined = self.nif_coder.decode(coarse, field, reliability)
                decoded[batch_idx, :count, :4] = refined[:, :4].to(decoded.dtype)
                decoded[batch_idx, :count, -1] = refined[:, 4].to(decoded.dtype)
            return decoded, raw
        scores = raw_one["scores"].sigmoid().amax(1)
        count = min(self.nif_max_candidates, scores.shape[1])
        for batch_idx in range(decoded.shape[0]):
            ids = scores[batch_idx].topk(count).indices
            # Dense OBB output is [xywh, class probabilities..., angle].
            # Angle is the LAST channel; channel 4 is the first class score.
            coarse = torch.cat((decoded[batch_idx, :4, ids].T,
                                decoded[batch_idx, -1:, ids].T), -1)
            rois = torch.cat((decoded.new_full((count, 1), batch_idx), coarse), -1)
            levels = NIFOBBLoss._levels(rois[:, 1:])
            field, reliability = self.nif_refiner(raw_one["feats"], rois, levels)
            refined = self.nif_coder.decode(coarse, field, reliability)
            decoded[batch_idx, :4, ids] = refined[:, :4].T.to(decoded.dtype)
            decoded[batch_idx, -1, ids] = refined[:, 4].to(decoded.dtype)
        return decoded, raw


class NIFOBBTrainer(OBBTrainer):
    """Ultralytics OBB trainer that freezes the carrier and trains only NIF."""

    val_interval = 5

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        # The native carrier is model.0..model.N; NIF is attached outside that
        # sequence. A large index range keeps every carrier parameter frozen.
        overrides.setdefault("freeze", 1000)
        super().__init__(cfg=cfg or {}, overrides=overrides, _callbacks=_callbacks)
        # Ultralytics launches a temporary DDP child for multi-GPU runs.  The
        # child inherits the environment, while Python class mutations made in
        # the parent do not, so carry the validation period explicitly.
        self.val_interval = max(1, int(os.environ.get(
            "NIF_VAL_INTERVAL", self.val_interval)))
        # DDP unpickles callbacks, so function identity may differ between
        # parent and worker. Remove stale copies and append exactly one copy.
        event_callbacks = self.callbacks["on_train_epoch_end"]
        event_callbacks[:] = [callback for callback in event_callbacks
                              if getattr(callback, "__name__", None) != "_set_periodic_validation"]
        event_callbacks.append(_set_periodic_validation)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(
            NIFYOLOOBBModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        )
        if weights:
            model.load(weights)
        return model

    def validate(self):
        if not self.args.val:
            return self.metrics, None
        return super().validate()

    def final_eval(self):
        if self.args.val:
            return super().final_eval()
        with torch_distributed_zero_first(LOCAL_RANK):
            if RANK in {-1, 0} and self.last.exists():
                strip_optimizer(self.last)

    def build_optimizer(self, model, name="AdamW", lr=1e-3, momentum=0.9, decay=5e-4, iterations=1e5):
        """Build an optimizer containing NIF tensors only."""
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("NIF optimizer has no trainable parameters")
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=(momentum, 0.999), weight_decay=decay)
        print(f"NIF optimizer: AdamW with {sum(p.numel() for p in parameters):,} trainable parameters")
        return optimizer
