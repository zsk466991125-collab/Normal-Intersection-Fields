"""NIF refinement of final O2 queries; carrier graph and scores stay frozen."""
import torch
import torch.nn.functional as F
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS
from mmrotate.registry import MODELS
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETR

from .core import (NIFLocalHead, NIFOBBCoder, exact_aligned_rotated_iou,
                   stratified_indices)


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class NIFOnlyOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Exclude every frozen parameter, including aliased RT-DETR branches."""

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module
        from mmengine.optim import OptimWrapper
        cfg = self.optimizer_cfg.copy()
        optimizer_type = cfg.pop('type')
        if isinstance(optimizer_type, str):
            optimizer_type = getattr(torch.optim, optimizer_type)
        parameters = [p for p in model.nif_head.parameters() if p.requires_grad]
        if set(map(id, parameters)) != set(
                id(p) for p in model.parameters() if p.requires_grad):
            raise RuntimeError('Only nif_head parameters may be optimized')
        return OptimWrapper(optimizer_type(parameters, **cfg),
                            **{k: v for k, v in self.optim_wrapper_cfg.items()
                               if k != 'type'})


@MODELS.register_module()
class FrozenO2RTDETRNIF(RotatedRTDETR):
    """Keep native checkpoint keys; append a single shared NIF local head.

    Final decoder queries provide coarse boxes. NIF reads the hybrid-encoder
    FPN at strides 8/16/32. Training uses the native Hungarian matcher; only
    matched queries supervise NIF. There are no fake GT/denoising queries,
    quality branch, EMA, or gradients through the frozen detector.
    """

    def __init__(self, *args, nif_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = dict(nif_cfg or {})
        self.nif_budget = cfg.pop('max_pos_per_img', 128)
        self.nif_bins = cfg.pop('iou_bins', [0.3, 0.45, 0.6, 0.75, 0.9, 1.01])
        self.nif_power = cfg.pop('reliability_power', 1.5)
        self.nif_error_scale = cfg.pop('reliability_error_scale', 0.25)
        self.nif_reliability_weight = cfg.pop('reliability_loss_weight', 0.05)
        self.nif_coder = NIFOBBCoder(num_points=cfg.pop('num_points', 5),
                                     sample_extent=cfg.pop('sample_extent', 0.75))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.nif_head = NIFLocalHead(
            self.nif_coder, in_channels=256, projection_channels=64,
            hidden_channels=64, strip_tangent=3, strip_normal=9,
            preserve_normal_position=True, reliability_enabled=True,
            reliability_prior=0.8, reliability_min=0.05,
            chunk_size=64, with_checkpoint=True, aligned=True, **cfg)
        self.train(self.training)

    def train(self, mode=True):
        # The root training flag controls BaseModel's loss dispatch, but the
        # complete carrier (BN buffers and dropout included) remains in eval.
        super().train(False)
        self.training = mode
        if hasattr(self, 'nif_head'):
            self.nif_head.train(mode)
        return self

    @torch.no_grad()
    def carrier_forward(self, inputs, samples):
        training = self.training
        self.training = False  # pre_decoder must not inject DN during NIF training
        try:
            feats = self.extract_feat(inputs)
            encoder_in, decoder_in = self.pre_transformer(feats, samples)
            encoded = self.forward_encoder(**encoder_in)
            memory = encoded['memory']
            shapes = encoded['spatial_shapes'].tolist()
            pyramid, start = [], 0
            for h, w in shapes:
                pyramid.append(memory[:, start:start+h*w].transpose(1, 2).reshape(
                    len(inputs), self.embed_dims, h, w).contiguous())
                start += h*w
            prepared, _ = self.pre_decoder(**encoded, batch_data_samples=samples)
            decoder_in.update(prepared)
            decoded = self.forward_decoder(**decoder_in)
            classes, coords = self.bbox_head(
                decoded['hidden_states'], decoded['references'])
            return tuple(pyramid), classes[-1].float(), coords[-1].float()
        finally:
            self.training = training

    def predict_field(self, pyramid, rois):
        # RoIAlign's finest-scale=56 corresponds to stride 4. The O2 pyramid
        # starts at stride 8, so shift physical level assignment by one.
        scale = rois[:, 3:5].prod(-1).clamp_min(1e-6).sqrt()
        levels = torch.floor(torch.log2(scale / 56 + 1e-6)).long() - 1
        levels = levels.clamp(0, len(pyramid)-1)
        # Run the small geometry branch in FP32, including grid_sample/solve.
        with torch.cuda.amp.autocast(enabled=False):
            return self.nif_head(tuple(f.float() for f in pyramid), rois.float(),
                                 levels, [8, 16, 32], return_reliability=True)

    @staticmethod
    def factor(boxes, sample, angle_factor):
        h, w = sample.metainfo['img_shape'][:2]
        return boxes.new_tensor([w, h, w, h, angle_factor])

    def loss(self, batch_inputs, batch_data_samples):
        pyramid, classes, coords = self.carrier_forward(batch_inputs, batch_data_samples)
        rois, targets = [], []
        with torch.no_grad():
            for batch, sample in enumerate(batch_data_samples):
                factor = self.factor(coords, sample, self.bbox_head.angle_factor)
                gt = sample.gt_instances.clone()
                result = self.bbox_head._get_targets_single(
                    classes[batch], coords[batch], gt, sample.metainfo)
                _, _, normalized_targets, _, positive, _ = result
                coarse = coords[batch, positive] * factor
                target = normalized_targets[positive] * factor
                ious = exact_aligned_rotated_iou(coarse, target)
                selected = stratified_indices(ious, self.nif_budget, self.nif_bins)
                coarse, target = coarse[selected], target[selected]
                rois.append(torch.cat((coarse.new_full((len(coarse), 1), batch), coarse), 1))
                targets.append(target)
        rois, target_boxes = torch.cat(rois), torch.cat(targets)
        field, reliability = self.predict_field(pyramid, rois)
        zero = field.sum() * 0 + reliability.sum() * 0
        if not len(rois):
            return dict(loss_nif_obs=zero, loss_nif_ref=zero,
                        loss_nif_reliability=zero, nif_iou_before=zero.detach(),
                        nif_iou_after=zero.detach(), nif_positive_count=zero.detach())
        with torch.cuda.amp.autocast(enabled=False):
            target_field = self.nif_coder.encode(rois[:, 1:], target_boxes).detach()
            weights = reliability.detach().pow(self.nif_power)
            weights = weights / weights.mean((1, 2), keepdim=True).clamp_min(1e-6)
            refined = self.nif_coder.decode(
                rois[:, 1:], field, weights=reliability.pow(self.nif_power),
                canonicalize=False)
            iou_after = exact_aligned_rotated_iou(refined, target_boxes)
            reliability_target = torch.exp(
                -(field.detach()-target_field).abs() / self.nif_error_scale).clamp(0.05, 1)
            return dict(
                loss_nif_obs=(weights*(field-target_field).square()).mean()+zero,
                loss_nif_ref=(1-iou_after).mean()+zero,
                loss_nif_reliability=self.nif_reliability_weight * F.smooth_l1_loss(
                    reliability, reliability_target, beta=0.1)+zero,
                nif_iou_before=exact_aligned_rotated_iou(
                    rois[:, 1:], target_boxes).mean().detach(),
                nif_iou_after=iou_after.mean().detach(),
                nif_positive_count=field.new_tensor(len(rois)/len(batch_data_samples)))

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        pyramid, classes, coords = self.carrier_forward(batch_inputs, batch_data_samples)
        rois = []
        factors = []
        for batch, sample in enumerate(batch_data_samples):
            factor = self.factor(coords, sample, self.bbox_head.angle_factor)
            factors.append(factor)
            coarse = coords[batch] * factor
            rois.append(torch.cat((coarse.new_full((len(coarse), 1), batch), coarse), 1))
        rois = torch.cat(rois)
        field, reliability = self.predict_field(pyramid, rois)
        with torch.cuda.amp.autocast(enabled=False):
            refined = self.nif_coder.decode(
                rois[:, 1:], field, weights=reliability.pow(self.nif_power),
                canonicalize=False).reshape_as(coords)
            normalized = torch.stack([box/factor for box, factor in zip(refined, factors)])
        # Native scoring, top-k, rescale and optional NMS are retained.
        results = self.bbox_head.predict_by_feat(
            [classes], [normalized], [s.metainfo for s in batch_data_samples], rescale=rescale)
        return self.add_pred_to_datasample(batch_data_samples, results)
