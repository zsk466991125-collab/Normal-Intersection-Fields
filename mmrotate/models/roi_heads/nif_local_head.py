"""Shared local strip predictor for displacements and reliability."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class NIFLocalHead(nn.Module):
    """Sample (tangent, normal) strips of shape 3 x 9 in x-, x+, y-, y+."""

    def __init__(self,
                 coder,
                 in_channels=256,
                 hidden_channels=64,
                 strip_normal=9,
                 strip_tangent=3,
                 chunk_size=64,
                 aligned=True,
                 with_checkpoint=True,
                 projection_channels=64,
                 preserve_normal_position=True,
                 reliability_enabled=True,
                 reliability_prior=0.8,
                 reliability_min=0.05):
        super().__init__()
        if strip_normal < 3 or strip_tangent < 3 or chunk_size < 1:
            raise ValueError(
                'strip dimensions >= 3 and chunk_size >= 1 required')
        self.coder = coder
        self.strip_normal = strip_normal
        self.strip_tangent = strip_tangent
        self.chunk_size = chunk_size
        self.aligned = aligned
        self.with_checkpoint = with_checkpoint
        self.preserve_normal_position = bool(preserve_normal_position)
        self.reliability_enabled = bool(reliability_enabled)
        self.reliability_prior = float(reliability_prior)
        self.reliability_min = float(reliability_min)
        if not 0 < self.reliability_prior < 1:
            raise ValueError('reliability_prior must be in (0, 1)')
        if not 0 < self.reliability_min <= self.reliability_prior:
            raise ValueError('reliability_min must be in (0, reliability_prior]')
        if (in_channels < 1 or hidden_channels < 1
                or (projection_channels is not None
                    and projection_channels < 1)):
            raise ValueError('NIF channel counts must be positive')
        self.fpn_channels = in_channels
        self.projection_channels = projection_channels
        self.in_channels = (in_channels if projection_channels is None
                            else projection_channels)
        # One NIF-only linear projection, with weights shared across
        # FPN levels. Its result is reused by every RoI/chunk on that image
        # and level. None keeps the compact descriptor state dict.
        self.feature_projection = (None if projection_channels is None else
                                   nn.Conv2d(in_channels, projection_channels, 1))
        self.hidden_channels = hidden_channels
        visual_layers = [
            nn.Conv2d(self.in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True)]
        if self.preserve_normal_position:
            # Pool only along the tangent axis (3 samples). The 9 normal
            # positions remain ordered and are projected jointly afterward.
            self.visual = nn.Sequential(*visual_layers)
            self.normal_position = nn.Sequential(
                nn.Linear(hidden_channels * strip_normal, hidden_channels),
                nn.ReLU(inplace=True))
        else:
            # Keep the compact pooled descriptor when position preservation is off.
            self.visual = nn.Sequential(*visual_layers,
                                         nn.AdaptiveAvgPool2d(1))
            self.normal_position = None
        # edge type, sigma, s, a/ell, b/ell, radius/ell, stride/ell.
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels + 7, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 2 if self.reliability_enabled else 1))
        nn.init.zeros_(self.predictor[-1].weight)
        if self.reliability_enabled:
            bias = self.predictor[-1].bias.detach()
            bias.zero_()
            # The configured prior is the *final* bounded reliability.  Map it
            # back through the affine sigmoid parameterization so a freshly
            # initialized head, including the empty-RoI fallback, has exactly
            # the same value rather than ``min + (1-min) * prior``.
            sigmoid_prior = ((self.reliability_prior - self.reliability_min) /
                             (1.0 - self.reliability_min))
            bias[1] = torch.logit(
                bias.new_tensor(sigmoid_prior).clamp(1e-6, 1 - 1e-6))
        else:
            nn.init.zeros_(self.predictor[-1].bias)

    def strip_points(self, boxes, stride):
        """Return image points (N,4m,3,9,2) and geometry (N,4m,7)."""
        n, m = boxes.size(0), self.coder.num_points
        a, b, ell = self.coder.frame(boxes)
        s = self.coder.sample_positions(boxes)
        normal_offsets = torch.linspace(
            -1, 1, self.strip_normal, device=boxes.device, dtype=boxes.dtype)
        tangent_offsets = torch.linspace(
            -stride,
            stride,
            self.strip_tangent,
            device=boxes.device,
            dtype=boxes.dtype)
        u = torch.stack((boxes[:, 4].cos(), boxes[:, 4].sin()), dim=-1)
        v = torch.stack((-boxes[:, 4].sin(), boxes[:, 4].cos()), dim=-1)
        strips, geometry = [], []
        for edge_type, sigma, half, span, normal, tangent in ((0, -1, a, b, -u,
                                                               v), (0, 1, a, b,
                                                                    u, v),
                                                              (1, -1, b, a, -v,
                                                               u), (1, 1, b, a,
                                                                    v, u)):
            radius = (0.5 * half).clamp_min(2 * stride)
            center = boxes[:, None, :2] + half[:, None, None] * normal[:, None]
            center = center + s[None, :, None] * span[:, None,
                                                      None] * tangent[:, None]
            points = (
                center[:, :, None, None, :] +
                normal_offsets[None, None, None, :, None] *
                radius[:, None, None, None, None] *
                normal[:, None, None, None, :] +
                tangent_offsets[None, None, :, None, None] *
                tangent[:, None, None, None, :])
            strips.append(points)
            geometry.append(
                torch.stack((boxes.new_full(
                    (n, m), edge_type), boxes.new_full(
                        (n, m), sigma), s[None].expand(n, m),
                             (a / ell)[:, None].expand(n, m),
                             (b / ell)[:, None].expand(n, m),
                             (radius / ell)[:, None].expand(n, m),
                             (stride / ell)[:, None].expand(n, m)),
                            dim=-1))
        return torch.cat(strips, dim=1), torch.cat(geometry, dim=1)

    def sample_strips(self, feature, boxes, stride):
        """Sample one image/level without copying its feature map per RoI.

        Match RoIAlignRotated(aligned=True): feature coordinate = p/stride
        - 0.5. With grid_sample(align_corners=False), the grid is therefore
        2*p/(stride*size)-1. Outside the image is zero padded.
        """
        points, geometry = self.strip_points(boxes, stride)
        n, k, rows, cols = points.shape[:4]
        height, width = feature.shape[-2:]
        xy = points / stride - (0.5 if self.aligned else 0.0)
        size = points.new_tensor([width, height])
        grid = 2 * (xy + 0.5) / size - 1
        # Treat RoIs as output rows of a SINGLE feature image.
        grid = grid.reshape(1, n, k * rows * cols, 2)
        sampled = F.grid_sample(
            feature,
            grid,
            align_corners=False,
            mode='bilinear',
            padding_mode='zeros')
        sampled = sampled[0].permute(1, 2, 0)
        sampled = sampled.reshape(n, k, rows, cols, self.in_channels)
        sampled = sampled.permute(0, 1, 4, 2, 3).contiguous()
        return sampled, geometry

    def _predict_group(self,
                       feature,
                       boxes,
                       stride,
                       return_visual=False,
                       return_reliability=False):
        strips, geometry = self.sample_strips(feature, boxes, stride)
        n, k = strips.shape[:2]
        visual_map = self.visual(strips.flatten(0, 1))
        if self.preserve_normal_position:
            # visual_map: (N*4m, C, tangent=3, normal=9)
            visual = self.normal_position(
                visual_map.mean(dim=2).flatten(1))
        else:
            visual = visual_map.flatten(1)
        descriptor = torch.cat((visual, geometry.flatten(0, 1)), -1)
        output = self.predictor(descriptor)
        if self.reliability_enabled:
            output = output.reshape(n, 4, self.coder.num_points, 2)
            field = output[..., 0]
            reliability = torch.sigmoid(output[..., 1])
            reliability = self.reliability_min + (
                1 - self.reliability_min) * reliability
        else:
            field = output.reshape(n, 4, self.coder.num_points)
            reliability = None
        values = [field]
        if return_visual:
            # Reuse the convolution output in edge order.
            # Average points within each edge, retaining four edge vectors.
            edges = visual.reshape(n, 4, self.coder.num_points,
                                   self.hidden_channels).mean(2).flatten(1)
            values.append(edges)
        if return_reliability:
            if not self.reliability_enabled:
                raise ValueError('return_reliability requires reliability_enabled')
            values.append(reliability)
        return tuple(values) if len(values) > 1 else values[0]

    def forward(self,
                features,
                rois,
                levels,
                strides,
                return_visual=False,
                return_reliability=False):
        """Use exactly the FPN assignment supplied by the baseline extractor."""
        # Keep every parameter and feature in the graph on empty-positive
        # batches as well, so DDP does not lose the refinement branch.
        zero = sum(p.flatten()[0] * 0 for p in self.parameters())
        zero = zero + sum(f.flatten()[0] * 0 for f in features)
        fields = rois.new_zeros((len(rois), 4, self.coder.num_points)) + zero
        if return_visual:
            visuals = rois.new_zeros((len(rois), 4 * self.hidden_channels)) + zero
        if return_reliability:
            if not self.reliability_enabled:
                raise ValueError('return_reliability requires reliability_enabled')
            reliabilities = rois.new_full(
                (len(rois), 4, self.coder.num_points),
                self.reliability_prior) + zero
        for level, (feature, stride) in enumerate(zip(features, strides)):
            for batch in range(feature.size(0)):
                indices = ((levels == level) & (rois[:, 0] == batch)).nonzero(
                    as_tuple=False).flatten()
                if not len(indices):
                    continue
                image = feature[batch:batch + 1]
                if self.feature_projection is not None:
                    # Project before sampling, once per active image/level,
                    # outside the chunk loop and checkpoint recomputation.
                    # No persistent cache: each forward owns its autograd graph.
                    image = self.feature_projection(image)
                for chunk in indices.split(self.chunk_size):
                    if chunk.numel() == 0:
                        continue
                    boxes = rois[chunk, 1:]
                    if (self.with_checkpoint and self.training
                            and image.requires_grad):
                        # Bind stride now; checkpoint recomputes during backward.
                        def run(feat, box, stride=stride):
                            if return_visual or return_reliability:
                                return self._predict_group(
                                    feat, box, stride,
                                    return_visual=return_visual,
                                    return_reliability=return_reliability)
                            return self._predict_group(feat, box, stride)

                        values = checkpoint(
                            run, image, boxes, use_reentrant=False)
                    else:
                        if return_visual or return_reliability:
                            values = self._predict_group(
                                image, boxes, stride,
                                return_visual=return_visual,
                                return_reliability=return_reliability)
                        else:
                            values = self._predict_group(image, boxes, stride)
                    if return_visual or return_reliability:
                        cursor = 0
                        field_value = values[cursor]
                        cursor += 1
                        fields = fields.index_copy(0, chunk, field_value)
                        if return_visual:
                            visual_value = values[cursor]
                            cursor += 1
                            visuals = visuals.index_copy(0, chunk, visual_value)
                        if return_reliability:
                            reliability_value = values[cursor]
                            reliabilities = reliabilities.index_copy(
                                0, chunk, reliability_value)
                    else:
                        fields = fields.index_copy(0, chunk, values)
        if return_visual or return_reliability:
            result = [fields]
            if return_visual:
                result.append(visuals)
            if return_reliability:
                result.append(reliabilities)
            return tuple(result)
        return fields

