"""Normal-intersection fields in a coarse OBB's local coordinate frame.

Fields have shape (N, 4, m), ordered x-, x+, y-, y+. Each value is an
outward signed distance divided by ell = sqrt(a**2 + b**2). Intersections
are with the *supporting lines*, not necessarily the finite GT edges.
"""
import math

import torch
try:
    from mmdet.core.bbox.builder import BBOX_CODERS
except ModuleNotFoundError:
    # MMDetection 3 adapters reuse the same geometry without importing the
    # legacy mmrotate package (which requires MMCV 1).
    from mmdet.registry import TASK_UTILS as BBOX_CODERS


@BBOX_CODERS.register_module()
class NIFOBBCoder:
    """Fixed OLS projection and an orthogonal-distance fitting control.

    Auxiliary coordinates are (Tx/ell, Ty/ell, Da/ell, Db/ell, kappa*tan(phi)).
    Equivalent GT edge assignments keep phi in [-pi/4, pi/4). OLS is the
    unconstrained projection in proposal Eq. (12); its slope is NOT clipped.
    Orthogonal fitting requires an explicit common angle limit, also applied
    to OLS when requested, for a controlled bounded-angle comparison.
    Size clipping is a safeguard, not a constrained least-squares solution.
    The output uses MMRotate's clockwise, long-edge ``le90`` convention.
    """

    def __init__(self,
                 num_points=5,
                 sample_extent=0.75,
                 min_size=1e-3,
                 angle_limit=None):
        if num_points < 2:
            raise ValueError('NIF requires at least two points per edge')
        if not 0 < sample_extent < 1:
            raise ValueError('sample_extent must be in (0, 1)')
        if min_size <= 0:
            raise ValueError('min_size must be positive')
        self.num_points = num_points
        self.sample_extent = sample_extent
        self.min_size = min_size
        if angle_limit is not None and not 0 < angle_limit < math.pi / 2:
            raise ValueError('angle_limit must be in (0, pi/2) or None')
        self.angle_limit = angle_limit

    def sample_positions(self, reference):
        return torch.linspace(
            -self.sample_extent,
            self.sample_extent,
            self.num_points,
            device=reference.device,
            dtype=reference.dtype)

    def frame(self, boxes):
        """Return positive half sizes and their Euclidean norm."""
        a, b = (boxes[:, 2:4].clamp_min(self.min_size) * 0.5).unbind(-1)
        return a, b, torch.sqrt(a.square() + b.square())

    @staticmethod
    def canonicalize(boxes):
        cx, cy, w, h, angle = boxes.unbind(-1)
        swap = w < h
        angle = angle + swap.to(angle.dtype) * (math.pi / 2)
        angle = (angle + math.pi / 2) % math.pi - math.pi / 2
        return torch.stack(
            (cx, cy, torch.where(swap, h, w), torch.where(swap, w, h), angle),
            dim=-1)

    def align_target(self, coarse, target):
        """Select target axes in the half-open [-pi/4, pi/4) branch."""
        delta = (target[:, 4] - coarse[:, 4] + math.pi / 2) % math.pi
        delta = delta - math.pi / 2
        swap = (delta >= math.pi / 4) | (delta < -math.pi / 4)
        phi = torch.where(swap, delta - delta.sign() * math.pi / 2, delta)
        ga = torch.where(swap, target[:, 3], target[:, 2]) * 0.5
        gb = torch.where(swap, target[:, 2], target[:, 3]) * 0.5
        return ga, gb, phi

    def encode_aux(self, coarse, target):
        """Align equivalent target axes, then form exact auxiliary labels."""
        if coarse.shape != target.shape or coarse.shape[-1] != 5:
            raise ValueError('coarse and target must both have shape (N, 5)')
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
                            (gb / phi.cos() - b) / ell, kappa * t),
                           dim=-1)

    def field_from_aux(self, coarse, aux):
        """Apply the fixed five-dimensional field design matrix."""
        a, b, ell = self.frame(coarse)
        s = self.sample_positions(coarse)[None, :]
        tx, ty, da, db, zt = (v[:, None] for v in aux.unbind(-1))
        kappa = self.sample_positions(coarse).square().mean().sqrt()
        t = zt / kappa
        xslope = (b / ell)[:, None] * s * t
        yslope = (a / ell)[:, None] * s * t
        return torch.stack((-tx + da + xslope, tx + da - xslope,
                            -ty + db - yslope, ty + db + yslope),
                           dim=1)

    def encode(self, coarse, target):
        return self.field_from_aux(coarse, self.encode_aux(coarse, target))

    def design_matrix(self, coarse):
        """Return J with shape (N, 4*m, 5), satisfying J.T @ J = 2*m*I."""
        n = coarse.size(0)
        basis = torch.eye(5, device=coarse.device, dtype=coarse.dtype)
        boxes = coarse[:, None, :].expand(n, 5, 5).reshape(-1, 5)
        fields = self.field_from_aux(boxes, basis.repeat(n, 1))
        return fields.reshape(n, 5, 4 * self.num_points).transpose(1, 2)

    def project(self, coarse, field, solver='ols', weights=None):
        """Fit the same observations using algebraic or orthogonal error.

        With the four edge means eliminated, Q(t)=A*t*t+2*B*t+C.
        OLS minimizes Q. Orthogonal fitting minimizes Q/(1+t*t), comparing
        the stationary angle and both interval endpoints. The isotropic
        (constant objective) case deterministically keeps the coarse angle.

        ``weights`` optionally supplies one positive reliability per observed
        edge/sample with shape ``(N, 4, num_points)``. The weighted path uses
        the full five-dimensional design matrix and reduces to ordinary OLS
        for uniform weights.
        """
        if solver not in ('ols', 'orthogonal'):
            raise ValueError('solver must be ols or orthogonal')
        if field.shape != (coarse.shape[0], 4, self.num_points):
            raise ValueError('field must have shape (N, 4, num_points)')
        if weights is not None:
            if weights.shape != field.shape:
                raise ValueError('weights must have the same shape as field')
            if not torch.isfinite(weights).all():
                raise ValueError('weights must be finite')
            weights = weights.to(dtype=field.dtype).clamp_min(1e-4)
            # The scale of the weights does not affect the minimizer. Normalize
            # it to keep the linear system numerically stable across RoIs.
            weights = weights / weights.mean(dim=(1, 2), keepdim=True).clamp_min(1e-4)
            design = self.design_matrix(coarse)
            observations = field.flatten(1)
            diagonal = weights.flatten(1)
            normal = design.transpose(1, 2) @ (design * diagonal[..., None])
            rhs = design.transpose(1, 2) @ (observations * diagonal)[..., None]
            return torch.linalg.solve(normal, rhs).squeeze(-1)
        a, b, ell = self.frame(coarse)
        s = self.sample_positions(field)
        means = field.mean(dim=-1)
        tx = (means[:, 1] - means[:, 0]) * 0.5
        ty = (means[:, 3] - means[:, 2]) * 0.5
        da = (means[:, 1] + means[:, 0]) * 0.5
        db = (means[:, 3] + means[:, 2]) * 0.5
        xmoment = ((field[:, 1] - field[:, 0]) * s).sum(-1)
        ymoment = ((field[:, 3] - field[:, 2]) * s).sum(-1)
        aa = 2 * s.square().sum()
        bb = (b * xmoment - a * ymoment) / ell
        if solver == 'ols':
            t = -bb / aa
            if self.angle_limit is not None:
                limit = math.tan(self.angle_limit)
                t = t.clamp(-limit, limit)
        else:
            if self.angle_limit is None:
                raise ValueError('orthogonal fitting needs an angle_limit; '
                                 'use the same limit for its OLS control')
            cc = (field - means[:, :, None]).square().sum(dim=(1, 2))
            gap = aa - cc
            isotropic = (gap.abs() + 2 * bb.abs()) < (
                torch.finfo(field.dtype).eps * aa)
            # Avoid atan2(0, 0), whose backward pass is undefined.
            y = torch.where(isotropic, torch.zeros_like(bb), -2 * bb)
            x = torch.where(isotropic, torch.ones_like(gap), gap)
            phi = (0.5 * torch.atan2(y, x)).clamp(-self.angle_limit,
                                                  self.angle_limit)
            limit = torch.full_like(bb, math.tan(self.angle_limit))
            candidates = torch.stack((phi.tan(), -limit, limit), dim=-1)
            costs = (aa * candidates.square() + 2 * bb[:, None] * candidates +
                     cc[:, None]) / (1 + candidates.square())
            t = candidates.gather(1, costs.argmin(-1, keepdim=True)).squeeze(1)
        kappa = self.sample_positions(field).square().mean().sqrt()
        return torch.stack((tx, ty, da, db, kappa * t), dim=-1)

    def decode_aux(self, coarse, aux, canonicalize=True, return_clipped=False):
        a, b, ell = self.frame(coarse)
        tx, ty, da, db, zt = aux.unbind(-1)
        kappa = self.sample_positions(coarse).square().mean().sqrt()
        t = zt / kappa
        if self.angle_limit is not None:
            limit = math.tan(self.angle_limit)
            t = t.clamp(-limit, limit)
        norm2 = 1 + t.square()
        dx = ell * (tx - t * ty) / norm2
        dy = ell * (t * tx + ty) / norm2
        cos, sin = coarse[:, 4].cos(), coarse[:, 4].sin()
        cx = coarse[:, 0] + cos * dx - sin * dy
        cy = coarse[:, 1] + sin * dx + cos * dy
        raw_w = 2 * (a + ell * da) / norm2.sqrt()
        raw_h = 2 * (b + ell * db) / norm2.sqrt()
        clipped = (raw_w < self.min_size) | (raw_h < self.min_size)
        w = raw_w.clamp_min(self.min_size)
        h = raw_h.clamp_min(self.min_size)
        angle = coarse[:, 4] + t.atan()
        boxes = torch.stack((cx, cy, w, h, angle), dim=-1)
        if canonicalize:
            boxes = self.canonicalize(boxes)
        return (boxes, clipped) if return_clipped else boxes

    def decode(self, coarse, field, solver='ols', weights=None, **kwargs):
        return self.decode_aux(coarse,
                               self.project(coarse, field, solver, weights),
                               **kwargs)
