"""Differentiable exact aligned rotated IoU using polygon intersections."""
import torch
import torch.nn as nn

try:
    from ..builder import ROTATED_LOSSES
except ImportError:
    # Standalone reuse by MMDetection 3 carriers.
    from mmdet.registry import MODELS as ROTATED_LOSSES


def _cross(first, second):
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _corners(boxes):
    signs = boxes.new_tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    points = signs[None] * boxes[:, None, 2:4] * 0.5
    cos, sin = boxes[:, 4].cos()[:, None], boxes[:, 4].sin()[:, None]
    return torch.stack(
        (cos * points[..., 0] - sin * points[..., 1],
         sin * points[..., 0] + cos * points[..., 1]), -1) + boxes[:, None, :2]


def _inside(points, boxes):
    offset = points - boxes[:, None, :2]
    cos, sin = boxes[:, 4].cos()[:, None], boxes[:, 4].sin()[:, None]
    x = cos * offset[..., 0] + sin * offset[..., 1]
    y = -sin * offset[..., 0] + cos * offset[..., 1]
    return ((x.abs() <= boxes[:, 2:3] * .5 + 1e-10)
            & (y.abs() <= boxes[:, 3:4] * .5 + 1e-10))


def exact_aligned_rotated_iou(pred, target):
    """Return (N,) exact polygon IoU, differentiable through box coordinates.

    A common translation/scale improves conditioning without changing IoU.
    At touching/parallel boundaries gradients use the selected piecewise
    polygon topology, as in other exact IoU implementations.
    """
    if pred.shape != target.shape or pred.ndim != 2 or pred.size(1) != 5:
        raise ValueError('pred and target must have matching shape (N,5)')
    if not len(pred):
        return pred.sum(-1) * 0
    dtype = pred.dtype
    p, g = pred.double(), target.double()
    origin = g[:, :2].detach()
    scale = torch.maximum(p[:, 2:4].amax(-1),
                          g[:, 2:4].amax(-1)).detach().clamp_min(1e-12)
    p = torch.cat(((p[:, :2] - origin) / scale[:, None],
                   p[:, 2:4] / scale[:, None], p[:, 4:]), -1)
    g = torch.cat(((g[:, :2] - origin) / scale[:, None],
                   g[:, 2:4] / scale[:, None], g[:, 4:]), -1)
    first, second = _corners(p), _corners(g)
    # Pair all 4 x 4 edges and collect genuine segment intersections.
    a, b = first[:, :, None], second[:, None, :]
    r = (first.roll(-1, dims=1) - first)[:, :, None]
    s = (second.roll(-1, dims=1) - second)[:, None, :]
    denominator = _cross(r, s)
    relative_tol = 1e-12 * r.norm(dim=-1) * s.norm(dim=-1)
    nonparallel = denominator.abs() > relative_tol
    safe = torch.where(nonparallel, denominator, torch.ones_like(denominator))
    t = _cross(b - a, s) / safe
    u = _cross(b - a, r) / safe
    mask = nonparallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    intersections = a + t[..., None] * r
    vertices = torch.cat((first, second, intersections.flatten(1, 2)), dim=1)
    valid = torch.cat((_inside(first, g), _inside(second, p), mask.flatten(1)),
                      dim=1)
    count = valid.sum(1)
    center = (vertices * valid[..., None]).sum(1) / count.clamp_min(1)[:, None]
    # Sorting has no coordinate derivative, but selected vertices do.
    with torch.no_grad():
        local = vertices.detach() - center.detach()[:, None]
        angles = torch.atan2(local[..., 1], local[..., 0])
        order = angles.masked_fill(~valid, float('inf')).argsort(1)
    polygon = vertices.gather(1, order[..., None].expand(-1, -1, 2))
    positions = torch.arange(
        vertices.size(1), device=pred.device)[None].expand(len(pred), -1)
    next_index = (positions + 1) % count.clamp_min(1)[:, None]
    following = polygon.gather(1, next_index[..., None].expand(-1, -1, 2))
    twice_area = (_cross(polygon, following) *
                  (positions < count[:, None])).sum(1).abs()
    area_p, area_g = p[:, 2] * p[:, 3], g[:, 2] * g[:, 3]
    intersection = torch.minimum(twice_area * .5,
                                 torch.minimum(area_p, area_g))
    union = (area_p + area_g - intersection).clamp_min(
        torch.finfo(p.dtype).tiny)
    return (intersection / union).clamp(0, 1).to(dtype)


@ROTATED_LOSSES.register_module()
class NIFRotatedIoULoss(nn.Module):
    """Mean of exact 1-rIoU, shared by every NIF refinement control."""

    def forward(self, pred, target):
        return (1 - exact_aligned_rotated_iou(pred, target)).mean()
