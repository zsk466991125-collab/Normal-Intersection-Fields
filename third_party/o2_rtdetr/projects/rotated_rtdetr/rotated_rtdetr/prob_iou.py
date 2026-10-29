from __future__ import annotations

import math
import numpy as np
import torch

def _get_covariance_matrix(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate covariance matrix from oriented bounding boxes.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

def probiou(obb1: torch.Tensor, obb2: torch.Tensor, CIoU: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
            x, y, w, and h are all real-valued coordinates, not normalized
            to the range [0, 1]. For example, for an image of size 1024×1024,
            the values of x, y, w, and h all lie within the range [0, 1024].
            The angle r is given in radians and is also a real value,
            not normalized to [0, 1]. For instance, its range is [0, π].
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
            x, y, w, and h are all real-valued coordinates, not normalized
            to the range [0, 1]. For example, for an image of size 1024×1024,
            the values of x, y, w, and h all lie within the range [0, 1024].
            The angle r is given in radians and is also a real value,
            not normalized to [0, 1]. For instance, its range is [0, π].
        CIoU (bool, optional): If True, calculate CIoU.
        eps (float, optional): Small value to avoid division by zero.

    Returns:
        (torch.Tensor): OBB similarities, shape (N, 1).

    Notes:
        OBB format: [center_x, center_y, width, height, rotation_angle].

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou

def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate the probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd

if __name__ == '__main__':

    img_size = 1024.0
    base_pixel = torch.tensor([[512.0, 512.0, 200.0, 100.0, math.pi / 4]])

    base_norm = base_pixel.clone()
    base_norm[:, :4] /= img_size

    iou_wrong = probiou(base_norm, base_norm)
    print(f"Normalized input → ProbIoU = {iou_wrong.item():.6f} (Should be ≈1.0)")

    iou_correct = probiou(base_pixel, base_pixel)
    print(f"Original scale input → ProbIoU = {iou_correct.item():.6f} (Should be ≈1.0)")

    box1_pixel = torch.tensor([[0.0, 0.0, 100.0, 50.0, 0.5]])
    box2_pixel = torch.tensor([[0.0, 0.0, 100.0, 50.0, 1.5]])

    iou_scale = probiou(box1_pixel, box2_pixel)
    print(f"ProbIoU = {iou_scale.item():.6f}")

    box1_norm = box1_pixel.clone()
    box2_norm = box2_pixel.clone()
    box1_norm[:, 4] /= math.pi
    box2_norm[:, 4] /= math.pi
    iou_scale_wrong = probiou(box1_norm, box2_norm)
    print(f"ProbIoU = {iou_scale_wrong.item():.6f} (Different from above！)")


    box1_norm[:, :4] /= img_size
    box2_norm[:, :4] /= img_size
    iou_scale_wrong = probiou(box1_norm, box2_norm)
    print(f"ProbIoU = {iou_scale_wrong.item():.6f} (Different from above！)")
