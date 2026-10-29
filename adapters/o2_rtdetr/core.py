"""Load the shared NIF geometry components for the O2-RTDETR adapter."""
import importlib.util
import sys
from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
COMPONENT_ROOT = PACKAGE_ROOT / 'mmrotate'


def source_module(name, relative):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, COMPONENT_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


NIFLocalHead = source_module(
    '_o2_nif_local', 'models/roi_heads/nif_local_head.py').NIFLocalHead
NIFOBBCoder = source_module(
    '_o2_nif_coder', 'core/bbox/coder/nif_obb_coder.py').NIFOBBCoder
exact_aligned_rotated_iou = source_module(
    '_o2_nif_iou', 'models/losses/nif_iou_loss.py').exact_aligned_rotated_iou


def stratified_indices(ious, budget, bins):
    """Use the same budget/bin/fill rule as the two-stage adapter."""
    candidates = torch.arange(len(ious), device=ious.device)
    if budget <= 0 or len(candidates) <= budget:
        return candidates
    quota = max(1, (budget + len(bins) - 2) // (len(bins) - 1))
    groups = []
    for left, right in zip(bins, bins[1:]):
        group = candidates[(ious >= left) & (ious < right)]
        groups.append(group[torch.randperm(len(group), device=group.device)[:quota]])
    selected = torch.unique(torch.cat(groups), sorted=False)
    if len(selected) > budget:
        selected = selected[torch.randperm(len(selected), device=ious.device)[:budget]]
    if len(selected) < budget:
        used = torch.zeros(len(ious), dtype=torch.bool, device=ious.device)
        used[selected] = True
        remaining = candidates[~used]
        order = torch.randperm(len(remaining), device=ious.device)
        selected = torch.cat((selected, remaining[order[:budget-len(selected)]]))
    return selected
