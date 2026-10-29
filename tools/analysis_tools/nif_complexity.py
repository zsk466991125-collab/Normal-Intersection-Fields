"""MMCV supported-module FLOP accounting."""
from mmcv.cnn.utils.flops_counter import get_modules_mapping

EXCLUDED = [
    'Rotated RoIAlign custom CUDA operator',
    'grid_sample bilinear interpolation and strip-coordinate construction',
    'Functional tensor arithmetic, attention gating, GELU, SiLU, sigmoid, softmax',
    'Box encoding/decoding, NIF analytic geometry, proposal filtering/sorting/NMS',
    'Training losses, backward pass, optimizer updates, memory operations',
]

def parameter_counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))

def group_name(name):
    if name.startswith('roi_head.nif_head.feature_projection'):
        return 'nif_projection'
    if name.startswith('roi_head.nif_head'):
        return 'nif_local'
    if name.startswith('roi_head'):
        return 'roi_head'
    return name.split('.')[0]

class ModuleCounter:
    """Run the exact MMCV hook formulas, recording all calls and components."""

    def __init__(self, model):
        self.model = model
        self.modules = []
        self.handles = []
        self.mapping = get_modules_mapping()

    def __enter__(self):
        for name, module in self.model.named_modules():
            if type(module) in self.mapping:
                if hasattr(module, '__flops__'):
                    raise RuntimeError(f'Already instrumented module: {name}')
                module.__flops__ = 0
                self.modules.append((name, module))
                self.handles.append(module.register_forward_hook(self.mapping[type(module)]))
        return self

    def values(self):
        components, layers = {}, []
        for name, module in self.modules:
            flops = int(module.__flops__)
            key = group_name(name)
            components[key] = components.get(key, 0) + flops
            layers.append(dict(name=name, type=type(module).__name__, flops=flops))
        return dict(flops=sum(components.values()), components=components, layers=layers)

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        for _, module in self.modules:
            del module.__flops__
