"""Strict carrier loading and NIF-only optimization."""
import torch
from mmcv.runner import HOOKS, Hook, OPTIMIZER_BUILDERS
from mmcv.runner.checkpoint import _load_checkpoint
from mmcv.parallel import is_module_wrapper


@OPTIMIZER_BUILDERS.register_module()
class NIFOnlyOptimizerConstructor:
    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        self.cfg = optimizer_cfg.copy()

    def __call__(self, model):
        model = model.module if is_module_wrapper(model) else model
        parameters = list(model.roi_head.nif_head.parameters())
        if {id(p) for p in parameters} != {id(p) for p in model.parameters() if p.requires_grad}:
            raise RuntimeError('Only NIF parameters may be trained')
        cfg = self.cfg.copy()
        optimizer = getattr(torch.optim, cfg.pop('type'))
        return optimizer(parameters, **cfg)


@HOOKS.register_module()
class StrictFrozenCarrierHook(Hook):
    """Check every detector tensor before allowing a frozen training run."""
    def before_run(self, runner):
        model = runner.model.module if is_module_wrapper(runner.model) else runner.model
        # The training entry point stores the actual load/resume filename.
        filename = getattr(model, '_nif_loaded_checkpoint', None)
        if filename is None:
            raise RuntimeError('Frozen training requires load_from or resume_from')
        state = _load_checkpoint(filename, map_location='cpu')
        state = state.get('state_dict', state)
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        expected = model.state_dict()
        required = {k for k in expected if not k.startswith('roi_head.nif_head.')}
        missing = required - state.keys()
        extra = state.keys() - expected.keys()
        mismatch = [k for k in required & state.keys() if state[k].shape != expected[k].shape]
        if missing or extra or mismatch:
            raise RuntimeError(f'Carrier mismatch: missing={sorted(missing)}, extra={sorted(extra)}, shape={mismatch}')
        if any(p.requires_grad for n, p in model.named_parameters() if not n.startswith('roi_head.nif_head.')):
            raise RuntimeError('Detector parameters must be frozen')
        runner.logger.info('Strict carrier check passed; optimizing NIF only.')
