"""Reject partial carrier loading; permit new NIF initialization only."""
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmrotate.registry import HOOKS


@HOOKS.register_module()
class StrictFrozenNIFLoadHook(Hook):
    def after_load_checkpoint(self, runner, checkpoint):
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        state = checkpoint['state_dict']
        expected = model.state_dict()
        nif_keys = {k for k in expected if k.startswith('nif_head.')}
        new_nif = not any(k.startswith('nif_head.') for k in state)
        required = set(expected) - nif_keys if new_nif else set(expected)
        missing = sorted(required - set(state))
        unexpected = sorted(set(state) - set(expected))
        mismatch = [k for k in state if k in expected and state[k].shape != expected[k].shape]
        if missing or unexpected or mismatch:
            raise RuntimeError(f'Checkpoint mismatch: missing={missing}, '
                               f'unexpected={unexpected}, shape={mismatch}')
        if new_nif:
            for key in nif_keys:
                state[key] = expected[key]
        runner.logger.info(
            f'Strict carrier load: {len(required)} entries; '
            f'NIF {"initialized" if new_nif else "restored"}; only NIF is trainable.')

    def before_train(self, runner):
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        if any(p.requires_grad for n, p in model.named_parameters()
               if not n.startswith('nif_head.')):
            raise RuntimeError('Carrier must be entirely frozen')
