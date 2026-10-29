"""Native MMEngine training/testing with the bundled O2 runtime."""
import argparse
import os
import sys
from types import ModuleType
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'third_party/o2_rtdetr'))
if os.environ.get('NIF_O2_DEPS'):
    sys.path.insert(0, os.environ['NIF_O2_DEPS'])

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmdet.utils import register_all_modules as register_mmdet
from mmrotate.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train','test'])
    parser.add_argument('config')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--launcher', default='none', choices=['none','pytorch'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--local-rank','--local_rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    register_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    for name, value in list(cfg.items()):
        if isinstance(value, ModuleType):
            cfg.pop(name)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.launcher = args.launcher
    cfg.work_dir = str(Path(args.work_dir).resolve())
    cfg.resume = args.resume if args.mode == 'train' else False
    if args.mode == 'test':
        if not args.checkpoint:
            parser.error('--checkpoint is required for testing')
        cfg.load_from = args.checkpoint
        cfg.test_evaluator.outfile_prefix = str(Path(cfg.work_dir)/'Task1')
    runner = Runner.from_cfg(cfg)
    runner.train() if args.mode == 'train' else runner.test()


if __name__ == '__main__':
    main()
