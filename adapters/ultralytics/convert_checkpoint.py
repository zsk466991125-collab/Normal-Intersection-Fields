"""Convert a released serialized NIF model to this repository's module path."""
import argparse
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'third_party/ultralytics'))
import torch
from adapters.ultralytics import nif_obb


def convert(source, target):
    aliases = {'scripts': types.ModuleType('scripts'),
               'scripts.ultra_nif': types.ModuleType('scripts.ultra_nif'),
               'scripts.ultra_nif.yolo26_obb': nif_obb}
    previous = {name: sys.modules.get(name) for name in aliases}
    try:
        sys.modules.update(aliases)
        checkpoint = torch.load(source, map_location='cpu')
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('target')
    args = parser.parse_args()
    print(convert(args.source, args.target))
