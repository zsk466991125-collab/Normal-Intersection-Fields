"""Train a frozen YOLO26 OBB carrier with NIF."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

NIF_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NIF_ROOT))
sys.path.insert(0, str(NIF_ROOT / "third_party/ultralytics"))
os.environ["PYTHONPATH"] = os.pathsep.join([str(NIF_ROOT / "third_party/ultralytics"), str(NIF_ROOT), os.environ.get("PYTHONPATH", "")])

from ultralytics.utils import DEFAULT_CFG

from adapters.ultralytics.nif_obb import NIFOBBTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26l-obb.pt")
    parser.add_argument("--data", default="data/yolo_dota10/DOTA10_nif.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max-det", type=int, default=700,
                        help="Maximum detections per image during validation/inference")
    parser.add_argument("--device", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="work_dirs/yolo26l_obb_nif_dota10")
    parser.add_argument("--name", default="train_50e_batch32_val1_maxdet700")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-interval", type=int, default=1,
                        help="Run validation every N epochs (1 = every epoch)")
    parser.add_argument("--resume", default=False, help="Resume from a checkpoint path")
    args = parser.parse_args()
    if args.val_interval < 1:
        parser.error("--val-interval must be at least 1")
    os.environ["NIF_VAL_INTERVAL"] = str(args.val_interval)
    NIFOBBTrainer.val_interval = args.val_interval
    overrides = vars(args).copy()
    overrides.pop("val_interval")
    overrides.pop("project")
    overrides["save_dir"] = f"{args.project}/{args.name}"
    overrides.update(
        task="obb",
        mode="train",
        model=args.model,
        data=args.data,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.0,
        close_mosaic=5,
        pretrained=True,
        fraction=args.fraction,
        plots=True,
        val=True,
        max_det=args.max_det,
    )
    trainer = NIFOBBTrainer(cfg=DEFAULT_CFG, overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
