# Copyright (c) OpenMMLab. All rights reserved.

from .yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS
from .dota import DOTAv15Dataset


@DATASETS.register_module()
class YOLOv5DOTA15Dataset(BatchShapePolicyDataset, DOTAv15Dataset):
    """Dataset for YOLOv5 DOTA Dataset.

    We only add `BatchShapePolicy` function compared with DOTADataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)