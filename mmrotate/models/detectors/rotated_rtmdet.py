# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector


@ROTATED_DETECTORS.register_module()
class RotatedRTMDet(RotatedSingleStageDetector):
    """Rotated RTMDet using the existing MMRotate 0.x train/test interfaces."""
