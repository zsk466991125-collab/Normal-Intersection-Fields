# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import ROTATED_DETECTORS
from .two_stage import RotatedTwoStageDetector


@ROTATED_DETECTORS.register_module()
class OrientedRCNN(RotatedTwoStageDetector):
    """Implementation of `Oriented R-CNN for Object Detection.`__

    __ https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf  # noqa: E501, E261.
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 freeze_except_nif=False):
        super(OrientedRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.freeze_except_nif = bool(freeze_except_nif)
        if self.freeze_except_nif:
            nif_head = getattr(self.roi_head, 'nif_head', None)
            if nif_head is None:
                raise ValueError('freeze_except_nif requires a NIF RoI head')
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in nif_head.parameters():
                parameter.requires_grad_(True)
            if not any(parameter.requires_grad for parameter in self.parameters()):
                raise ValueError('The NIF RoI head has no trainable parameters')

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_except_nif:
            # Keep frozen BatchNorm statistics and all carrier modules fixed.
            self.backbone.eval()
            if self.with_neck:
                self.neck.eval()
            if self.with_rpn:
                self.rpn_head.eval()
            self.roi_head.bbox_roi_extractor.eval()
            self.roi_head.bbox_head.eval()
        return self

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmrotate/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 6).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs
