# Copyright (c) Open-CD. All rights reserved.
# Copyright (c) AI4RS. All rights reserved.
from typing import List

import torch
from torch import Tensor

from .siamencoder_decoder import SiamEncoderDecoder
from mmrotate.registry import MODELS

@MODELS.register_module()
class DIEncoderDecoder(SiamEncoderDecoder):
    """Dual Input Encoder Decoder segmentors.

    DIEncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # `in_channels` is not in the ATTRIBUTE for some backbone CLASS.
        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        x = self.backbone(img_from, img_to)
        if self.with_neck:
            x = self.neck(x)
        return x