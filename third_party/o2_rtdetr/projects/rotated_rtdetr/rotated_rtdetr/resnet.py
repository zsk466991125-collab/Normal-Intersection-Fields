from mmdet.models.backbones import ResNetV1d
from .res_layer import ResLayer


class ResNetV1dPaddle(ResNetV1d):
    '''
        The downsampling in the 1st layer is different between mmdet and Paddle.
        The mmdet does not have backbone.layer1.0.downsample.
        The Paddel has backbone.layer1.0.downsample.
        Add backbone.layer1.0.downsample.
    '''
    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)