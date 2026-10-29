from mmengine.config import read_base
with read_base():
    from projects.rotated_dino.configs.rotated_dino_4scale_r50_2xb4_12e_dotav15 import *
from mmdet.models.backbones import SwinTransformer
from mmdet.models.necks import ChannelMapper

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model.update(
    backbone=dict(
        type=SwinTransformer,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type=ChannelMapper,
        in_channels=[192, 384, 768],
        out_channels=256,
        num_outs=4))