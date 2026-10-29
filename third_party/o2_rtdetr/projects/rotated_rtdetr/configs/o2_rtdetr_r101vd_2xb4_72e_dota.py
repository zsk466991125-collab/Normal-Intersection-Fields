from mmengine.config import read_base
with read_base():
    from .o2_rtdetr_r50vd_2xb4_72e_dota import *

pretrained = ('https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/'
              'master/rtdetr/resnet101vd_ssld_pretrained_64ed664a.pth')  # noqa

model.update(
    backbone=dict(
        depth=101, init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(out_channels=384),
    encoder=dict(
        in_channels=[384, 384, 384],
        fpn_cfg=dict(in_channels=[384, 384, 384]),
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=384),
            ffn_cfg=dict(embed_dims=384, feedforward_channels=2048))))

# optimizer
optim_wrapper.update(
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.01)}, norm_decay_mult=1))