from mmengine.config import read_base
with read_base():
    from .o2_rtdetr_r50vd_2xb4_coco_pretrain_72e_dota_ms import *

pretrained = ('https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/'
              'master/rtdetr/rtdetr_r18vd_8xb2-72e_coco_3dda8dd4.pth')  # noqa

model.update(
    init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    backbone=dict(
        depth=18,
        frozen_stages=-1,
        norm_cfg=dict(requires_grad=True),
        norm_eval=False,
        init_cfg=dict()),
    neck=dict(in_channels=[128, 256, 512]),
    encoder=dict(fpn_cfg=dict(expansion=0.5)),
    decoder=dict(num_layers=3))

# set all norm layers in backbone to lr_mult=0.1 and decay_mult=0.0
# set all other layers in backbone to lr_mult=0.1
num_blocks_list = (2, 2, 2, 2)  # r18
downsample_norm_idx_list = (2, 3, 3, 3)  # r18
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
custom_keys = {'backbone': dict(lr_mult=0.1)}
custom_keys.update({
    f'backbone.layer{stage_id + 1}.{block_id}.bn': backbone_norm_multi
    for stage_id, num_blocks in enumerate(num_blocks_list)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.layer{stage_id + 1}.{block_id}.downsample.{downsample_norm_idx - 1}':  # noqa
    backbone_norm_multi
    for stage_id, (num_blocks, downsample_norm_idx) in enumerate(
        zip(num_blocks_list, downsample_norm_idx_list))
    for block_id in range(num_blocks)
})

# optimizer
optim_wrapper.paramwise_cfg.pop('custom_keys')
optim_wrapper.update(
    paramwise_cfg=dict(custom_keys=dict(**custom_keys)))