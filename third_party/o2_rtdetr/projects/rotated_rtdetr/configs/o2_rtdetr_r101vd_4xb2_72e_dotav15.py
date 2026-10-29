from mmengine.config import read_base
with read_base():
    from .o2_rtdetr_r50vd_4xb2_72e_dotav15 import *

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

train_dataloader.update(batch_size=2, num_workers=4)
val_dataloader.update(batch_size=4, num_workers=4)
test_dataloader.update(batch_size=4, num_workers=4)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (4 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(enable=False, base_batch_size=8)