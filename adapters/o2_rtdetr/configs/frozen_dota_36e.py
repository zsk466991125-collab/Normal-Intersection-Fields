import os

from mmengine.config import read_base
from adapters.o2_rtdetr import (FrozenO2RTDETRNIF, NIFOnlyOptimWrapperConstructor,
                            DOTASlidingWindowDataset, LoadDOTAWindow,
                            StrictFrozenNIFLoadHook)
from mmengine.optim.scheduler.lr_scheduler import MultiStepLR

with read_base():
    from projects.rotated_rtdetr.configs.o2_rtdetr_r50vd_2xb4_72e_dota import *

# Preserve the entire pretrained O2 architecture; append only the NIF head.
model.update(type=FrozenO2RTDETRNIF, use_syncbn=False, nif_cfg=dict(
    num_points=5, sample_extent=0.75, max_pos_per_img=128,
    iou_bins=[0.3, 0.45, 0.6, 0.75, 0.9, 1.01],
    reliability_power=1.5, reliability_error_scale=0.25,
    reliability_loss_weight=0.05))
model['backbone']['init_cfg'] = None

data_root = os.path.join(os.environ.get('NIF_DATA_ROOT', 'data'), 'dota') + '/'
# Same 1024 single-scale resize, 3-direction flip (.25 each), and random
# rotation (.5, +/-180 degrees, rectangle classes 9/11) as frozen LSK FAAHead.
# Mean/std and channel order stay native O2; equivalent ImageNet BGR/RGB pairs.
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.75,
         direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', prob=0.5, angle_range=180, rect_obj_labels=[9, 11]),
    dict(type='mmdet.Pad', size_divisor=32, pad_val=dict(img=(104, 116, 124))),
    dict(type='mmdet.PackDetInputs')]
train_dataloader.update(batch_size=2, num_workers=4)
train_dataloader['dataset'].update(
    data_root=data_root, ann_file='trainval/annfiles/',
    data_prefix=dict(img_path='trainval/images/'), diff_thr=100,
    pipeline=train_pipeline)

# Official unlabeled test: virtual patches, native rotated-NMS patch merge,
# Task1 ZIP. No test GT or validation is consulted during training.
test_dataloader.update(batch_size=2, num_workers=4)
test_dataloader['dataset'] = dict(
    type=DOTASlidingWindowDataset, data_root='',
    data_prefix=dict(img_path=os.path.join(
        os.environ.get('NIF_DATA_ROOT', 'data'), 'dota', 'test', 'images')),
    test_mode=True, pipeline=[dict(type=LoadDOTAWindow),
        dict(type='mmdet.PackDetInputs', meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))])
test_evaluator.update(outfile_prefix=os.path.join(
    os.environ.get('NIF_WORK_ROOT', 'work_dirs'), 'o2_nif_submission'))

max_epochs = 36
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=36, val_interval=37)
val_cfg = None
val_dataloader = None
val_evaluator = None
custom_hooks = [dict(type=StrictFrozenNIFLoadHook)]  # No EMA.
optim_wrapper = dict(
    type=OptimWrapper, constructor=NIFOnlyOptimWrapperConstructor,
    optimizer=dict(type=AdamW, lr=0.0001, weight_decay=0.05, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(type=LinearLR, start_factor=1/3, by_epoch=False, begin=0, end=500),
    dict(type=MultiStepLR, by_epoch=True, milestones=[24, 33], gamma=0.1)]
default_hooks['checkpoint'].update(interval=1, max_keep_ckpts=5)
default_hooks.pop('visualization', None)
auto_scale_lr = dict(enable=False, base_batch_size=8)
model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=False,
                         broadcast_buffers=False)
load_from = os.environ.get('NIF_O2_CHECKPOINT')
resume = False
randomness = dict(seed=0, deterministic=False)
work_dir = os.path.join(os.environ.get('NIF_WORK_ROOT', 'work_dirs'),
                        'o2_rtdetr_frozen_dota_36e')
