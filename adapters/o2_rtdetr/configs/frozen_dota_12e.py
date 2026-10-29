import os

from mmengine.config import read_base
with read_base():
    from adapters.o2_rtdetr.configs.frozen_dota_36e import *

max_epochs = 12
train_cfg.update(max_epochs=12, val_interval=13)
param_scheduler[1].update(milestones=[8, 11])
work_dir = os.path.join(os.environ.get('NIF_WORK_ROOT', 'work_dirs'),
                        'o2_rtdetr_frozen_dota_12e')
