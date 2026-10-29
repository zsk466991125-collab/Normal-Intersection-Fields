from mmdet.datasets import CocoDataset
from mmrotate.registry import DATASETS

@DATASETS.register_module()
class NWPUDataset(CocoDataset):

    METAINFO = {
        'classes':  ('airplane', 'ship', 'storage_tank', 'baseball_diamond',
                    'tennis_court', 'basketball_court', 'ground_track_field',
                    'harbor', 'bridge', 'vehicle'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230),
                    (119, 11, 32), (0, 255, 0), (0, 0, 255)]
    }