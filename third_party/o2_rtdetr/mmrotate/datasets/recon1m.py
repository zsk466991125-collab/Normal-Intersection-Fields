from mmrotate.registry import DATASETS
from mmrotate.datasets import DOTADataset


@DATASETS.register_module()
class ReCon1MDataset(DOTADataset):

    METAINFO = {
        'classes':
        (
            'airplane', 'airport', 'baseball-field', 'basketball-court', 'block',
            'boarding_bridge', 'bridge', 'building', 'bus', 'cargo',
            'cargo-truck', 'chimney', 'construction-site', 'container', 'control-tower',
            'crane', 'dam', 'dry-cargo-ship', 'dump-truck', 'engineering-ship',
            'excavator', 'exhaust-fan', 'expressway-service-area', 'factory', 'farmland',
            'fishing-boat', 'football-field', 'gas-station', 'greenbelt', 'harbor',
            'helicopter-apron', 'intersection', 'liquid-cargo-ship', 'locomotive', 'motorboat',
            'other-ship', 'other-vehicle', 'parking-lot', 'passenger-ship', 'pool',
            'railway', 'road', 'roundabout', 'runway', 'small-car',
            'smoke', 'solar-panel', 'stadium', 'storage-tank', 'storage-tank-group',
            'tennis-court', 'terminal', 'tractor', 'trailer', 'train-carriage',
            'truck-tractor', 'tugboat', 'van', 'warship', 'water'

        ),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0), (0, 128, 0), (128, 0, 128),
            (0, 128, 128), (0, 0, 128), (255, 165, 0), (255, 192, 203), (165, 42, 42), (189, 183, 107),
            (0, 100, 0), (75, 0, 130), (255, 69, 0), (255, 20, 147), (70, 130, 180), (50, 205, 50),
            (255, 140, 0), (106, 90, 205), (220, 20, 60), (173, 255, 47), (255, 215, 0), (95, 158, 160),
            (176, 48, 96), (127, 255, 212), (105, 105, 105), (218, 112, 214), (135, 206, 235), (0, 191, 255),
            (124, 252, 0), (255, 99, 71), (210, 105, 30), (100, 149, 237), (128, 0, 128), (255, 105, 180),
            (173, 216, 230), (240, 230, 140), (230, 230, 250), (245, 222, 179), (205, 133, 63), (250, 250, 210),
            (240, 128, 128), (221, 160, 221), (144, 238, 144), (175, 238, 238), (255, 182, 193), (216, 191, 216),
            (220, 220, 220), (140, 180, 210), (180, 140, 210), (210, 140, 180), (140, 210, 180), (180, 210, 140),
        ]
    }