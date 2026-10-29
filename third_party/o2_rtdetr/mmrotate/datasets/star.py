from mmrotate.registry import DATASETS
from mmrotate.datasets import DOTADataset

@DATASETS.register_module()
class STARDataset(DOTADataset):
    """STAR dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """

    METAINFO = {
        'classes':
            ('ship', 'boat', 'crane', 'goods_yard', 'tank', 'storehouse', 'breakwater', 'dock',
               'airplane', 'boarding_bridge', 'runway', 'taxiway', 'terminal', 'apron', 'gas_station',
               'truck', 'car', 'truck_parking', 'car_parking', 'bridge', 'cooling_tower', 'chimney',
               'vapor', 'smoke', 'genset', 'coal_yard', 'lattice_tower', 'substation', 'wind_mill',
               'cement_concrete_pavement', 'toll_gate', 'flood_dam', 'gravity_dam', 'ship_lock',
               'ground_track_field', 'basketball_court', 'engineering_vehicle', 'foundation_pit',
               'intersection', 'soccer_ball_field', 'tennis_court', 'tower_crane', 'unfinished_building',
               'arch_dam', 'roundabout', 'baseball_diamond', 'stadium', 'containment_vessel'),
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255),
               (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208), (166, 196, 102), (208, 195, 210), (255, 109, 65),
               (0, 143, 149), (179, 0, 194)]
    }