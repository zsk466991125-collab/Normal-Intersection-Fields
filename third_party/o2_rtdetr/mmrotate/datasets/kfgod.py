import glob
import os.path as osp
from typing import List, Optional
import torch
from mmrotate.registry import DATASETS
from mmrotate.datasets import DOTADataset
from mmrotate.structures.bbox import qbox2rbox


@DATASETS.register_module()
class KFGODDataset(DOTADataset):

    METAINFO = {
        'classes':
        (
            'aquaculture_farm', 'barge', 'bridge', 'bus', 'container', 'container_group',
            'container_ship', 'crane', 'dam', 'drill_ship', 'ferry', 'fighter_aircraft',
            'fishing_boat', 'helicopter', 'helipad',
            'large_civilian_aircraft', 'large_military_aircraft', 'marine_research_station',
            'motorboat', 'oil_tanker', 'roundabout', 'sailboat', 'small_civilian_aircraft', 'small_vehicle',
            'sports_field', 'stadium', 'storage_tank', 'swimming_pool', 'train', 'truck',
            'tugboat', 'warship', 'wind_turbine'

        ),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0), (0, 128, 0), (128, 0, 128),
            (0, 128, 128), (0, 0, 128), (255, 165, 0), (255, 192, 203), (165, 42, 42), (189, 183, 107),
            (0, 100, 0), (75, 0, 130), (255, 69, 0), (255, 20, 147), (70, 130, 180), (50, 205, 50),
            (255, 140, 0), (106, 90, 205), (220, 20, 60), (173, 255, 47), (255, 215, 0), (95, 158, 160),
            (176, 48, 96), (127, 255, 212), (105, 105, 105)
        ]
    }

    @property
    def bbox_min_size(self) -> Optional[str]:
        """Return the minimum size of bounding boxes in the images."""
        if self.filter_cfg is not None:
            return self.filter_cfg.get('bbox_min_size', None)
        else:
            return None

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in '
                                 f'{self.ann_file}')
            for txt_file in txt_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)

                instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        ignore = False
                        instance = {}
                        bbox_info = si.split()
                        qbox = [float(i) for i in bbox_info[:8]]
                        instance['bbox'] = qbox
                        rbbox = qbox2rbox(torch.tensor(qbox))
                        w = rbbox[2]
                        h = rbbox[3]
                        if self.bbox_min_size is not None:
                            assert not self.test_mode
                            if w < self.bbox_min_size or h < self.bbox_min_size:
                                ignore = True
                        cls_name = bbox_info[8]
                        instance['bbox_label'] = cls_map[cls_name]
                        difficulty = int(bbox_info[9])
                        if difficulty > self.diff_thr:
                            ignore = True
                        if ignore:
                            instance['ignore_flag'] = 1
                        else:
                            instance['ignore_flag'] = 0
                        instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list