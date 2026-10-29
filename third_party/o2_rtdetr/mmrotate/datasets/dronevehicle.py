import glob
import os.path as osp
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict

import mmcv
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import get, get_local_path, join_path
from mmengine.utils import is_abs

from mmrotate.registry import DATASETS


@DATASETS.register_module()
class DroneVehicleDataset(BaseDataset):
    """DroneVehicle dataset for detection.

    Args:
        modality (str): Modality type to load. Options are:
            - 'rgb': Load only RGB images and annotations.
            - 'ir': Load only Infrared images and annotations.
            - 'rgb_ir': Load both RGB and IR data for fused detection.
            Defaults to 'rgb'.
        img_suffix (str): Suffix of the image files (e.g., 'jpg', 'png').
            Defaults to 'jpg'.
        ann_file_rgb (str, optional): Path to the directory or file
            containing RGB XML annotations. Defaults to None.
        ann_file_infrared (str, optional): Path to the directory or file
            containing Infrared (IR) XML annotations. Defaults to None.
        file_client_args (dict): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    METAINFO = {
        'classes':
        ('car', 'freight car', 'truck', 'bus', 'van'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226)]
    }

    def __init__(self,
                 modality: str = 'rgb',
                 img_suffix: str = 'jpg',
                 ann_file_rgb: Optional[str] = None,
                 ann_file_infrared: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 **kwargs) -> None:
        valid_modalities = ['rgb', 'ir', 'rgb_ir']
        assert modality in valid_modalities, \
            f"Unsupported modality: '{modality}'. Must be one of {valid_modalities}"
        self.modality = modality.lower()
        self.img_suffix = img_suffix
        self.backend_args = backend_args
        self.ann_file_rgb = ann_file_rgb
        self.ann_file_infrared = ann_file_infrared
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/dev-1.x/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )
        super().__init__(**kwargs)

    def _join_prefix(self):
        super()._join_prefix()
        modality_map = {
            'rgb': ['ann_file_rgb'],
            'ir': ['ann_file_infrared'],
            'rgb_ir': ['ann_file_rgb', 'ann_file_infrared']
        }
        target_attrs = modality_map[self.modality]
        for attr_name in target_attrs:
            ann_file = getattr(self, attr_name, None)
            assert ann_file, f"In {self.modality} mode, {attr_name} must be provided."
            if not is_abs(ann_file) and self.data_root:
                full_path = join_path(self.data_root, ann_file)
                setattr(self, attr_name, full_path)

    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        assert self._metainfo.get('classes', None) is not None, \
            'classes in `DroneVehicleDataset` can not be None.'
        self.cat2label = {cat: i for i, cat in enumerate(self.metainfo['classes'])}

        main_ann_dir = self.ann_file_rgb or self.ann_file_infrared
        if not main_ann_dir:
            raise ValueError("Either 'ann_file_rgb' or 'ann_file_infrared' must be provided.")

        xml_files = glob.glob(osp.join(main_ann_dir, '*.xml'))
        if not xml_files:
            raise ValueError(f'No XML files found in {main_ann_dir}')

        data_list = []
        for xml_path in xml_files:
            img_id = osp.splitext(osp.basename(xml_path))[0]
            data_info = {'img_id': img_id, 'file_name': f"{img_id}.{self.img_suffix}"}

            modalities_to_load = []
            if self.modality in ['rgb', 'rgb_ir']:
                modalities_to_load.append(('rgb', self.ann_file_rgb, 'img_rgb_path'))
            if self.modality in ['ir', 'rgb_ir']:
                modalities_to_load.append(('ir', self.ann_file_infrared, 'img_infrared_path'))

            for mode_name, ann_dir, prefix_key in modalities_to_load:
                curr_xml = osp.join(ann_dir, f"{img_id}.xml")
                curr_img = osp.join(self.data_prefix[prefix_key], data_info['file_name'])

                res = self._get_xml_and_img_info(curr_xml, curr_img)

                if 'width' not in data_info:
                    data_info.update({'width': res['width'], 'height': res['height']})

                suffix = f"_{mode_name}" if self.modality == 'rgb_ir' else ""
                data_info[f'img_path{suffix}'] = curr_img
                data_info[f'instances{suffix}'] = res['instances']

            data_list.append(data_info)
        return data_list

    def _get_xml_and_img_info(self, xml_path: str, img_path: str) -> Dict:
        with get_local_path(xml_path, backend_args=self.backend_args) as local_path:
            root = ET.parse(local_path).getroot()

        size_node = root.find('size')
        if size_node is not None:
            width = int(size_node.find('width').text)
            height = int(size_node.find('height').text)
        else:
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, backend='cv2')
            height, width = img.shape[:2]

        instances = []
        alias_map = {
            'feright car': 'freight car', 'feright_car': 'freight car',
            'feright': 'freight car', 'truvk': 'truck'
        }

        for obj in root.findall('object'):
            cls_name = obj.find('name').text.lower().strip()
            if not cls_name or cls_name == '*':
                continue

            cls_name = alias_map.get(cls_name, cls_name)
            if cls_name not in self.cat2label:
                raise ValueError(f'Unknown class name: {cls_name}')

            poly = obj.find('polygon')
            bbox_node = obj.find('bndbox')

            if bbox_node is not None:
                coords = [bbox_node.find(k).text for k in
                          ['xmin', 'ymin', 'xmax', 'ymin', 'xmax', 'ymax', 'xmin', 'ymax']]
            elif poly is not None:
                coords = [poly.find(f'{k}{i}').text for i in range(1, 5) for k in ['x', 'y']]
            else:
                continue

            bbox = np.array(coords, dtype=np.float32)

            ignore = False
            if self.bbox_min_size is not None:
                assert not self.test_mode
                if width < self.bbox_min_size or height < self.bbox_min_size:
                    ignore = True
            if ignore:
                ignore_flag = 1
            else:
                ignore_flag = 0

            instances.append({
                'bbox': bbox,
                'bbox_label': self.cat2label[cls_name],
                'ignore_flag': ignore_flag
            })

        return {'width': width, 'height': height, 'instances': instances}

    @property
    def bbox_min_size(self) -> Optional[str]:
        """Return the minimum size of bounding boxes in the images."""
        if self.filter_cfg is not None:
            return self.filter_cfg.get('bbox_min_size', None)
        else:
            return None

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False
        min_size = self.filter_cfg.get('min_size', 0) \
            if self.filter_cfg is not None else 0

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get DIOR category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]