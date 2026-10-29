"""Virtual test patches: reuse existing DOTA pixels, no extra dataset copy."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from mmcv.transforms import BaseTransform
from mmengine.dataset import BaseDataset
from mmrotate.datasets import DOTADataset
from mmrotate.registry import DATASETS, TRANSFORMS
from .core import source_module

get_sliding_window = source_module(
    '_o2_dota_split', '../data/dota/split/img_split.py').get_sliding_window


@DATASETS.register_module()
class DOTASlidingWindowDataset(BaseDataset):
    """Same 1024/200/.6 windows as the current LSK frozen submission script."""
    METAINFO = DOTADataset.METAINFO

    def __init__(self, *args, max_samples=None, **kwargs):
        # A small cap is useful for validating the test pipeline without
        # changing the full submission dataset when the option is omitted.
        self.max_samples = max_samples
        super().__init__(*args, **kwargs)

    def load_data_list(self):
        records = []
        paths = sorted(Path(self.data_prefix['img_path']).glob('*'))
        for path in paths:
            if path.suffix.lower() not in ('.png', '.jpg', '.tif', '.tiff', '.bmp'):
                continue
            with Image.open(path) as image:
                width, height = image.size
            windows = get_sliding_window(dict(width=width, height=height), [1024], [200], 0.6)
            for window in windows:
                x, y, _, _ = window.tolist()
                records.append(dict(img_id=f'{path.stem}__1024__{x}___{y}',
                                    img_path=str(path), window=window.tolist(),
                                    instances=[]))
        if not records:
            raise RuntimeError(f'No DOTA test images in {self.data_prefix}')
        if self.max_samples is not None:
            records = records[:int(self.max_samples)]
        return records


@lru_cache(maxsize=1)
def read_original(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f'Cannot read {path}')
    return image


@TRANSFORMS.register_module()
class LoadDOTAWindow(BaseTransform):
    def transform(self, results):
        image = read_original(results['img_path'])
        x1, y1, x2, y2 = results['window']
        crop = image[y1:y2, x1:x2]
        patch = np.empty((y2-y1, x2-x1, 3), dtype=np.uint8)
        patch[:] = [104, 116, 124]
        patch[:crop.shape[0], :crop.shape[1]] = crop
        results.update(img=patch, img_shape=patch.shape[:2],
                       ori_shape=patch.shape[:2], scale_factor=(1., 1.))
        return results
