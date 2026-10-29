# Copyright (c) OpenMMLab. All rights reserved.
"""RTMDet's MMRotate 1.x augmentation geometry with the 0.x array API."""
import cv2
import mmcv
import numpy as np
from mmdet.datasets.pipelines import RandomFlip

from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class RTMDetRandomFlip(RandomFlip):
    def bbox_flip(self, bboxes, img_shape, direction):
        # RotatedBoxes in 1.x uses width/height rather than width-1/height-1.
        boxes = bboxes.copy()
        if direction in ('horizontal', 'diagonal'):
            boxes[..., 0] = img_shape[1] - boxes[..., 0]
        if direction in ('vertical', 'diagonal'):
            boxes[..., 1] = img_shape[0] - boxes[..., 1]
        if direction in ('horizontal', 'vertical'):
            boxes[..., 4] = -boxes[..., 4]
        return boxes


@ROTATED_PIPELINES.register_module()
class RTMDetRandomRotate:
    def __init__(self, prob=0.5, angle_range=180, rect_obj_labels=(9, 11)):
        self.prob = prob
        self.angle_range = angle_range
        self.rect_obj_labels = rect_obj_labels

    def __call__(self, results):
        if np.random.rand() >= self.prob:
            return results
        angle = self.angle_range * (2 * np.random.rand() - 1)
        if np.isin(results['gt_labels'], self.rect_obj_labels).any():
            angle = np.random.choice([90, 180, -90, -180])
        height, width = results['img_shape'][:2]
        results['img'] = mmcv.imrotate(results['img'], angle, border_value=0)
        # Match RotatedBoxes.rotate_: its box center is (W/2,H/2), while
        # mmcv.imrotate uses the image pixel center ((W-1)/2,(H-1)/2).
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -angle, 1)
        for name in results.get('bbox_fields', []):
            boxes = results[name].copy()
            centers = np.column_stack((boxes[:, :2], np.ones(len(boxes))))
            boxes[:, :2] = centers @ matrix.T
            boxes[:, 4] += angle * np.pi / 180
            valid = ((boxes[:, 0] >= 0) & (boxes[:, 0] <= width) &
                     (boxes[:, 1] >= 0) & (boxes[:, 1] <= height))
            results[name] = boxes[valid]
            if name == 'gt_bboxes':
                results['gt_labels'] = results['gt_labels'][valid]
        return results
