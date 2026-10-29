# Copyright (c) mmrotate. All rights reserved.
from mmdet.datasets import ConcatDataset as MMDET_ConcatDataset
from mmrotate.registry import DATASETS

@DATASETS.register_module()
class ConcatDataset(MMDET_ConcatDataset):

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys. It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys
