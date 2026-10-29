"""Frozen O2-RTDETR with the NIF geometry refinement branch."""
from .model import FrozenO2RTDETRNIF, NIFOnlyOptimWrapperConstructor
from .data import DOTASlidingWindowDataset, LoadDOTAWindow
from .hooks import StrictFrozenNIFLoadHook

__all__ = ['FrozenO2RTDETRNIF', 'NIFOnlyOptimWrapperConstructor',
           'DOTASlidingWindowDataset', 'LoadDOTAWindow']
