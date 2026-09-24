"""TSDF construction port."""

from abc import ABC, abstractmethod

from ..domain.types import TSDFResult


class TSDFPort(ABC):
    @abstractmethod
    def build(self, depth, K, mask=None, cloud=None,
              T_cam_volume=None):
        raise NotImplementedError
