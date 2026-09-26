"""TSDF construction contract."""

from abc import ABC, abstractmethod


class TSDFPort(ABC):
    @abstractmethod
    def build(self, depth, K, mask=None, cloud=None, T_cam_volume=None):
        raise NotImplementedError
