"""Depth contract: RGB + camera calibration -> metric-scaled depth."""

from abc import ABC, abstractmethod

from .types import DepthResult


class DepthPort(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, image, camera_K=None, fov_x=None):
        """Return DepthResult for one frame."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
