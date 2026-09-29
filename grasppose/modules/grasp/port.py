"""Grasp inference contract: TSDF -> grasp poses."""

from abc import ABC, abstractmethod


class GraspPort(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, tsdf):
        """Return a grasp result from a TSDF result."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
