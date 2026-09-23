"""Grasp inference port: TSDF -> grasp poses."""

from abc import ABC, abstractmethod

from ..domain.types import GraspResult, TSDFResult


class GraspPort(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, tsdf):
        """Return GraspResult from a TSDFResult."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
