"""Vision port: RGB + prompt -> detections and target mask."""

from abc import ABC, abstractmethod

from ..domain.types import VisionResult


class VisionPort(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, image, prompt):
        """Return VisionResult for one frame."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
