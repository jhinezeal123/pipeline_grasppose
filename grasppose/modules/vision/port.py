"""Vision contract: RGB + prepared prompt ID -> detections and target mask."""

from abc import ABC, abstractmethod

from .types import VisionResult


class VisionPort(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, image, prompt_id):
        """Return VisionResult for one frame."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
