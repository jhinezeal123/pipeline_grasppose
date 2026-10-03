"""Port and data contracts for depth estimation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DepthResult:
    depth: np.ndarray
    intrinsics: np.ndarray
    fov_x_deg: float
    scale: float = 1.0
    reason: Optional[str] = None

    @classmethod
    def empty(cls, height, width, reason=None):
        return cls(
            depth=np.zeros((height, width), np.float32),
            intrinsics=np.eye(3, dtype=np.float32),
            fov_x_deg=0.0,
            scale=1.0,
            reason=reason,
        )


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

# Giữ identity/pickle của đường import đã công khai.
DepthPort.__module__ = "grasppose.modules.depth.port"
