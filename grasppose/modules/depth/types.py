"""Data contracts for depth estimation."""

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
