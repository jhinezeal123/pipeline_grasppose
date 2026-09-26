"""Data contracts for grasp inference."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GraspResult:
    graspgroup: np.ndarray
    reason: Optional[str] = None

    @classmethod
    def empty(cls, reason=None):
        return cls(
            graspgroup=np.zeros((0, 17), np.float64),
            reason=reason,
        )
