"""Data contracts for TSDF construction."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TSDFResult:
    grid: np.ndarray
    voxel_size: float
    T_cam_volume: np.ndarray
    observed_voxels: int
