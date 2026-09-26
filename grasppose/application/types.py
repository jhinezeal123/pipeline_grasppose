"""Application-level aggregate results."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..modules.depth.types import DepthResult
from ..modules.grasp.types import GraspResult
from ..modules.tsdf.types import TSDFResult
from ..modules.vision.types import VisionResult


@dataclass
class PipelineResult:
    vision: VisionResult
    depth: DepthResult
    cloud: np.ndarray
    tsdf: Optional[TSDFResult]
    grasp: GraspResult
    camera_K: np.ndarray
    depth_m: Optional[float]
