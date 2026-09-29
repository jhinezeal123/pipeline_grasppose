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


@dataclass(frozen=True)
class GraspPose:
    score: float
    width_m: float
    translation_m: tuple
    rotation: tuple


@dataclass(frozen=True)
class EstimateResult:
    grasps: tuple
    depth_m: Optional[float]
    detection_count: int
    mask_pixels: int
    grasp_count: int
    request_id: Optional[str] = None
    snapshot_available: bool = False
    latency_ms: Optional[float] = None
