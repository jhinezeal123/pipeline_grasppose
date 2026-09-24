"""Typed data contracts shared between pipeline modules."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class DetectionResult:
    boxes: np.ndarray
    scores: np.ndarray
    labels: List[str]
    reason: Optional[str] = None

    @classmethod
    def empty(cls, reason=None):
        return cls(
            boxes=np.zeros((0, 4), np.float32),
            scores=np.zeros((0,), np.float32),
            labels=[],
            reason=reason,
        )


@dataclass
class SegmentationResult:
    mask: np.ndarray
    scores: np.ndarray
    best_index: int
    candidate_count: int
    reason: Optional[str] = None

    @classmethod
    def empty(cls, height, width, reason=None):
        return cls(
            mask=np.zeros((height, width), bool),
            scores=np.zeros((0,), np.float32),
            best_index=-1,
            candidate_count=0,
            reason=reason,
        )


@dataclass
class VisionResult:
    detection: DetectionResult
    segmentation: SegmentationResult


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


@dataclass
class TSDFResult:
    grid: np.ndarray
    voxel_size: float
    T_cam_volume: np.ndarray
    observed_voxels: int


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


@dataclass
class PipelineResult:
    vision: VisionResult
    depth: DepthResult
    cloud: np.ndarray
    tsdf: Optional[TSDFResult]
    grasp: GraspResult
    camera_K: np.ndarray
    depth_m: Optional[float]
