"""Data contracts for the vision feature."""

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
