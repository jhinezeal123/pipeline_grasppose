"""Short-lived inference results for optional, out-of-process rendering."""

from collections import OrderedDict
from dataclasses import dataclass
import threading
import time
import uuid

import numpy as np


ARRAY_NAMES = (
    "image", "mask", "depth", "boxes", "scores", "graspgroup", "camera_k",
)


@dataclass
class OutputSnapshot:
    image: np.ndarray
    mask: np.ndarray
    depth: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    graspgroup: np.ndarray
    camera_k: np.ndarray
    labels: list
    detection_reason: str
    segmentation_reason: str
    depth_reason: str
    grasp_reason: str
    max_width: float
    top: int

    @classmethod
    def capture(cls, image, result, max_width, top):
        # Own every frame buffer: adapters may reuse their output allocations
        # on the next camera frame while this snapshot remains cached.
        return cls(
            image=np.array(image, dtype=np.uint8, order="C", copy=True),
            mask=np.array(
                result.vision.segmentation.mask,
                dtype=np.bool_, order="C", copy=True),
            depth=np.array(
                result.depth.depth, dtype=np.float32, order="C", copy=True),
            boxes=np.array(
                result.vision.detection.boxes,
                dtype=np.float32, order="C", copy=True),
            scores=np.array(
                result.vision.detection.scores,
                dtype=np.float32, order="C", copy=True),
            graspgroup=np.array(
                result.grasp.graspgroup,
                dtype=np.float64, order="C", copy=True),
            camera_k=np.array(
                result.camera_K, dtype=np.float64, order="C", copy=True),
            labels=list(result.vision.detection.labels),
            detection_reason=result.vision.detection.reason or "",
            segmentation_reason=result.vision.segmentation.reason or "",
            depth_reason=result.depth.reason or "",
            grasp_reason=result.grasp.reason or "",
            max_width=float(max_width),
            top=int(top),
        )

    @property
    def nbytes(self):
        return sum(getattr(self, name).nbytes for name in ARRAY_NAMES)

    def metadata(self):
        return {
            "labels": self.labels,
            "detection_reason": self.detection_reason,
            "segmentation_reason": self.segmentation_reason,
            "depth_reason": self.depth_reason,
            "grasp_reason": self.grasp_reason,
            "max_width": self.max_width,
            "top": self.top,
        }

    @classmethod
    def from_wire(cls, metadata, arrays):
        if set(arrays) != set(ARRAY_NAMES):
            raise ValueError("incomplete output snapshot")
        return cls(
            **arrays,
            labels=list(metadata["labels"]),
            detection_reason=metadata["detection_reason"],
            segmentation_reason=metadata["segmentation_reason"],
            depth_reason=metadata["depth_reason"],
            grasp_reason=metadata["grasp_reason"],
            max_width=float(metadata["max_width"]),
            top=int(metadata["top"]),
        )


class SnapshotCache:
    """Bounded RAM cache. A fetched snapshot remains valid after eviction."""

    def __init__(self, max_entries=8, max_bytes=128 * 1024 * 1024,
                 ttl_seconds=600):
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.ttl_seconds = float(ttl_seconds)
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def _expire(self, now):
        for run_id, (created, snapshot) in list(self._entries.items()):
            if now - created >= self.ttl_seconds:
                self._entries.pop(run_id)
                self._bytes -= snapshot.nbytes

    def put(self, snapshot):
        if snapshot.nbytes > self.max_bytes:
            return None
        run_id = uuid.uuid4().hex
        with self._lock:
            self._expire(time.monotonic())
            while self._entries and (
                    len(self._entries) >= self.max_entries or
                    self._bytes + snapshot.nbytes > self.max_bytes):
                _, (_, removed) = self._entries.popitem(last=False)
                self._bytes -= removed.nbytes
            self._entries[run_id] = (time.monotonic(), snapshot)
            self._bytes += snapshot.nbytes
        return run_id

    def get(self, run_id):
        with self._lock:
            self._expire(time.monotonic())
            item = self._entries.get(run_id)
            return None if item is None else item[1]
