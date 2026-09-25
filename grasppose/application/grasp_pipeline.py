"""Application orchestration over abstract model ports."""

import os
import threading
import time

import numpy as np

from ..config import GRIP_HW_OPEN_M
from ..domain.geometry import depth_range_str, depth_to_cloud
from ..domain.types import (
    DepthResult,
    GraspResult,
    PipelineResult,
)
from ..ports.depth import DepthPort
from ..ports.grasp import GraspPort
from ..ports.tsdf import TSDFPort
from ..ports.vision import VisionPort
from ..runtime import log, log_exception, log_vram


class GraspPipeline:
    """Coordinate the grasp workflow without knowing model frameworks."""

    def __init__(self, vision: VisionPort, depth: DepthPort,
                 tsdf_builder: TSDFPort, grasper: GraspPort):
        self._vision = vision
        self._depth = depth
        self._tsdf_builder = tsdf_builder
        self._grasper = grasper
        self._loaded = False
        self._warmed_up = False
        self._lock = threading.Lock()

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        """Load all heavyweight model resources exactly once."""
        with self._lock:
            self._load_unlocked()
        return self

    def _load_unlocked(self):
        if self._loaded:
            return

        started = time.time()
        log("Loading persistent models into RAM/VRAM ...")
        self._vision.load()
        log_vram(" after YOLOE load")
        self._depth.load()
        log_vram(" after Lite-Mono load")
        self._grasper.load()
        log_vram(" after VGN TensorRT load")
        self._loaded = True
        log("Persistent models ready in %.2fs" % (
            time.time() - started))

    def warmup(self):
        """Initialize the three inference paths before accepting requests."""
        with self._lock:
            self._load_unlocked()
            if self._warmed_up:
                return self
            for name, resource in (
                ("vision", self._vision),
                ("depth", self._depth),
                ("grasp", self._grasper),
            ):
                warmup = getattr(resource, "warmup", None)
                if warmup is not None:
                    started = time.time()
                    warmup()
                    log("%s warmup %.3fs" % (name, time.time() - started))
            self._warmed_up = True
        return self

    def close(self):
        """Release every resource, even if one adapter fails to close."""
        with self._lock:
            failures = []
            for name, resource in (
                ("vision", self._vision),
                ("depth", self._depth),
                ("grasp", self._grasper),
            ):
                try:
                    resource.close()
                except Exception as exc:
                    failures.append((name, exc))
                    log_exception(
                        "ERROR closing %s: %s: %s" % (
                            name, type(exc).__name__, exc)
                    )
            self._loaded = False
            self._warmed_up = False
            log_vram(" after pipeline close")
            if failures:
                details = "; ".join(
                    "%s=%s: %s" % (
                        name, type(exc).__name__, exc)
                    for name, exc in failures
                )
                raise RuntimeError(
                    "pipeline close failed: %s" % details
                ) from failures[0][1]

    def run(self, image, prompt, camera_K=None, fov_x=None,
            T_cam_volume=None):
        """Run one frame using the already resident model objects."""
        with self._lock:
            self._load_unlocked()
            return self._run_frame(
                image=image,
                prompt=prompt,
                camera_K=camera_K,
                fov_x=fov_x,
                T_cam_volume=T_cam_volume,
            )

    def _run_frame(self, image, prompt, camera_K, fov_x,
                   T_cam_volume):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(
                "input image must have shape (H,W,3+)")
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("input image is empty")

        started = time.time()
        vision = self._vision.predict(image, prompt)
        mask = np.asarray(vision.segmentation.mask, bool)
        mask_pixels = int(np.count_nonzero(mask))
        log("YOLOE: %d boxes | mask=%d px | %.3fs" % (
            len(vision.detection.boxes),
            mask_pixels,
            time.time() - started,
        ))

        if mask_pixels == 0:
            reason = (
                vision.segmentation.reason or
                "empty target mask"
            )
            return PipelineResult(
                vision=vision,
                depth=DepthResult.empty(
                    height, width, reason),
                cloud=np.zeros((0, 3), np.float32),
                tsdf=None,
                grasp=GraspResult.empty(
                    "empty target mask"),
                camera_K=np.eye(3, dtype=np.float64),
                depth_m=None,
            )

        started = time.time()
        depth = self._depth.predict(
            image,
            camera_K=camera_K,
            fov_x=fov_x,
        )
        K = np.asarray(
            depth.intrinsics, np.float64).reshape(3, 3)
        depth_summary = (
            depth_range_str(depth.depth)
            if os.environ.get("GRASP_VERBOSE_DEPTH") == "1"
            else "ready"
        )
        log("Lite-Mono: depth %s | scale=%.5g | %.3fs" % (
            depth_summary,
            float(depth.scale),
            time.time() - started,
        ))

        cloud = depth_to_cloud(
            depth.depth, K, mask=mask)
        depth_m = (
            float(np.median(cloud[:, 2]))
            if len(cloud) else None
        )
        if len(cloud) == 0:
            return PipelineResult(
                vision=vision,
                depth=depth,
                cloud=cloud,
                tsdf=None,
                grasp=GraspResult.empty(
                    "target mask has no valid depth; "
                    "point cloud is empty"
                ),
                camera_K=K,
                depth_m=depth_m,
            )

        started = time.time()
        tsdf = self._tsdf_builder.build(
            depth.depth,
            K,
            mask=mask,
            cloud=cloud,
            T_cam_volume=T_cam_volume,
        )
        log("TSDF: %s | observed=%d | %.3fs" % (
            tuple(tsdf.grid.shape),
            tsdf.observed_voxels,
            time.time() - started,
        ))

        if tsdf.observed_voxels == 0:
            return PipelineResult(
                vision=vision,
                depth=depth,
                cloud=cloud,
                tsdf=tsdf,
                grasp=GraspResult.empty(
                    "TSDF contains no observed voxels"),
                camera_K=K,
                depth_m=depth_m,
            )

        started = time.time()
        try:
            grasp = self._grasper.predict(tsdf)
        except Exception:
            log_exception(
                "VGN TensorRT inference failed; "
                "propagating runtime error"
            )
            raise

        graspgroup = grasp.graspgroup
        fit_count = int(
            (graspgroup[:, 1] <= GRIP_HW_OPEN_M).sum()
        ) if len(graspgroup) else 0
        log("VGN TensorRT: %d grasps | %d fit %.1f mm | %.3fs" % (
            len(graspgroup),
            fit_count,
            GRIP_HW_OPEN_M * 1000,
            time.time() - started,
        ))

        return PipelineResult(
            vision=vision,
            depth=depth,
            cloud=cloud,
            tsdf=tsdf,
            grasp=grasp,
            camera_K=K,
            depth_m=depth_m,
        )
