"""Persistent-model pipeline orchestration for Jetson Xavier.

YOLOE, Lite-Mono and VGN TensorRT are loaded once and kept resident. Per-frame
execution only prepares inputs and runs inference; weights are released only by close().
"""

import threading
import time

import numpy as np

from .config import GRIP_HW_OPEN_M
from .geometry import depth_range_str, depth_to_cloud
from .runtime import _log, _vram


class GraspPipeline:
    def __init__(self, vision_factory, depth_factory, tsdf_factory, grasper_factory):
        self.vision_factory = vision_factory
        self.depth_factory = depth_factory
        self.tsdf_factory = tsdf_factory
        self.grasper_factory = grasper_factory
        self.vision = None
        self.depther = None
        self.tsdf_builder = None
        self.grasper = None
        self._loaded = False
        self._lock = threading.Lock()

    def load(self):
        """Load all model weights/engines once and keep them resident."""
        with self._lock:
            if self._loaded:
                return self
            _log("Loading persistent models into RAM/VRAM ...")
            self.vision = self.vision or self.vision_factory()
            self.depther = self.depther or self.depth_factory()
            self.tsdf_builder = self.tsdf_builder or self.tsdf_factory()
            self.grasper = self.grasper or self.grasper_factory()

            t0 = time.time()
            if hasattr(self.vision, "load"):
                self.vision.load()
            _vram(" after YOLOE load")
            if hasattr(self.depther, "load"):
                self.depther.load()
            _vram(" after Lite-Mono load")
            if hasattr(self.grasper, "load"):
                self.grasper.load()
            _vram(" after VGN TensorRT load")
            self._loaded = True
            _log("Persistent models ready in %.2fs" % (time.time() - t0))
            return self

    def close(self):
        """Release resident models. Call only when the process is shutting down."""
        with self._lock:
            for obj in (self.vision, self.depther, self.grasper):
                if obj is not None and hasattr(obj, "release"):
                    obj.release()
            self.vision = self.depther = self.grasper = None
            self.tsdf_builder = None
            self._loaded = False
            _vram(" after pipeline close")

    def run(self, image, prompt, camera_K=None, fov_x=None, T_cam_volume=None,
            vision=None, depther=None, tsdf_builder=None, grasper=None):
        """Run one frame while keeping default model instances resident."""
        # Adapters keep per-frame state, therefore serialize access to the shared
        # resident instances. This avoids races without unloading any model.
        with self._lock:
            if not self._loaded and all(x is None for x in
                                        (vision, depther, tsdf_builder, grasper)):
                # Inline load to avoid re-entering the non-reentrant lock.
                self.vision = self.vision or self.vision_factory()
                self.depther = self.depther or self.depth_factory()
                self.tsdf_builder = self.tsdf_builder or self.tsdf_factory()
                self.grasper = self.grasper or self.grasper_factory()
                if hasattr(self.vision, "load"):
                    self.vision.load()
                if hasattr(self.depther, "load"):
                    self.depther.load()
                if hasattr(self.grasper, "load"):
                    self.grasper.load()
                self._loaded = True
                _vram(" persistent models loaded")
            return self._run_frame(
                image, prompt, camera_K=camera_K, fov_x=fov_x,
                T_cam_volume=T_cam_volume,
                vision=vision or self.vision,
                depther=depther or self.depther,
                tsdf_builder=tsdf_builder or self.tsdf_builder,
                grasper=grasper or self.grasper,
            )

    def _run_frame(self, image, prompt, camera_K, fov_x, T_cam_volume,
                   vision, depther, tsdf_builder, grasper):
        image = np.asarray(image)
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("input image is empty")
        out = {}

        t0 = time.time()
        vision.prepare(image, prompt)
        vis = vision.inference()
        out["det"], out["seg"] = vis["det"], vis["seg"]
        _log("YOLOE: %d boxes | mask=%d px | %.3fs" % (
            len(out["det"]["boxes"]), int(out["seg"]["mask"].sum()),
            time.time() - t0))

        mask = np.asarray(out["seg"]["mask"], bool)
        if not mask.any():
            out.update({
                "dep": {"depth": np.zeros((h, w), np.float32),
                        "intrinsics": np.eye(3, dtype=np.float32),
                        "fov_x_deg": 0.0,
                        "reason": out["seg"].get("reason") or "empty target mask"},
                "depth_m": None,
                "cloud": np.zeros((0, 3), np.float32),
                "tsdf": None,
                "grasp": {"graspgroup": np.zeros((0, 17), np.float64),
                          "reason": "empty target mask"},
                "K": np.eye(3, dtype=np.float64),
            })
            return out

        t0 = time.time()
        depther.prepare(image, camera_K=camera_K, fov_x=fov_x)
        dep = depther.inference()
        out["dep"] = dep
        K = np.asarray(dep["intrinsics"], np.float64)
        out["K"] = K
        _log("Lite-Mono: depth %s | scale=%.5g | %.3fs" % (
            depth_range_str(dep["depth"]), float(dep.get("scale", 1.0)),
            time.time() - t0))

        dm = np.asarray(dep["depth"], np.float32)[mask]
        dm = dm[np.isfinite(dm) & (dm > 0.05)]
        out["depth_m"] = float(np.median(dm)) if dm.size else None

        cloud = depth_to_cloud(dep["depth"], K, mask=mask)
        out["cloud"] = cloud
        if len(cloud) == 0:
            out["tsdf"] = None
            out["grasp"] = {
                "graspgroup": np.zeros((0, 17), np.float64),
                "reason": "target mask has no valid depth; point cloud is empty",
            }
            return out

        t0 = time.time()
        tsdf = tsdf_builder.build(
            dep["depth"], K, mask=mask, cloud=cloud,
            T_cam_volume=T_cam_volume,
        )
        out["tsdf"] = tsdf
        _log("TSDF: %s | observed=%d | %.3fs" % (
            tuple(tsdf["grid"].shape), tsdf["observed_voxels"],
            time.time() - t0))
        if tsdf["observed_voxels"] == 0:
            out["grasp"] = {
                "graspgroup": np.zeros((0, 17), np.float64),
                "reason": "TSDF contains no observed voxels",
            }
            return out

        t0 = time.time()
        try:
            grasper.prepare(
                tsdf["grid"], tsdf["voxel_size"], tsdf["T_cam_volume"])
            out["grasp"] = grasper.inference()
        except Exception as exc:
            out["grasp"] = {
                "graspgroup": np.zeros((0, 17), np.float64),
                "reason": "%s: %s" % (type(exc).__name__, exc),
            }
        gg = out["grasp"]["graspgroup"]
        n_ok = int((gg[:, 1] <= GRIP_HW_OPEN_M).sum()) if len(gg) else 0
        _log("VGN TensorRT: %d grasps | %d fit %.1f mm | %.3fs" % (
            len(gg), n_ok, GRIP_HW_OPEN_M * 1000, time.time() - t0))
        return out
