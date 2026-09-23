"""Sequential edge pipeline orchestration.

Only one GPU-heavy model is alive at a time:
YOLOE -> release -> Lite-Mono -> release -> CPU TSDF -> VGN TensorRT -> release.
"""

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

    def run(self, image, prompt, camera_K=None, fov_x=None, T_cam_volume=None,
            vision=None, depther=None, tsdf_builder=None, grasper=None):
        image = np.asarray(image)
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("input image is empty")
        out = {}

        _log("PHASE 1: YOLOE-26s detection + segmentation")
        vision = vision or self.vision_factory(); t0 = time.time()
        try:
            vision.prepare(image, prompt)
            vis = vision.inference(); out["det"], out["seg"] = vis["det"], vis["seg"]
        finally:
            vision.release(); _vram(" after YOLOE release")
        _log("  YOLOE: %d boxes | mask=%d px | %.2fs" % (
            len(out["det"]["boxes"]), int(out["seg"]["mask"].sum()), time.time() - t0))

        mask = np.asarray(out["seg"]["mask"], bool)
        if not mask.any():
            out.update({
                "dep": {"depth": np.zeros((h, w), np.float32),
                        "intrinsics": np.eye(3, dtype=np.float32), "fov_x_deg": 0.0,
                        "reason": out["seg"].get("reason") or "empty target mask"},
                "depth_m": None, "cloud": np.zeros((0, 3), np.float32), "tsdf": None,
                "grasp": {"graspgroup": np.zeros((0, 17), np.float64),
                          "reason": "empty target mask"},
                "K": np.eye(3, dtype=np.float64),
            })
            return out

        _log("PHASE 2: Lite-Mono depth")
        depther = depther or self.depth_factory(); t0 = time.time()
        try:
            depther.prepare(image, camera_K=camera_K, fov_x=fov_x)
            dep = depther.inference(); out["dep"] = dep
        finally:
            depther.release(); _vram(" after Lite-Mono release")
        K = np.asarray(dep["intrinsics"], np.float64); out["K"] = K
        _log("  Lite-Mono: depth %s | scale=%.5g | %.2fs" % (
            depth_range_str(dep["depth"]), float(dep.get("scale", 1.0)), time.time() - t0))
        dm = np.asarray(dep["depth"], np.float32)[mask]
        dm = dm[np.isfinite(dm) & (dm > 0.05)]
        out["depth_m"] = float(np.median(dm)) if dm.size else None

        cloud = depth_to_cloud(dep["depth"], K, mask=mask); out["cloud"] = cloud
        if len(cloud) == 0:
            out["tsdf"] = None
            out["grasp"] = {"graspgroup": np.zeros((0, 17), np.float64),
                            "reason": "target mask has no valid depth; point cloud is empty"}
            return out
        _log("PHASE 3: point cloud %d points" % len(cloud))

        _log("PHASE 4: TSDF voxel grid")
        tsdf_builder = tsdf_builder or self.tsdf_factory()
        tsdf = tsdf_builder.build(dep["depth"], K, mask=mask, cloud=cloud,
                                  T_cam_volume=T_cam_volume)
        out["tsdf"] = tsdf
        if tsdf["observed_voxels"] == 0:
            out["grasp"] = {"graspgroup": np.zeros((0, 17), np.float64),
                            "reason": "TSDF contains no observed voxels"}
            return out
        _log("  TSDF: %s | observed=%d | voxel=%.2f mm" % (
            tuple(tsdf["grid"].shape), tsdf["observed_voxels"], tsdf["voxel_size"] * 1000))

        _log("PHASE 5: VGN TensorRT")
        grasper = grasper or self.grasper_factory(); t0 = time.time()
        try:
            grasper.prepare(tsdf["grid"], tsdf["voxel_size"], tsdf["T_cam_volume"])
            out["grasp"] = grasper.inference()
        except Exception as exc:
            out["grasp"] = {"graspgroup": np.zeros((0, 17), np.float64),
                            "reason": "%s: %s" % (type(exc).__name__, exc)}
        finally:
            grasper.release(); _vram(" after VGN TensorRT release")
        gg = out["grasp"]["graspgroup"]
        n_ok = int((gg[:, 1] <= GRIP_HW_OPEN_M).sum()) if len(gg) else 0
        _log("  VGN: %d grasps | %d fit %.1f mm gripper | %.2fs" % (
            len(gg), n_ok, GRIP_HW_OPEN_M * 1000, time.time() - t0))
        return out
