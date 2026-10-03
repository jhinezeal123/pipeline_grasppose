"""Xử lý một ảnh; độc lập với socket và HTTP/CLI.

Estimator, prompt catalog và cache được truyền vào constructor. Mỗi request
chỉ giữ dữ liệu cục bộ; model resident vẫn do worker quản lý.
"""

import os
import time
import uuid

import numpy as np
from PIL import Image

from ..output.snapshot import OutputSnapshot
from ..runtime import log


def reject_inline_render(request):
    if request.get("render", False):
        raise ValueError(
            "inline worker rendering is unavailable; request inference "
            "then queue output with scripts/output.sh RUN_ID"
        )


class WorkerInference:
    def __init__(self, estimator, catalog, snapshots):
        self._estimator = estimator
        self._catalog = catalog
        self._snapshots = snapshots

    def infer(self, request):
        reject_inline_render(request)
        prompt_id = str(request.get("prompt_id", ""))
        self._catalog.require(prompt_id)
        image_path = request.get("image")
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            raise ValueError("input image does not exist: %r" % image_path)
        profile = os.environ.get("GRASP_PROFILE_INFER") == "1"
        started = time.perf_counter()
        image_started = time.perf_counter()
        image = np.asarray(Image.open(image_path).convert("RGB"))
        if profile:
            log("profile image decode %.3f s" % (time.perf_counter() - image_started))
        max_width = float(request.get("max_width", 0.080))
        top = int(request.get("top", 1))
        if max_width <= 0 or top < 1:
            raise ValueError("top and max_width must be positive")
        estimate, result = self._estimator.estimate_with_details(
            image,
            prompt_id=prompt_id,
            camera_K=request.get("camera_k"),
            camera_K_size=request.get("camera_k_size"),
            fov_x=request.get("fov_x"),
            fov_y=request.get("fov_y"),
            max_width=max_width,
            top=top,
            T_cam_volume=request.get("T_cam_volume"),
        )
        snapshot = OutputSnapshot.capture(image, result, max_width, top)
        run_id = self._snapshots.put(snapshot)
        snapshot_available = run_id is not None
        if run_id is None:
            run_id = uuid.uuid4().hex
        files = []
        render_ms = None
        grasp_poses = [
            {
                "score": pose.score,
                "width_m": pose.width_m,
                "translation_m": list(pose.translation_m),
                "rotation": [list(row) for row in pose.rotation],
            }
            for pose in estimate.grasps
        ]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if profile:
            log("profile worker total %.3f s" % (elapsed_ms / 1000.0))
        return {
            "ok": True,
            "run_id": run_id,
            "snapshot_available": snapshot_available,
            "files": files,
            "grasps": grasp_poses,
            "depth_m": estimate.depth_m,
            "detection_count": estimate.detection_count,
            "mask_pixels": estimate.mask_pixels,
            "grasp_count": estimate.grasp_count,
            "server_ms": elapsed_ms,
            "render_ms": render_ms,
        }
