"""Stable outer API shared by CLI and UI."""

import os
import time

import numpy as np

from .bootstrap import build_default_pipeline
from .config import GRIP_MAX_OPEN_M
from .presentation.rendering import (
    draw_box,
    draw_depth,
    draw_grasp,
    draw_mask,
)
from .runtime import log


class GraspService:
    """Application facade returning the four rendered pipeline images."""

    def __init__(self, core):
        self._core = core

    @property
    def core(self):
        return self._core

    def load(self):
        self._core.load()
        return self

    def close(self):
        self._core.close()

    def infer(self, image, prompt_id,
              camera_K=None, fov_x=None, T_cam_volume=None,
              max_width=GRIP_MAX_OPEN_M, top=1):
        if isinstance(image, str):
            from PIL import Image
            image = np.array(
                Image.open(image).convert("RGB"))

        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(
                "input image must have shape (H,W,3+)")
        image = image[:, :, :3]

        profile = os.environ.get("GRASP_PROFILE_INFER") == "1"
        started = time.perf_counter()
        result = self._core.run(
            image,
            prompt_id,
            camera_K=camera_K,
            fov_x=fov_x,
            T_cam_volume=T_cam_volume,
        )
        if profile:
            log("profile service core %.3f s" % (time.perf_counter() - started))
        renderers = (
            ("box", lambda: draw_box(image, result.vision.detection)),
            ("mask", lambda: draw_mask(image, result.vision.segmentation)),
            ("depthmap", lambda: draw_depth(result.depth)),
            ("grasp", lambda: draw_grasp(
                image, result.grasp, result.camera_K,
                max_width=max_width, top=top)),
        )
        rendered = {}
        for name, render in renderers:
            started = time.perf_counter()
            rendered[name] = render()
            if profile:
                log("profile render %s %.3f s" % (
                    name, time.perf_counter() - started))
        return {
            "detection_count": int(len(result.vision.detection.boxes)),
            "mask_pixels": int(np.asarray(result.vision.segmentation.mask, bool).sum()),
            "grasp_count": int(len(result.grasp.graspgroup)),
            "box": rendered["box"],
            "mask": rendered["mask"],
            "depthmap": rendered["depthmap"],
            "grasp": rendered["grasp"],
            "depth_m": result.depth_m,
        }


DEFAULT_SERVICE = GraspService(build_default_pipeline())
