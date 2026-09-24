"""Stable outer API shared by CLI and UI."""

import numpy as np

from .bootstrap import build_default_pipeline
from .config import DEFAULT_PROMPT, GRIP_MAX_OPEN_M
from .presentation.rendering import (
    draw_box,
    draw_depth,
    draw_grasp,
    draw_mask,
)


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

    def infer(self, image, prompt=DEFAULT_PROMPT,
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

        result = self._core.run(
            image,
            prompt,
            camera_K=camera_K,
            fov_x=fov_x,
            T_cam_volume=T_cam_volume,
        )
        return {
            "box": draw_box(
                image, result.vision.detection),
            "mask": draw_mask(
                image, result.vision.segmentation),
            "depthmap": draw_depth(result.depth),
            "grasp": draw_grasp(
                image,
                result.grasp,
                result.camera_K,
                max_width=max_width,
                top=top,
            ),
            "depth_m": result.depth_m,
        }


DEFAULT_SERVICE = GraspService(build_default_pipeline())
