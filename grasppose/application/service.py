"""Application service exposing grasp estimation independently of delivery adapters."""

import os

import numpy as np

from .interface import GraspEstimator
from .types import EstimateResult, GraspPose
from ..modules.depth.geometry import fov_x_from_fovy, scale_camera_intrinsics


class LocalGraspEstimator(GraspEstimator):
    """Run the grasp pipeline in-process behind the public estimator contract."""

    def __init__(self, pipeline):
        self._pipeline = pipeline

    def load(self):
        self._pipeline.load()
        return self

    def warmup(self):
        self._pipeline.warmup()
        return self

    def close(self):
        self._pipeline.close()

    def estimate(self, image, prompt_id, camera_K=None, fov_x=None,
                 fov_y=None, camera_K_size=None, max_width=0.080,
                 top=1, T_cam_volume=None):
        estimate, _ = self.estimate_with_details(
            image,
            prompt_id,
            camera_K=camera_K,
            fov_x=fov_x,
            fov_y=fov_y,
            camera_K_size=camera_K_size,
            max_width=max_width,
            top=top,
            T_cam_volume=T_cam_volume,
        )
        return estimate

    def estimate_with_details(self, image, prompt_id, camera_K=None,
                              fov_x=None, fov_y=None,
                              camera_K_size=None, max_width=0.080,
                              top=1, T_cam_volume=None):
        if isinstance(image, (str, os.PathLike)):
            from PIL import Image
            image = np.asarray(Image.open(image).convert("RGB"))
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("input image must have shape (H,W,3+)")
        image = image[:, :, :3]
        if max_width <= 0 or int(top) < 1:
            raise ValueError("top and max_width must be positive")

        K = _camera_matrix(camera_K)
        if camera_K_size is not None:
            if K is None:
                raise ValueError("camera_K_size requires camera_K")
            K = scale_camera_intrinsics(
                K, camera_K_size, (image.shape[1], image.shape[0]))
        if fov_x is None and fov_y is not None:
            fov_x = fov_x_from_fovy(
                fov_y, image.shape[1], image.shape[0])

        T = (
            None if T_cam_volume is None
            else np.asarray(T_cam_volume, np.float64).reshape(4, 4)
        )
        result = self._pipeline.run(
            image,
            prompt_id=prompt_id,
            camera_K=K,
            fov_x=fov_x,
            T_cam_volume=T,
        )
        grasps = np.asarray(result.grasp.graspgroup, np.float64).reshape(-1, 17)
        valid = grasps[grasps[:, 1] <= float(max_width)]
        valid = valid[np.argsort(-valid[:, 0])[:int(top)]]
        poses = tuple(
            GraspPose(
                score=float(row[0]),
                width_m=float(row[1]),
                translation_m=tuple(float(value) for value in row[13:16]),
                rotation=tuple(
                    tuple(float(value) for value in axis)
                    for axis in row[4:13].reshape(3, 3)
                ),
            )
            for row in valid
        )
        estimate = EstimateResult(
            grasps=poses,
            depth_m=result.depth_m,
            detection_count=int(len(result.vision.detection.boxes)),
            mask_pixels=int(np.count_nonzero(result.vision.segmentation.mask)),
            grasp_count=int(len(grasps)),
        )
        return estimate, result


def _camera_matrix(camera_K):
    if camera_K is None:
        return None
    values = np.asarray(camera_K, np.float64)
    if values.size == 4:
        fx, fy, cx, cy = values.reshape(4)
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    return values.reshape(3, 3)
