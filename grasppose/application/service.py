"""Application service exposing grasp estimation independently of delivery adapters."""

import numpy as np

from .interface import GraspEstimator
from .types import EstimateResult
from .input import EstimateInput, _camera_matrix as _camera_matrix
from .selection import GraspSelector


class LocalGraspEstimator(GraspEstimator):
    """Run the grasp pipeline in-process behind the public estimator contract."""

    def __init__(self, pipeline, selector=None):
        self._pipeline = pipeline
        self._selector = GraspSelector() if selector is None else selector

    def load(self):
        self._pipeline.load()
        return self

    def warmup(self):
        self._pipeline.warmup()
        return self

    def close(self):
        self._pipeline.close()

    def estimate(
        self,
        image,
        prompt_id,
        camera_K=None,
        fov_x=None,
        fov_y=None,
        camera_K_size=None,
        max_width=0.080,
        top=1,
        T_cam_volume=None,
    ):
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

    def estimate_with_details(
        self,
        image,
        prompt_id,
        camera_K=None,
        fov_x=None,
        fov_y=None,
        camera_K_size=None,
        max_width=0.080,
        top=1,
        T_cam_volume=None,
    ):
        request = EstimateInput.from_values(
            image, camera_K, fov_x, fov_y, camera_K_size, max_width, top, T_cam_volume
        )
        result = self._pipeline.run(
            request.image,
            prompt_id=prompt_id,
            camera_K=request.camera_K,
            fov_x=request.fov_x,
            T_cam_volume=request.T_cam_volume,
        )
        grasps = np.asarray(result.grasp.graspgroup, np.float64).reshape(-1, 17)
        poses = self._selector.select(grasps, max_width, top)
        estimate = EstimateResult(
            grasps=poses,
            depth_m=result.depth_m,
            detection_count=int(len(result.vision.detection.boxes)),
            mask_pixels=int(np.count_nonzero(result.vision.segmentation.mask)),
            grasp_count=int(len(grasps)),
        )
        return estimate, result
