"""Public application contract shared by CLI, UI, and robot clients."""

from abc import ABC, abstractmethod


class GraspEstimator(ABC):
    @abstractmethod
    def estimate(self, image, prompt_id, camera_K=None, fov_x=None,
                 fov_y=None, camera_K_size=None, max_width=0.080,
                 top=1, T_cam_volume=None):
        """Estimate grasp poses for one image and return an EstimateResult."""
        raise NotImplementedError
