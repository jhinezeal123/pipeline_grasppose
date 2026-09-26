"""Public Python interface for grasp estimation clients."""

from .application.interface import GraspEstimator
from .application.types import EstimateResult, GraspPose
from .infrastructure.composition import DEFAULT_ESTIMATOR
from .modules.grasp.constants import GRIP_MAX_OPEN_M

DEFAULT_MAX_WIDTH_M = GRIP_MAX_OPEN_M


def get_estimator() -> GraspEstimator:
    return DEFAULT_ESTIMATOR


def load_models():
    DEFAULT_ESTIMATOR.load()
    DEFAULT_ESTIMATOR.warmup()
    return DEFAULT_ESTIMATOR


def close_models():
    DEFAULT_ESTIMATOR.close()


def estimate(image, prompt_id, camera_K=None, fov_x=None, fov_y=None,
             camera_K_size=None, max_width=DEFAULT_MAX_WIDTH_M,
             top=1, T_cam_volume=None) -> EstimateResult:
    return DEFAULT_ESTIMATOR.estimate(
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


pipeline = estimate

__all__ = [
    "GraspEstimator",
    "EstimateResult",
    "GraspPose",
    "DEFAULT_MAX_WIDTH_M",
    "get_estimator",
    "load_models",
    "close_models",
    "estimate",
    "pipeline",
]
