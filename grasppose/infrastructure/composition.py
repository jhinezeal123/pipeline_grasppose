"""Composition root for the production Jetson pipeline."""

from ..modules.depth.da3_metric import Da3MetricDepth
from ..modules.depth.lite_mono import LiteMonoDepth
from ..modules.grasp.vgn_trt import VgnTensorRT
from ..modules.vision.yoloe import Yoloe26sVision
from ..application import GraspPipeline
from ..application.service import LocalGraspEstimator
from .settings import (
    DEPTH_BACKEND,
    TSDF_RESOLUTION,
    TSDF_SIZE_M,
    TSDF_TRUNC_VOXELS,
)
from ..modules.tsdf.projective import ProjectiveTSDFBuilder


def build_depth(backend=None):
    """Chọn backend ở một nơi; không fallback âm thầm sang model khác."""
    selected = DEPTH_BACKEND if backend is None else backend
    if selected == "lite-mono":
        return LiteMonoDepth()
    if selected == "da3":
        return Da3MetricDepth()
    raise ValueError(
        "GRASP_DEPTH_BACKEND phải là lite-mono hoặc da3; nhận %r" % selected
    )


def build_default_pipeline(depth=None):
    """Wire concrete adapters to application ports in one place.

    ``depth`` is an injection seam, not a configuration option. Validation
    harnesses that need to drive the pipeline with a supplied depth map -- the
    MuJoCo ground-truth bridge, for instance -- pass a ``DepthPort`` here rather
    than patching a concrete adapter. The production call site passes nothing
    and gets the real model.
    """
    return GraspPipeline(
        vision=Yoloe26sVision(),
        depth=build_depth() if depth is None else depth,
        tsdf_builder=ProjectiveTSDFBuilder(
            size_m=TSDF_SIZE_M,
            resolution=TSDF_RESOLUTION,
            trunc_voxels=TSDF_TRUNC_VOXELS,
        ),
        grasper=VgnTensorRT(),
    )


def build_default_estimator():
    return LocalGraspEstimator(build_default_pipeline())


DEFAULT_PIPELINE = build_default_pipeline()
DEFAULT_ESTIMATOR = LocalGraspEstimator(DEFAULT_PIPELINE)
