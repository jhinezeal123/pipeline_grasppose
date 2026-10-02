"""Composition root for the production Jetson pipeline."""

from ..modules.depth.da3_metric import Da3MetricDepth
from ..modules.grasp.vgn_trt import VgnTensorRT
from ..modules.vision.yoloe import Yoloe26sVision
from ..application import GraspPipeline
from ..application.service import LocalGraspEstimator
from .settings import (
    TSDF_RESOLUTION,
    TSDF_SIZE_M,
    TSDF_TRUNC_VOXELS,
)
from ..modules.tsdf.projective import ProjectiveTSDFBuilder


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
        depth=Da3MetricDepth() if depth is None else depth,
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
