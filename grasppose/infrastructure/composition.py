"""Composition root for the production Jetson pipeline."""

from ..modules.depth.lite_mono import LiteMonoDepth
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


def build_default_pipeline():
    """Wire concrete adapters to application ports in one place."""
    return GraspPipeline(
        vision=Yoloe26sVision(),
        depth=LiteMonoDepth(),
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
