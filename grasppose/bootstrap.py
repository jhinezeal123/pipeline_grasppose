"""Composition root for the production Jetson pipeline."""

from .adapters import LiteMonoDepth, VgnTensorRT, Yoloe26sVision
from .application import GraspPipeline
from .config import (
    TSDF_RESOLUTION,
    TSDF_SIZE_M,
    TSDF_TRUNC_VOXELS,
)
from .domain.tsdf import ProjectiveTSDFBuilder


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
