#!/usr/bin/env python3

import numpy as np

import pipeline as P
from grasppose.application.grasp_pipeline import GraspPipeline
from grasppose.domain.types import (
    DepthResult,
    DetectionResult,
    GraspResult,
    SegmentationResult,
    TSDFResult,
    VisionResult,
)
from grasppose.facade import GraspService


class Vision:
    def load(self):
        return self

    def predict(self, image, prompt):
        height, width = image.shape[:2]
        mask = np.zeros((height, width), bool)
        mask[
            height // 4:3 * height // 4,
            width // 4:3 * width // 4,
        ] = True
        return VisionResult(
            detection=DetectionResult(
                boxes=np.array([[
                    width / 4,
                    height / 4,
                    3 * width / 4,
                    3 * height / 4,
                ]], np.float32),
                scores=np.array([0.9], np.float32),
                labels=["object"],
            ),
            segmentation=SegmentationResult(
                mask=mask,
                scores=np.array([0.9], np.float32),
                best_index=0,
                candidate_count=1,
            ),
        )

    def close(self):
        pass


class Depth:
    def load(self):
        return self

    def predict(self, image, camera_K=None, fov_x=None):
        height, width = image.shape[:2]
        return DepthResult(
            depth=np.full(
                (height, width), 0.6, np.float32),
            intrinsics=np.asarray(
                camera_K, np.float32),
            fov_x_deg=60.0,
            scale=1.0,
        )

    def close(self):
        pass


class TSDF:
    def build(self, *args, **kwargs):
        return TSDFResult(
            grid=np.ones(
                (1, 40, 40, 40), np.float32),
            voxel_size=0.0075,
            T_cam_volume=np.eye(4),
            observed_voxels=100,
        )


class Grasp:
    def load(self):
        return self

    def predict(self, tsdf):
        graspgroup = np.zeros((1, 17), np.float64)
        graspgroup[0, 0] = 0.9
        # Deliberately above the 69.4 mm hardware opening but below
        # the 80 mm visualization limit, so the warning path is rendered.
        graspgroup[0, 1] = 0.075
        graspgroup[0, 4:13] = np.eye(3).reshape(-1)
        graspgroup[0, 13:16] = [0, 0, 0.6]
        return GraspResult(graspgroup=graspgroup)

    def close(self):
        pass


def main():
    image = np.zeros((120, 160, 3), np.uint8)
    K = np.array([
        [120.0, 0, 80.0],
        [0, 120.0, 60.0],
        [0, 0, 1.0],
    ])

    mock_service = GraspService(GraspPipeline(
        vision=Vision(),
        depth=Depth(),
        tsdf_builder=TSDF(),
        grasper=Grasp(),
    ))
    original = P.DEFAULT_SERVICE
    P.DEFAULT_SERVICE = mock_service
    try:
        result = P.pipeline(
            image, "object", camera_K=K)
    finally:
        P.DEFAULT_SERVICE = original

    assert set(result) == {
        "box", "mask", "depthmap",
        "grasp", "depth_m",
        "detection_count", "mask_pixels", "grasp_count",
    }
    assert result["detection_count"] == 1
    assert result["mask_pixels"] == 4800
    assert result["grasp_count"] == 1
    assert all(
        result[key].shape == image.shape
        for key in (
            "box", "mask", "depthmap", "grasp")
    )
    assert abs(result["depth_m"] - 0.6) < 1e-5
    # Regression guard: correct shape alone is insufficient. Each renderer
    # must actually draw content (this previously caught a P0 rendering bug).
    for key in ("box", "mask", "depthmap", "grasp"):
        assert np.any(result[key]), "%s rendering is empty" % key
    print("TAT CA MUC DEU PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
