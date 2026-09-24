import unittest
from unittest.mock import patch

import numpy as np

from grasppose.application.grasp_pipeline import GraspPipeline
from grasppose.domain.types import (
    DepthResult,
    DetectionResult,
    GraspResult,
    SegmentationResult,
    TSDFResult,
    VisionResult,
)


class Vision:
    def __init__(self, calls):
        self.calls = calls

    def load(self):
        self.calls.append("vision.load")
        return self

    def predict(self, image, prompt):
        self.calls.append("vision.predict")
        height, width = image.shape[:2]
        mask = np.zeros((height, width), bool)
        mask[1:-1, 1:-1] = True
        return VisionResult(
            detection=DetectionResult(
                boxes=np.array(
                    [[1, 1, width - 1, height - 1]],
                    np.float32,
                ),
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
        self.calls.append("vision.close")


class Depth:
    def __init__(self, calls):
        self.calls = calls

    def load(self):
        self.calls.append("depth.load")
        return self

    def predict(self, image, camera_K=None, fov_x=None):
        self.calls.append("depth.predict")
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
        self.calls.append("depth.close")


class TSDF:
    def __init__(self, calls):
        self.calls = calls

    def build(self, *args, **kwargs):
        self.calls.append("tsdf.build")
        return TSDFResult(
            grid=np.full(
                (1, 40, 40, 40), 0.5, np.float32),
            voxel_size=0.0075,
            T_cam_volume=np.eye(4, dtype=np.float32),
            observed_voxels=10,
        )


class Grasper:
    def __init__(self, calls):
        self.calls = calls

    def load(self):
        self.calls.append("grasp.load")
        return self

    def predict(self, tsdf):
        self.calls.append("grasp.predict")
        return GraspResult(
            graspgroup=np.zeros((0, 17), np.float64))

    def close(self):
        self.calls.append("grasp.close")


class ArchitectureTests(unittest.TestCase):
    def _pipeline(self, calls):
        return GraspPipeline(
            vision=Vision(calls),
            depth=Depth(calls),
            tsdf_builder=TSDF(calls),
            grasper=Grasper(calls),
        )

    def test_models_load_once_and_stay_resident_across_frames(self):
        calls = []
        pipeline = self._pipeline(calls)
        K = np.array([
            [100.0, 0, 2],
            [0, 100.0, 2],
            [0, 0, 1],
        ])
        image = np.zeros((4, 4, 3), np.uint8)

        pipeline.run(image, "object", camera_K=K)
        pipeline.run(image, "object", camera_K=K)

        self.assertEqual(calls.count("vision.load"), 1)
        self.assertEqual(calls.count("depth.load"), 1)
        self.assertEqual(calls.count("grasp.load"), 1)
        self.assertEqual(calls.count("vision.predict"), 2)
        self.assertEqual(calls.count("depth.predict"), 2)
        self.assertEqual(calls.count("grasp.predict"), 2)
        self.assertNotIn("vision.close", calls)
        self.assertNotIn("depth.close", calls)
        self.assertNotIn("grasp.close", calls)

        pipeline.close()
        self.assertEqual(calls.count("vision.close"), 1)
        self.assertEqual(calls.count("depth.close"), 1)
        self.assertEqual(calls.count("grasp.close"), 1)

    def test_explicit_load_is_idempotent(self):
        calls = []
        pipeline = self._pipeline(calls)
        pipeline.load()
        pipeline.load()
        self.assertEqual(calls.count("vision.load"), 1)
        self.assertEqual(calls.count("depth.load"), 1)
        self.assertEqual(calls.count("grasp.load"), 1)

    def test_empty_mask_skips_frame_depth_and_grasp(self):
        calls = []

        class EmptyVision(Vision):
            def predict(self, image, prompt):
                result = super().predict(image, prompt)
                result.segmentation.mask[:] = False
                result.segmentation.reason = "none"
                return result

        pipeline = GraspPipeline(
            vision=EmptyVision(calls),
            depth=Depth(calls),
            tsdf_builder=TSDF(calls),
            grasper=Grasper(calls),
        )
        result = pipeline.run(
            np.zeros((4, 4, 3), np.uint8),
            "x",
            camera_K=np.eye(3),
        )

        self.assertEqual(len(result.cloud), 0)
        # Resources are resident, but unnecessary per-frame work is skipped.
        self.assertIn("depth.load", calls)
        self.assertIn("grasp.load", calls)
        self.assertNotIn("depth.predict", calls)
        self.assertNotIn("grasp.predict", calls)


    def test_empty_cloud_skips_tsdf_and_grasp(self):
        calls = []

        class EmptyDepth(Depth):
            def predict(self, image, camera_K=None, fov_x=None):
                self.calls.append("depth.predict")
                height, width = image.shape[:2]
                return DepthResult(
                    depth=np.zeros((height, width), np.float32),
                    intrinsics=np.asarray(camera_K, np.float32),
                    fov_x_deg=60.0,
                    scale=1.0,
                )

        pipeline = GraspPipeline(
            vision=Vision(calls),
            depth=EmptyDepth(calls),
            tsdf_builder=TSDF(calls),
            grasper=Grasper(calls),
        )
        result = pipeline.run(
            np.zeros((4, 4, 3), np.uint8),
            "object",
            camera_K=np.eye(3),
        )

        self.assertEqual(len(result.cloud), 0)
        self.assertIsNone(result.tsdf)
        self.assertIn("point cloud is empty", result.grasp.reason)
        self.assertNotIn("tsdf.build", calls)
        self.assertNotIn("grasp.predict", calls)

    def test_zero_observed_tsdf_skips_grasp(self):
        calls = []

        class EmptyTSDF(TSDF):
            def build(self, *args, **kwargs):
                self.calls.append("tsdf.build")
                return TSDFResult(
                    grid=np.zeros((1, 40, 40, 40), np.float32),
                    voxel_size=0.0075,
                    T_cam_volume=np.eye(4, dtype=np.float32),
                    observed_voxels=0,
                )

        pipeline = GraspPipeline(
            vision=Vision(calls),
            depth=Depth(calls),
            tsdf_builder=EmptyTSDF(calls),
            grasper=Grasper(calls),
        )
        result = pipeline.run(
            np.zeros((4, 4, 3), np.uint8),
            "object",
            camera_K=np.eye(3),
        )

        self.assertIsNotNone(result.tsdf)
        self.assertEqual(result.tsdf.observed_voxels, 0)
        self.assertIn("no observed voxels", result.grasp.reason)
        self.assertNotIn("grasp.predict", calls)

    def test_vgn_runtime_failure_is_not_converted_to_empty_grasp(self):
        calls = []

        class FailingGrasper(Grasper):
            def predict(self, tsdf):
                self.calls.append("grasp.predict")
                raise RuntimeError("CUDA out of memory")

        pipeline = GraspPipeline(
            vision=Vision(calls),
            depth=Depth(calls),
            tsdf_builder=TSDF(calls),
            grasper=FailingGrasper(calls),
        )
        with self.assertRaisesRegex(
                RuntimeError, "CUDA out of memory"):
            pipeline.run(
                np.zeros((4, 4, 3), np.uint8),
                "object",
                camera_K=np.eye(3),
            )

    def test_close_attempts_all_resources_and_surfaces_failure(self):
        calls = []

        class FailingVision(Vision):
            def close(self):
                self.calls.append("vision.close")
                raise RuntimeError("vision close failed")

        pipeline = GraspPipeline(
            vision=FailingVision(calls),
            depth=Depth(calls),
            tsdf_builder=TSDF(calls),
            grasper=Grasper(calls),
        )
        pipeline.load()

        with patch(
                "grasppose.application.grasp_pipeline.log_exception"
        ) as log_exception:
            with self.assertRaisesRegex(
                    RuntimeError, "vision close failed"):
                pipeline.close()

        log_exception.assert_called_once()
        self.assertIn("vision.close", calls)
        self.assertIn("depth.close", calls)
        self.assertIn("grasp.close", calls)
        self.assertFalse(pipeline.loaded)

    def test_application_depends_on_ports_not_concrete_adapters(self):
        import inspect
        import grasppose.application.grasp_pipeline as module

        source = inspect.getsource(module)
        self.assertNotIn("ultralytics", source)
        self.assertNotIn("tensorrt", source)
        self.assertNotIn("LiteMonoDepth", source)
        self.assertNotIn("Yoloe26sVision", source)
        self.assertNotIn("VgnTensorRT", source)


if __name__ == "__main__":
    unittest.main()
