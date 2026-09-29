import subprocess
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from grasppose.modules.grasp.vgn_trt import classify_vgn_outputs
from grasppose.modules.vision.yoloe import Yoloe26sVision
from grasppose.modules.depth.geometry import (
    depth_range_str,
    depth_to_cloud,
    resolve_camera_intrinsics,
    scale_camera_intrinsics,
)
from grasppose.modules.tsdf.projective import ProjectiveTSDFBuilder
from grasppose.modules.grasp.vgn import vgn_to_graspgroup
from grasppose.application.service import LocalGraspEstimator
from grasppose.presentation.rendering import hw_open_note
import grasppose.infrastructure.runtime as runtime

ROOT = Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
    def test_depth_to_cloud_uses_mask_and_K(self):
        depth = np.ones((3, 3), np.float32)
        K = np.array([
            [2.0, 0, 1.0],
            [0, 2.0, 1.0],
            [0, 0, 1.0],
        ])
        mask = np.zeros((3, 3), bool)
        mask[1, 1] = True
        cloud = depth_to_cloud(depth, K, mask)

        self.assertEqual(cloud.shape, (1, 3))
        self.assertTrue(
            np.allclose(cloud[0], [0, 0, 1]))

    def test_intrinsics_scale_with_full_frame_resize(self):
        K = np.array([
            [957.746642, 0.0, 636.883856],
            [0.0, 948.820235, 352.232764],
            [0.0, 0.0, 1.0],
        ])
        scaled = scale_camera_intrinsics(K, (1280, 720), (1920, 1080))
        np.testing.assert_allclose(
            scaled,
            [[1436.619963, 0.0, 955.325784],
             [0.0, 1423.2303525, 528.349146],
             [0.0, 0.0, 1.0]],
            rtol=0, atol=1e-9,
        )
        np.testing.assert_array_equal(
            scale_camera_intrinsics(K, (1280, 720), (1280, 720)), K)
        self.assertEqual(K[0, 0], 957.746642)

    def test_intrinsics_reject_bad_calibration_size(self):
        K = np.eye(3)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            scale_camera_intrinsics(K, (0, 720), (1920, 1080))
        with self.assertRaisesRegex(ValueError, "WIDTH HEIGHT"):
            scale_camera_intrinsics(K, (1280,), (1920, 1080))

    def test_depth_range_empty_is_safe(self):
        text = depth_range_str(
            np.zeros((2, 2), np.float32))
        self.assertIn("no valid depth", text)

    def test_intrinsics_required(self):
        with self.assertRaises(ValueError):
            resolve_camera_intrinsics(
                None, None, 640, 480)
        self.assertEqual(
            resolve_camera_intrinsics(
                None, 60.0, 640, 480).shape,
            (3, 3),
        )


class TSDFTests(unittest.TestCase):
    def test_projective_grid_shape_and_encoding(self):
        height, width = 48, 64
        depth = np.full(
            (height, width), 0.6, np.float32)
        K = np.array([
            [60.0, 0, width / 2],
            [0, 60.0, height / 2],
            [0, 0, 1.0],
        ])
        mask = np.ones((height, width), bool)
        cloud = depth_to_cloud(depth, K, mask)
        result = ProjectiveTSDFBuilder().build(
            depth, K, mask, cloud)

        self.assertEqual(
            result.grid.shape, (1, 40, 40, 40))
        self.assertGreater(
            result.observed_voxels, 0)
        self.assertGreaterEqual(
            float(result.grid.min()), 0.0)
        self.assertLessEqual(
            float(result.grid.max()), 1.0)


class VGNTests(unittest.TestCase):
    def test_output_classification_by_names(self):
        outputs = {
            "quality": np.zeros(
                (1, 1, 40, 40, 40)),
            "rotation": np.zeros(
                (1, 4, 40, 40, 40)),
            "width": np.zeros(
                (1, 1, 40, 40, 40)),
        }
        quality, rotation, width = (
            classify_vgn_outputs(outputs)
        )
        self.assertEqual(
            quality.shape, (40, 40, 40))
        self.assertEqual(
            rotation.shape, (4, 40, 40, 40))
        self.assertEqual(
            width.shape, (40, 40, 40))

    def test_vgn_conversion(self):
        tsdf = np.ones(
            (1, 40, 40, 40), np.float32)
        quality = np.zeros(
            (40, 40, 40), np.float32)
        quality[20, 20, 20] = 1.0
        rotation = np.zeros(
            (4, 40, 40, 40), np.float32)
        rotation[3, ...] = 1.0
        width = np.full(
            (40, 40, 40), 5.0, np.float32)

        graspgroup = vgn_to_graspgroup(
            tsdf,
            quality,
            rotation,
            width,
            0.0075,
            np.eye(4),
            threshold=0.01,
        )

        self.assertEqual(graspgroup.shape[1], 17)
        self.assertGreaterEqual(len(graspgroup), 1)
        self.assertAlmostEqual(
            graspgroup[0, 1], 0.0375, places=5)


class RuntimeCleanupTests(unittest.TestCase):
    def test_release_attributes_drops_references_before_cuda_cleanup(self):
        events = []

        class Target:
            model = object()

        target = Target()

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def synchronize():
                events.append("synchronize")

            @staticmethod
            def empty_cache():
                events.append("empty_cache")

        fake_torch = SimpleNamespace(cuda=FakeCuda())

        def collect():
            self.assertIsNone(target.model)
            events.append("gc")

        with patch.dict(sys.modules, {"torch": fake_torch}), \
                patch.object(runtime.gc, "collect", side_effect=collect):
            runtime.release_attributes(target, "model")

        self.assertEqual(
            events, ["gc", "synchronize", "empty_cache"])


class YoloeInputTests(unittest.TestCase):
    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

        def __len__(self):
            return len(self.value)

    class Catalog:
        def __init__(self):
            self.prompts = [{
                "id": "blue_cube",
                "text": "the blue cube",
                "class_index": 0,
            }]
            self.by_id = {"blue_cube": self.prompts[0]}
            self.manifest = {"engine": {"imgsz": 640}}
            self.artifact_dir = "/tmp/fake-yoloe"

        def require(self, prompt_id):
            if prompt_id not in self.by_id:
                raise ValueError("unknown prompt ID")
            return self.by_id[prompt_id]

    def test_rgb_is_converted_and_only_selected_fixed_prompt_is_returned(self):
        captured = {}

        class Boxes:
            cls = YoloeInputTests.Tensor([0, 1])
            conf = YoloeInputTests.Tensor([0.7, 0.9])
            xyxy = YoloeInputTests.Tensor([
                [1, 2, 11, 12],
                [20, 20, 30, 30],
            ])

            def __len__(self):
                return 2

        class FakeModel:
            def predict(self, **kwargs):
                captured["kwargs"] = dict(kwargs)
                captured["source"] = kwargs["source"].copy()
                masks = np.zeros((2, 1, 2), np.float32)
                masks[0, 0, 0] = 1
                masks[1, 0, 1] = 1
                return [SimpleNamespace(
                    boxes=Boxes(),
                    masks=SimpleNamespace(
                        data=YoloeInputTests.Tensor(masks)),
                )]

        adapter = Yoloe26sVision()
        adapter._model = FakeModel()
        adapter._catalog = self.Catalog()
        rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        result = adapter.predict(rgb, "blue_cube")

        np.testing.assert_array_equal(
            captured["source"],
            np.array([[[30, 20, 10], [60, 50, 40]]], dtype=np.uint8),
        )
        self.assertEqual(captured["kwargs"]["device"], 0)
        self.assertEqual(captured["kwargs"]["imgsz"], 640)
        self.assertTrue(captured["kwargs"]["retina_masks"])
        self.assertEqual(result.detection.labels, ["the blue cube"])
        np.testing.assert_array_equal(
            result.detection.boxes, np.array([[1, 2, 11, 12]], np.float32))
        self.assertEqual(result.segmentation.mask.shape, (1, 2))
        self.assertTrue(result.segmentation.mask[0, 0])

    def test_unknown_prompt_id_fails_before_model_prediction(self):
        class FakeModel:
            def predict(self, **kwargs):
                raise AssertionError("prediction must not run")

        adapter = Yoloe26sVision()
        adapter._model = FakeModel()
        adapter._catalog = self.Catalog()
        with self.assertRaisesRegex(ValueError, "unknown prompt ID"):
            adapter.predict(np.zeros((2, 2, 3), np.uint8), "free text")

    def test_default_service_construction_does_not_import_torch(self):
        code = (
            "import sys; "
            "import grasppose.infrastructure.composition; "
            "assert 'torch' not in sys.modules, "
            "f'torch imported during estimator composition: {sys.modules.get(\"torch\")}'"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            completed.returncode, 0,
            msg=completed.stdout + completed.stderr,
        )


class RenderingGuardTests(unittest.TestCase):
    def test_hardware_width_warning_is_live(self):
        self.assertEqual(hw_open_note(0.050), "")
        note = hw_open_note(0.075)
        self.assertIn("75.0 mm", note)
        self.assertIn("HW 69.4 mm", note)


class EstimatorInputGuardTests(unittest.TestCase):
    def test_2d_image_raises_value_error_before_core_run(self):
        class Core:
            def run(self, *args, **kwargs):
                raise AssertionError("core must not be called")

        estimator = LocalGraspEstimator(Core())
        with self.assertRaisesRegex(
                ValueError, "input image must have shape"):
            estimator.estimate(
                np.zeros((10, 10), np.uint8),
                prompt_id="object",
            )


if __name__ == "__main__":
    unittest.main()
