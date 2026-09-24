import subprocess
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from grasppose.adapters.lite_mono import _disp_to_depth
from grasppose.adapters.vgn_trt import classify_vgn_outputs
from grasppose.adapters.yoloe import Yoloe26sVision
from grasppose.domain.geometry import (
    depth_range_str,
    depth_to_cloud,
    resolve_camera_intrinsics,
)
from grasppose.domain.tsdf import ProjectiveTSDFBuilder
from grasppose.domain.vgn import vgn_to_graspgroup
from grasppose.facade import GraspService
from grasppose.presentation.rendering import hw_open_note
import grasppose.runtime as runtime

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


class LiteMonoTensorRTTests(unittest.TestCase):
    def test_disp_to_depth_matches_litemono_conversion(self):
        disparity = np.array([0.0, 1.0], dtype=np.float32)
        depth = _disp_to_depth(disparity)
        self.assertAlmostEqual(float(depth[0]), 100.0, places=4)
        self.assertAlmostEqual(float(depth[1]), 0.1, places=5)

    def test_adapter_has_no_pytorch_runtime_dependency(self):
        source = (ROOT / "grasppose/adapters/lite_mono.py").read_text()
        self.assertNotIn("import torch", source)
        self.assertIn("litemono_infer", source)
        self.assertIn("LITEMONO_TRT_LIBRARY", source)


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
    def test_rgb_pipeline_input_is_converted_to_bgr_for_ultralytics(self):
        captured = {}

        class FakeModel:
            def set_classes(self, classes):
                captured["classes"] = list(classes)

            def predict(self, **kwargs):
                captured["kwargs"] = dict(kwargs)
                captured["source"] = kwargs["source"].copy()
                return [SimpleNamespace(boxes=[])]

        adapter = Yoloe26sVision(device="cpu")
        adapter._model = FakeModel()
        rgb = np.array(
            [[[10, 20, 30], [40, 50, 60]]],
            dtype=np.uint8,
        )

        result = adapter.predict(rgb, "cube")

        np.testing.assert_array_equal(
            captured["source"],
            np.array(
                [[[30, 20, 10], [60, 50, 40]]],
                dtype=np.uint8,
            ),
        )
        self.assertEqual(captured["classes"], ["cube"])
        self.assertEqual(captured["kwargs"]["device"], "cpu")
        self.assertNotIn("quantize", captured["kwargs"])
        self.assertNotIn("half", captured["kwargs"])
        self.assertEqual(len(result.detection.boxes), 0)

    def test_default_device_and_precision_are_left_to_ultralytics(self):
        captured = {}

        class FakeModel:
            def set_classes(self, classes):
                pass

            def predict(self, **kwargs):
                captured.update(kwargs)
                return [SimpleNamespace(boxes=[])]

        adapter = Yoloe26sVision()
        adapter._model = FakeModel()
        adapter.predict(
            np.zeros((2, 2, 3), np.uint8),
            "cube",
        )

        self.assertIsNone(adapter.device)
        self.assertNotIn("device", captured)
        self.assertNotIn("quantize", captured)
        self.assertNotIn("half", captured)

    def test_cpu_device_never_enables_fp16(self):
        captured = {}

        class FakeModel:
            def set_classes(self, classes):
                pass

            def predict(self, **kwargs):
                captured.update(kwargs)
                return [SimpleNamespace(boxes=[])]

        adapter = Yoloe26sVision(device="cpu", half=True)
        adapter._model = FakeModel()
        adapter.predict(
            np.zeros((2, 2, 3), np.uint8),
            "cube",
        )

        self.assertEqual(captured["device"], "cpu")
        self.assertNotIn("quantize", captured)

    def test_cuda_fp16_uses_official_quantize_flag(self):
        captured = {}

        class FakeModel:
            def set_classes(self, classes):
                pass

            def predict(self, **kwargs):
                captured.update(kwargs)
                return [SimpleNamespace(boxes=[])]

        adapter = Yoloe26sVision(device=0, half=True)
        adapter._model = FakeModel()
        adapter.predict(
            np.zeros((2, 2, 3), np.uint8),
            "cube",
        )

        self.assertEqual(captured["device"], 0)
        self.assertEqual(captured["quantize"], 16)
        self.assertNotIn("half", captured)

    def test_default_service_construction_does_not_import_torch(self):
        code = (
            "import sys; "
            "import grasppose.facade; "
            "assert 'torch' not in sys.modules, "
            "f'torch imported during service construction: {sys.modules.get(\"torch\")}'"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


class RenderingGuardTests(unittest.TestCase):
    def test_hardware_width_warning_is_live(self):
        self.assertEqual(hw_open_note(0.050), "")
        note = hw_open_note(0.075)
        self.assertIn("75.0 mm", note)
        self.assertIn("HW 69.4 mm", note)


class FacadeInputGuardTests(unittest.TestCase):
    def test_2d_image_raises_value_error_before_core_run(self):
        class Core:
            def run(self, *args, **kwargs):
                raise AssertionError("core must not be called")

        service = GraspService(Core())
        with self.assertRaisesRegex(
                ValueError, "input image must have shape"):
            service.infer(
                np.zeros((10, 10), np.uint8),
                prompt="object",
            )


if __name__ == "__main__":
    unittest.main()
