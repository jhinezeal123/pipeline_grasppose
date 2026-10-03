"""Chọn backend thật ở composition/preflight, không cần GPU hoặc weights."""

import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from env import check_env
from grasppose.infrastructure import composition
from grasppose.modules.depth.da3_metric import Da3MetricDepth
from grasppose.modules.depth.lite_mono import LiteMonoDepth
from tests.test_architecture import Vision, Depth, TSDF, Grasper


class DepthBackendTests(unittest.TestCase):
    def test_clean_environment_keeps_da3_as_default(self):
        env = dict(os.environ)
        env.pop("GRASP_DEPTH_BACKEND", None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from grasppose.infrastructure.composition import build_depth; "
                "assert type(build_depth()).__name__ == 'Da3MetricDepth'",
            ],
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_both_backends_are_explicit_and_da3_keeps_its_scale(self):
        self.assertIsInstance(composition.build_depth("lite-mono"), LiteMonoDepth)
        model = composition.build_depth("da3")
        self.assertIsInstance(model, Da3MetricDepth)
        self.assertEqual(model.metric_scale, composition.Da3MetricDepth().metric_scale)
        with self.assertRaisesRegex(ValueError, "GRASP_DEPTH_BACKEND"):
            composition.build_depth("typo")

    def test_simulator_depth_is_still_used_instead_of_any_real_backend(self):
        calls = []
        with patch.object(
            composition, "Yoloe26sVision", lambda: Vision(calls)
        ), patch.object(
            composition, "VgnTensorRT", lambda: Grasper(calls)
        ), patch.object(
            composition, "ProjectiveTSDFBuilder", lambda **kwargs: TSDF(calls)
        ), patch.object(
            composition,
            "build_depth",
            side_effect=AssertionError("không nạp model depth"),
        ):
            pipeline = composition.build_default_pipeline(depth=Depth(calls))
            pipeline.run(np.zeros((4, 4, 3), np.uint8), "cube", camera_K=np.eye(3))
        self.assertEqual(calls.count("depth.predict"), 1)

    def test_preflight_artifacts_only_require_the_selected_backend(self):
        lite = [str(path) for path in check_env.depth_artifacts("lite-mono")]
        da3 = [str(path) for path in check_env.depth_artifacts("da3")]
        self.assertTrue(any(path.endswith("encoder.pth") for path in lite))
        self.assertTrue(any(path.endswith("CURRENT") for path in lite))
        self.assertEqual(len(da3), 1)
        self.assertTrue(da3[0].endswith("model.onnx"))
        self.assertFalse(any("da3metric" in path for path in lite))

    def _smoke(self, backend, values):
        calls = []
        self.smoke_calls = calls

        def predict(image, camera_K):
            calls.append((image.shape, camera_K.copy()))
            return SimpleNamespace(depth=np.full(image.shape[:2], values, np.float32))

        fake = SimpleNamespace(
            load=lambda: None, predict=predict, close=lambda: calls.append("close")
        )
        with patch.object(check_env, "build_depth", return_value=fake):
            result = check_env.depth_smoke(backend)
        self.assertEqual(calls[-1], "close")
        return calls, result

    def test_lite_mono_smoke_uses_original_shape_and_camera(self):
        calls, _ = self._smoke("lite-mono", 0.6)
        self.assertEqual(calls[0][0], (192, 640, 3))
        np.testing.assert_array_equal(
            calls[0][1], [[500.0, 0.0, 320.0], [0.0, 500.0, 96.0], [0.0, 0.0, 1.0]]
        )

    def test_da3_constant_output_is_rejected_and_resource_closed(self):
        with self.assertRaisesRegex(RuntimeError, "spatially constant"):
            self._smoke("da3", 1.0)
        self.assertEqual(self.smoke_calls[-1], "close")


if __name__ == "__main__":
    unittest.main()
