"""Khóa JSON và lỗi của worker trước khi tách xử lý khỏi Unix socket."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from grasppose.application.service import LocalGraspEstimator
from grasppose.infrastructure.worker.server import Handler
from tests.test_output_runtime import FixtureCore, fixture_result


class WorkerInferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.image, result = fixture_result()
        self.core = FixtureCore(result)
        self.estimator = LocalGraspEstimator(self.core)
        self.snapshots = []
        self.handler = Handler.__new__(Handler)
        self.handler.server = SimpleNamespace(
            estimator=self.estimator,
            catalog=SimpleNamespace(require=lambda prompt: None),
            snapshots=SimpleNamespace(put=self._retain),
        )

    def _retain(self, snapshot):
        self.snapshots.append(snapshot)
        return "fixed-run-id"

    def test_inference_keeps_json_fields_pose_and_snapshot_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            Image.fromarray(self.image).save(path)
            with patch("time.perf_counter", side_effect=[10.0, 10.0, 10.125]):
                response = self.handler._infer(
                    {
                        "image": str(path),
                        "prompt_id": "cube",
                        "camera_k": [100.0, 100.0, 40.0, 30.0],
                        "T_cam_volume": np.eye(4).tolist(),
                    }
                )
        self.assertEqual(
            response,
            {
                "ok": True,
                "run_id": "fixed-run-id",
                "snapshot_available": True,
                "files": [],
                "grasps": [
                    {
                        "score": 0.95,
                        "width_m": 0.04,
                        "translation_m": [-0.01, -0.01, 0.5],
                        "rotation": np.eye(3).tolist(),
                    }
                ],
                "depth_m": 0.6,
                "detection_count": 1,
                "mask_pixels": 1600,
                "grasp_count": 1,
                "server_ms": 125.0,
                "render_ms": None,
            },
        )
        np.testing.assert_array_equal(self.snapshots[0].image, self.image)
        np.testing.assert_array_equal(self.core.last_kwargs["T_cam_volume"], np.eye(4))

    def test_render_is_rejected_before_loading_an_image(self):
        with self.assertRaisesRegex(
            ValueError, "inline worker rendering is unavailable"
        ):
            self.handler._infer({"render": True})

    def test_missing_image_is_rejected_before_estimation(self):
        with self.assertRaisesRegex(ValueError, "input image does not exist"):
            self.handler._infer({"prompt_id": "cube", "image": "missing.png"})


if __name__ == "__main__":
    unittest.main()
