"""Keep the per-frame CLI light while preserving its output contract."""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apps.cli import infer as cli


ROOT = Path(__file__).resolve().parents[1]


class CliFastPathTests(unittest.TestCase):
    def test_cli_starts_without_site_packages(self):
        completed = subprocess.run(
            [sys.executable, "-S", "-m", "apps.cli.infer", "--help"],
            cwd=str(ROOT), text=True, capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--prompt-id", completed.stdout)

    def test_cli_reports_worker_result_and_queues_render(self):
        response = {
            "run_id": "a" * 32,
            "snapshot_available": True,
            "render_job": {"state": "queued", "output_dir": "/tmp/output"},
            "detection_count": 1,
            "mask_pixels": 100053,
            "grasp_count": 2,
            "depth_m": 0.319,
            "server_ms": 478.8,
            "grasps": [{"score": 0.9, "width_m": 0.03,
                        "translation_m": [0.1, 0.2, 0.3],
                        "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}],
        }
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.jpg"
            image.write_bytes(b"fixture")
            output = io.StringIO()
            with patch.object(cli, "infer_image", return_value=response) as infer, \
                    contextlib.redirect_stdout(output):
                status = cli.main([
                    str(image), "--prompt-id", "cube", "--camera-k",
                    "100", "100", "40", "30", "--camera-k-size",
                    "1280", "720", "--render", "--out", directory,
                ])
        self.assertEqual(status, 0)
        self.assertEqual(infer.call_args.kwargs["camera_k_size"], [1280, 720])
        self.assertTrue(infer.call_args.kwargs["render"])
        self.assertEqual(infer.call_args.kwargs["output_dir"], directory)
        self.assertIn("DETECTIONS: 1 MASK_PIXELS=100053 GRASPS=2", output.getvalue())
        self.assertIn("RENDER_JOB: queued", output.getvalue())
        self.assertIn("worker: 478.8 ms", output.getvalue())
        self.assertIn("GRASP_POSES:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
