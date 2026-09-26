"""Fast inference and opt-in diagnostic rendering runtime coverage."""

import os
import tempfile
import threading
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from grasppose.application.types import PipelineResult
from grasppose.modules.depth.types import DepthResult
from grasppose.modules.grasp.types import GraspResult
from grasppose.modules.vision.types import (
    DetectionResult, SegmentationResult, VisionResult,
)
from grasppose.facade import GraspService
from grasppose.infrastructure.output.renderer import (
    fetch_snapshot, render_and_save, render_images,
)
from grasppose.infrastructure.output.snapshot import (
    ARRAY_NAMES, OutputSnapshot, SnapshotCache,
)
from grasppose.infrastructure.worker.client import WorkerError, infer_image
from grasppose.infrastructure.output import control as output_control
try:
    from grasppose.infrastructure.worker.server import Handler, WorkerServer
except AttributeError:  # Windows Python has no socketserver.UnixStreamServer.
    Handler = WorkerServer = None


def fixture_result():
    height, width = 60, 80
    image = np.full((height, width, 3), 110, np.uint8)
    mask = np.zeros((height, width), bool)
    mask[10:50, 20:60] = True
    K = np.array([
        [100.0, 0.0, 40.0],
        [0.0, 100.0, 30.0],
        [0.0, 0.0, 1.0],
    ])
    vision = VisionResult(
        DetectionResult(
            np.array([[20, 10, 60, 50]], np.float32),
            np.array([0.93], np.float32), ["blue cube"],
        ),
        SegmentationResult(mask, np.array([0.93], np.float32), 0, 1),
    )
    depth = np.linspace(0.25, 1.5, height * width, dtype=np.float32).reshape(
        height, width)
    depth_result = DepthResult(depth, K, 43.6)
    graspgroup = np.zeros((1, 17), np.float64)
    graspgroup[0, 0] = 0.95
    graspgroup[0, 1:4] = (0.04, 0.02, 0.04)
    graspgroup[0, 4:13] = np.eye(3).reshape(-1)
    graspgroup[0, 13:16] = (-0.01, -0.01, 0.5)
    result = PipelineResult(
        vision=vision,
        depth=depth_result,
        cloud=np.zeros((0, 3), np.float32),
        tsdf=None,
        grasp=GraspResult(graspgroup),
        camera_K=K,
        depth_m=0.6,
    )
    return image, result


class FixtureCore:
    def __init__(self, result):
        self.result = result

    def run(self, *args, **kwargs):
        self.last_kwargs = kwargs
        return self.result


class CacheAndRenderingTests(unittest.TestCase):
    def test_snapshot_owns_stable_frame_buffers(self):
        image, result = fixture_result()
        snapshot = OutputSnapshot.capture(image, result, 0.08, 1)
        expected = {name: getattr(snapshot, name).copy() for name in ARRAY_NAMES}
        image[:] = 0
        result.vision.segmentation.mask[:] = False
        result.depth.depth[:] = 0
        result.vision.detection.boxes[:] = 0
        result.vision.detection.scores[:] = 0
        result.grasp.graspgroup[:] = 0
        result.camera_K[:] = 0
        for name in ARRAY_NAMES:
            np.testing.assert_array_equal(getattr(snapshot, name), expected[name])

    def test_cache_is_bounded_expires_and_rejects_oversized_frame(self):
        image, result = fixture_result()
        snapshot = OutputSnapshot.capture(image, result, 0.08, 1)
        cache = SnapshotCache(max_entries=1)
        first = cache.put(snapshot)
        self.assertIsNotNone(cache.get(first))
        second = cache.put(snapshot)
        self.assertIsNone(cache.get(first))
        self.assertIsNotNone(cache.get(second))

        expired = SnapshotCache(ttl_seconds=0)
        expired_id = expired.put(snapshot)
        self.assertIsNone(expired.get(expired_id))
        too_small = SnapshotCache(max_bytes=1)
        self.assertIsNone(too_small.put(snapshot))

    def test_rendered_pngs_match_legacy_facade_output(self):
        image, result = fixture_result()
        old_path = GraspService(FixtureCore(result)).infer(
            image, "blue_cube", camera_K=result.camera_K,
            max_width=0.08, top=1,
        )
        snapshot = OutputSnapshot.capture(image, result, 0.08, 1)
        new_path = render_images(snapshot)
        self.assertEqual(set(old_path) & {"box", "mask", "depthmap", "grasp"},
                         set(new_path))
        for name in ("box", "mask", "depthmap", "grasp"):
            np.testing.assert_array_equal(new_path[name], old_path[name])

        with tempfile.TemporaryDirectory() as directory:
            run_id = uuid.uuid4().hex
            files, _ = render_and_save(snapshot, run_id, directory)
            self.assertEqual(len(files), 4)
            for name, path in zip(
                    ("box", "mask", "depthmap", "grasp"), files):
                with Image.open(path) as rendered:
                    np.testing.assert_array_equal(
                        np.asarray(rendered.convert("RGB")), old_path[name])

    @unittest.skipIf(Handler is None, "Unix worker server is unavailable")
    def test_fast_worker_response_skips_render_and_png_writes(self):
        image, result = fixture_result()
        class Catalog:
            def require(self, prompt_id):
                if prompt_id != "cube":
                    raise ValueError("unknown prompt")

        server = SimpleNamespace(
            catalog=Catalog(),
            service=SimpleNamespace(core=FixtureCore(result)),
            snapshots=SnapshotCache(),
        )
        handler = object.__new__(Handler)
        handler.server = server
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "frame.png")
            Image.fromarray(image).save(image_path)
            output_dir = os.path.join(directory, "output")
            response = handler._infer({
                "prompt_id": "cube", "image": image_path,
                "camera_k": [100, 100, 40, 30],
            })
            self.assertTrue(response["ok"])
            self.assertTrue(response["snapshot_available"])
            self.assertEqual(response["files"], [])
            self.assertIsNone(response["render_ms"])
            self.assertTrue(response["run_id"])
            self.assertEqual(len(response["grasps"]), 1)
            self.assertFalse(os.path.exists(output_dir))

    @unittest.skipIf(Handler is None, "Unix worker server is unavailable")
    def test_worker_scales_calibrated_K_to_decoded_image(self):
        image, result = fixture_result()
        class Catalog:
            def require(self, prompt_id):
                if prompt_id != "cube":
                    raise ValueError("unknown prompt")

        core = FixtureCore(result)
        handler = object.__new__(Handler)
        handler.server = SimpleNamespace(
            catalog=Catalog(),
            service=SimpleNamespace(core=core),
            snapshots=SnapshotCache(),
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "frame.png")
            Image.fromarray(image).save(image_path)
            handler._infer({
                "prompt_id": "cube", "image": image_path,
                "camera_k": [50, 50, 20, 15],
                "camera_k_size": [40, 30],
            })
        np.testing.assert_allclose(
            core.last_kwargs["camera_K"],
            [[100, 0, 40], [0, 100, 30], [0, 0, 1]],
        )

    @unittest.skipIf(Handler is None, "Unix worker server is unavailable")
    def test_worker_rejects_inline_render(self):
        handler = object.__new__(Handler)
        with self.assertRaisesRegex(ValueError, "inline worker rendering"):
            handler._infer({"render": True})

    @unittest.skipIf(Handler is None, "Unix worker server is unavailable")
    def test_render_runs_in_a_separate_process(self):
        image, result = fixture_result()
        class Catalog:
            def require(self, prompt_id):
                if prompt_id != "cube":
                    raise ValueError("unknown prompt")

        with tempfile.TemporaryDirectory() as directory:
            runtime_dir = os.path.join(directory, "runtime")
            os.makedirs(runtime_dir)
            socket_path = os.path.join(runtime_dir, "worker.sock")
            image_path = os.path.join(directory, "frame.png")
            output_dir = os.path.join(directory, "output")
            Image.fromarray(image).save(image_path)
            server = WorkerServer(socket_path, Handler)
            server.catalog = Catalog()
            server.service = SimpleNamespace(core=FixtureCore(result))
            server.state = "ready"
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with patch("grasppose.infrastructure.worker.client.WORKER_SOCKET", socket_path), \
                        patch.object(output_control, "SOCKET_PATH", socket_path), \
                        patch.object(output_control, "JOB_DIR", os.path.join(
                            runtime_dir, "output-jobs")), \
                        patch.dict(os.environ, {
                            "GRASP_RUNTIME_DIR": runtime_dir,
                            "GRASP_WORKER_SOCKET": socket_path,
                        }):
                    response = infer_image(
                        image_path, "cube", camera_k=[100, 100, 40, 30],
                        render=True, output_dir=output_dir)
                    self.assertEqual(response["render_job"]["state"], "queued")
                    self.assertNotEqual(response["render_job"]["pid"],
                                        os.getpid())
                    self.assertEqual(response["files"], [])
                    self.assertIsNone(response["render_ms"])
                    job = output_control.wait(response["run_id"], timeout=30)
                    self.assertEqual(job["state"], "done", job.get("error"))
                    self.assertEqual(
                        [os.path.basename(path) for path in job["files"]],
                        ["box.png", "mask.png", "depthmap.png", "grasp.png"],
                    )
                    self.assertEqual(
                        os.path.dirname(job["files"][0]),
                        os.path.join(output_dir, response["run_id"]),
                    )
                    self.assertTrue(all(os.path.isfile(path)
                                        for path in job["files"]))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_client_defaults_to_no_render(self):
        with patch("grasppose.infrastructure.worker.client.request_worker", return_value={
                "ok": True}) as request:
            infer_image("frame.png", "cube", camera_k_size=[1280, 720])
        self.assertFalse(request.call_args.args[0]["render"])
        self.assertEqual(
            request.call_args.args[0]["camera_k_size"], [1280, 720])

    def test_client_reports_unavailable_render_snapshot(self):
        with patch("grasppose.infrastructure.worker.client.request_worker", return_value={
                "ok": True, "run_id": "a" * 32,
                "snapshot_available": False}):
            with self.assertRaisesRegex(WorkerError, "snapshot was not retained"):
                infer_image("frame.png", "cube", render=True)

    @unittest.skipIf(Handler is None, "Unix worker server is unavailable")
    def test_snapshot_can_be_retrieved_over_worker_socket(self):
        image, result = fixture_result()
        snapshot = OutputSnapshot.capture(image, result, 0.08, 1)
        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "worker.sock")
            server = WorkerServer(socket_path, Handler)
            run_id = server.snapshots.put(snapshot)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                restored = fetch_snapshot(run_id, socket_path=socket_path)
                for name in ARRAY_NAMES:
                    np.testing.assert_array_equal(
                        getattr(restored, name), getattr(snapshot, name))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
