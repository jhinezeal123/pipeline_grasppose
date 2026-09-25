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

from grasppose.domain.types import (
    DepthResult, DetectionResult, GraspResult, PipelineResult,
    SegmentationResult, VisionResult,
)
from grasppose.facade import GraspService
from grasppose.output_renderer import (
    fetch_snapshot, render_and_save, render_images,
)
from grasppose.output_snapshot import (
    ARRAY_NAMES, OutputSnapshot, SnapshotCache,
)
from grasppose.worker_client import infer_image
try:
    from grasppose.worker_server import Handler, WorkerServer
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
    def test_render_flag_returns_the_four_diagnostic_files(self):
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
            output_dir = os.path.join(directory, "output")
            Image.fromarray(image).save(image_path)
            response = handler._infer({
                "prompt_id": "cube", "image": image_path,
                "camera_k": [100, 100, 40, 30],
                "render": True, "output_dir": output_dir,
            })
            self.assertTrue(response["ok"])
            self.assertEqual(len(response["files"]), 4)
            self.assertGreaterEqual(response["render_ms"], 0)
            self.assertEqual(
                [os.path.basename(path) for path in response["files"]],
                ["box.png", "mask.png", "depthmap.png", "grasp.png"],
            )

    def test_client_defaults_to_no_render(self):
        with patch("grasppose.worker_client.request_worker", return_value={
                "ok": True}) as request:
            infer_image("frame.png", "cube", camera_k_size=[1280, 720])
        self.assertFalse(request.call_args.args[0]["render"])
        self.assertEqual(
            request.call_args.args[0]["camera_k_size"], [1280, 720])

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
