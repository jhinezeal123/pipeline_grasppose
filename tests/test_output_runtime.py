"""Fast paths must preserve geometry and optional rendering data."""

import unittest
import os
import socketserver
import tempfile
import threading

import numpy as np

from grasppose.domain.geometry import depth_to_cloud
from grasppose.domain.tsdf import ProjectiveTSDFBuilder
from grasppose.domain.types import (
    DepthResult, DetectionResult, GraspResult, PipelineResult,
    SegmentationResult, VisionResult,
)
from grasppose.output_snapshot import ARRAY_NAMES, OutputSnapshot, SnapshotCache
from grasppose.output_renderer import fetch_snapshot, render_images


class GeometryParityTests(unittest.TestCase):
    def test_masked_cloud_matches_full_grid_reference(self):
        random = np.random.RandomState(7)
        depth = random.uniform(0.03, 1.7, (35, 61)).astype(np.float32)
        depth[3, 4] = np.nan
        mask = random.rand(35, 61) > 0.65
        K = np.array([[957.7, 0, 30.2], [0, 948.8, 17.1], [0, 0, 1]])
        ys, xs = np.mgrid[:35, :61]
        z = depth.astype(np.float64)
        valid = np.isfinite(z) & (z > 0.05) & mask
        zz = z[valid]
        expected = np.stack([
            (xs[valid] - K[0, 2]) * zz / K[0, 0],
            (ys[valid] - K[1, 2]) * zz / K[1, 1],
            zz,
        ], axis=-1).astype(np.float32)
        np.testing.assert_array_equal(depth_to_cloud(depth, K, mask), expected)

    def test_tsdf_pose_combined_percentiles_match_reference(self):
        random = np.random.RandomState(9)
        cloud = random.uniform(-0.2, 0.5, (1000, 3)).astype(np.float32)
        builder = ProjectiveTSDFBuilder()
        low = np.percentile(cloud, 5.0, axis=0)
        high = np.percentile(cloud, 95.0, axis=0)
        expected = np.eye(4, dtype=np.float64)
        expected[:3, 3] = 0.5 * (low + high) - builder.size_m * 0.5
        np.testing.assert_allclose(
            builder._auto_pose(cloud), expected, rtol=0, atol=1e-7)


class SnapshotTests(unittest.TestCase):
    def make_snapshot(self):
        image = np.full((10, 12, 3), 100, np.uint8)
        mask = np.zeros((10, 12), bool)
        mask[2:7, 3:8] = True
        K = np.array([[50, 0, 6], [0, 50, 5], [0, 0, 1]], np.float64)
        vision = VisionResult(
            DetectionResult(
                np.array([[3, 2, 8, 7]], np.float32),
                np.array([0.9], np.float32), ["blue cube"]),
            SegmentationResult(mask, np.array([0.9]), 0, 1),
        )
        depth = DepthResult(np.full((10, 12), 0.5, np.float32), K, 40)
        result = PipelineResult(
            vision, depth, np.zeros((0, 3), np.float32), None,
            GraspResult.empty("none"), K, 0.5,
        )
        return OutputSnapshot.capture(image, result, 0.08, 1)

    def test_cache_is_bounded_and_expires(self):
        cache = SnapshotCache(max_entries=1, ttl_seconds=600)
        first = cache.put(self.make_snapshot())
        self.assertIsNotNone(cache.get(first))
        second = cache.put(self.make_snapshot())
        self.assertIsNone(cache.get(first))
        self.assertIsNotNone(cache.get(second))
        expired = SnapshotCache(ttl_seconds=0)
        self.assertIsNone(expired.get(expired.put(self.make_snapshot())))

    def test_wire_arrays_need_no_pickle(self):
        snapshot = self.make_snapshot()
        restored = OutputSnapshot.from_wire(
            snapshot.metadata(),
            {name: np.frombuffer(
                (memoryview(getattr(snapshot, name)).cast("B")
                 if getattr(snapshot, name).nbytes else b""),
                dtype=getattr(snapshot, name).dtype,
            ).reshape(getattr(snapshot, name).shape)
             for name in ARRAY_NAMES},
        )
        for name in ARRAY_NAMES:
            np.testing.assert_array_equal(
                getattr(restored, name), getattr(snapshot, name))
        self.assertEqual(restored.labels, ["blue cube"])

    def test_worker_streams_snapshot_to_separate_renderer(self):
        if not hasattr(socketserver, "UnixStreamServer"):
            self.skipTest("Unix domain socket server is unavailable")
        from grasppose.worker_server import Handler, WorkerServer

        snapshot = self.make_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "worker.sock")
            server = WorkerServer(socket_path, Handler)
            server.state = "ready"
            run_id = server.snapshots.put(snapshot)
            thread = threading.Thread(target=server.serve_forever,
                                      daemon=True)
            thread.start()
            try:
                restored = fetch_snapshot(run_id, socket_path=socket_path)
                for name in ARRAY_NAMES:
                    np.testing.assert_array_equal(
                        getattr(restored, name), getattr(snapshot, name))
                images = render_images(restored)
                self.assertEqual(set(images), {
                    "box", "mask", "depthmap", "grasp"})
                for image in images.values():
                    self.assertEqual(image.shape, (10, 12, 3))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
