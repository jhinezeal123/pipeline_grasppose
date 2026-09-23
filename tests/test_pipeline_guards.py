import unittest

import numpy as np

from grasppose.adapters.vgn_trt import classify_vgn_outputs
from grasppose.domain.geometry import (
    depth_range_str,
    depth_to_cloud,
    resolve_camera_intrinsics,
)
from grasppose.domain.tsdf import ProjectiveTSDFBuilder
from grasppose.domain.vgn import vgn_to_graspgroup


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


if __name__ == "__main__":
    unittest.main()
