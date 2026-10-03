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
from grasppose.modules.grasp.vgn import refine_to_surface, vgn_to_graspgroup
from grasppose.application.service import LocalGraspEstimator
from grasppose.presentation.rendering import hw_open_note
import grasppose.infrastructure.runtime as runtime

ROOT = Path(__file__).resolve().parents[1]

# Refinement fixtures: the production 40^3 grid of 7.5 mm voxels and the
# TSDF_TRUNC_VOXELS=4.0 truncation the encoded field saturates at.
VOXEL_SIZE = 0.0075
TRUNCATION = 4.0 * VOXEL_SIZE
CENTRE = np.full(3, 0.15)
GRID_INDICES = np.indices((40, 40, 40)).reshape(3, -1).T
CORNERS = GRID_INDICES * VOXEL_SIZE


def encode_sdf(sdf):
    """Encode a signed distance field the way ProjectiveTSDFBuilder does."""
    encoded = 0.5 * (
        np.clip(np.asarray(sdf) / TRUNCATION, -1.0, 1.0) + 1.0)
    return encoded.astype(np.float32).reshape(40, 40, 40)


def _plane_distance(points, normal, offset):
    return (np.asarray(points).reshape(-1, 3) - CENTRE) @ normal - offset


def _sphere_distance(points, radius):
    return np.linalg.norm(
        np.asarray(points).reshape(-1, 3) - CENTRE, axis=1) - radius


def _interior_band(distance, margin_voxels=2.0):
    """Lattice points whose whole 3x3x3 stencil is observed and unsaturated."""
    interior = np.all((GRID_INDICES >= 1) & (GRID_INDICES <= 38), axis=1)
    return np.flatnonzero(interior & (
        np.abs(distance(CORNERS)) <= TRUNCATION - margin_voxels * VOXEL_SIZE))


def _refine_errors(grid, distance, sample):
    """Per-point (corner error, refined error, shift) in voxels."""
    corner_error, refined_error, shift = [], [], []
    for n in sample:
        index = GRID_INDICES[n]
        corner = index * VOXEL_SIZE
        refined = refine_to_surface(grid, index, VOXEL_SIZE)
        corner_error.append(distance(corner[None])[0])
        refined_error.append(distance(refined[None])[0])
        shift.append(np.linalg.norm(refined - corner))
    return (np.abs(corner_error) / VOXEL_SIZE,
            np.abs(refined_error) / VOXEL_SIZE,
            np.array(shift) / VOXEL_SIZE)


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


class SubVoxelRefineTests(unittest.TestCase):
    def test_refined_point_lands_on_plane_surface(self):
        normal = np.array([0.6, 0.5, 0.6234])
        normal /= np.linalg.norm(normal)
        offset = 0.4 * TRUNCATION
        distance = lambda points: _plane_distance(points, normal, offset)
        grid = encode_sdf(distance(CORNERS))
        sample = _interior_band(distance)
        self.assertGreater(len(sample), 1000)

        corner, refined, shift = _refine_errors(grid, distance, sample)

        # The lattice really is coarse here: up to two voxels away from the
        # plane, and never exactly on it.
        self.assertGreater(corner.max(), 1.9)
        self.assertGreater(corner.min(), 0.0)
        # A single Gauss-Newton step on the encoded field is exact for a plane.
        self.assertLess(refined.max(), 0.02)
        self.assertLessEqual(shift.max(), 4.0)

    def test_refined_point_lands_on_sphere_surface(self):
        radius = 5.0 * VOXEL_SIZE  # 37.5 mm, the scale of the reported cube
        distance = lambda points: _sphere_distance(points, radius)
        grid = encode_sdf(distance(CORNERS))
        sample = _interior_band(distance)
        self.assertGreater(len(sample), 1000)

        corner, refined, shift = _refine_errors(grid, distance, sample)

        self.assertGreater(corner.max(), 1.9)
        # Curvature costs the one-step estimate some accuracy, but only a
        # twentieth of a voxel (measured 0.043).
        self.assertLess(refined.max(), 0.1)
        self.assertLessEqual(shift.max(), 4.0)

    def test_refinement_is_off_by_default(self):
        surface = 21.5 * VOXEL_SIZE
        distance = lambda points: (
            surface - np.asarray(points).reshape(-1, 3)[:, 0])
        tsdf = encode_sdf(distance(CORNERS))[None]
        quality = np.zeros((40, 40, 40), np.float32)
        quality[19, 20, 20] = 1.0
        rotation = np.zeros((4, 40, 40, 40), np.float32)
        rotation[3, ...] = 1.0
        width = np.full((40, 40, 40), 5.0, np.float32)
        args = (tsdf, quality, rotation, width, VOXEL_SIZE, np.eye(4))
        corner = np.array([19, 20, 20]) * VOXEL_SIZE

        default = vgn_to_graspgroup(*args, threshold=0.01)
        self.assertEqual(len(default), 1)
        np.testing.assert_array_equal(default[0, 13:16], corner)

        refined = vgn_to_graspgroup(
            *args, threshold=0.01, refine_subvoxel=True)
        self.assertEqual(len(refined), 1)
        point = refined[0, 13:16]
        self.assertLess(
            abs(float(distance(point[None])[0])), 0.02 * VOXEL_SIZE)
        self.assertGreater(
            np.linalg.norm(point - corner), 2.0 * VOXEL_SIZE)
        self.assertLessEqual(
            np.linalg.norm(point - corner), 4.0 * VOXEL_SIZE)

    def test_degenerate_and_out_of_range_refuse_to_move(self):
        corner = np.array([5, 6, 7]) * VOXEL_SIZE
        for grid in (
            np.zeros((40, 40, 40), np.float32),        # unobserved everywhere
            np.full((40, 40, 40), 0.7, np.float32),    # flat: no gradient
            np.full((40, 40, 40), np.nan, np.float32),
            np.full((40, 40, 40), np.inf, np.float32),
        ):
            refined = refine_to_surface(grid, (5, 6, 7), VOXEL_SIZE)
            np.testing.assert_array_equal(refined, corner)
            self.assertTrue(np.all(np.isfinite(refined)))

        plane = encode_sdf(
            _plane_distance(CORNERS, np.array([1.0, 0, 0]), 0.1))
        for index in ((0, 0, 0), (39, 39, 39), (-1, 5, 5), (5, 5, 99)):
            refined = refine_to_surface(plane, index, VOXEL_SIZE)
            np.testing.assert_array_equal(
                refined, np.array(index, np.float64) * VOXEL_SIZE)
            self.assertTrue(np.all(np.isfinite(refined)))

        # One unobserved voxel in the stencil is enough to refuse the step.
        holed = plane.copy()
        holed[6, 6, 8] = 0.0
        np.testing.assert_array_equal(
            refine_to_surface(holed, (5, 6, 7), VOXEL_SIZE), corner)

    def test_shift_bound_is_enforced(self):
        index = np.array([5, 6, 7])
        corner = index * VOXEL_SIZE

        # A shallow gradient predicts a crossing thousands of voxels away.
        shallow = 0.9 + 1e-4 * np.arange(40, dtype=np.float32)
        np.testing.assert_array_equal(
            refine_to_surface(
                np.broadcast_to(shallow, (40, 40, 40)).copy(),
                index, VOXEL_SIZE),
            corner)

        two_voxels = encode_sdf(
            (index[0] + 2.0) * VOXEL_SIZE - CORNERS[:, 0])
        refined = refine_to_surface(two_voxels, index, VOXEL_SIZE)
        self.assertAlmostEqual(
            np.linalg.norm(refined - corner) / VOXEL_SIZE, 2.0, places=6)
        np.testing.assert_array_equal(
            refine_to_surface(
                two_voxels, index, VOXEL_SIZE, max_shift_voxels=1.0),
            corner)

        # Beyond the truncation band the field saturates and says nothing.
        saturated = encode_sdf(
            (index[0] + 4.5) * VOXEL_SIZE - CORNERS[:, 0])
        np.testing.assert_array_equal(
            refine_to_surface(saturated, index, VOXEL_SIZE), corner)


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
