"""Khóa thứ tự grasp, đơn vị và calibration trước khi tách service."""

import unittest

import numpy as np

from grasppose.application.service import LocalGraspEstimator
from tests.test_output_runtime import FixtureCore, fixture_result


class EstimatorContractTests(unittest.TestCase):
    def test_selection_preserves_width_boundary_score_order_and_plain_floats(self):
        image, result = fixture_result()
        rows = np.repeat(result.grasp.graspgroup, 3, axis=0)
        rows[:, 0] = [0.4, 0.99, 0.8]
        rows[:, 1] = [0.04, 0.080001, 0.08]
        rows[:, 13] = [0.1, 0.2, 0.3]
        result.grasp.graspgroup = rows
        estimate = LocalGraspEstimator(FixtureCore(result)).estimate(
            image, "cube", camera_K=np.eye(3), max_width=0.08, top=2
        )
        self.assertEqual([g.score for g in estimate.grasps], [0.8, 0.4])
        self.assertEqual([g.translation_m[0] for g in estimate.grasps], [0.3, 0.1])
        self.assertEqual(estimate.grasp_count, 3)
        self.assertIs(type(estimate.grasps[0].score), float)
        self.assertIs(type(estimate.grasps[0].rotation[0][0]), float)

    def test_camera_resize_and_fovy_are_resolved_before_the_pipeline(self):
        image, result = fixture_result()
        core = FixtureCore(result)
        LocalGraspEstimator(core).estimate(
            image,
            "cube",
            camera_K=[200.0, 200.0, 80.0, 60.0],
            camera_K_size=(160, 120),
            fov_y=60.0,
        )
        np.testing.assert_array_equal(core.last_kwargs["camera_K"], result.camera_K)
        expected = 2 * np.degrees(np.arctan(np.tan(np.radians(30)) * 80 / 60))
        self.assertEqual(core.last_kwargs["fov_x"], float(expected))

    def test_invalid_request_does_not_start_the_pipeline(self):
        image, result = fixture_result()
        core = FixtureCore(result)
        estimator = LocalGraspEstimator(core)
        for kwargs in [{"top": 0}, {"max_width": 0}, {"camera_K_size": (80, 60)}]:
            with self.assertRaises(ValueError):
                estimator.estimate(image, "cube", **kwargs)
        self.assertFalse(hasattr(core, "last_kwargs"))
