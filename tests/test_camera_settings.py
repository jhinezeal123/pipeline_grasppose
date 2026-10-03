"""Characterization: CLI và Gradio giữ cùng contract camera trước/sau refactor."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from apps.cli import infer
from apps.gradio import app


class CameraSettingsContractTests(unittest.TestCase):
    def options(self, **values):
        return SimpleNamespace(camera_k=values.get("matrix"), camera_k_size=values.get("size"))

    def test_unconfigured_camera_remains_none(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(infer._camera_k(self.options()))
            self.assertIsNone(infer._camera_k_size(self.options()))
            self.assertIsNone(app._camera_k())
            self.assertIsNone(app._camera_k_size())

    def test_shared_matrix_and_comma_separated_resolution(self):
        with patch.dict(os.environ, {"CAMERA_K": "100 101 40 30", "CAMERA_K_SIZE": "1280,720"}, clear=True):
            self.assertEqual(infer._camera_k(self.options()), [100.0, 101.0, 40.0, 30.0])
            self.assertEqual(infer._camera_k(self.options()), app._camera_k())
            self.assertEqual(infer._camera_k_size(self.options()), [1280, 720])
            self.assertEqual(infer._camera_k_size(self.options()), app._camera_k_size())

    def test_explicit_cli_values_override_invalid_environment(self):
        options = self.options(matrix=[1, 2, 3, 4], size=[80, 60])
        with patch.dict(os.environ, {"CAMERA_K": "invalid", "CAMERA_K_SIZE": "invalid"}, clear=True):
            self.assertIs(infer._camera_k(options), options.camera_k)
            self.assertIs(infer._camera_k_size(options), options.camera_k_size)

    def test_matrix_cardinality_error_is_unchanged(self):
        with patch.dict(os.environ, {"CAMERA_K": "1 2 3"}, clear=True):
            for read in (lambda: infer._camera_k(self.options()), app._camera_k):
                with self.assertRaisesRegex(ValueError, "^CAMERA_K must contain FX FY CX CY$"):
                    read()

    def test_resolution_cardinality_error_is_unchanged(self):
        with patch.dict(os.environ, {"CAMERA_K_SIZE": "80"}, clear=True):
            for read in (lambda: infer._camera_k_size(self.options()), app._camera_k_size):
                with self.assertRaisesRegex(ValueError, "^CAMERA_K_SIZE must contain WIDTH HEIGHT$"):
                    read()

    def test_resolution_parsing_is_independent_of_matrix(self):
        with patch.dict(os.environ, {"CAMERA_K": "invalid", "CAMERA_K_SIZE": "80 60"}, clear=True):
            self.assertEqual(infer._camera_k_size(self.options()), [80, 60])
            self.assertEqual(app._camera_k_size(), [80, 60])

    def test_environment_changes_are_observed_on_every_call(self):
        with patch.dict(os.environ, {"CAMERA_K": "1 2 3 4"}, clear=True):
            self.assertEqual(app._camera_k(), [1, 2, 3, 4])
            os.environ["CAMERA_K"] = "5 6 7 8"
            self.assertEqual(app._camera_k(), [5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
