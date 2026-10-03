"""Giữ import/pickle/type identity của contract khi gộp implementation."""

import importlib
import pickle
import unittest


class FeatureContractCompatibilityTests(unittest.TestCase):
    def test_existing_port_and_type_paths_keep_class_identity(self):
        expected = {
            "vision": ("VisionPort", ("DetectionResult", "SegmentationResult", "VisionResult")),
            "depth": ("DepthPort", ("DepthResult",)),
            "grasp": ("GraspPort", ("GraspResult",)),
            "tsdf": ("TSDFPort", ("TSDFResult",)),
        }
        for feature, (port_name, result_names) in expected.items():
            prefix = "grasppose.modules." + feature
            for leaf, names in (("port", (port_name,)), ("types", result_names)):
                module = importlib.import_module(prefix + "." + leaf)
                for name in names:
                    with self.subTest(feature=feature, name=name):
                        contract = getattr(module, name)
                        self.assertEqual(contract.__module__, prefix + "." + leaf)
                        self.assertIs(pickle.loads(pickle.dumps(contract)), contract)


if __name__ == "__main__":
    unittest.main()
