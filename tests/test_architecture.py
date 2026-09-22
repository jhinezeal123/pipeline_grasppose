import unittest

import numpy as np

from grasppose.orchestrator import GraspPipeline


class _Detector:
    def __init__(self, calls):
        self.calls = calls

    def prepare(self, image, prompt):
        self.calls.append("det.prepare")
        return self

    def inference(self):
        self.calls.append("det.inference")
        return {
            "boxes": np.array([[0, 0, 2, 2]], np.float32),
            "scores": np.array([0.9], np.float32),
            "labels": ["object"],
            "reason": None,
        }

    def release(self):
        self.calls.append("det.release")


class _Depth:
    def __init__(self, calls):
        self.calls = calls
        self.shape = None

    def prepare(self, image, fov_x=None):
        self.calls.append("dep.prepare")
        self.shape = image.shape[:2]
        return self

    def inference(self):
        self.calls.append("dep.inference")
        h, w = self.shape
        return {
            "depth": np.ones((h, w), np.float32),
            "points": np.zeros((h, w, 3), np.float32),
            "intrinsics": np.array([[100.0, 0, w / 2], [0, 100.0, h / 2], [0, 0, 1]],
                                   np.float32),
            "fov_x_deg": 60.0,
            "reason": None,
        }

    def release(self):
        self.calls.append("dep.release")


class _Segmenter:
    def __init__(self, calls):
        self.calls = calls
        self.shape = None

    def prepare(self, image, boxes):
        self.calls.append("seg.prepare")
        self.shape = image.shape[:2]
        return self

    def inference(self):
        self.calls.append("seg.inference")
        h, w = self.shape
        return {
            "mask": np.ones((h, w), bool),
            "iou": np.array([0.9], np.float32),
            "best": 0,
            "n_pred": 1,
            "reason": None,
        }

    def release(self):
        self.calls.append("seg.release")


class _Grasper:
    def __init__(self, calls):
        self.calls = calls

    def prepare(self, cloud):
        self.calls.append("grasp.prepare")
        return self

    def inference(self):
        self.calls.append("grasp.inference")
        return {"graspgroup": np.zeros((0, 17), np.float64), "reason": None}

    def release(self):
        self.calls.append("grasp.release")


class ArchitectureTests(unittest.TestCase):
    def test_orchestrator_uses_injected_factories(self):
        calls = []
        p = GraspPipeline(
            detector_factory=lambda: _Detector(calls),
            segmenter_factory=lambda: _Segmenter(calls),
            depth_factory=lambda: _Depth(calls),
            grasper_factory=lambda: _Grasper(calls),
        )

        out = p.run(np.zeros((4, 4, 3), np.uint8), "object")

        self.assertIn("det", out)
        self.assertIn("seg", out)
        self.assertIn("dep", out)
        self.assertIn("grasp", out)
        self.assertIn("cloud", out)
        self.assertIn("K", out)
        self.assertLess(calls.index("dep.release"), calls.index("seg.prepare"))
        self.assertLess(calls.index("det.release"), calls.index("seg.prepare"))
        self.assertLess(calls.index("seg.release"), calls.index("grasp.prepare"))
        self.assertIn("grasp.release", calls)

    def test_orchestrator_has_no_concrete_model_defaults(self):
        with self.assertRaises(TypeError):
            GraspPipeline()


if __name__ == "__main__":
    unittest.main()
