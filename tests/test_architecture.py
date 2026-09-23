import unittest
import numpy as np
from grasppose.orchestrator import GraspPipeline


class Vision:
    def __init__(self, calls):
        self.calls = calls
    def load(self):
        self.calls.append("vision.load"); return self
    def prepare(self, image, prompt):
        self.calls.append("vision.prepare"); self.shape = image.shape[:2]; return self
    def inference(self):
        self.calls.append("vision.inference")
        h, w = self.shape
        m = np.zeros((h, w), bool); m[1:-1, 1:-1] = True
        return {
            "det": {"boxes": np.array([[1, 1, w-1, h-1]], np.float32),
                    "scores": np.array([.9], np.float32),
                    "labels": ["object"], "reason": None},
            "seg": {"mask": m, "iou": np.array([.9], np.float32),
                    "best": 0, "n_pred": 1, "reason": None},
        }
    def release(self):
        self.calls.append("vision.release")


class Depth:
    def __init__(self, calls):
        self.calls = calls
    def load(self):
        self.calls.append("depth.load"); return self
    def prepare(self, image, camera_K=None, fov_x=None):
        self.calls.append("depth.prepare"); self.shape = image.shape[:2]
        self.K = camera_K; return self
    def inference(self):
        self.calls.append("depth.inference")
        h, w = self.shape
        return {"depth": np.full((h, w), .6, np.float32),
                "intrinsics": np.asarray(self.K, np.float32),
                "fov_x_deg": 60., "scale": 1., "reason": None}
    def release(self):
        self.calls.append("depth.release")


class TSDF:
    def __init__(self, calls):
        self.calls = calls
    def build(self, *args, **kwargs):
        self.calls.append("tsdf.build")
        return {"grid": np.full((1, 40, 40, 40), .5, np.float32),
                "voxel_size": .0075,
                "T_cam_volume": np.eye(4, dtype=np.float32),
                "observed_voxels": 10}


class Grasper:
    def __init__(self, calls):
        self.calls = calls
    def load(self):
        self.calls.append("grasp.load"); return self
    def prepare(self, *args):
        self.calls.append("grasp.prepare"); return self
    def inference(self):
        self.calls.append("grasp.inference")
        return {"graspgroup": np.zeros((0, 17), np.float64), "reason": None}
    def release(self):
        self.calls.append("grasp.release")


class ArchitectureTests(unittest.TestCase):
    def _pipeline(self, calls, made):
        def factory(name, cls):
            def make():
                made[name] += 1
                return cls(calls)
            return make
        return GraspPipeline(
            factory("vision", Vision),
            factory("depth", Depth),
            factory("tsdf", TSDF),
            factory("grasp", Grasper),
        )

    def test_models_load_once_and_stay_resident_across_frames(self):
        calls = []
        made = {"vision": 0, "depth": 0, "tsdf": 0, "grasp": 0}
        p = self._pipeline(calls, made)
        K = np.array([[100., 0, 2], [0, 100., 2], [0, 0, 1.]])
        image = np.zeros((4, 4, 3), np.uint8)

        p.run(image, "object", camera_K=K)
        p.run(image, "object", camera_K=K)

        self.assertEqual(made, {"vision": 1, "depth": 1, "tsdf": 1, "grasp": 1})
        self.assertEqual(calls.count("vision.load"), 1)
        self.assertEqual(calls.count("depth.load"), 1)
        self.assertEqual(calls.count("grasp.load"), 1)
        self.assertEqual(calls.count("vision.prepare"), 2)
        self.assertEqual(calls.count("depth.prepare"), 2)
        self.assertEqual(calls.count("grasp.prepare"), 2)
        self.assertNotIn("vision.release", calls)
        self.assertNotIn("depth.release", calls)
        self.assertNotIn("grasp.release", calls)

        p.close()
        self.assertEqual(calls.count("vision.release"), 1)
        self.assertEqual(calls.count("depth.release"), 1)
        self.assertEqual(calls.count("grasp.release"), 1)

    def test_explicit_load_is_idempotent(self):
        calls = []
        made = {"vision": 0, "depth": 0, "tsdf": 0, "grasp": 0}
        p = self._pipeline(calls, made)
        p.load(); p.load()
        self.assertEqual(made, {"vision": 1, "depth": 1, "tsdf": 1, "grasp": 1})
        self.assertEqual(calls.count("vision.load"), 1)
        self.assertEqual(calls.count("depth.load"), 1)
        self.assertEqual(calls.count("grasp.load"), 1)

    def test_empty_mask_skips_per_frame_depth_and_vgn_inference(self):
        calls = []
        made = {"vision": 0, "depth": 0, "tsdf": 0, "grasp": 0}

        class EmptyVision(Vision):
            def inference(self):
                r = super().inference()
                r["seg"]["mask"][:] = False
                r["seg"]["reason"] = "none"
                return r

        p = GraspPipeline(
            lambda: EmptyVision(calls),
            lambda: Depth(calls),
            lambda: TSDF(calls),
            lambda: Grasper(calls),
        )
        out = p.run(np.zeros((4, 4, 3), np.uint8), "x", camera_K=np.eye(3))
        self.assertEqual(len(out["cloud"]), 0)
        # Models are preloaded, but unnecessary per-frame inference is skipped.
        self.assertIn("depth.load", calls)
        self.assertIn("grasp.load", calls)
        self.assertNotIn("depth.prepare", calls)
        self.assertNotIn("grasp.prepare", calls)


if __name__ == "__main__":
    unittest.main()
