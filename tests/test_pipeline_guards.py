"""Guard regressions cho pipeline.py: khong GPU, khong model, khong mang.

Bon bug nay deu thuoc loai "chi no khi model tra ve du lieu SUY BIEN" (MoGe
khong do duoc gi, GraspNess khong co tu the nao) — dung luc can anh chan doan
nhat thi pipeline lai chet. Test o day dung model GIA de ep dung cac dau vao
suy bien do, nen chay duoc tren CI ubuntu-latest khong GPU.
"""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pipeline', ROOT / 'pipeline.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

H, W = 240, 320


def _K():
    return P.K_from_fovy(62.0, W, H)


class _FakeDetector(P.Object_Detection):
    def __init__(self):
        self.model_path = 'fake'
        self.device = 'cpu'

    def prepare(self, image, prompt):
        return self

    def inference(self):
        return {'boxes': np.array([[100.0, 120.0, 300.0, 340.0]], np.float32),
                'scores': np.array([0.42], np.float32), 'labels': ['bag'],
                'reason': None}

    def release(self):
        pass


class _FakeSegmenter(P.Segmentation):
    """Mask CO pixel that su (khong rong) — day la diem mau chot cua P1-b."""

    def __init__(self):
        self.model_path = 'fake'
        self.device = 'cpu'

    def prepare(self, image, boxes):
        return self

    def inference(self):
        m = np.zeros((H, W), bool)
        m[120:200, 100:220] = True
        return {'mask': m, 'iou': np.array([0.91], np.float32), 'best': 0,
                'n_pred': 1, 'reason': None}

    def release(self):
        pass


class _FakeDepth(P.Depth_Estimate):
    """MoGe gia. `zeros=True` mo phong truong hop MoGe KHONG do duoc gi."""

    def __init__(self, zeros=False):
        self.model_path = 'fake'
        self.device = 'cpu'
        self.zeros = zeros

    def prepare(self, image, fov_x=None):
        return self

    def inference(self):
        if self.zeros:
            z = np.zeros((H, W), np.float32)
        else:
            z = np.full((H, W), 0.7, np.float32)
        return {'depth': z, 'points': np.zeros((H, W, 3), np.float32),
                'intrinsics': _K().astype(np.float32), 'fov_x_deg': 77.4,
                'reason': None}

    def release(self):
        pass


class _FakeGrasper(P.GraspNess):
    """Ghi lai xem prepare() co bi goi khong — de chung minh cloud rong bi bo qua."""

    def __init__(self):
        self.model_path = 'fake'
        self.device = 'cpu'
        self.prepared = False

    def prepare(self, points, num_point=15000):
        self.prepared = True
        self._n = len(points)
        return self

    def inference(self):
        raise AssertionError('GraspNess KHONG duoc goi khi cloud rong')

    def release(self):
        pass


class DepthRangeStrTests(unittest.TestCase):
    """P1-a: dong log chan doan khong duoc crash khi MoGe tra depth toan 0."""

    def test_all_zero_depth_does_not_raise(self):
        # Truoc day: dep["depth"][dep["depth"] > 0].min() -> mang rong ->
        # ValueError: zero-size array to reduction operation minimum.
        text = P.depth_range_str(np.zeros((4, 5), np.float32))
        self.assertIn('khong do duoc', text)

    def test_all_zero_says_not_measured_not_a_fake_number(self):
        """Phai NOI RO la khong do duoc, khong duoc nuot loi im lang."""
        text = P.depth_range_str(np.zeros((4, 5), np.float32))
        self.assertNotRegex(text, r'\d\.\d{3}\.\.')   # khong bia ra khoang so

    def test_nan_and_inf_only_does_not_raise(self):
        d = np.full((4, 4), np.nan, np.float32)
        d[0, 0] = np.inf
        self.assertIn('khong do duoc', P.depth_range_str(d))

    def test_valid_depth_reports_range(self):
        d = np.full((4, 4), 0.5, np.float32)
        d[0, 0] = 1.25
        text = P.depth_range_str(d)
        self.assertIn('0.500..1.250 m', text)
        self.assertIn('16 px hop le', text)

    def test_ignores_nonpositive_pixels(self):
        """Pixel 0 la 'khong do duoc', khong duoc keo min xuong 0."""
        d = np.full((4, 4), 0.8, np.float32)
        d[0, 0] = 0.0
        self.assertIn('0.800..0.800 m', P.depth_range_str(d))
        self.assertIn('15 px hop le', P.depth_range_str(d))


class EmptyCloudGuardTests(unittest.TestCase):
    """P1-b: mask CO pixel nhung depth trong mask khong hop le -> cloud rong."""

    def _run(self, zeros):
        grasper = _FakeGrasper()
        out = P.run_phases(np.zeros((H, W, 3), np.uint8), 'a bag',
                           detector=_FakeDetector(), segmenter=_FakeSegmenter(),
                           depther=_FakeDepth(zeros=zeros), grasper=grasper)
        return out, grasper

    def test_mask_has_pixels_but_no_valid_depth_does_not_crash(self):
        """Day chinh la ca P1-b: mask.any() True, cloud (0,3), cloud.max(0) no."""
        out, _ = self._run(zeros=True)
        self.assertEqual(len(out['cloud']), 0)
        self.assertEqual(out['grasp']['graspgroup'].shape, (0, 17))

    def test_empty_cloud_skips_graspness(self):
        """Khong duoc roi xuong goi GraspNess voi cloud rong."""
        _, grasper = self._run(zeros=True)
        self.assertFalse(grasper.prepared)

    def test_empty_cloud_reason_is_explicit(self):
        out, _ = self._run(zeros=True)
        reason = out['grasp']['reason']
        self.assertTrue(reason)
        self.assertIn('cloud rong', reason)

    def test_empty_cloud_matches_empty_mask_branch(self):
        """Nhat quan voi nhanh `mask rong`: cung cloud shape, cung kieu graspgroup."""
        empty_depth, _ = self._run(zeros=True)

        class _EmptySeg(_FakeSegmenter):
            def inference(self):
                return {'mask': np.zeros((H, W), bool),
                        'iou': np.zeros(0, np.float32), 'best': -1,
                        'n_pred': 0, 'reason': 'khong co hop'}

        out_empty_mask = P.run_phases(
            np.zeros((H, W, 3), np.uint8), 'a bag', detector=_FakeDetector(),
            segmenter=_EmptySeg(), depther=_FakeDepth(zeros=False),
            grasper=_FakeGrasper())
        self.assertEqual(empty_depth['cloud'].shape,
                         out_empty_mask['cloud'].shape)
        self.assertEqual(empty_depth['grasp']['graspgroup'].shape,
                         out_empty_mask['grasp']['graspgroup'].shape)
        # Ca hai deu la "khong co cloud" -> deu phai co ly do, khong phai None.
        self.assertTrue(empty_depth['grasp']['reason'])
        self.assertTrue(out_empty_mask['grasp']['reason'])

    def test_valid_depth_still_runs_graspness(self):
        """Chot chan: guard moi KHONG duoc chan nham duong binh thuong."""
        out, grasper = self._run(zeros=False)
        self.assertGreater(len(out['cloud']), 0)
        self.assertTrue(grasper.prepared)


class DrawGraspEarlyExitTests(unittest.TestCase):
    """P1-c: khong co tu the nao -> chi ve chu, khong duoc cham open3d/graspnetAPI."""

    def test_zero_grasps_does_not_import_open3d_or_graspnetapi(self):
        called = {'api': False}

        def _boom():
            called['api'] = True
            raise RuntimeError('khong thay graspnetAPI tai ...')

        im = np.zeros((H, W, 3), np.uint8)
        with patch.object(P, '_load_graspnetapi', _boom):
            out = P.draw_grasp(im, np.zeros((0, 17), np.float64), _K())
        self.assertFalse(called['api'],
                         '_load_graspnetapi() phai KHONG duoc goi khi khong co tu the')
        self.assertEqual(out.shape, im.shape)

    def test_all_too_wide_also_takes_early_exit(self):
        """Co tu the nhung TAT CA rong hon nguong -> cung phai thoat som."""
        called = {'api': False}

        def _boom():
            called['api'] = True
            raise RuntimeError('khong thay graspnetAPI tai ...')

        gg = np.zeros((2, 17), np.float64)
        gg[:, 0] = [0.4, 0.3]
        gg[:, 1] = [0.200, 0.150]           # 200 mm / 150 mm > 80 mm
        im = np.zeros((H, W, 3), np.uint8)
        with patch.object(P, '_load_graspnetapi', _boom):
            out = P.draw_grasp(im, gg, _K())
        self.assertFalse(called['api'])
        self.assertEqual(out.shape, im.shape)

    def test_reason_is_drawn_onto_image(self):
        """Anh 4/4 phai co CHU bao loi, khong duoc tra ve anh trong tron."""
        im = np.zeros((H, W, 3), np.uint8)
        with patch.object(P, '_load_graspnetapi',
                          side_effect=RuntimeError('nope')):
            out = P.draw_grasp(im, np.zeros((0, 17), np.float64), _K(),
                               reason='thieu MinkowskiEngine')
        self.assertTrue((out != 0).any(), 'phai ve chu len anh')

    def test_graspnetapi_failure_does_not_break_empty_path(self):
        """Du graspnetAPI VA open3d deu thieu, duong rong van phai chay."""
        import sys
        im = np.zeros((H, W, 3), np.uint8)
        with patch.dict(sys.modules, {'open3d': None}):
            out = P.draw_grasp(im, np.zeros((0, 17), np.float64), _K(),
                               reason='model nap hong')
        self.assertEqual(out.shape, im.shape)


class ReleaseFailureTests(unittest.TestCase):
    """release() nem khi inference DA THANH CONG -> phai surfaced, khong duoc che.

    Truoc day loi release duoc ghi vao errs, nhung errs chi duoc doc khi
    `"dep" not in out`. Inference thanh cong => out["dep"] ton tai => loi khong
    bao gio duoc doc. Ma release that bai nghia la VRAM chua duoc nha: invariant
    "DINO/MoGe da nha truoc SAM" khong con dung, phase sau co the OOM voi thong
    bao khong lien quan gi toi nguyen nhan that.
    """

    class _DepthReleaseBoom(_FakeDepth):
        def release(self):
            raise ModuleNotFoundError("No module named 'torch'")

    class _DetReleaseBoom(_FakeDetector):
        def release(self):
            raise RuntimeError("release DINO hong")

    class _SegReleaseBoom(_FakeSegmenter):
        def release(self):
            raise RuntimeError('release SAM hong')

    def _run(self, det, dep, seg=None):
        # Truyen DU segmenter/grasper gia: khong thi run_phases di tiep toi SAM
        # THAT, ma SamSegmenter can torch + model tren dia -> CI khong co.
        return P.run_phases(np.zeros((H, W, 3), np.uint8), 'a little bag',
                            fov_x=62.0, detector=det, depther=dep,
                            segmenter=seg or _FakeSegmenter(),
                            grasper=_FakeGrasper())

    def test_release_failure_is_surfaced(self):
        """inference OK + release nem -> run_phases() phai raise RO rang."""
        with self.assertRaises(RuntimeError) as cm:
            self._run(self._DetReleaseBoom(), self._DepthReleaseBoom())
        msg = str(cm.exception)
        self.assertIn('khong nha duoc', msg)
        # Phai neu CA HAI worker, khong chi worker dau tien gap.
        self.assertIn('dep', msg)
        self.assertIn('det', msg)

    def test_single_side_release_failure_also_surfaced(self):
        """Chi mot ben nem cung phai raise — khong duoc im lang."""
        with self.assertRaises(RuntimeError) as cm:
            self._run(self._DetReleaseBoom(), _FakeDepth())
        self.assertIn('khong nha duoc', str(cm.exception))

    def test_sam_release_failure_is_surfaced(self):
        """SAM release nem trong finally -> phai surfaced, khong duoc nuot.

        O day release() nam trong finally cua PHASE 2, nen neu no thoat ra thi no
        de luon ca out['seg'] vua tinh xong: mask tot bi vut di va run_phases chet.
        """
        with self.assertRaises(RuntimeError) as cm:
            self._run(_FakeDetector(), _FakeDepth(), seg=self._SegReleaseBoom())
        self.assertIn('khong nha duoc', str(cm.exception))
        self.assertIn('seg', str(cm.exception))

    def test_release_ok_still_runs_normally(self):
        """Chot chan: guard moi KHONG duoc chan nham duong binh thuong."""
        out = self._run(_FakeDetector(), _FakeDepth())
        self.assertIn('dep', out)
        self.assertIn('det', out)


class HwOpenNoteTests(unittest.TestCase):
    """P1-d: nhanh 'VUOT' cu la code chet (g da loc <= max_width)."""

    def test_wide_grasp_warns_against_hardware_opening(self):
        self.assertIn('VUOT khe mo that', P.hw_open_note(0.075))

    def test_narrow_grasp_has_no_note(self):
        self.assertEqual(P.hw_open_note(0.050), '')

    def test_boundary_is_hardware_not_display_threshold(self):
        """75 mm: LOT nguong ve (80 mm) nhung VUOT khe mo that (69.4 mm)."""
        self.assertEqual(P.hw_open_note(0.075, hw_open=P.GRIP_HW_OPEN_M),
                         P.hw_open_note(0.075))
        self.assertNotEqual(P.hw_open_note(0.075), '')

    def test_default_threshold_unchanged(self):
        """80 mm la CHU Y (comment dau file) — khong duoc doi thanh 69.4 mm."""
        self.assertEqual(P.GRIP_MAX_OPEN_M, 0.080)
        self.assertEqual(P.GRIP_HW_OPEN_M, 0.0694)


class _FakeGeom(object):
    """Mesh gia: du truong ma draw_grasp doc (vertices/triangles/vertex_colors)."""

    def __init__(self, *a, **k):
        self.vertices = np.zeros((8, 3))
        self.triangles = np.zeros((12, 3), np.int32)
        self.vertex_colors = np.zeros((8, 3))


class _FakeGraspGroup(object):
    def __init__(self, arr):
        self._a = np.asarray(arr, np.float64)

    def __len__(self):
        return len(self._a)

    def to_open3d_geometry_list(self):
        return [_FakeGeom() for _ in range(len(self._a))]


class _FakeGraspnetAPI(object):
    GraspGroup = _FakeGraspGroup


class DrawGraspNonEmptyTests(unittest.TestCase):
    """Duong VE THAT (co grasp hop le) — nhanh ma 4 test early-exit KHONG cham toi.

    Day la lo hong da de lot mot P0: draw_grasp() dinh nghia hw_open_note() nhung
    lai goi _hw_open_note(), nen MOI lan chay co grasp hop le deu NameError. CI
    khong bat duoc vi test render non-empty bi SKIP khi thieu open3d (xem
    test_pipeline_mock.py). Mock ca open3d lan _load_graspnetapi de di toi tan
    nhanh render that.
    """

    def _run(self, gg):
        import sys
        import types
        o3d = types.ModuleType('open3d')
        o3d.geometry = types.SimpleNamespace(TriangleMesh=_FakeGeom,
                                             LineSet=_FakeGeom)
        o3d.utility = types.SimpleNamespace(Vector3dVector=lambda v: np.asarray(v))
        o3d.io = types.SimpleNamespace()
        im = np.full((H, W, 3), 255, np.uint8)
        with patch.dict(sys.modules, {'open3d': o3d}), \
                patch.object(P, '_add_sys_path', lambda: None), \
                patch.object(P, '_load_graspnetapi', lambda: _FakeGraspnetAPI):
            return P.draw_grasp(im, gg, _K(), top=1), im

    def test_non_empty_grasp_renders_without_nameerror(self):
        """1 grasp hop le (55 mm) -> phai ve duoc, khong NameError."""
        gg = np.zeros((1, 17), np.float64)
        gg[0, 0] = 0.44
        gg[0, 1] = 0.055                    # 55 mm: lot ca 80 mm lan 69.4 mm
        gg[0, 13:16] = (0.0, 0.0, 0.5)
        out, im = self._run(gg)
        self.assertEqual(out.shape, im.shape)
        self.assertTrue((out != 255).any(), 'phai ve gripper len anh')

    def test_hw_open_note_is_reachable_on_widest_drawable_grasp(self):
        """75 mm: ve duoc (< 80 mm) nhung VUOT khe mo that -> phai di qua
        hw_open_note() va khong no. Chot chan rang ten ham dung."""
        gg = np.zeros((1, 17), np.float64)
        gg[0, 0] = 0.44
        gg[0, 1] = 0.075                    # 75 mm
        gg[0, 13:16] = (0.0, 0.0, 0.5)
        out, im = self._run(gg)             # NameError o day neu ten sai
        self.assertEqual(out.shape, im.shape)
        self.assertTrue((out != 255).any())


if __name__ == '__main__':
    unittest.main()
