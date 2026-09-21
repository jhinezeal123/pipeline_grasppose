#!/usr/bin/env python3
"""test_pipeline_mock.py — kiem tra pipeline.py KHONG can model that.

Thay 4 model bang lop gia (tra ket qua dung dinh dang), roi chay pipeline() va
kiem tra 4 anh dau ra. Muc dich: bat loi lap trinh (thu tu phase, chia se du
lieu, ve anh) ma khong ton GPU / khong ton 10 phut nap trong so.

Chay:  python test_pipeline_mock.py
"""
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pipeline as P                                          # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           ("  — " + detail) if detail else ""))
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------------------
# Model gia
# ---------------------------------------------------------------------------
CALLS = []


class FakeDetector(P.Object_Detection):
    def __init__(self):
        self.model_path = "fake"
        self.device = "cpu"

    def prepare(self, image, prompt):
        CALLS.append("det.prepare")
        self._p = prompt
        return self

    def inference(self):
        CALLS.append("det.inference")
        return {"boxes": np.array([[100.0, 120.0, 300.0, 340.0]], np.float32),
                "scores": np.array([0.42], np.float32), "labels": ["bag"],
                "reason": None}

    def release(self):
        CALLS.append("det.release")


class FakeSegmenter(P.Segmentation):
    def __init__(self):
        self.model_path = "fake"
        self.device = "cpu"

    def prepare(self, image, boxes):
        CALLS.append("seg.prepare")
        self._n = len(boxes)
        return self

    def inference(self):
        CALLS.append("seg.inference")
        h, w = 240, 320
        m = np.zeros((h, w), bool)
        m[120:200, 100:220] = True
        return {"mask": m, "iou": np.array([0.91, 0.55], np.float32),
                "best": 0, "n_pred": 2, "reason": None}

    def release(self):
        CALLS.append("seg.release")


class FakeDepth(P.Depth_Estimate):
    def __init__(self):
        self.model_path = "fake"
        self.device = "cpu"
        self._fov = None

    def prepare(self, image, fov_x=None):
        CALLS.append("dep.prepare")
        self._fov = fov_x
        self._h, self._w = image.shape[:2]
        return self

    def inference(self):
        CALLS.append("dep.inference")
        h, w = self._h, self._w
        # Mat phang nghieng don gian: z tang dan theo y
        z = 0.5 + 0.5 * (np.arange(h, dtype=np.float32) / h)[:, None] * np.ones((1, w), np.float32)
        ys, xs = np.mgrid[0:h, 0:w]
        K = P.K_from_fovy(62.0, w, h)
        px = (xs - K[0, 2]) * z / K[0, 0]
        py = (ys - K[1, 2]) * z / K[1, 1]
        return {"depth": z.astype(np.float32),
                "points": np.stack([px, py, z], -1).astype(np.float32),
                "intrinsics": K.astype(np.float32),
                "fov_x_deg": (P.fov_x_from_fovy(62.0, w, h) if self._fov is None
                              else float(self._fov)),
                "reason": None}

    def release(self):
        CALLS.append("dep.release")


class FakeGrasper(P.GraspNess):
    def __init__(self):
        self.model_path = "fake"
        self.device = "cpu"

    def prepare(self, points, num_point=15000):
        CALLS.append("grasp.prepare")
        self._n_in = len(points)
        return self

    def inference(self):
        CALLS.append("grasp.inference")
        K = P.K_from_fovy(62.0, 320, 240)
        g = np.zeros((3, 17), np.float64)
        for i, (sc, wd) in enumerate([(0.45, 0.060), (0.30, 0.090), (0.20, 0.050)]):
            g[i, 0] = sc
            g[i, 1] = wd
            g[i, 2] = 0.02
            g[i, 3] = 0.03
            g[i, 4:13] = np.eye(3).reshape(9)
            g[i, 13:16] = [0.02 * i, 0.0, 0.6]
        return {"graspgroup": g, "reason": None}

    def release(self):
        CALLS.append("grasp.release")


# ---------------------------------------------------------------------------
def main():
    print("=" * 66)
    print(" TEST pipeline.py bang model GIA (khong can GPU / trong so)")
    print("=" * 66)

    # ---- 1. Tien ich hinh hoc ----
    print("\n1. Tien ich hinh hoc")
    K = P.K_from_fovy(62.0, 640, 480)
    check("K_from_fovy: fx = (H/2)/tan(fovy/2)",
          abs(K[0, 0] - 399.4271) < 1e-3, "fx=%.4f (mong doi 399.4271)" % K[0, 0])
    check("K_from_fovy: pixel vuong", abs(K[0, 0] - K[1, 1]) < 1e-9)
    check("K_from_fovy: tam anh", abs(K[0, 2] - 320) < 1e-9 and abs(K[1, 2] - 240) < 1e-9)
    fx = P.fov_x_from_fovy(62.0, 640, 480)
    check("fov_x_from_fovy(62,640,480) = 77.400 do",
          abs(fx - 77.3998) < 1e-3, "= %.4f" % fx)

    # ---- 2. depth_to_cloud: mask vs khong mask ----
    print("\n2. depth_to_cloud")
    d = np.full((4, 4), 1.0, np.float32)
    Kk = np.array([[2.0, 0, 2.0], [0, 2.0, 2.0], [0, 0, 1.0]])
    full = P.depth_to_cloud(d, Kk)
    m = np.zeros((4, 4), bool)
    m[0, 0] = True
    masked = P.depth_to_cloud(d, Kk, mask=m)
    check("khong mask -> 16 diem", len(full) == 16, "%d" % len(full))
    check("co mask   -> 1 diem", len(masked) == 1, "%d" % len(masked))
    check("MASK THUAN: masked la tap con cua full",
          len(masked) == 1 and np.allclose(masked[0], full[0]))
    d0 = np.zeros((4, 4), np.float32)
    check("depth=0 bi loai (z_min)", len(P.depth_to_cloud(d0, Kk)) == 0)

    # ---- 3. nms_grasps ----
    print("\n3. nms_grasps")
    g = np.zeros((3, 17), np.float64)
    g[0, 0] = 0.9; g[0, 4:13] = np.eye(3).reshape(9); g[0, 13:16] = [0, 0, 0.5]
    g[1, 0] = 0.8; g[1, 4:13] = np.eye(3).reshape(9); g[1, 13:16] = [0.001, 0, 0.5]
    g[2, 0] = 0.7; g[2, 4:13] = np.eye(3).reshape(9); g[2, 13:16] = [0.5, 0, 0.5]
    kept = P.nms_grasps(g)
    check("grasp trung tam <3cm bi loai", len(kept) == 2, "giu %d/3" % len(kept))
    check("grasp diem cao nhat duoc giu", abs(kept[0, 0] - 0.9) < 1e-9)

    # ---- 4. _phrase_match ----
    print("\n4. _phrase_match")
    check("khop truc tiep", P._phrase_match(["bag"], "a little bag") == 0)
    check("manh WordPiece '##rmos' KHONG khop 'thermos' nhung dung 1 hop -> van lay",
          P._phrase_match(["##rmos"], "thermos") == 0)
    check("nhieu hop, khong khop -> None",
          P._phrase_match(["cat", "dog"], "thermos") is None)
    check("khop qua token 'black'",
          P._phrase_match(["blackrmos"], "a black thermos") == 0)

    # ---- 5. Chay pipeline() voi model gia ----
    print("\n5. pipeline() voi model gia")
    img = np.zeros((240, 320, 3), np.uint8)
    img[100:210, 90:230] = (180, 160, 140)
    dep, seg, gr, det = FakeDepth(), FakeSegmenter(), FakeGrasper(), FakeDetector()

    orig = P.run_phases
    def patched(image, prompt, fov_x=None, **kw):
        return orig(image, prompt, fov_x=fov_x, detector=det, segmenter=seg,
                    depther=dep, grasper=gr)
    P.run_phases = patched
    try:
        res = P.pipeline(img, prompt="a little bag")
    except Exception:
        traceback.print_exc()
        check("pipeline() khong raise", False, "xem traceback")
        P.run_phases = orig
        return report()
    finally:
        P.run_phases = orig

    check("tra ve 4 anh + 1 con so",
          set(res.keys()) == {"box", "mask", "depthmap", "grasp", "depth_m"},
          str(sorted(res.keys())))
    for k in ("box", "mask", "depthmap", "grasp"):
        a = res.get(k)
        check("anh '%s' la numpy uint8 3 kenh" % k,
              isinstance(a, np.ndarray) and a.dtype == np.uint8 and a.ndim == 3
              and a.shape[2] == 3,
              "" if a is None else "%s %s" % (a.shape, a.dtype))
    check("anh 'grasp' cung kich thuoc anh goc",
          res["grasp"].shape == img.shape, str(res["grasp"].shape))
    check("anh 'depthmap' la anh THAT (khong phai anh goc)",
          not np.array_equal(res["depthmap"], img))
    check("anh 'mask' co pixel xanh la (mask duoc ve)",
          bool(((res["mask"][:, :, 1].astype(int) - res["mask"][:, :, 0].astype(int)) > 40).any()))

    # ---- 5b. depth_m: TRUNG VI do sau TRONG MASK, khong phai ca anh ----
    # Dung lai dung cong thuc cua FakeDepth de tinh ky vong DOC LAP voi pipeline.
    print("\n5b. depth_m (do sau cua VAT)")
    z1d = 0.5 + 0.5 * (np.arange(240, dtype=np.float32) / 240)
    zz = np.repeat(z1d[:, None], 320, axis=1)
    mref = np.zeros((240, 320), bool)
    mref[120:200, 100:220] = True
    expect_in_mask = float(np.median(zz[mref]))
    expect_whole = float(np.median(zz))
    dm = res["depth_m"]
    check("depth_m la so float", isinstance(dm, float), repr(dm))
    check("depth_m = trung vi depth TRONG MASK (%.4f)" % expect_in_mask,
          dm is not None and abs(dm - expect_in_mask) < 1e-3,
          "nhan duoc %s" % dm)
    check("depth_m KHONG phai trung vi ca anh (%.4f)" % expect_whole,
          dm is not None and abs(dm - expect_whole) > 1e-2,
          "lech %.4f" % (abs(dm - expect_whole) if dm is not None else float("nan")))

    # ---- 6. Thu tu phase ----
    print("\n6. Thu tu 3 phase (VRAM khong bao gio giu 2 model nang)")
    idx = {n: i for i, n in enumerate(CALLS)}
    # Ghi lai thu tu thuc: prepare/inference/release cua tung model
    seq = [c for c in CALLS]
    print("     thu tu thuc te: %s" % " -> ".join(seq))
    check("MoGe release TRUOC khi SAM prepare",
          seq.index("dep.release") < seq.index("seg.prepare"))
    check("DINO release TRUOC khi SAM prepare",
          seq.index("det.release") < seq.index("seg.prepare"))
    check("SAM release TRUOC khi GraspNess prepare",
          seq.index("seg.release") < seq.index("grasp.prepare"))
    check("ca 4 model deu duoc release",
          all(s in CALLS for s in ("det.release", "seg.release",
                                   "dep.release", "grasp.release")))
    # MoGe va DINO phai chay truoc khi cai nao release (tuc la song song)
    check("MoGe va DINO cung prepare truoc khi cai nao release",
          seq.index("dep.prepare") < seq.index("det.release")
          and seq.index("det.prepare") < seq.index("dep.release"),
          "chung minh phase 1 chay song song")

    # ---- 7. Mask rong -> khong crash ----
    print("\n7. Mask rong")
    class EmptySeg(FakeSegmenter):
        def inference(self):
            return {"mask": np.zeros((240, 320), bool), "iou": np.zeros(0, np.float32),
                    "best": -1, "n_pred": 0, "reason": "khong co hop"}
    seg2 = EmptySeg()
    def patched2(image, prompt, fov_x=None, **kw):
        return orig(image, prompt, fov_x=fov_x, detector=FakeDetector(),
                    segmenter=seg2, depther=FakeDepth(), grasper=FakeGrasper())
    P.run_phases = patched2
    try:
        res2 = P.pipeline(img, prompt="khong co gi")
        check("mask rong -> van tra 4 anh + 1 so, khong raise", len(res2) == 5)
        check("mask rong -> depth_m = None (khong bia ra so)",
              res2["depth_m"] is None, repr(res2["depth_m"]))
    except Exception:
        traceback.print_exc()
        check("mask rong -> khong raise", False)
    finally:
        P.run_phases = orig

    print("\n8. grasp_empty_msg: ba nguyen nhan phai ra ba cau KHAC NHAU")
    # Truoc day ca ba truong hop deu hien dung mot cau "khong co tu the nao",
    # nen nhin anh khong the biet GraspNess nap hong hay chi la loc qua chat.
    m_fail = P.grasp_empty_msg(0, "thieu MinkowskiEngine", 0.080)
    m_none = P.grasp_empty_msg(0, None, 0.080)
    m_narrow = P.grasp_empty_msg(7, None, 0.080)
    check("nap hong -> noi ro 'KHONG CHAY DUOC' + ly do",
          "KHONG CHAY DUOC" in m_fail and "MinkowskiEngine" in m_fail, m_fail)
    check("khong co tu the nao -> cau rieng", "khong co tu the nao" in m_none
          and "KHONG CHAY DUOC" not in m_none, m_none)
    check("co tu the nhung rong qua -> noi RO SO LUONG va nguong",
          "7" in m_narrow and "80" in m_narrow, m_narrow)
    check("ba cau that su khac nhau", len({m_fail, m_none, m_narrow}) == 3)

    print("\n9. draw_grasp: phai ve HINH CHU U, khong phai may cai que")
    # Loi that da gap: to_open3d_geometry_list() tra ve LineSet (4 hop RONG),
    # nhung code cu doc geom.triangles — LineSet khong co truong do nen numpy
    # tra mang rong, vong lap ve canh khong chay dong nao, va anh ket qua chi
    # con may vach roi rac trong nhu "cai que".
    #
    # Dung hinh gia mo phong DUNG kieu LineSet cua upstream: 4 hop, mỗi hop 8
    # dinh / 12 canh, kich thuoc that (do tren mesh that: 84 x 77 x 4 mm).
    class _LS(object):
        def __init__(self, V, E, C):
            self.vertices = V
            self.lines = E
            self.vertex_colors = C
            # CO Y khong co .triangles — giong LineSet that.

    def _box(x0, x1, y0, y1, z0, z1):
        v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1)
                      for z in (z0, z1)], np.float64)
        e = []
        for i in range(8):
            for j in range(i + 1, 8):
                if bin(i ^ j).count("1") == 1:
                    e.append([i, j])
        return v, np.array(e)

    parts = [(-0.024, 0.020, -0.0385, -0.0345, -0.002, 0.002),   # ngon trai
             (-0.024, 0.020, 0.0345, 0.0385, -0.002, 0.002),     # ngon phai
             (-0.024, -0.020, -0.0345, 0.0345, -0.002, 0.002),   # thanh noi truoc
             (-0.064, -0.024, -0.002, 0.002, -0.002, 0.002)]     # duoi
    Vs, Es = [], []
    for (x0, x1, y0, y1, z0, z1) in parts:
        v, e = _box(x0, x1, y0, y1, z0, z1)
        Es.append(e + len(Vs))
        Vs.append(v)
    VV = np.concatenate(Vs)
    EE = np.concatenate(Es)
    CC = np.tile(np.array([[0.5, 0.0, 0.5]]), (len(VV), 1))
    ls = _LS(VV, EE, CC)

    class _GG(object):
        def __init__(self, a):
            self.a = a

        def to_open3d_geometry_list(self):
            return [ls]

    class _API(object):
        GraspGroup = _GG

    gg = np.zeros((1, 17), np.float64)
    gg[0, 0] = 0.44
    gg[0, 1] = 0.055                     # khe kep 55 mm -> nam trong nguong
    gg[0, 15] = 0.5
    Kd = P.K_from_fovy(60.0, 640, 480)
    canvas = np.full((480, 640, 3), 255, np.uint8)

    orig_api = P._load_graspnetapi
    P._load_graspnetapi = lambda: _API()
    try:
        out = P.draw_grasp(canvas, gg, Kd, top=1)
        n_pix = int((out != 255).any(-1).sum())
        # "May cai que" cu cho khoang ~600 diem anh; chu U day du cho > 1500.
        check("co ve ra nhieu hon may vach roi rac (>1500 px)", n_pix > 1500,
              "%d px" % n_pix)
        # Chu U phai trai RONG theo truc y (2 ngon cach nhau) chu khong phai
        # vai duong song song manh.
        ys, xs = np.where((out != 255).any(-1))
        check("hinh trai rong theo CA HAI truc (khong phai 1 vach)",
              (xs.max() - xs.min()) > 40 and (ys.max() - ys.min()) > 40,
              "rong %d x %d px" % (xs.max() - xs.min(), ys.max() - ys.min()))
    except Exception:
        traceback.print_exc()
        check("draw_grasp voi LineSet khong raise", False)
    finally:
        P._load_graspnetapi = orig_api

    return report()


def report():
    print("\n" + "=" * 66)
    if FAIL:
        print(" %d MUC THAT BAI:" % len(FAIL))
        for f in FAIL:
            print("   - %s" % f)
    else:
        print(" TAT CA MUC DEU PASS")
    print("=" * 66)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
