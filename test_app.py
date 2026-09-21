#!/usr/bin/env python3
"""test_app.py - kiem tra app.py KHONG can gradio va KHONG can model that.

app.py co y KHONG import gradio o cap module, nen o day chi can thay
pipeline.pipeline() bang ham gia la test duoc toan bo phan loi: tra ve dung thu
tu, khong bao gio raise, va noi ro khi thieu so do sau.

Chay:  python test_app.py
"""
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import app as A                                                 # noqa: E402
import pipeline as P                                            # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           ("  — " + detail) if detail else ""))
    if not cond:
        FAIL.append(name)


H, W = 60, 80
IMG = np.zeros((H, W, 3), np.uint8)


def fake_result(depth_m=0.4712):
    """Ket qua gia DUNG dinh dang pipeline() that tra ve."""
    return {
        "box": np.full((H, W, 3), 10, np.uint8),
        "mask": np.full((H, W, 3), 20, np.uint8),
        "depthmap": np.full((H, W, 3), 30, np.uint8),
        "grasp": np.full((H, W, 3), 40, np.uint8),
        "depth_m": depth_m,
    }


def main():
    print("=" * 66)
    print(" TEST app.py (khong can gradio, khong can model)")
    print("=" * 66)

    # ---- 0. app.py khong duoc keo gradio vao ----
    print("\n0. app.py nap duoc ma khong can gradio")
    check("import app thanh cong", True)
    check("gradio KHONG bi nap o cap module",
          "gradio" not in sys.modules,
          "co trong sys.modules -> se hong tren may thieu gradio")
    check("run_one ton tai", callable(getattr(A, "run_one", None)))
    check("build_ui ton tai", callable(getattr(A, "build_ui", None)))

    orig = P.pipeline
    try:
        # ---- 1. Duong di tot ----
        print("\n1. Duong di tot")
        seen = {}

        def ok(image, prompt=None, top=None, **kw):
            seen["prompt"] = prompt
            seen["top"] = top
            seen["shape"] = np.asarray(image).shape
            return fake_result(0.4712)

        P.pipeline = ok
        out = A.run_one(IMG, "a little bag")
        check("tra ve dung 6 phan tu", len(out) == 6, "%d" % len(out))
        b, m, d, g, dm, st = out
        for nm, arr, val in (("box", b, 10), ("mask", m, 20),
                             ("depthmap", d, 30), ("grasp", g, 40)):
            check("anh '%s' dung thu tu/du lieu" % nm,
                  isinstance(arr, np.ndarray) and arr.shape == (H, W, 3)
                  and int(arr[0, 0, 0]) == val)
        check("depth_m la float 0.4712", isinstance(dm, float) and abs(dm - 0.4712) < 1e-9,
              repr(dm))
        check("prompt duoc chuyen nguyen ven", seen.get("prompt") == "a little bag",
              repr(seen.get("prompt")))
        check("top = %d duoc truyen" % A.TOP_GRASPS, seen.get("top") == A.TOP_GRASPS,
              repr(seen.get("top")))
        check("trang thai co con so met", "0.471" in st, st[:70])

        # ---- 2. Khong co anh ----
        print("\n2. Khong tai anh len")
        out = A.run_one(None, "a little bag")
        check("khong raise, tra ve 6 phan tu", len(out) == 6)
        check("4 anh deu None", all(x is None for x in out[:4]))
        check("depth_m = None", out[4] is None)
        check("bao ro chua co anh", "Chua co anh" in out[5], out[5][:60])

        # ---- 3. Pipeline raise ----
        print("\n3. Pipeline nem loi (thieu trong so / het VRAM)")
        def boom(image, **kw):
            raise RuntimeError("grasp_models khong tra mask/depth: no box for prompt")

        P.pipeline = boom
        try:
            out = A.run_one(IMG, "the blue cylinder")
            check("KHONG raise ra ngoai", True)
            check("4 anh None + depth_m None",
                  all(x is None for x in out[:5]))
            check("trang thai co chu LOI va ten loi",
                  "LOI" in out[5] and "RuntimeError" in out[5], out[5][:80])
            check("trang thai giu nguyen van ban loi goc",
                  "no box for prompt" in out[5])
        except Exception:
            traceback.print_exc()
            check("KHONG raise ra ngoai", False, "xem traceback")

        # ---- 4. Chay duoc nhung khong co so ----
        print("\n4. Chay xong nhung mask rong (depth_m = None)")
        P.pipeline = lambda image, **kw: fake_result(None)
        out = A.run_one(IMG, "xyzzynotathing")
        check("4 anh VAN duoc tra (khong mat ket qua)",
              all(isinstance(x, np.ndarray) for x in out[:4]))
        check("depth_m = None", out[4] is None)
        check("noi ro vi sao khong co so",
              "KHONG co so do sau" in out[5] and "xyzzynotathing" in out[5],
              out[5][:80])

        # ---- 5. Prompt rong -> dung mac dinh ----
        print("\n5. Prompt rong")
        seen.clear()
        P.pipeline = ok
        for bad in ("", "   ", None):
            A.run_one(IMG, bad)
            check("prompt %r -> dung DEFAULT_PROMPT" % (bad,),
                  seen.get("prompt") == P.DEFAULT_PROMPT, repr(seen.get("prompt")))

        # ---- 6. Khong bao gio raise voi dau vao ki quac ----
        print("\n6. Dau vao ki quac - run_one khong duoc raise")
        P.pipeline = orig
        for nm, im, pr in (("anh 1x1", np.zeros((1, 1, 3), np.uint8), "x"),
                           ("anh int rong", np.zeros((0, 0, 3), np.uint8), "x"),
                           ("prompt so", IMG, 12345),
                           ("prompt list", IMG, ["a"])):
            try:
                A.run_one(im, pr)
                check("%s -> khong raise" % nm, True)
            except Exception as e:
                check("%s -> khong raise" % nm, False,
                      "%s: %s" % (type(e).__name__, e))
    finally:
        P.pipeline = orig

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
