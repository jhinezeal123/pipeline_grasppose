#!/usr/bin/env python3
"""Kiem tra moi truong co du dieu kien chay pipeline khong.

Chay TRUOC khi chay run.sh de biet ngay thieu gi, thay vi chet mo ho o buoc
GraspNess sau nay (no chi bao "khong co tu the nao", rat de chan doan nham).

    python env/check_env.py

Tra ve 0 neu du dieu kien, 1 neu thieu thu gi do. Khong import model, khong
can GPU that de chay — chi kiem tra sach se moi thu co mat.
"""
import importlib
import os
import shutil
import subprocess
import sys

# (ten module, phien ban da do duoc, bat buoc?)
# Phien ban lay tu requirements.lock.txt — do tren mot lan chay That.
WANT = [
    ("numpy",        "1.23.5",   True),
    ("scipy",        "1.10.1",   True),
    ("torch",        "2.1.0",    True),
    ("transformers", "4.44.2",   True),
    ("PIL",          "10.4.0",   True),
    ("cv2",          "4.5.4",    True),
    ("huggingface_hub", "0.24.7", True),
    ("transforms3d", "0.4.2",    True),
    ("timm",         "1.0.9",    False),
    ("einops",       "0.8.1",    False),
    ("gradio",       "4.44.1",   False),   # chi can khi chay web UI
]

OK = "\033[32mOK\033[0m"
BAD = "\033[31mTHIEU\033[0m"
WARN = "\033[33mCU\033[0m"

problems = []


def head(t):
    print("\n" + t)
    print("-" * 62)


def main():
    problems.clear()
    head("1. Python")
    print("  phien ban: %d.%d.%d" % sys.version_info[:3])
    if sys.version_info[:2] < (3, 8):
        problems.append(
            "Python %d.%d — can >=3.8" % sys.version_info[:2])

    head("2. Thu vien Python")
    for name, want, must in WANT:
        try:
            m = importlib.import_module(name)
            got = getattr(m, "__version__", "?")
            same = got == want or got.startswith(want) or want.startswith(got)
            mark = OK if same else WARN
            note = "" if same else "  (da kiem chung voi %s)" % want
            print("  [%s] %-16s %s%s" % (mark, name, got, note))
            # Snapshot differences are informational, not compatibility failures.
        except Exception as exc:
            print("  [%s] %-16s: %s" % (BAD, name, exc))
            if must:
                problems.append("thieu %s (bat buoc)" % name)

    for name in ("torchvision", "MinkowskiEngine", "moge.model.v2"):
        try:
            importlib.import_module(name)
            print("  [%s] %s" % (OK, name))
        except Exception as exc:
            problems.append("%s: %s" % (name, exc))

    head("3. GPU + CUDA")
    try:
        import torch
        print("  torch build CUDA : %s" % torch.version.cuda)
        if torch.cuda.is_available():
            print("  GPU              : %s" % torch.cuda.get_device_name(0))
            cap = torch.cuda.get_device_capability(0)
            print("  compute cap.     : sm_%d%d" % cap)
            archs = torch.cuda.get_arch_list()
            print("  arch torch co    : %s" % ", ".join(archs))
            if "sm_%d%d" % cap not in archs:
                problems.append(
                    "torch KHONG build cho sm_%d%d -> _ext se loi khi chay" % cap)
        else:
            # Rat hay gap tren Kaggle: co GPU that nhung torch khong thay, vi
            # thieu LD_LIBRARY_PATH tro toi driver. run.sh co export bien nay,
            # con chay tay thi khong — nen phai phan biet ro hai truong hop,
            # khong thi de chan doan nham la "may khong co GPU".
            nv = "/usr/local/nvidia/lib64"
            has_nv = os.path.isdir(nv)
            ld = os.environ.get("LD_LIBRARY_PATH", "")
            if has_nv and nv not in ld:
                print("  GPU              : KHONG THAY — thu kiem tra driver/path %s" % nv)
                print("       -> thieu LD_LIBRARY_PATH. Chay lai bang:")
                print("          LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH \\" % nv)
                print("              python env/check_env.py")
                print("       (run.sh tu export bien nay, nen chay qua run.sh thi khong gap)")
                problems.append(
                    "torch khong thay GPU; kiem tra driver va LD_LIBRARY_PATH=%s" % nv)
            else:
                print("  GPU              : KHONG THAY")
                problems.append("torch khong thay CUDA GPU")
    except Exception as exc:
        problems.append("torch/CUDA: %s" % exc)

    head("4. Cong cu bien dich _ext (pointnet2)")
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if shutil.which("nvcc") or shutil.which(nvcc):
        try:
            out = subprocess.run([nvcc, "--version"], capture_output=True,
                                 text=True, timeout=20).stdout
            line = [l for l in out.splitlines() if "release" in l]
            print("  [%s] nvcc: %s" % (OK, line[0].strip() if line else "co"))
        except Exception as e:
            print("  [%s] nvcc: %s" % (WARN, e))
    else:
        print("  [%s] nvcc khong co tren PATH" % BAD)
        print("       -> run.sh se KHONG build duoc pointnet2._ext")
        print("       -> Jetson: export PATH=/usr/local/cuda/bin:$PATH")
        problems.append("khong co nvcc -> khong bien dich duoc _ext")
    print("  gcc: %s" % (shutil.which("gcc") or "khong co"))

    head("5. Extension da build chua")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hits = []
    for base in (os.environ.get("GRASPNESS_HOME", "model/graspness_unofficial"),
                 ".venv/native/graspness_unofficial", "model/graspness_unofficial_build"):
        d = os.path.join(root, base, "pointnet2")
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith("_ext") and f.endswith(".so"):
                    hits.append(os.path.join(base, "pointnet2", f))
    if hits:
        for h in hits:
            print("  [%s] %s" % (OK, h))
    else:
        print("  [%s] chua build" % WARN)
        print("       -> run.sh se tu build (mat 2-4 phut, can nvcc)")

    head("6. Ket luan")
    if not problems:
        print("  [%s] Moi truong DU dieu kien chay pipeline." % OK)
        return 0
    print("  [%s] %d van de:" % (BAD, len(problems)))
    for p in problems:
        print("     - %s" % p)
    return 1


if __name__ == "__main__":
    sys.exit(main())
