#!/usr/bin/env python3
"""Kiem tra moi truong co du dieu kien chay pipeline khong.

Chay TRUOC khi chay run.sh de biet ngay thieu gi, thay vi chet mo ho o buoc
GraspNess sau nay (no chi bao "khong co tu the nao", rat de chan doan nham).

    python env/check_env.py

Tra ve 0 neu du dieu kien, 1 neu thieu thu gi do. Khong import model, khong
can GPU that de chay — chi kiem tra sach se moi thu co mat.
"""
import importlib
import shutil
import subprocess
import sys

# (ten module, phien ban da do duoc, bat buoc?)
# Phien ban lay tu requirements.lock.txt — do tren mot lan chay That.
WANT = [
    ("numpy",        "2.0.2",   True),
    ("scipy",        "1.16.3",  True),
    ("torch",        "2.10.0",  True),
    ("transformers", "5.0.0",   True),
    ("PIL",          "11.3.0",  True),
    ("cv2",          "4.13.0",  True),
    ("open3d",       "0.20.0",  True),
    ("huggingface_hub", "1.32.0", True),
    ("transforms3d", "0.4.2",   True),
    ("timm",         "1.0.26",  False),
    ("einops",       "0.8.2",   False),
    ("gradio",       "6.28.0",  False),   # chi can khi chay web UI
]

OK = "\033[32mOK\033[0m"
BAD = "\033[31mTHIEU\033[0m"
WARN = "\033[33mCU\033[0m"

problems = []


def head(t):
    print("\n" + t)
    print("-" * 62)


def main():
    head("1. Python")
    print("  phien ban: %d.%d.%d" % sys.version_info[:3])
    if sys.version_info[:2] != (3, 12):
        problems.append(
            "Python %d.%d — da kiem chung tren 3.12; ban khac co the van chay "
            "nhung _ext phai bien dich lai" % sys.version_info[:2])

    head("2. Thu vien Python")
    for name, want, must in WANT:
        try:
            m = importlib.import_module(name)
            got = getattr(m, "__version__", "?")
            same = got == want or got.startswith(want) or want.startswith(got)
            mark = OK if same else WARN
            note = "" if same else "  (da kiem chung voi %s)" % want
            print("  [%s] %-16s %s%s" % (mark, name, got, note))
            if not same:
                problems.append("%s la %s, da kiem chung voi %s" % (name, got, want))
        except ImportError:
            print("  [%s] %-16s" % (BAD, name))
            if must:
                problems.append("thieu %s (bat buoc)" % name)

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
            print("  GPU              : KHONG THAY")
            problems.append("torch khong thay GPU (can T4 tro len)")
    except ImportError:
        print("  (bo qua: chua co torch)")

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
        print("       -> tren Kaggle: export PATH=/usr/local/cuda/bin:$PATH")
        problems.append("khong co nvcc -> khong bien dich duoc _ext")
    print("  gcc: %s" % (shutil.which("gcc") or "khong co"))

    head("5. Extension da build chua")
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hits = []
    for base in ("model/graspness_unofficial_build", "model/graspness_unofficial"):
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
