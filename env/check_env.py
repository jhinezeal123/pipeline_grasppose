#!/usr/bin/env python3
"""Quick Jetson runtime check for the edge grasp pipeline."""
import importlib
import os
import shutil
import sys

REQUIRED=("numpy","scipy","torch","torchvision","cv2","PIL")
OPTIONAL=("ultralytics","timm","tensorrt")


def main():
    problems=[]
    print("Python %d.%d.%d" % sys.version_info[:3])
    if sys.version_info[:2] < (3,8): problems.append("Python >=3.8 is required")
    for name in REQUIRED+OPTIONAL:
        try:
            m=importlib.import_module(name); print("[OK] %-12s %s" % (name,getattr(m,"__version__","")))
        except Exception as exc:
            print("[--] %-12s %s" % (name,exc))
            if name in REQUIRED: problems.append("missing %s" % name)
    try:
        import torch
        print("CUDA available:",torch.cuda.is_available())
        if not torch.cuda.is_available(): problems.append("PyTorch cannot see CUDA")
        else: print("GPU:",torch.cuda.get_device_name(0),"| torch CUDA:",torch.version.cuda)
    except Exception as exc: problems.append("torch/CUDA: %s" % exc)
    print("trtexec:",shutil.which("trtexec") or shutil.which("/usr/src/tensorrt/bin/trtexec") or "not on PATH")
    for path in ("model/yoloe-26s-seg.pt","model/lite-mono/encoder.pth","model/lite-mono/depth.pth","model/vgn.engine"):
        print("[%s] %s" % ("OK" if os.path.isfile(path) else "--",path))
    if problems:
        print("Problems:"); [print(" -",p) for p in problems]; return 1
    print("Environment looks ready."); return 0

if __name__=="__main__": sys.exit(main())
