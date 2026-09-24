#!/usr/bin/env python3
"""Bake fixed YOLOE prompts and export one FP16 TensorRT engine on Jetson."""

import argparse
from pathlib import Path
import shutil

from ultralytics import YOLOE


FIXED_CLASSES = (
    "blue cube",
    "yellow ball",
    "blue cyclinder",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workspace", type=float, default=2.0)
    args = parser.parse_args()

    model_path = Path(args.model)
    out_path = Path(args.out)
    if not model_path.is_file():
        raise SystemExit("YOLOE source checkpoint not found: %s" % model_path)

    model = YOLOE(str(model_path))
    model.set_classes(list(FIXED_CLASSES))
    exported = Path(model.export(
        format="engine",
        imgsz=args.imgsz,
        batch=1,
        dynamic=False,
        simplify=False,
        device=0,
        quantize=16,
        workspace=args.workspace,
    ))

    if not exported.is_file():
        raise SystemExit("Ultralytics did not create TensorRT engine: %s" % exported)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != out_path.resolve():
        shutil.copy2(str(exported), str(out_path))

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise SystemExit("TensorRT engine output is missing/empty: %s" % out_path)

    print("YOLOE TensorRT classes:", FIXED_CLASSES)
    print("YOLOE TensorRT engine:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
