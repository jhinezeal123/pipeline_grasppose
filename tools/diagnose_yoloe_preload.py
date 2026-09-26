#!/usr/bin/env python3
"""Isolate which preload stage changes YOLOE behavior on Jetson Xavier."""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

from grasppose.modules.depth.lite_mono import LiteMonoDepth
from grasppose.modules.grasp.vgn_trt import VgnTensorRT
from grasppose.modules.vision.yoloe import Yoloe26sVision


def _summary(result):
    scores = np.asarray(result.detection.scores, np.float32)
    top = float(scores.max()) if scores.size else None
    return len(result.detection.boxes), top


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Probe YOLOE after selected resident-model preload stages. "
            "Run each stage as a separate process for clean CUDA state."
        )
    )
    parser.add_argument("--img", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("vision-only", "after-depth", "after-vgn", "full"),
    )
    args = parser.parse_args()

    image = np.array(Image.open(args.img).convert("RGB"))
    vision = Yoloe26sVision(conf=args.conf)
    depth = LiteMonoDepth()
    grasper = VgnTensorRT()
    loaded = []

    try:
        vision.load()
        loaded.append(("vision", vision))

        if args.stage in ("after-depth", "full"):
            depth.load()
            loaded.append(("depth", depth))

        if args.stage in ("after-vgn", "full"):
            grasper.load()
            loaded.append(("grasp", grasper))

        result = vision.predict(image, args.prompt)
        boxes, top = _summary(result)
        print(
            "stage=%s boxes=%d top_score=%s"
            % (
                args.stage,
                boxes,
                "none" if top is None else "%.6f" % top,
            )
        )
        print(
            "loaded=%s"
            % ",".join(name for name, _ in loaded)
        )
        return 0
    finally:
        for _, resource in reversed(loaded):
            try:
                resource.close()
            except Exception as exc:
                print(
                    "WARNING: close failed: %s: %s"
                    % (type(exc).__name__, exc),
                    file=sys.stderr,
                )


if __name__ == "__main__":
    sys.exit(main())
