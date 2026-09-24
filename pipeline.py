#!/usr/bin/env python3
"""CLI/public API for the Jetson grasp-pose service."""

import argparse
import os
import sys
import time

import numpy as np

from grasppose.config import (
    DEFAULT_PROMPT,
    GRIP_MAX_OPEN_M,
    YOLOE_CLASSES,
)
from grasppose.domain.geometry import fov_x_from_fovy
from grasppose.facade import DEFAULT_SERVICE
from grasppose.runtime import log


DEFAULT_OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
)


# Compatibility alias for callers that need the typed core pipeline.
DEFAULT_PIPELINE = DEFAULT_SERVICE.core


def load_models():
    """Load YOLOE, Lite-Mono and VGN TensorRT once."""
    return DEFAULT_SERVICE.load()


def close_models():
    """Release persistent model resources."""
    DEFAULT_SERVICE.close()


def pipeline(img, prompt=DEFAULT_PROMPT, camera_K=None, fov_x=None,
             T_cam_volume=None, max_width=GRIP_MAX_OPEN_M, top=1):
    """Run one image and return box/mask/depth/grasp renderings."""
    return DEFAULT_SERVICE.infer(
        img,
        prompt=prompt,
        camera_K=camera_K,
        fov_x=fov_x,
        T_cam_volume=T_cam_volume,
        max_width=max_width,
        top=top,
    )


def _camera_K_from_args(args):
    if args.camera_k is None:
        return None
    fx, fy, cx, cy = map(float, args.camera_k)
    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        np.float64,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "YOLOE-26s -> Lite-Mono -> TSDF -> VGN TensorRT"
        )
    )
    parser.add_argument("--img", required=True)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        choices=YOLOE_CLASSES,
        help="target class baked into the YOLOE TensorRT engine",
    )
    parser.add_argument(
        "--camera-k",
        nargs=4,
        type=float,
        metavar=("FX", "FY", "CX", "CY"),
        help="camera intrinsics in pixels; preferred for real cameras",
    )
    parser.add_argument("--fov-x", type=float, default=None)
    parser.add_argument("--fov-y", type=float, default=None)
    parser.add_argument(
        "--max-width", type=float, default=GRIP_MAX_OPEN_M)
    parser.add_argument("--top", type=int, default=1)
    args = parser.parse_args()

    from PIL import Image

    image = np.array(
        Image.open(args.img).convert("RGB"))
    height, width = image.shape[:2]

    fov_x = args.fov_x
    if fov_x is None and args.fov_y is not None:
        fov_x = fov_x_from_fovy(
            args.fov_y, width, height)

    camera_K = _camera_K_from_args(args)
    if (camera_K is None and fov_x is None and
            not os.environ.get("CAMERA_K")):
        parser.error(
            "real-camera pipeline needs "
            "--camera-k FX FY CX CY (or CAMERA_K env)"
        )

    load_models()
    started = time.time()
    result = pipeline(
        image,
        prompt=args.prompt,
        camera_K=camera_K,
        fov_x=fov_x,
        max_width=args.max_width,
        top=args.top,
    )
    log("TOTAL: %.2fs" % (time.time() - started))

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(
        os.path.basename(args.img))[0]
    saved = []
    for key in ("box", "mask", "depthmap", "grasp"):
        path = os.path.join(
            args.out, "%s_%s.png" % (stem, key))
        Image.fromarray(result[key]).save(path)
        saved.append(path)

    if result["depth_m"] is not None:
        log("target depth: %.3f m" % result["depth_m"])

    print("\n".join(saved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
