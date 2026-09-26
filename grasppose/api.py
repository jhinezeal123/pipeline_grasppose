#!/usr/bin/env python3
"""Python API and local worker-client CLI for the grasp pipeline."""

import argparse
import os
import sys

import numpy as np

from grasppose.config import GRIP_MAX_OPEN_M
from grasppose.modules.depth.geometry import fov_x_from_fovy
from grasppose.facade import DEFAULT_SERVICE
from grasppose.runtime import log
from grasppose.infrastructure.worker.client import infer_image


DEFAULT_OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "output"),
)
DEFAULT_PIPELINE = DEFAULT_SERVICE.core


def load_models():
    DEFAULT_SERVICE.load()
    DEFAULT_SERVICE.core.warmup()
    return DEFAULT_SERVICE


def close_models():
    DEFAULT_SERVICE.close()


def pipeline(img, prompt_id, camera_K=None, fov_x=None,
             T_cam_volume=None, max_width=GRIP_MAX_OPEN_M, top=1):
    """Run one frame using a prepared prompt ID."""
    return DEFAULT_SERVICE.infer(
        img,
        prompt_id=prompt_id,
        camera_K=camera_K,
        fov_x=fov_x,
        T_cam_volume=T_cam_volume,
        max_width=max_width,
        top=top,
    )


def _camera_values(args):
    if args.camera_k is not None:
        return args.camera_k
    configured = os.environ.get("CAMERA_K", "").split()
    if configured:
        if len(configured) != 4:
            raise ValueError("CAMERA_K must contain FX FY CX CY")
        return list(map(float, configured))
    return None


def _camera_k_size(args):
    if args.camera_k_size is not None:
        return args.camera_k_size
    configured = os.environ.get("CAMERA_K_SIZE", "").replace(",", " ").split()
    if not configured:
        return None
    if len(configured) != 2:
        raise ValueError("CAMERA_K_SIZE must contain WIDTH HEIGHT")
    return [int(value) for value in configured]


def main():
    parser = argparse.ArgumentParser(
        description="Client for the resident TensorRT grasp worker")
    parser.add_argument("--img", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument(
        "--camera-k", nargs=4, type=float,
        metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument(
        "--camera-k-size", nargs=2, type=int,
        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--fov-x", type=float, default=None)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-width", type=float, default=GRIP_MAX_OPEN_M)
    parser.add_argument("--top", type=int, default=1)
    args = parser.parse_args()
    from PIL import Image

    image_path = os.path.abspath(args.img)
    if not os.path.isfile(image_path):
        parser.error("input image not found: %s" % image_path)
    width, height = Image.open(image_path).size
    camera_k = _camera_values(args)
    camera_k_size = _camera_k_size(args)
    if camera_k_size is not None and camera_k is None:
        parser.error("--camera-k-size requires --camera-k or CAMERA_K")
    if camera_k is None and args.fov_x is None:
        parser.error("provide --camera-k FX FY CX CY, CAMERA_K, or --fov-x")
    response = infer_image(
        image_path,
        args.prompt_id,
        camera_k=camera_k,
        camera_k_size=camera_k_size,
        fov_x=args.fov_x,
        output_dir=args.out,
        top=args.top,
        max_width=args.max_width,
    )
    print("\n".join(response["files"]))
    if response.get("depth_m") is not None:
        log("target depth: %.3f m" % response["depth_m"])
    log("worker inference + four PNG writes: %.1f ms" % response["server_ms"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
