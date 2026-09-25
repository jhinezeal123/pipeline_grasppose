#!/usr/bin/env python3
"""CLI client for the resident worker."""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from grasppose.config import GRIP_MAX_OPEN_M
from grasppose.worker_client import WorkerError, infer_image


def _camera_k(args):
    values = args.camera_k
    if values is None:
        configured = os.environ.get("CAMERA_K", "").split()
        if configured:
            if len(configured) != 4:
                raise ValueError("CAMERA_K must contain FX FY CX CY")
            values = [float(value) for value in configured]
    return values


def _camera_k_size(args):
    values = args.camera_k_size
    if values is None:
        configured = os.environ.get("CAMERA_K_SIZE", "").replace(",", " ").split()
        if configured:
            if len(configured) != 2:
                raise ValueError("CAMERA_K_SIZE must contain WIDTH HEIGHT")
            values = [int(value) for value in configured]
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a fixed-prompt TensorRT pipeline through cold.sh worker."
    )
    parser.add_argument("image")
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument(
        "--camera-k", nargs=4, type=float,
        metavar=("FX", "FY", "CX", "CY"),
    )
    parser.add_argument(
        "--camera-k-size", nargs=2, type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="resolution at which --camera-k was calibrated",
    )
    parser.add_argument("--fov-x", type=float, default=None)
    parser.add_argument("--fov-y", type=float, default=None)
    parser.add_argument("--render", action="store_true",
                        help="also create the four diagnostic PNG images")
    parser.add_argument("--out", default=os.environ.get(
        "OUTPUT_DIR",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
    ))
    parser.add_argument("--max-width", type=float, default=GRIP_MAX_OPEN_M)
    parser.add_argument("--top", type=int, default=1)
    args = parser.parse_args(argv)

    image = os.path.abspath(args.image)
    if not os.path.isfile(image):
        parser.error("input image not found: %s" % image)
    try:
        camera_k = _camera_k(args)
        camera_k_size = _camera_k_size(args)
        if camera_k_size is not None and camera_k is None:
            parser.error("--camera-k-size requires --camera-k or CAMERA_K")
        if camera_k is None and args.fov_x is None and args.fov_y is None:
            parser.error(
                "provide --camera-k FX FY CX CY, CAMERA_K env, --fov-x or --fov-y"
            )
        started = time.perf_counter()
        response = infer_image(
            image,
            args.prompt_id,
            camera_k=camera_k,
            camera_k_size=camera_k_size,
            fov_x=args.fov_x,
            fov_y=args.fov_y,
            output_dir=args.out if args.render else None,
            render=args.render,
            top=args.top,
            max_width=args.max_width,
        )
    except (WorkerError, ValueError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    print("RUN_ID: %s" % response["run_id"])
    if not response.get("snapshot_available", True):
        print("SNAPSHOT: unavailable (frame exceeded the retained-output cache limit)")
    if response.get("files"):
        print("\n".join(response["files"]))
    print("DETECTIONS: %d MASK_PIXELS=%d GRASPS=%d" % (
        response.get("detection_count", 0),
        response.get("mask_pixels", 0),
        response.get("grasp_count", 0),
    ))
    if response.get("depth_m") is not None:
        print("target depth: %.3f m" % response["depth_m"])
    print("TOTAL: %.1f ms" % (
        (time.perf_counter() - started) * 1000.0))
    print("worker: %.1f ms" % response["server_ms"])
    if response.get("render_ms") is not None:
        print("render: %.1f ms" % response["render_ms"])
    print("GRASP_POSES: %s" % json.dumps(
        response.get("grasps", []), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
