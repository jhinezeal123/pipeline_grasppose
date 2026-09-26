#!/usr/bin/env python3
"""CLI client for the resident worker."""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from grasppose.api import DEFAULT_MAX_WIDTH_M
from grasppose.infrastructure.output.control import start as start_output
from grasppose.infrastructure.worker.client import WorkerError, WorkerGraspEstimator

ESTIMATOR = WorkerGraspEstimator()


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
        description="Run a fixed-prompt TensorRT pipeline through scripts/worker.sh worker."
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
                        help="queue four diagnostic PNGs in a separate process")
    parser.add_argument("--out", default=os.environ.get(
        "OUTPUT_DIR",
        os.path.join(ROOT, "artifacts", "output"),
    ))
    parser.add_argument("--max-width", type=float, default=DEFAULT_MAX_WIDTH_M)
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
        estimate = ESTIMATOR.estimate(
            image,
            args.prompt_id,
            camera_K=camera_k,
            camera_K_size=camera_k_size,
            fov_x=args.fov_x,
            fov_y=args.fov_y,
            top=args.top,
            max_width=args.max_width,
        )
        render_job = None
        if args.render:
            if not estimate.snapshot_available or not estimate.request_id:
                raise WorkerError(
                    "cannot render asynchronously: output snapshot was not retained"
                )
            render_job = start_output(
                estimate.request_id, output_dir=args.out)
    except (WorkerError, ValueError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    print("RUN_ID: %s" % estimate.request_id)
    if not estimate.snapshot_available:
        print("SNAPSHOT: unavailable (frame exceeded the retained-output cache limit)")
    if render_job is not None:
        print("RENDER_JOB: %s" % render_job["state"])
        if render_job.get("output_dir"):
            print("OUTPUT_PENDING: %s" % render_job["output_dir"])
    print("DETECTIONS: %d MASK_PIXELS=%d GRASPS=%d" % (
        estimate.detection_count,
        estimate.mask_pixels,
        estimate.grasp_count,
    ))
    if estimate.depth_m is not None:
        print("target depth: %.3f m" % estimate.depth_m)
    print("TOTAL: %.1f ms" % (
        (time.perf_counter() - started) * 1000.0))
    if estimate.latency_ms is not None:
        print("worker: %.1f ms" % estimate.latency_ms)
    print("GRASP_POSES: %s" % json.dumps([
        {
            "score": pose.score,
            "width_m": pose.width_m,
            "translation_m": list(pose.translation_m),
            "rotation": [list(row) for row in pose.rotation],
        }
        for pose in estimate.grasps
    ], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
