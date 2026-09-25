#!/usr/bin/env python3
"""Measure infer.sh latency over the resident worker (fast path by default)."""

import argparse
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETECTION = re.compile(r"DETECTIONS:\s*(\d+)\s+MASK_PIXELS=(\d+)")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument(
        "--camera-k", nargs=4, type=float,
        metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument(
        "--camera-k-size", nargs=2, type=int,
        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--out", default=str(ROOT / "output"))
    parser.add_argument(
        "--render", action="store_true",
        help="also measure the optional four-image PNG render path",
    )
    args = parser.parse_args(argv)
    if args.runs < 20:
        parser.error("--runs must be at least 20 for the acceptance check")
    image = Path(args.image).resolve()
    if not image.is_file():
        parser.error("image does not exist: %s" % image)

    command = ["bash", str(ROOT / "infer.sh"), str(image),
               "--prompt-id", args.prompt_id]
    if args.render:
        command.extend(["--render", "--out", args.out])
    if args.camera_k:
        command.extend(["--camera-k"] + [str(value) for value in args.camera_k])
    if args.camera_k_size:
        command.extend(
            ["--camera-k-size"] + [str(value) for value in args.camera_k_size])
    samples = []
    failures = []
    for index in range(args.runs):
        started = time.perf_counter()
        completed = subprocess.run(
            command, cwd=str(ROOT), text=True, capture_output=True)
        elapsed = (time.perf_counter() - started) * 1000.0
        if completed.returncode != 0:
            failures.append(
                "run %d exited %d: %s"
                % (index + 1, completed.returncode,
                   (completed.stderr or completed.stdout).strip())
            )
            continue
        match = DETECTION.search(completed.stdout)
        if not match:
            failures.append(
                "run %d did not report target detection metadata" % (index + 1)
            )
            continue
        boxes, mask_pixels = map(int, match.groups())
        if boxes <= 0 or mask_pixels <= 0:
            failures.append(
                "run %d found no target (boxes=%d, mask_pixels=%d)"
                % (index + 1, boxes, mask_pixels)
            )
            continue
        samples.append(elapsed)

    if not samples:
        print("successful runs: 0/%d" % args.runs)
        for failure in failures:
            print("ERROR:", failure, file=sys.stderr)
        return 1
    ordered = sorted(samples)
    p95 = ordered[max(0, int(math.ceil(0.95 * len(ordered))) - 1)]
    median = ordered[len(ordered) // 2]
    print("successful runs: %d/%d" % (len(samples), args.runs))
    mode = "with PNG rendering" if args.render else "fast inference"
    print("mode: %s" % mode)
    print("median infer.sh wall time: %.1f ms" % median)
    print("P95 infer.sh wall time: %.1f ms" % p95)
    print("maximum infer.sh wall time: %.1f ms" % ordered[-1])
    if args.render:
        print("acceptance: NOT APPLICABLE (render path selected)")
    elif len(samples) == args.runs and p95 < 1000.0:
        print("acceptance: PASS (P95 < 1000 ms)")
    else:
        print("acceptance: FAIL")
    for failure in failures:
        print("ERROR:", failure, file=sys.stderr)
    if args.render:
        return 0 if len(samples) == args.runs else 1
    return 0 if len(samples) == args.runs and p95 < 1000.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
