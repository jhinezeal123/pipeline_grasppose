#!/usr/bin/env python3
"""Smoke-test app.py without Gradio or real models."""

import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as A


FAIL = []
H, W = 60, 80
IMG = np.zeros((H, W, 3), np.uint8)


def check(name, condition, detail=""):
    print("  [%s] %s%s" % (
        "PASS" if condition else "FAIL",
        name,
        (" - " + detail) if detail else "",
    ))
    if not condition:
        FAIL.append(name)


def main():
    check("gradio is lazy-imported", "gradio" not in sys.modules)
    check("run_one exists", callable(A.run_one))
    check("build_ui exists", callable(A.build_ui))

    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        paths = []
        for name, value in (
            ("box", 10), ("mask", 20), ("depthmap", 30), ("grasp", 40),
        ):
            path = temp_path / (name + ".png")
            Image.fromarray(np.full((H, W, 3), value, np.uint8)).save(path)
            paths.append(str(path))

        seen = {}

        def success(image_path, prompt_id, **kwargs):
            seen["image_path"] = image_path
            seen["image_exists_at_call"] = Path(image_path).is_file()
            seen["prompt_id"] = prompt_id
            seen.update(kwargs)
            return {
                "files": paths,
                "depth_m": 0.4712,
                "server_ms": 57.2,
            }

        with patch.object(A, "OUTPUT_DIR", temp), \
                patch.object(A, "infer_image", side_effect=success):
            output = A.run_one(IMG, "blue_cube")

        check("six UI outputs", len(output) == 6)
        box, mask, depth, grasp, depth_m, status = output
        for name, array, value in (
            ("box", box, 10), ("mask", mask, 20),
            ("depthmap", depth, 30), ("grasp", grasp, 40),
        ):
            check(
                "%s output" % name,
                isinstance(array, np.ndarray)
                and array.shape == (H, W, 3)
                and int(array[0, 0, 0]) == value,
            )
        check(
            "prompt ID forwarded",
            seen.get("prompt_id") == "blue_cube",
        )
        check(
            "image sent by local path",
            isinstance(seen.get("image_path"), str)
            and seen.get("image_exists_at_call") is True,
        )
        check(
            "no prompt text sent to worker",
            "the blue cube" not in repr(seen),
        )
        check(
            "top grasps forwarded",
            seen.get("top") == A.TOP_GRASPS,
        )
        check("depth value", abs(depth_m - 0.4712) < 1e-9)
        check("status has timing", "57.2 ms" in status)

        none_output = A.run_one(None, "blue_cube")
        check("None image returns six outputs", len(none_output) == 6)
        check(
            "None image has no renderings",
            all(value is None for value in none_output[:5]),
        )

        def failure(*args, **kwargs):
            raise RuntimeError("synthetic worker failure")

        with patch.object(A, "OUTPUT_DIR", temp), \
                patch.object(A, "infer_image", side_effect=failure):
            failed = A.run_one(IMG, "blue_cube")
        check(
            "worker errors are displayed",
            "RuntimeError" in failed[5]
            and "synthetic worker failure" in failed[5],
        )

        empty = A.run_one(IMG, "")
        check("empty ID is rejected", "chon prompt id" in empty[5].lower())

    if FAIL:
        print("%d TESTS FAILED" % len(FAIL))
        for name in FAIL:
            print(" -", name)
        return 1

    print("TAT CA MUC DEU PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
