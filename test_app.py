#!/usr/bin/env python3
"""Smoke-test app.py without Gradio or real models."""

import sys
import traceback

import numpy as np

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


def fake_result(depth_m=0.4712):
    return {
        "box": np.full((H, W, 3), 10, np.uint8),
        "mask": np.full((H, W, 3), 20, np.uint8),
        "depthmap": np.full((H, W, 3), 30, np.uint8),
        "grasp": np.full((H, W, 3), 40, np.uint8),
        "depth_m": depth_m,
    }


class FakeService:
    def __init__(self, fn):
        self.fn = fn

    def infer(self, image, **kwargs):
        return self.fn(image, **kwargs)


def main():
    check(
        "gradio is lazy-imported",
        "gradio" not in sys.modules,
    )
    check("run_one exists", callable(A.run_one))
    check("build_ui exists", callable(A.build_ui))

    original = A.SERVICE
    try:
        seen = {}

        def success(image, prompt=None, top=None, **kwargs):
            seen["prompt"] = prompt
            seen["top"] = top
            seen["shape"] = np.asarray(image).shape
            return fake_result(0.4712)

        A.SERVICE = FakeService(success)
        output = A.run_one(IMG, "a little bag")
        check("six UI outputs", len(output) == 6)
        box, mask, depth, grasp, depth_m, status = output

        for name, array, value in (
            ("box", box, 10),
            ("mask", mask, 20),
            ("depthmap", depth, 30),
            ("grasp", grasp, 40),
        ):
            check(
                "%s output" % name,
                isinstance(array, np.ndarray)
                and array.shape == (H, W, 3)
                and int(array[0, 0, 0]) == value,
            )

        check(
            "depth value",
            isinstance(depth_m, float)
            and abs(depth_m - 0.4712) < 1e-9,
        )
        check(
            "prompt forwarded",
            seen.get("prompt") == "a little bag",
        )
        check(
            "top grasps forwarded",
            seen.get("top") == A.TOP_GRASPS,
        )
        check("status contains metric depth", "0.471" in status)

        output = A.run_one(None, "object")
        check("None image returns six outputs", len(output) == 6)
        check(
            "None image has no renderings",
            all(value is None for value in output[:5]),
        )

        def failure(image, **kwargs):
            raise RuntimeError("synthetic pipeline failure")

        A.SERVICE = FakeService(failure)
        try:
            output = A.run_one(IMG, "object")
            check("service errors do not escape", True)
            check(
                "error status preserved",
                "RuntimeError" in output[5]
                and "synthetic pipeline failure" in output[5],
            )
        except Exception:
            traceback.print_exc()
            check("service errors do not escape", False)

        A.SERVICE = FakeService(
            lambda image, **kwargs: fake_result(None))
        output = A.run_one(IMG, "unknown object")
        check(
            "no-depth frame still returns four images",
            all(isinstance(value, np.ndarray)
                for value in output[:4]),
        )
        check("no-depth value is None", output[4] is None)

        seen.clear()
        A.SERVICE = FakeService(success)
        for prompt in ("", "   ", None):
            A.run_one(IMG, prompt)
            check(
                "empty prompt uses default",
                seen.get("prompt") == A.DEFAULT_PROMPT,
            )

        A.SERVICE = FakeService(failure)
        for name, image, prompt in (
            ("1x1 image", np.zeros((1, 1, 3), np.uint8), "x"),
            ("empty image", np.zeros((0, 0, 3), np.uint8), "x"),
            ("numeric prompt", IMG, 12345),
            ("list prompt", IMG, ["a"]),
        ):
            try:
                A.run_one(image, prompt)
                check("%s does not raise" % name, True)
            except Exception as exc:
                check(
                    "%s does not raise" % name,
                    False,
                    "%s: %s" % (
                        type(exc).__name__, exc),
                )
    finally:
        A.SERVICE = original

    if FAIL:
        print("%d TESTS FAILED" % len(FAIL))
        for name in FAIL:
            print(" -", name)
        return 1

    print("TAT CA MUC DEU PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
