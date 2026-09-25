#!/usr/bin/env python3
"""Smoke-test the on-demand Gradio flow without models or Gradio import."""

import sys

import numpy as np

import app as A


def main():
    assert "gradio" not in sys.modules
    assert callable(A.run_one) and callable(A.show_output)

    class FakeService:
        def __init__(self):
            self.calls = []

        def infer(self, image, **kwargs):
            self.calls.append((image, kwargs))
            return {
                "run_id": "a" * 32,
                "depth_m": 0.4712,
                "detection_count": 2,
                "grasp_count": 1,
                "grasps": [{"score": 0.95}],
            }

        def output(self, run_id):
            assert run_id == "a" * 32
            return [np.full((5, 6, 3), n, np.uint8)
                    for n in (10, 20, 30, 40)]

    original = A.SERVICE
    try:
        fake = FakeService()
        A.SERVICE = fake
        result = A.run_one("/tmp/input.png", "blue cube")
        assert len(result) == 9
        assert result[0] == "a" * 32 and result[1] == "a" * 32
        assert abs(result[2] - 0.4712) < 1e-9
        assert "0.95" in result[3]
        assert "2 box" in result[4]
        assert result[5:] == (None, None, None, None)
        assert fake.calls[0][1]["top"] == A.TOP_GRASPS

        images = A.show_output(result[0])
        assert len(images) == 5
        assert [int(image[0, 0, 0]) for image in images[:4]] == [
            10, 20, 30, 40,
        ]
        assert "output" in images[4]

        assert A.run_one(None, "blue cube")[0] is None
        assert A.show_output(None)[0] is None
        A.run_one("/tmp/input.png", "   ")
        assert fake.calls[-1][1]["prompt"] == A.DEFAULT_PROMPT

        def fail(*_args, **_kwargs):
            raise RuntimeError("synthetic failure")

        fake.infer = fail
        assert "synthetic failure" in A.run_one("/tmp/x", "x")[4]
        fake.output = fail
        assert "synthetic failure" in A.show_output("a" * 32)[4]
    finally:
        A.SERVICE = original
    print("On-demand Gradio flow PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
