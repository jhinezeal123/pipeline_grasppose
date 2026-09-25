#!/usr/bin/env python3
"""Fast stdlib-only CLI client for the warm inference worker."""

import json
import os
import socket
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DIR = os.environ.get("GRASP_RUNTIME_DIR", os.path.join(ROOT, ".runtime"))
SOCKET_PATH = os.environ.get(
    "GRASP_WORKER_SOCKET", os.path.join(RUNTIME_DIR, "worker.sock"))


def _camera_k(values):
    if values is None:
        configured = os.environ.get("CAMERA_K", "").split()
        if configured:
            if len(configured) != 4:
                raise ValueError("CAMERA_K must contain FX FY CX CY")
            values = [float(value) for value in configured]
    return values


def _request(payload):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(300)
            client.connect(SOCKET_PATH)
            client.sendall((json.dumps(payload, separators=(",", ":"))
                            + "\n").encode("utf-8"))
            with client.makefile("rb") as stream:
                line = stream.readline(1024 * 1024)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "inference worker is not running; start it with: bash cold.sh"
        ) from exc
    except OSError as exc:
        raise RuntimeError("cannot connect to inference worker: %s" % exc) from exc
    if not line:
        raise RuntimeError("inference worker closed the connection")
    response = json.loads(line.decode("utf-8"))
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "inference worker failed"))
    return response


def _arguments(argv):
    usage = (
        "Usage: infer.sh IMAGE [--prompt NAME] "
        "[--camera-k FX FY CX CY | --fov-x DEG | --fov-y DEG] "
        "[--max-width METERS] [--top N]"
    )
    if not argv or argv[0] in ("-h", "--help"):
        print(usage)
        raise SystemExit(0 if argv else 2)
    values = {
        "image": argv[0], "prompt": None, "camera_k": None,
        "fov_x": None, "fov_y": None,
        "max_width": 0.080, "top": 1,
    }
    i = 1
    while i < len(argv):
        option = argv[i]
        if option == "--camera-k":
            if i + 4 >= len(argv):
                raise ValueError("--camera-k requires FX FY CX CY")
            values["camera_k"] = [float(x) for x in argv[i + 1:i + 5]]
            i += 5
        elif option in ("--prompt", "--fov-x", "--fov-y",
                        "--max-width", "--top"):
            if i + 1 >= len(argv):
                raise ValueError("%s requires a value" % option)
            key = option[2:].replace("-", "_")
            conversion = (
                str if key == "prompt" else
                int if key == "top" else float
            )
            values[key] = conversion(argv[i + 1])
            i += 2
        else:
            raise ValueError("unknown option: %s\n%s" % (option, usage))
    return values


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        args = _arguments(argv)
        image = os.path.abspath(args["image"])
        if not os.path.isfile(image):
            raise ValueError("input image not found: %s" % image)
        camera_k = _camera_k(args["camera_k"])
        if camera_k is None and args["fov_x"] is None and args["fov_y"] is None:
            raise ValueError(
                "provide --camera-k FX FY CX CY, CAMERA_K env, "
                "--fov-x or --fov-y")
        started = time.perf_counter()
        response = _request({
            "op": "infer",
            "image": image,
            "prompt": args["prompt"],
            "camera_k": camera_k,
            "fov_x": args["fov_x"],
            "fov_y": args["fov_y"],
            "top": args["top"],
            "max_width": args["max_width"],
        })
    except (OSError, ValueError, RuntimeError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    print("RUN_ID: %s" % response["run_id"])
    print("DETECTIONS: %d MASK_PIXELS=%d GRASPS=%d" % (
        response.get("detection_count", 0),
        response.get("mask_pixels", 0),
        response.get("grasp_count", 0),
    ))
    if response.get("depth_m") is not None:
        print("target depth: %.3f m" % response["depth_m"])
    print("GRASP_POSES: %s" % json.dumps(
        response.get("grasps", []), separators=(",", ":")))
    print("TOTAL: %.1f ms" % (
        (time.perf_counter() - started) * 1000.0))
    print("worker: %.1f ms" % response["server_ms"])
    if (response.get("detection_count", 0) <= 0
            or response.get("mask_pixels", 0) <= 0):
        print("ERROR: no segmented target; RUN_ID is available for diagnostics",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
