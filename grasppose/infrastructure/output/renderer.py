"""Render one cached inference result without importing or loading models."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import socket
import sys
import time

import numpy as np
from PIL import Image

from ...config import RUNTIME_DIR, WORKER_SOCKET
from ...modules.depth.types import DepthResult
from ...modules.grasp.types import GraspResult
from ...modules.vision.types import DetectionResult, SegmentationResult
from .snapshot import ARRAY_NAMES, OutputSnapshot
from ...presentation.rendering import draw_box, draw_depth, draw_grasp, draw_mask


def _write_json(path, value):
    temporary = "%s.tmp.%d" % (path, os.getpid())
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_exact(stream, size):
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        count = stream.readinto(view[offset:])
        if not count:
            raise RuntimeError("incomplete snapshot array")
        offset += count
    return data


def fetch_snapshot(run_id, socket_path=WORKER_SOCKET):
    """Fetch binary NumPy arrays over the local socket; never unpickle data."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(60)
        client.connect(socket_path)
        client.sendall((json.dumps({
            "op": "snapshot", "run_id": run_id,
        }) + "\n").encode("utf-8"))
        with client.makefile("rb") as stream:
            header = stream.readline(1024 * 1024)
            if not header:
                raise RuntimeError("worker closed the snapshot connection")
            response = json.loads(header.decode("utf-8"))
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "snapshot unavailable"))
            specs = response.get("arrays", ())
            if tuple(spec.get("name") for spec in specs) != ARRAY_NAMES:
                raise RuntimeError("worker snapshot protocol mismatch")
            arrays = {}
            for spec in specs:
                dtype = np.dtype(spec["dtype"])
                shape = tuple(int(size) for size in spec["shape"])
                nbytes = int(spec["nbytes"])
                if (dtype.hasobject or nbytes < 0 or
                        nbytes > 256 * 1024 * 1024 or
                        int(np.prod(shape)) * dtype.itemsize != nbytes):
                    raise RuntimeError("invalid snapshot array metadata")
                raw = _read_exact(stream, nbytes)
                arrays[spec["name"]] = np.frombuffer(
                    raw, dtype=dtype).reshape(shape)
    return OutputSnapshot.from_wire(response["metadata"], arrays)


def render_images(snapshot):
    detection = DetectionResult(
        snapshot.boxes, snapshot.scores, snapshot.labels,
        snapshot.detection_reason or None,
    )
    segmentation = SegmentationResult(
        snapshot.mask, snapshot.scores, 0, len(snapshot.scores),
        snapshot.segmentation_reason or None,
    )
    depth = DepthResult(
        snapshot.depth, snapshot.camera_k, 0.0,
        reason=snapshot.depth_reason or None,
    )
    grasp = GraspResult(
        snapshot.graspgroup, snapshot.grasp_reason or None,
    )
    return {
        "box": draw_box(snapshot.image, detection),
        "mask": draw_mask(snapshot.image, segmentation),
        "depthmap": draw_depth(depth),
        "grasp": draw_grasp(
            snapshot.image, grasp, snapshot.camera_k,
            max_width=snapshot.max_width, top=snapshot.top,
        ),
    }


def render_and_save(snapshot, run_id, output_dir):
    """Render a retained result to four PNGs in an atomic run directory."""
    started = time.perf_counter()
    images = render_images(snapshot)
    os.makedirs(output_dir, exist_ok=True)
    temporary_dir = os.path.join(
        output_dir, ".%s.tmp.%d" % (run_id, os.getpid()))
    final_dir = os.path.join(output_dir, run_id)
    os.makedirs(temporary_dir, exist_ok=False)
    try:
        compression = int(os.environ.get("GRASP_PNG_COMPRESSION_LEVEL", "1"))
        if not 0 <= compression <= 9:
            raise ValueError("GRASP_PNG_COMPRESSION_LEVEL must be 0..9")

        def save(item):
            name, array = item
            path = os.path.join(temporary_dir, name + ".png")
            Image.fromarray(array).save(
                path, format="PNG", compress_level=compression)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(save, images.items()))
        _write_json(os.path.join(temporary_dir, "manifest.json"), {
            "run_id": run_id,
            "files": list(images),
        })
        if os.path.exists(final_dir):
            raise RuntimeError("output already exists: %s" % final_dir)
        os.replace(temporary_dir, final_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return [os.path.join(final_dir, name + ".png") for name in images], round(
        (time.perf_counter() - started) * 1000.0, 1)


def export(run_id, output_dir):
    snapshot = fetch_snapshot(run_id)
    files, render_ms = render_and_save(snapshot, run_id, output_dir)
    return {
        "state": "done",
        "run_id": run_id,
        "files": files,
        "render_ms": render_ms,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)
    job_path = os.path.join(RUNTIME_DIR, "output-jobs", args.run_id + ".json")
    try:
        try:
            os.nice(10)
        except OSError:
            pass
        _write_json(job_path, {
            "state": "running", "run_id": args.run_id, "pid": os.getpid(),
        })
        result = export(args.run_id, args.output_dir)
        _write_json(job_path, result)
        return 0
    except Exception as exc:
        _write_json(job_path, {
            "state": "failed", "run_id": args.run_id,
            "error": "%s: %s" % (type(exc).__name__, exc),
        })
        print("output render failed: %s: %s" % (
            type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
