"""Resident single-process model worker over a local Unix socket."""

import json
import os
import signal
import socketserver
import sys
import threading
import time
import traceback
import uuid

import numpy as np
from PIL import Image

from .config import RUNTIME_DIR, WORKER_PID, WORKER_SOCKET
from .domain.geometry import fov_x_from_fovy
from .facade import DEFAULT_SERVICE
from .output_snapshot import ARRAY_NAMES, OutputSnapshot, SnapshotCache
from .prompt_catalog import PromptCatalog
from .runtime import log


class WorkerServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler):
        self.state = "starting"
        self.error = None
        self.service = DEFAULT_SERVICE
        self.catalog = None
        self.snapshots = SnapshotCache()
        super().__init__(address, handler)


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            raw = self.rfile.readline(1024 * 1024)
            if not raw:
                return
            request = json.loads(raw.decode("utf-8"))
            operation = request.get("op")
            if operation == "status":
                self._respond({
                    "ok": self.server.state == "ready",
                    "state": self.server.state,
                    "pid": os.getpid(),
                    "prompt_ids": list(self.server.catalog.by_id)
                    if self.server.catalog else [],
                    "prompts": (
                        [{"id": item["id"], "text": item["text"]}
                         for item in self.server.catalog.prompts]
                        if self.server.catalog else []
                    ),
                    "error": self.server.error,
                })
                return
            if operation in ("snapshot_status", "snapshot"):
                run_id = request.get("run_id")
                snapshot = self.server.snapshots.get(run_id)
                if snapshot is None:
                    raise ValueError("RUN_ID is unknown or expired: %r" % run_id)
                if operation == "snapshot_status":
                    self._respond({"ok": True, "run_id": run_id})
                else:
                    specs = [{
                        "name": name,
                        "dtype": str(getattr(snapshot, name).dtype),
                        "shape": list(getattr(snapshot, name).shape),
                        "nbytes": int(getattr(snapshot, name).nbytes),
                    } for name in ARRAY_NAMES]
                    self._respond({
                        "ok": True, "run_id": run_id,
                        "metadata": snapshot.metadata(), "arrays": specs,
                    })
                    for name in ARRAY_NAMES:
                        array = getattr(snapshot, name)
                        if array.nbytes:
                            self.wfile.write(memoryview(array).cast("B"))
                    self.wfile.flush()
                return
            if operation == "stop":
                self._respond({"ok": True, "state": "stopping"})
                threading.Thread(
                    target=self.server.shutdown, daemon=True
                ).start()
                return
            if operation != "infer":
                raise ValueError("unsupported worker operation")
            if self.server.state != "ready":
                raise RuntimeError(
                    "worker is %s: %s"
                    % (self.server.state, self.server.error or "not ready")
                )
            self._respond(self._infer(request))
        except Exception as exc:
            self._respond({
                "ok": False,
                "error": "%s: %s" % (type(exc).__name__, exc),
            })

    def _infer(self, request):
        prompt_id = str(request.get("prompt_id", ""))
        self.server.catalog.require(prompt_id)
        image_path = request.get("image")
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            raise ValueError("input image does not exist: %r" % image_path)
        profile = os.environ.get("GRASP_PROFILE_INFER") == "1"
        started = time.perf_counter()
        image_started = time.perf_counter()
        image = np.asarray(Image.open(image_path).convert("RGB"))
        if profile:
            log("profile image decode %.3f s" % (
                time.perf_counter() - image_started))
        camera_k = request.get("camera_k")
        if camera_k is not None:
            camera_k = np.asarray(camera_k, dtype=np.float64).reshape(4)
            fx, fy, cx, cy = camera_k
            camera_k = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        fov_x = request.get("fov_x")
        if fov_x is None and request.get("fov_y") is not None:
            fov_x = fov_x_from_fovy(
                request["fov_y"], image.shape[1], image.shape[0])
        max_width = float(request.get("max_width", 0.080))
        top = int(request.get("top", 1))
        if max_width <= 0 or top < 1:
            raise ValueError("top and max_width must be positive")
        output_dir = request.get("output_dir")
        if request.get("render", False):
            if not isinstance(output_dir, str) or not output_dir:
                raise ValueError("output directory is required with render")
            os.makedirs(output_dir, exist_ok=True)
            if not os.access(output_dir, os.W_OK | os.X_OK):
                raise ValueError("output directory is not writable: %s" % output_dir)
        result = self.server.service.core.run(
            image,
            prompt_id=prompt_id,
            camera_K=camera_k,
            fov_x=fov_x,
        )
        snapshot = OutputSnapshot.capture(image, result, max_width, top)
        run_id = self.server.snapshots.put(snapshot)
        snapshot_available = run_id is not None
        if run_id is None:
            run_id = uuid.uuid4().hex
        files = []
        render_ms = None
        if request.get("render", False):
            from .output_renderer import render_and_save
            files, render_ms = render_and_save(snapshot, run_id, output_dir)
        grasps = snapshot.graspgroup
        valid_grasps = grasps[grasps[:, 1] <= max_width]
        valid_grasps = valid_grasps[np.argsort(-valid_grasps[:, 0])[:top]]
        grasp_poses = [{
            "score": float(row[0]),
            "width_m": float(row[1]),
            "translation_m": row[13:16].tolist(),
            "rotation": row[4:13].reshape(3, 3).tolist(),
        } for row in valid_grasps]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if profile:
            log("profile worker total %.3f s" % (elapsed_ms / 1000.0))
        return {
            "ok": True,
            "run_id": run_id,
            "snapshot_available": snapshot_available,
            "files": files,
            "grasps": grasp_poses,
            "depth_m": result.depth_m,
            "detection_count": int(len(snapshot.boxes)),
            "mask_pixels": int(np.count_nonzero(snapshot.mask)),
            "grasp_count": int(len(snapshot.graspgroup)),
            "server_ms": elapsed_ms,
            "render_ms": render_ms,
        }

    def _respond(self, value):
        self.wfile.write(
            (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")
        )
        self.wfile.flush()


def _write_pid():
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    temporary = WORKER_PID + ".tmp"
    with open(temporary, "w", encoding="ascii") as handle:
        handle.write("%d\n" % os.getpid())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, WORKER_PID)


def serve():
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    os.umask(0o077)
    lock_path = os.path.join(RUNTIME_DIR, "worker.lock")
    lock_handle = open(lock_path, "w")
    try:
        import fcntl
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (ImportError, BlockingIOError) as exc:
        raise RuntimeError("another inference worker is already starting/running") from exc

    if os.path.lexists(WORKER_SOCKET):
        os.unlink(WORKER_SOCKET)
    _write_pid()

    server = None
    try:
        catalog = PromptCatalog.load(verify_engine=True, require_full_pipeline=True)
        expected_artifact_id = catalog.manifest["artifact_id"]
        DEFAULT_SERVICE.load()
        loaded_catalog = DEFAULT_SERVICE.core._vision._catalog
        if loaded_catalog.manifest["artifact_id"] != expected_artifact_id:
            raise RuntimeError(
                "YOLOE prompt artifact changed during worker startup; "
                "retry cold.sh start"
            )
        catalog = loaded_catalog
        DEFAULT_SERVICE.core.warmup()
        server = WorkerServer(WORKER_SOCKET, Handler)
        server.catalog = catalog
        server.state = "ready"
        os.chmod(WORKER_SOCKET, 0o600)
        print(
            "worker ready pid=%d prompt_ids=%s"
            % (os.getpid(), ",".join(catalog.by_id)),
            flush=True,
        )

        def shutdown(signum, _frame):
            if server is not None:
                threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        server.serve_forever(poll_interval=0.2)
    except Exception:
        if server is not None:
            server.state = "failed"
            server.error = traceback.format_exc()
        traceback.print_exc()
        raise
    finally:
        if server is not None:
            server.server_close()
        try:
            os.unlink(WORKER_SOCKET)
        except OSError:
            pass
        try:
            if os.path.isfile(WORKER_PID):
                with open(WORKER_PID, "r", encoding="ascii") as handle:
                    if handle.read().strip() == str(os.getpid()):
                        os.unlink(WORKER_PID)
        except OSError:
            pass
        lock_handle.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("serve",))
    parser.parse_args(argv)
    serve()


if __name__ == "__main__":
    sys.exit(main())
