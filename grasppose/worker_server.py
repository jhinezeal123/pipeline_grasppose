"""Resident single-process model worker over a local Unix socket."""

import json
import os
import signal
from concurrent.futures import ThreadPoolExecutor
import socketserver
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from .config import RUNTIME_DIR, WORKER_PID, WORKER_SOCKET
from .facade import DEFAULT_SERVICE
from .prompt_catalog import PromptCatalog
from .runtime import log


class WorkerServer(socketserver.UnixStreamServer):
    allow_reuse_address = True

    def __init__(self, address, handler):
        self.state = "starting"
        self.error = None
        self.service = DEFAULT_SERVICE
        self.catalog = None
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
                    "prompt_ids": (
                        list(self.server.catalog.by_id)
                        if self.server.catalog else []
                    ),
                    "prompts": (
                        [{"id": item["id"], "text": item["text"]}
                         for item in self.server.catalog.prompts]
                        if self.server.catalog else []
                    ),
                    "error": self.server.error,
                })
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
        output_dir = request.get("output_dir")
        if not isinstance(output_dir, str) or not output_dir:
            raise ValueError("output directory is required")
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK | os.X_OK):
            raise ValueError("output directory is not writable: %s" % output_dir)

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
        result = self.server.service.infer(
            image,
            prompt_id=prompt_id,
            camera_K=camera_k,
            fov_x=request.get("fov_x"),
            max_width=float(request.get("max_width", 0.080)),
            top=int(request.get("top", 1)),
        )
        stem = Path(image_path).stem
        compression = int(os.environ.get("GRASP_PNG_COMPRESSION_LEVEL", "1"))
        workers = int(os.environ.get("GRASP_PNG_WORKERS", "4"))
        if not 0 <= compression <= 9:
            raise ValueError("GRASP_PNG_COMPRESSION_LEVEL must be from 0 to 9")
        if workers < 1:
            raise ValueError("GRASP_PNG_WORKERS must be at least 1")

        def save_png(item):
            key, array = item
            path = os.path.join(
                output_dir, "%s_%s.png" % (stem, key))
            save_started = time.perf_counter()
            Image.fromarray(array).save(
                path,
                format="PNG",
                compress_level=compression,
            )
            if profile:
                log("profile PNG %s %.3f s" % (
                    key, time.perf_counter() - save_started))
            return path

        with ThreadPoolExecutor(max_workers=workers) as pool:
            saved = list(pool.map(save_png, (
                (key, result[key])
                for key in ("box", "mask", "depthmap", "grasp")
            )))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if profile:
            log("profile worker total %.3f s" % (elapsed_ms / 1000.0))
        return {
            "ok": True,
            "files": saved,
            "depth_m": result["depth_m"],
            "detection_count": result["detection_count"],
            "mask_pixels": result["mask_pixels"],
            "grasp_count": result["grasp_count"],
            "server_ms": elapsed_ms,
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
