"""Resident single-process model worker over a local Unix socket."""

import json
import os
import signal
import socketserver
import sys
import threading
import traceback

from ..settings import RUNTIME_DIR, WORKER_PID, WORKER_SOCKET
from ..composition import DEFAULT_ESTIMATOR, DEFAULT_PIPELINE
from ..output.snapshot import ARRAY_NAMES, SnapshotCache
from ...modules.vision.prompt_catalog import PromptCatalog
from .inference import WorkerInference, reject_inline_render


class WorkerServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler):
        self.state = "starting"
        self.error = None
        self.estimator = DEFAULT_ESTIMATOR
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
                self._respond(
                    {
                        "ok": self.server.state == "ready",
                        "state": self.server.state,
                        "pid": os.getpid(),
                        "prompt_ids": list(self.server.catalog.by_id)
                        if self.server.catalog
                        else [],
                        "prompts": (
                            [
                                {"id": item["id"], "text": item["text"]}
                                for item in self.server.catalog.prompts
                            ]
                            if self.server.catalog
                            else []
                        ),
                        "error": self.server.error,
                    }
                )
                return
            if operation in ("snapshot_status", "snapshot"):
                run_id = request.get("run_id")
                snapshot = self.server.snapshots.get(run_id)
                if snapshot is None:
                    raise ValueError("RUN_ID is unknown or expired: %r" % run_id)
                if operation == "snapshot_status":
                    self._respond({"ok": True, "run_id": run_id})
                else:
                    specs = [
                        {
                            "name": name,
                            "dtype": str(getattr(snapshot, name).dtype),
                            "shape": list(getattr(snapshot, name).shape),
                            "nbytes": int(getattr(snapshot, name).nbytes),
                        }
                        for name in ARRAY_NAMES
                    ]
                    self._respond(
                        {
                            "ok": True,
                            "run_id": run_id,
                            "metadata": snapshot.metadata(),
                            "arrays": specs,
                        }
                    )
                    for name in ARRAY_NAMES:
                        array = getattr(snapshot, name)
                        if array.nbytes:
                            self.wfile.write(memoryview(array).cast("B"))
                    self.wfile.flush()
                return
            if operation == "stop":
                self._respond({"ok": True, "state": "stopping"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
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
            self._respond(
                {
                    "ok": False,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }
            )

    def _infer(self, request):
        reject_inline_render(request)
        return WorkerInference(
            self.server.estimator, self.server.catalog, self.server.snapshots
        ).infer(request)

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
        raise RuntimeError(
            "another inference worker is already starting/running"
        ) from exc

    if os.path.lexists(WORKER_SOCKET):
        os.unlink(WORKER_SOCKET)
    _write_pid()

    server = None
    try:
        catalog = PromptCatalog.load(verify_engine=True, require_full_pipeline=True)
        expected_artifact_id = catalog.manifest["artifact_id"]
        DEFAULT_ESTIMATOR.load()
        loaded_catalog = DEFAULT_PIPELINE._vision._catalog
        if loaded_catalog.manifest["artifact_id"] != expected_artifact_id:
            raise RuntimeError(
                "YOLOE prompt artifact changed during worker startup; "
                "retry scripts/worker.sh start"
            )
        catalog = loaded_catalog
        DEFAULT_ESTIMATOR.warmup()
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
