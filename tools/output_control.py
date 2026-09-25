#!/usr/bin/env python3
"""Small standard-library-only launcher for optional image rendering."""

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DIR = os.environ.get("GRASP_RUNTIME_DIR", os.path.join(ROOT, ".runtime"))
SOCKET_PATH = os.environ.get(
    "GRASP_WORKER_SOCKET", os.path.join(RUNTIME_DIR, "worker.sock"))
JOB_DIR = os.path.join(RUNTIME_DIR, "output-jobs")
OUTPUT_DIR = os.path.abspath(os.environ.get(
    "OUTPUT_DIR", os.path.join(ROOT, "output")))


def _run_id(value):
    if not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("RUN_ID must be the 32-character ID from infer.sh")
    return value


def _status(run_id):
    path = os.path.join(JOB_DIR, run_id + ".json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _check_snapshot(run_id):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(3)
        client.connect(SOCKET_PATH)
        client.sendall((json.dumps({
            "op": "snapshot_status", "run_id": run_id,
        }) + "\n").encode("utf-8"))
        with client.makefile("rb") as stream:
            line = stream.readline(1024 * 1024)
    if not line:
        raise RuntimeError("worker closed the connection")
    response = json.loads(line.decode("utf-8"))
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "snapshot unavailable"))


def start(run_id, retry=False, output_dir=None):
    run_id = _run_id(run_id)
    output_dir = os.path.abspath(output_dir or OUTPUT_DIR)
    path = os.path.join(JOB_DIR, run_id + ".json")
    if os.path.isfile(path):
        previous = _status(run_id)
        if retry and previous.get("state") == "failed":
            os.unlink(path)
        elif retry:
            raise ValueError("only failed output jobs can be retried")
        else:
            return previous
    _check_snapshot(run_id)
    os.makedirs(output_dir, exist_ok=True)
    if not os.access(output_dir, os.W_OK | os.X_OK):
        raise PermissionError("output directory is not writable: %s" % output_dir)
    os.makedirs(JOB_DIR, exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as handle:
            json.dump({
                "state": "queued", "run_id": run_id,
                "output_dir": os.path.join(output_dir, run_id),
            }, handle)
    except FileExistsError:
        return _status(run_id)
    log_path = os.path.join(JOB_DIR, run_id + ".log")
    try:
        with open(log_path, "ab") as log_handle:
            child = subprocess.Popen(
                [sys.executable, "-m", "grasppose.output_renderer",
                 run_id, output_dir],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except Exception:
        os.unlink(path)
        raise
    threading.Thread(target=child.wait, daemon=True).start()
    return {
        "state": "queued", "run_id": run_id,
        "pid": child.pid,
        "output_dir": os.path.join(output_dir, run_id),
    }


def wait(run_id, timeout=120):
    run_id = _run_id(run_id)
    deadline = time.monotonic() + timeout
    while True:
        result = _status(run_id)
        if result.get("state") in ("done", "failed"):
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError("render job has not completed in %d s" % timeout)
        time.sleep(0.1)


def main(argv=None):
    os.umask(0o077)
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 1:
        command, run_id = "start", argv[0]
    elif len(argv) == 2 and argv[0] in (
            "start", "status", "wait", "retry"):
        command, run_id = argv
    else:
        print("Usage: get_output.sh RUN_ID | status RUN_ID | wait RUN_ID | retry RUN_ID",
              file=sys.stderr)
        return 2
    try:
        run_id = _run_id(run_id)
        if command in ("start", "retry"):
            result = start(run_id, retry=command == "retry")
        elif command == "status":
            result = _status(run_id)
        else:
            result = wait(run_id)
        print(json.dumps(result, separators=(",", ":")))
        return 1 if result.get("state") == "failed" else 0
    except (OSError, ValueError, RuntimeError) as exc:
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
