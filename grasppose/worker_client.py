"""Client for the local Unix-socket inference worker."""

import json
import os
import socket

from .config import WORKER_SOCKET


class WorkerError(RuntimeError):
    pass


def request_worker(payload, timeout=300.0, socket_path=None):
    path = socket_path or WORKER_SOCKET
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(path)
        client.sendall(
            (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        )
        stream = client.makefile("rb")
        line = stream.readline()
        if not line:
            raise WorkerError("inference worker closed the connection")
        response = json.loads(line.decode("utf-8"))
    except FileNotFoundError as exc:
        raise WorkerError(
            "inference worker is not running; start it with: bash cold.sh"
        ) from exc
    except socket.timeout as exc:
        raise WorkerError("timed out waiting for inference worker") from exc
    except OSError as exc:
        raise WorkerError("cannot connect to inference worker: %s" % exc) from exc
    finally:
        client.close()
    if not response.get("ok"):
        raise WorkerError(response.get("error", "inference worker failed"))
    return response


def infer_image(image_path, prompt, camera_k=None, fov_x=None, fov_y=None,
                top=1, max_width=0.080, timeout=300.0):
    image_path = os.path.abspath(image_path)
    payload = {
        "op": "infer",
        "image": image_path,
        "prompt": str(prompt),
        "camera_k": camera_k,
        "fov_x": fov_x,
        "fov_y": fov_y,
        "top": int(top),
        "max_width": float(max_width),
    }
    return request_worker(payload, timeout=timeout)
