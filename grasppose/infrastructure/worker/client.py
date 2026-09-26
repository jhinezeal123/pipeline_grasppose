"""Client for the local Unix-socket inference worker."""

import json
import os
import socket

from ...application.interface import GraspEstimator
from ...application.types import EstimateResult, GraspPose
from ..settings import RUNTIME_DIR, WORKER_SOCKET


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
            "inference worker is not running; start it with: bash scripts/worker.sh"
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


def infer_image(image_path, prompt_id, camera_k=None, fov_x=None,
                output_dir=None, top=1, max_width=0.080, timeout=300.0,
                render=False, fov_y=None, camera_k_size=None,
                socket_path=None, T_cam_volume=None):
    image_path = os.path.abspath(image_path)
    payload = {
        "op": "infer",
        "image": image_path,
        "prompt_id": str(prompt_id),
        "camera_k": camera_k,
        "camera_k_size": camera_k_size,
        "fov_x": fov_x,
        "fov_y": fov_y,
        "render": False,
        "output_dir": None,
        "top": int(top),
        "max_width": float(max_width),
        "T_cam_volume": T_cam_volume,
    }
    response = request_worker(payload, timeout=timeout, socket_path=socket_path)
    if render:
        if not response.get("snapshot_available") or not response.get("run_id"):
            raise WorkerError(
                "cannot render asynchronously: output snapshot was not retained"
            )
        from ..output.control import start
        try:
            response["render_job"] = start(
                response["run_id"], output_dir=output_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkerError("cannot queue output render: %s" % exc) from exc
    return response



class WorkerGraspEstimator(GraspEstimator):
    """GraspEstimator adapter backed by the resident Unix-socket worker."""

    def __init__(self, socket_path=None, timeout=300.0):
        self.socket_path = socket_path
        self.timeout = float(timeout)

    def estimate(self, image, prompt_id, camera_K=None, fov_x=None,
                 fov_y=None, camera_K_size=None, max_width=0.080,
                 top=1, T_cam_volume=None):
        path, cleanup = _image_path(image)
        try:
            camera_values = _camera_values(camera_K)
            response = infer_image(
                path,
                prompt_id,
                camera_k=camera_values,
                camera_k_size=camera_K_size,
                fov_x=fov_x,
                fov_y=fov_y,
                top=top,
                max_width=max_width,
                timeout=self.timeout,
                socket_path=self.socket_path,
                T_cam_volume=(
                    None if T_cam_volume is None
                    else _matrix4(T_cam_volume)
                ),
            )
        finally:
            if cleanup is not None:
                try:
                    os.unlink(cleanup)
                except OSError:
                    pass
        poses = tuple(
            GraspPose(
                score=float(item["score"]),
                width_m=float(item["width_m"]),
                translation_m=tuple(item["translation_m"]),
                rotation=tuple(tuple(row) for row in item["rotation"]),
            )
            for item in response.get("grasps", ())
        )
        return EstimateResult(
            grasps=poses,
            depth_m=response.get("depth_m"),
            detection_count=int(response.get("detection_count", 0)),
            mask_pixels=int(response.get("mask_pixels", 0)),
            grasp_count=int(response.get("grasp_count", 0)),
            request_id=response.get("run_id"),
            snapshot_available=bool(response.get("snapshot_available", False)),
            latency_ms=response.get("server_ms"),
        )


def _camera_values(camera_K):
    if camera_K is None:
        return None
    import numpy as np
    values = np.asarray(camera_K, dtype=np.float64)
    if values.size == 4:
        return values.reshape(4).tolist()
    matrix = values.reshape(3, 3)
    return [
        float(matrix[0, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
    ]


def _image_path(image):
    if isinstance(image, (str, os.PathLike)):
        return os.path.abspath(os.fspath(image)), None
    import tempfile
    import numpy as np
    from PIL import Image

    incoming = os.path.join(RUNTIME_DIR, "incoming")
    os.makedirs(incoming, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="estimate-", suffix=".png", dir=incoming)
    os.close(fd)
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        os.unlink(path)
        raise ValueError("input image must have shape (H,W,3+)")
    Image.fromarray(array[:, :, :3].astype(np.uint8)).save(path, format="PNG")
    return path, path



def _matrix4(value):
    import numpy as np
    return np.asarray(value, dtype=np.float64).reshape(4, 4).tolist()
