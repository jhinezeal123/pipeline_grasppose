"""Client for the local Unix-socket inference worker."""

import os

from ...application.interface import GraspEstimator
from ...application.types import EstimateResult, GraspPose
from ..settings import RUNTIME_DIR
from .rpc import WorkerError, infer_image, request_worker



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
