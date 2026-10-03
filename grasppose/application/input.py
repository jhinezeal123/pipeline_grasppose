"""Chuẩn hóa request và camera trước inference; giữ nguyên phép tính cũ."""

import os
from dataclasses import dataclass

import numpy as np

from ..modules.depth.geometry import fov_x_from_fovy, scale_camera_intrinsics


@dataclass(frozen=True)
class EstimateInput:
    image: np.ndarray
    camera_K: object
    fov_x: object
    T_cam_volume: object

    @classmethod
    def from_values(
        cls, image, camera_K, fov_x, fov_y, camera_K_size, max_width, top, T_cam_volume
    ):
        if isinstance(image, (str, os.PathLike)):
            from PIL import Image

            image = np.asarray(Image.open(image).convert("RGB"))
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("input image must have shape (H,W,3+)")
        image = image[:, :, :3]
        if max_width <= 0 or int(top) < 1:
            raise ValueError("top and max_width must be positive")

        K = _camera_matrix(camera_K)
        if camera_K_size is not None:
            if K is None:
                raise ValueError("camera_K_size requires camera_K")
            K = scale_camera_intrinsics(
                K, camera_K_size, (image.shape[1], image.shape[0])
            )
        if fov_x is None and fov_y is not None:
            fov_x = fov_x_from_fovy(fov_y, image.shape[1], image.shape[0])

        T = (
            None
            if T_cam_volume is None
            else np.asarray(T_cam_volume, np.float64).reshape(4, 4)
        )

        return cls(image, K, fov_x, T)


def _camera_matrix(camera_K):
    if camera_K is None:
        return None
    values = np.asarray(camera_K, np.float64)
    if values.size == 4:
        fx, fy, cx, cy = values.reshape(4)
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    return values.reshape(3, 3)
