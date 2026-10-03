"""Cấu hình camera chung cho CLI và Gradio; chỉ dùng thư viện chuẩn."""

import os


class CameraSettings:
    """Đọc calibration; giá trị explicit luôn ưu tiên hơn environment."""

    def __init__(self, environ=None):
        self._environ = os.environ if environ is None else environ

    def matrix(self, explicit=None):
        if explicit is not None:
            return explicit
        configured = self._environ.get("CAMERA_K", "").split()
        if not configured:
            return None
        if len(configured) != 4:
            raise ValueError("CAMERA_K must contain FX FY CX CY")
        return [float(value) for value in configured]

    def size(self, explicit=None):
        if explicit is not None:
            return explicit
        configured = self._environ.get("CAMERA_K_SIZE", "").replace(",", " ").split()
        if not configured:
            return None
        if len(configured) != 2:
            raise ValueError("CAMERA_K_SIZE must contain WIDTH HEIGHT")
        return [int(value) for value in configured]
