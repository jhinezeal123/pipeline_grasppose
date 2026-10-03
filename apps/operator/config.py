"""Profile riêng của menu, lưu atomic; không sửa settings của pipeline."""

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from apps.camera_settings import CameraSettings


@dataclass
class Profile:
    socket: str = ""
    camera_k: str = ""
    camera_k_size: str = ""
    backend: str = "da3"
    confidence: float = 0.20
    ui_port: int = 8082
    last_image: str = ""
    last_run: str = ""

    def environment(self):
        return {
            "GRASP_WORKER_SOCKET": self.socket,
            "GRASP_DEPTH_BACKEND": self.backend,
            "YOLOE_CONF": str(self.confidence),
            "CAMERA_K": self.camera_k,
            "CAMERA_K_SIZE": self.camera_k_size.replace(",", " "),
        }

    def validate(self):
        for name in (
            "socket",
            "camera_k",
            "camera_k_size",
            "backend",
            "last_image",
            "last_run",
        ):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"Cấu hình {name} phải là chuỗi.")
        if self.backend not in ("da3", "lite-mono"):
            raise ValueError("Backend phải là da3 hoặc lite-mono.")
        if not self.socket:
            raise ValueError("Socket worker không được để trống.")
        if (
            type(self.confidence) not in (int, float)
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("Confidence phải trong 0..1.")
        if type(self.ui_port) is not int or not 1 <= self.ui_port <= 65535:
            raise ValueError("Port giao diện phải trong 1..65535.")
        camera = CameraSettings(self.environment())
        matrix, size = camera.matrix(), camera.size()
        if matrix is not None and (
            not all(math.isfinite(v) for v in matrix)
            or matrix[0] <= 0
            or matrix[1] <= 0
        ):
            raise ValueError("Camera K cần FX/FY dương và bốn số hữu hạn.")
        if size is not None and (matrix is None or min(size) <= 0):
            raise ValueError("Resolution cần K và hai kích thước dương.")


class ProfileStore:
    def __init__(self, root, path=None):
        self.root = Path(root)
        self.path = Path(path) if path else self.root / ".runtime" / "operator.json"

    def load(self):
        profile = Profile(
            socket=os.environ.get(
                "GRASP_WORKER_SOCKET", str(self.root / ".runtime" / "worker.sock")
            ),
            camera_k=os.environ.get("CAMERA_K", ""),
            camera_k_size=os.environ.get("CAMERA_K_SIZE", ""),
            backend=os.environ.get("GRASP_DEPTH_BACKEND", "da3"),
            confidence=float(os.environ.get("YOLOE_CONF", "0.20")),
        )
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.pop("version", None) != 1:
                raise ValueError(f"Profile không đúng version 1: {self.path}")
            allowed = {f.name for f in fields(Profile)}
            if not set(data).issubset(allowed):
                raise ValueError(f"Profile có trường không được hỗ trợ: {self.path}")
            profile = Profile(**{**asdict(profile), **data})
        profile.validate()
        return profile

    def save(self, profile):
        profile.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(self.path.parent), delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump(
                    {"version": 1, **asdict(profile)},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
            os.replace(str(temporary), str(self.path))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
