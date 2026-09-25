"""Lite-Mono TensorRT adapter with the checkpoint's fixed input shape."""

import json
import os
import re

import numpy as np
from PIL import Image

from ..artifacts import verify_sha256
from ..config import (
    LITEMONO_ARTIFACT_ROOT,
    LITEMONO_CURRENT_FILE,
    LITEMONO_DEPTH_SCALE,
    LITEMONO_MODEL,
    LITEMONO_WEIGHTS,
)
from ..domain.geometry import resolve_camera_intrinsics
from ..domain.types import DepthResult
from ..ports.depth import DepthPort
from ..runtime import log
from ..trt_engine import TensorRTEngine


class LiteMonoDepth(DepthPort):
    """Static TensorRT encoder+decoder graph; no Torch model fallback."""

    def __init__(self, engine_path=None, depth_scale=None):
        self.engine_path = engine_path
        self.depth_scale = (
            LITEMONO_DEPTH_SCALE if depth_scale is None else float(depth_scale)
        )
        self._engine = None
        self._feed_hw = None
        self._manifest = None

    def load(self):
        if self._engine is not None:
            return self
        if not os.path.isfile(LITEMONO_CURRENT_FILE):
            raise RuntimeError(
                "Lite-Mono TensorRT artifacts missing; run prepare.sh")
        with open(LITEMONO_CURRENT_FILE, "r", encoding="utf-8") as handle:
            artifact_id = handle.read().strip()
        if not re.fullmatch(r"[a-f0-9]{64}", artifact_id):
            raise RuntimeError("invalid Lite-Mono CURRENT artifact pointer")
        artifact_dir = os.path.join(LITEMONO_ARTIFACT_ROOT, artifact_id)
        manifest_path = os.path.join(artifact_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise RuntimeError("Lite-Mono manifest missing: %s" % manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("artifact_id") != artifact_id:
            raise RuntimeError("Lite-Mono artifact ID does not match manifest")
        if manifest.get("schema_version") != 1:
            raise RuntimeError("unsupported Lite-Mono artifact manifest schema")
        if manifest.get("model", {}).get("name") != LITEMONO_MODEL:
            raise RuntimeError("Lite-Mono model name does not match artifact")
        weights_dir = LITEMONO_WEIGHTS
        for name in ("encoder.pth", "depth.pth"):
            expected = manifest.get("weights", {}).get(name)
            verify_sha256(os.path.join(weights_dir, name), expected, "Lite-Mono weight")
        engine = manifest.get("engine", {})
        engine_name = engine.get("file")
        if not isinstance(engine_name, str) or os.path.basename(engine_name) != engine_name:
            raise RuntimeError("invalid Lite-Mono engine path in manifest")
        engine_path = self.engine_path or os.path.join(
            artifact_dir, engine_name)
        self._engine = TensorRTEngine(
            engine_path,
            expected_sha256=engine.get("sha256"),
            label="Lite-Mono TensorRT",
        ).load()
        self._feed_hw = tuple(int(x) for x in manifest["input"]["hw"])
        self._manifest = manifest
        log("Lite-Mono TensorRT precision: %s" % engine["precision"])
        return self

    def predict(self, image, camera_K=None, fov_x=None):
        self.load()
        import torch
        import torch.nn.functional as F

        rgb = np.asarray(image)[:, :, :3]
        height, width = rgb.shape[:2]
        K = resolve_camera_intrinsics(camera_K, fov_x, width, height)
        feed_h, feed_w = self._feed_hw
        resized = Image.fromarray(rgb.astype(np.uint8)).resize(
            (feed_w, feed_h), Image.LANCZOS)
        input_array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)
        input_array = np.ascontiguousarray(input_array[None] / 255.0)
        disparity = self._engine.infer_cuda(input_array)["disp"].float()
        disparity = F.interpolate(
            disparity,
            (height, width),
            mode="bilinear",
            align_corners=False,
        )
        min_disp = float(self._manifest["depth"]["min_disp"])
        max_disp = float(self._manifest["depth"]["max_disp"])
        depth = 1.0 / (min_disp + (max_disp - min_disp) * disparity)
        depth = depth.squeeze().detach().cpu().numpy().astype(np.float32)
        depth *= self.depth_scale
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[(depth < 0.05) | (depth > 10.0)] = 0.0

        fx = float(K[0, 0])
        fov_x_deg = float(2.0 * np.degrees(np.arctan(
            width / (2.0 * max(fx, 1e-6)))))
        return DepthResult(
            depth=depth,
            intrinsics=K.astype(np.float32),
            fov_x_deg=fov_x_deg,
            scale=float(self.depth_scale),
        )

    def warmup(self):
        self.load()
        feed_h, feed_w = self._feed_hw
        self._engine.warmup((1, 3, feed_h, feed_w))

    def close(self):
        if self._engine is not None:
            self._engine.close()
        self._engine = None
        self._feed_hw = None
        self._manifest = None
