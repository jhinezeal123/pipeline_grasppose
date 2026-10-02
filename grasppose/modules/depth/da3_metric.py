"""Depth Anything 3 metric-large adapter, running the ONNX graph on the CPU.

The checkpoint predicts canonical depth through a 300-pixel reference camera,
so metres come from one multiply by the feed focal length (upstream FAQ):

    metric = raw * focal_at_network_input / 300

Two deliberate choices, both measured on this project's Jetson AGX Xavier:

* The CPU provider is used by default. The TensorRT and CUDA providers return a
  silent *constant* for this fp16 graph: every pixel 0.9741, identical for a
  real photograph, a simulation frame, uniform grey and pure noise, with no
  error raised. The CPU provider produces a real depth map (0.174-2.697 over
  3610 distinct values).

* ``DA3_METRIC_SCALE`` corrects the upstream formula by an empirically measured
  factor. See the setting's comment: it is fitted on the simulation test bed and
  has NOT been validated against real-world ground truth.
"""

import os
import time

import numpy as np

from ...infrastructure.artifacts import verify_sha256
from ...infrastructure.runtime import log
from ...infrastructure.settings import (
    DA3_INPUT_SIZE,
    DA3_METRIC_SCALE,
    DA3_PROVIDER,
    DA3_WEIGHTS,
    DA3_WEIGHTS_SHA256,
)
from .geometry import resolve_camera_intrinsics
from .port import DepthPort
from .types import DepthResult

# ImageNet statistics, as required by the upstream preprocessing recipe.
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Upstream reference-camera focal length, in pixels.
_REFERENCE_FOCAL_PX = 300.0

# The sky head marks pixels where the depth prediction is meaningless. Zeros are
# used rather than infinities so that the downstream validity test (depth > 0.05)
# rejects them without special-casing.
_SKY_THRESHOLD = 0.5


class Da3MetricDepth(DepthPort):
    """Metric monocular depth from the DA3 metric-large ONNX graph."""

    def __init__(self, model_path=None, provider=None, metric_scale=None,
                 input_size=None):
        self.model_path = model_path or DA3_WEIGHTS
        self.provider = provider or DA3_PROVIDER
        self.metric_scale = (
            DA3_METRIC_SCALE if metric_scale is None else float(metric_scale)
        )
        self.input_size = int(input_size or DA3_INPUT_SIZE)
        self._session = None
        self._input_name = None
        self._output_names = None

    def load(self):
        if self._session is not None:
            return self

        path = os.path.abspath(self.model_path)
        if not os.path.isfile(path):
            raise RuntimeError(
                "Depth Anything 3 metric-large weights missing: %s" % path)
        verify_sha256(path, DA3_WEIGHTS_SHA256, "Depth Anything 3 metric weights")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Depth Anything 3 requires onnxruntime; run prepare.sh") from exc

        available = ort.get_available_providers()
        if self.provider not in available:
            raise RuntimeError(
                "requested provider %s is unavailable; have %s"
                % (self.provider, available))

        options = ort.SessionOptions()
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        options.log_severity_level = 3
        started = time.time()
        session = ort.InferenceSession(path, options, providers=[self.provider])

        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_names = [output.name for output in session.get_outputs()]
        if "depth" not in self._output_names:
            raise RuntimeError(
                "Depth Anything 3 graph has no 'depth' output: %s"
                % self._output_names)
        log(
            "Depth Anything 3 metric-large loaded on %s in %.2fs (input %s %s)"
            % (self.provider, time.time() - started,
               self._input_name, session.get_inputs()[0].shape)
        )
        return self

    def predict(self, image, camera_K=None, fov_x=None):
        self.load()
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError("Depth Anything 3 expects an HxWx3 RGB image")
        rgb = rgb[:, :, :3]
        if not np.issubdtype(rgb.dtype, np.integer):
            if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
                raise ValueError("RGB values must be finite and in [0, 255]")
            rgb = np.rint(rgb).astype(np.uint8)
        elif rgb.dtype != np.uint8:
            if rgb.min() < 0 or rgb.max() > 255:
                raise ValueError("RGB values must be in [0, 255]")
            rgb = rgb.astype(np.uint8)
        height, width = rgb.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("RGB image cannot be empty")

        K = resolve_camera_intrinsics(camera_K, fov_x, width, height)

        # The upstream recipe resizes the whole frame to the network input, so
        # the feed is aspect-squashed rather than cropped. A centre crop was
        # measured and is worse in absolute terms (cube error ~900 mm against
        # ~490 mm), even though its rank correlation is slightly higher.
        size = self.input_size
        feed = _resize(rgb, size, size)
        values = ((feed.astype(np.float32) / 255.0) - _MEAN) / _STD
        tensor = values.transpose(2, 0, 1)[None].astype(np.float32)

        outputs = self._session.run(None, {self._input_name: tensor})
        raw = outputs[self._output_names.index("depth")][0, 0].astype(np.float32)
        if "sky" in self._output_names:
            sky = outputs[self._output_names.index("sky")][0, 0].astype(np.float32)
        else:
            sky = np.zeros_like(raw)

        # Canonical -> metres. The focal length must be expressed in pixels at
        # the network input, which is the frame focal scaled by size / width.
        focal_at_input = float(K[0, 0]) * float(size) / float(width)
        depth = raw * (focal_at_input / _REFERENCE_FOCAL_PX) * self.metric_scale
        depth = _resize(depth, width, height)

        sky_full = _resize(sky, width, height)
        masked = int(np.count_nonzero(sky_full >= _SKY_THRESHOLD))
        if masked:
            depth = np.where(sky_full >= _SKY_THRESHOLD, 0.0, depth)

        if not np.isfinite(depth).all():
            raise RuntimeError("Depth Anything 3 returned non-finite distances")
        if np.count_nonzero(depth > 0.05) == 0:
            raise RuntimeError("Depth Anything 3 returned no usable depth")

        fx = float(K[0, 0])
        fov_x_deg = float(2.0 * np.degrees(np.arctan(
            width / (2.0 * max(fx, 1e-6)))))
        return DepthResult(
            depth=np.ascontiguousarray(depth, np.float32),
            intrinsics=K.astype(np.float32),
            fov_x_deg=fov_x_deg,
            scale=self.metric_scale,
            reason=("sky masked on %d px" % masked) if masked else None,
        )

    def warmup(self):
        self.load()
        self.predict(
            np.zeros((240, 320, 3), np.uint8),
            camera_K=np.array(
                [[320.0, 0.0, 160.0], [0.0, 320.0, 120.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
        )

    def close(self):
        self._session = None
        self._input_name = None
        self._output_names = None


def _resize(field, width, height):
    """Bilinear resize without pulling a heavy imaging dependency into import."""
    import cv2

    array = np.asarray(field, np.float32)
    if array.shape[0] == height and array.shape[1] == width:
        return array
    interpolation = cv2.INTER_AREA if (
        array.shape[0] > height or array.shape[1] > width) else cv2.INTER_LINEAR
    return cv2.resize(array, (int(width), int(height)),
                      interpolation=interpolation)
