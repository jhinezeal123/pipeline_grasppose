"""Lite-Mono Tiny TensorRT adapter implementing the depth port."""

import ctypes
import os
import time

import numpy as np

from ..config import (
    LITEMONO_DEPTH_SCALE,
    LITEMONO_ENGINE,
    LITEMONO_TRT_LIBRARY,
)
from ..domain.geometry import resolve_camera_intrinsics
from ..domain.types import DepthResult
from ..ports.depth import DepthPort
from ..runtime import log


_ERROR_CAP = 1024


def _native_error(buffer):
    text = buffer.value.decode("utf-8", "replace").strip()
    return text or "unknown native TensorRT error"


def _disp_to_depth(disparity, min_depth=0.1, max_depth=100.0):
    """Match Lite-Mono/Monodepth2 disp_to_depth without importing Torch."""
    min_disp = 1.0 / float(max_depth)
    max_disp = 1.0 / float(min_depth)
    scaled_disp = min_disp + (max_disp - min_disp) * disparity
    return 1.0 / np.maximum(scaled_disp, 1e-12)


class LiteMonoDepth(DepthPort):
    """Lite-Mono Tiny using a resident TensorRT engine on Jetson."""

    def __init__(self, engine=None, library=None, depth_scale=None):
        self.engine_path = engine or LITEMONO_ENGINE
        self.library_path = library or LITEMONO_TRT_LIBRARY
        self.depth_scale = (
            LITEMONO_DEPTH_SCALE if depth_scale is None
            else float(depth_scale)
        )
        self._scale_is_default = (
            depth_scale is None and
            "LITEMONO_DEPTH_SCALE" not in os.environ
        )
        self._lib = None
        self._handle = None
        self._feed_hw = None
        self._output_elements = None

    def _configure_native_api(self):
        char_ptr = ctypes.POINTER(ctypes.c_char)
        float_ptr = ctypes.POINTER(ctypes.c_float)

        self._lib.litemono_create.argtypes = [
            ctypes.c_char_p, char_ptr, ctypes.c_size_t]
        self._lib.litemono_create.restype = ctypes.c_void_p

        self._lib.litemono_get_input_hw.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            char_ptr,
            ctypes.c_size_t,
        ]
        self._lib.litemono_get_input_hw.restype = ctypes.c_int

        self._lib.litemono_get_output_elements.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            char_ptr,
            ctypes.c_size_t,
        ]
        self._lib.litemono_get_output_elements.restype = ctypes.c_int

        self._lib.litemono_infer.argtypes = [
            ctypes.c_void_p,
            float_ptr,
            float_ptr,
            char_ptr,
            ctypes.c_size_t,
        ]
        self._lib.litemono_infer.restype = ctypes.c_int

        self._lib.litemono_destroy.argtypes = [ctypes.c_void_p]
        self._lib.litemono_destroy.restype = None

    def load(self):
        if self._handle is not None:
            return self

        if not os.path.isfile(self.engine_path):
            raise RuntimeError(
                "Lite-Mono TensorRT engine not found at %r; run prepare.sh"
                % self.engine_path
            )
        if not os.path.isfile(self.library_path):
            raise RuntimeError(
                "Lite-Mono TensorRT native library not found at %r; "
                "run prepare.sh" % self.library_path
            )

        started = time.time()
        self._lib = ctypes.CDLL(self.library_path)
        self._configure_native_api()

        error = ctypes.create_string_buffer(_ERROR_CAP)
        handle = self._lib.litemono_create(
            os.fsencode(self.engine_path), error, len(error))
        if not handle:
            raise RuntimeError(
                "failed to load Lite-Mono TensorRT engine: %s"
                % _native_error(error)
            )

        try:
            height = ctypes.c_int()
            width = ctypes.c_int()
            error.value = b""
            if self._lib.litemono_get_input_hw(
                    handle,
                    ctypes.byref(height),
                    ctypes.byref(width),
                    error,
                    len(error)) != 0:
                raise RuntimeError(_native_error(error))

            elements = ctypes.c_size_t()
            error.value = b""
            if self._lib.litemono_get_output_elements(
                    handle,
                    ctypes.byref(elements),
                    error,
                    len(error)) != 0:
                raise RuntimeError(_native_error(error))

            expected = int(height.value) * int(width.value)
            if int(elements.value) != expected:
                raise RuntimeError(
                    "unexpected Lite-Mono disp_output size: %d != %d"
                    % (int(elements.value), expected)
                )
        except Exception:
            self._lib.litemono_destroy(handle)
            raise

        self._handle = handle
        self._feed_hw = (int(height.value), int(width.value))
        self._output_elements = int(elements.value)

        log("Lite-Mono Tiny TensorRT loaded in %.1fs (%dx%d)" % (
            time.time() - started,
            self._feed_hw[1],
            self._feed_hw[0],
        ))
        if self._scale_is_default:
            log(
                "WARNING: Lite-Mono is monocular/scale-ambiguous; "
                "calibrate LITEMONO_DEPTH_SCALE before metric TSDF/VGN use"
            )
        return self

    def _infer_disparity(self, tensor):
        output = np.empty(
            self._output_elements, dtype=np.float32)
        error = ctypes.create_string_buffer(_ERROR_CAP)
        rc = self._lib.litemono_infer(
            self._handle,
            tensor.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)),
            error,
            len(error),
        )
        if rc != 0:
            raise RuntimeError(
                "Lite-Mono TensorRT inference failed: %s"
                % _native_error(error)
            )
        return output

    def predict(self, image, camera_K=None, fov_x=None):
        from PIL import Image
        import cv2

        self.load()
        rgb = np.asarray(image)[:, :, :3]
        height, width = rgb.shape[:2]
        K = resolve_camera_intrinsics(
            camera_K, fov_x, width, height)

        feed_h, feed_w = self._feed_hw
        pil = Image.fromarray(
            rgb.astype(np.uint8)).resize(
                (feed_w, feed_h), Image.LANCZOS)
        # Official Lite-Mono inference uses RGB ToTensor(): [0,1], CHW.
        # Do not apply ImageNet mean/std normalization here.
        tensor = np.asarray(pil, dtype=np.float32) / 255.0
        tensor = np.ascontiguousarray(
            tensor.transpose(2, 0, 1)[None, ...],
            dtype=np.float32,
        )

        disparity = self._infer_disparity(tensor).reshape(
            feed_h, feed_w)
        disparity = cv2.resize(
            disparity,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)

        depth = _disp_to_depth(disparity)
        depth = depth.astype(np.float32, copy=False)
        depth *= self.depth_scale
        depth = np.nan_to_num(
            depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[(depth < 0.05) | (depth > 10.0)] = 0.0

        fx = float(K[0, 0])
        fov_x_deg = float(2.0 * np.degrees(np.arctan(
            width / (2.0 * max(fx, 1e-6))
        )))
        return DepthResult(
            depth=depth,
            intrinsics=K.astype(np.float32),
            fov_x_deg=fov_x_deg,
            scale=float(self.depth_scale),
        )

    def close(self):
        if self._handle is not None and self._lib is not None:
            self._lib.litemono_destroy(self._handle)
        self._handle = None
        self._lib = None
        self._feed_hw = None
        self._output_elements = None
