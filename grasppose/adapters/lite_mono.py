"""Lite-Mono adapter implementing the depth port."""

import importlib
import os
import sys
import time

import numpy as np

from ..config import (
    LITEMONO_DEPTH_SCALE,
    LITEMONO_HOME,
    LITEMONO_MODEL,
    LITEMONO_WEIGHTS,
)
from ..domain.geometry import resolve_camera_intrinsics
from ..domain.types import DepthResult
from ..ports.depth import DepthPort
from ..runtime import cuda_available, log, release_attributes


class LiteMonoDepth(DepthPort):
    """Monocular depth with an explicit metric scale calibration."""

    def __init__(self, weights=None, home=None, model_name=None,
                 device=None, depth_scale=None):
        self.model_path = weights or LITEMONO_WEIGHTS
        self.home = home or LITEMONO_HOME
        self.model_name = model_name or LITEMONO_MODEL
        self.device = device or (
            "cuda" if cuda_available() else "cpu")
        self.depth_scale = (
            LITEMONO_DEPTH_SCALE if depth_scale is None
            else float(depth_scale)
        )
        self._scale_is_default = (
            depth_scale is None and
            "LITEMONO_DEPTH_SCALE" not in os.environ
        )
        self._encoder = None
        self._decoder = None
        self._layers = None
        self._feed_hw = None

    def load(self):
        if self._encoder is not None:
            return self

        import torch

        if not os.path.isdir(self.home):
            raise RuntimeError(
                "Lite-Mono source not found at %r" % self.home)
        encoder_path = os.path.join(
            self.model_path, "encoder.pth")
        decoder_path = os.path.join(
            self.model_path, "depth.pth")
        if not (os.path.isfile(encoder_path) and
                os.path.isfile(decoder_path)):
            raise RuntimeError(
                "Lite-Mono weights need encoder.pth + depth.pth in %r"
                % self.model_path
            )

        if self.home not in sys.path:
            sys.path.insert(0, self.home)

        networks = importlib.import_module("networks")
        layers = importlib.import_module("layers")
        encoder_checkpoint = _torch_load(torch, encoder_path)
        decoder_checkpoint = _torch_load(torch, decoder_path)
        feed_h = int(encoder_checkpoint["height"])
        feed_w = int(encoder_checkpoint["width"])

        started = time.time()
        encoder = networks.LiteMono(
            model=self.model_name, height=feed_h, width=feed_w)
        encoder_state = encoder.state_dict()
        encoder_weights = {
            key: value
            for key, value in encoder_checkpoint.items()
            if key in encoder_state
        }
        # Explicit strict=True guarantees that every model parameter is
        # present after filtering checkpoint metadata such as height/width.
        encoder.load_state_dict(
            encoder_weights, strict=True)

        decoder = networks.DepthDecoder(
            encoder.num_ch_enc, scales=range(3))
        decoder_state = decoder.state_dict()
        decoder_weights = {
            key: value
            for key, value in decoder_checkpoint.items()
            if key in decoder_state
        }
        decoder.load_state_dict(
            decoder_weights, strict=True)

        # Keep Lite-Mono in FP32. Its upstream positional encoding creates
        # explicit float32 tensors before Conv2d; blindly calling .half() on
        # the module can cause input/weight dtype mismatches on CUDA.
        encoder.to(self.device).eval()
        decoder.to(self.device).eval()

        self._encoder = encoder
        self._decoder = decoder
        self._layers = layers
        self._feed_hw = (feed_h, feed_w)
        log("Lite-Mono loaded in %.1fs (%s)" % (
            time.time() - started, self.device))
        if self._scale_is_default:
            log(
                "WARNING: Lite-Mono is monocular/scale-ambiguous; "
                "calibrate LITEMONO_DEPTH_SCALE before metric TSDF/VGN use"
            )
        return self

    def predict(self, image, camera_K=None, fov_x=None):
        from PIL import Image
        from torchvision import transforms
        import torch
        import torch.nn.functional as F

        self.load()
        rgb = np.asarray(image)[:, :, :3]
        height, width = rgb.shape[:2]
        K = resolve_camera_intrinsics(
            camera_K, fov_x, width, height)

        feed_h, feed_w = self._feed_hw
        pil = Image.fromarray(
            rgb.astype(np.uint8)).resize(
                (feed_w, feed_h), Image.LANCZOS)
        tensor = (
            transforms.ToTensor()(pil)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.inference_mode():
            disparity = self._decoder(
                self._encoder(tensor))[("disp", 0)].float()
            disparity = F.interpolate(
                disparity,
                (height, width),
                mode="bilinear",
                align_corners=False,
            )
            _, depth = self._layers.disp_to_depth(
                disparity, 0.1, 100.0)
            depth = (
                depth.squeeze().cpu().numpy()
                .astype(np.float32)
            )

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
        release_attributes(self, "_encoder", "_decoder")
        self._layers = None
        self._feed_hw = None


def _torch_load(torch_module, path):
    try:
        return torch_module.load(
            path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")
