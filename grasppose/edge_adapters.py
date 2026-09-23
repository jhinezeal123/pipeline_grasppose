"""Jetson-oriented model adapters: YOLOE-26s, Lite-Mono and VGN TensorRT."""

import importlib
import os
import sys
import time

import numpy as np

from .config import (
    LITEMONO_DEPTH_SCALE, LITEMONO_HOME, LITEMONO_MODEL, LITEMONO_WEIGHTS,
    VGN_ENGINE, VGN_QUAL_THRESHOLD, YOLOE_CONF, YOLOE_IMGSZ, YOLOE_MODEL,
)
from .runtime import _cuda, _free, _log
from .vgn import vgn_to_graspgroup


class Yoloe26sVision:
    """YOLOE-26s open-vocabulary detection + instance segmentation in one pass."""

    def __init__(self, model_path=None, device=None, conf=None, imgsz=None):
        self.model_path = model_path or YOLOE_MODEL
        self.device = device or (0 if _cuda() else "cpu")
        self.conf = YOLOE_CONF if conf is None else float(conf)
        self.imgsz = YOLOE_IMGSZ if imgsz is None else int(imgsz)
        self.model = None
        self._image = None
        self._prompt = None
        self._result = None

    def prepare(self, image, prompt):
        from ultralytics import YOLOE
        self._image = np.asarray(image)[:, :, :3]
        self._prompt = str(prompt).strip() or "object"
        if self.model is None:
            t0 = time.time()
            self.model = YOLOE(self.model_path)
            _log("YOLOE-26s loaded in %.1fs" % (time.time() - t0))
        self.model.set_classes([self._prompt])
        self._result = self.model.predict(
            source=self._image, conf=self.conf, imgsz=self.imgsz,
            device=self.device, half=bool(_cuda()), retina_masks=True,
            verbose=False,
        )[0]
        return self

    def inference(self):
        h, w = self._image.shape[:2]
        empty_det = {"boxes": np.zeros((0, 4), np.float32),
                     "scores": np.zeros(0, np.float32), "labels": [], "reason": None}
        empty_seg = {"mask": np.zeros((h, w), bool),
                     "iou": np.zeros(0, np.float32), "best": -1,
                     "n_pred": 0, "reason": None}
        r = self._result
        if r is None or r.boxes is None or len(r.boxes) == 0:
            reason = "YOLOE did not find an object for prompt %r" % self._prompt
            empty_det["reason"] = empty_seg["reason"] = reason
            return {"det": empty_det, "seg": empty_seg}
        boxes = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]
        det = {"boxes": boxes, "scores": scores,
               "labels": [self._prompt for _ in order], "reason": None}
        if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
            empty_seg["reason"] = "YOLOE returned boxes but no instance mask"
            return {"det": det, "seg": empty_seg}
        masks = r.masks.data.detach().cpu().numpy()[order]
        if masks.shape[-2:] != (h, w):
            import cv2
            masks = np.stack([cv2.resize(m.astype(np.float32), (w, h),
                            interpolation=cv2.INTER_NEAREST) for m in masks])
        seg = {"mask": masks[0] > 0.5, "iou": scores.copy(), "best": 0,
               "n_pred": int(len(masks)), "reason": None}
        return {"det": det, "seg": seg}

    def release(self):
        _free(self, "model", "_result")


class LiteMonoDepth:
    """Lite-Mono monocular depth with an explicit metric scale calibration."""

    def __init__(self, weights=None, home=None, model_name=None, device=None,
                 depth_scale=None):
        self.model_path = weights or LITEMONO_WEIGHTS
        self.home = home or LITEMONO_HOME
        self.model_name = model_name or LITEMONO_MODEL
        self.device = device or ("cuda" if _cuda() else "cpu")
        self.depth_scale = LITEMONO_DEPTH_SCALE if depth_scale is None else float(depth_scale)
        self._scale_is_default = depth_scale is None and "LITEMONO_DEPTH_SCALE" not in os.environ
        self.encoder = self.decoder = self._layers = None
        self._image = self._K = self._input = self._feed_hw = None

    def _load(self):
        import torch
        if not os.path.isdir(self.home):
            raise RuntimeError("Lite-Mono source not found at %r" % self.home)
        enc_path = os.path.join(self.model_path, "encoder.pth")
        dec_path = os.path.join(self.model_path, "depth.pth")
        if not (os.path.isfile(enc_path) and os.path.isfile(dec_path)):
            raise RuntimeError("Lite-Mono weights need encoder.pth + depth.pth in %r" % self.model_path)
        if self.home not in sys.path:
            sys.path.insert(0, self.home)
        networks = importlib.import_module("networks")
        layers = importlib.import_module("layers")
        enc_ckpt, dec_ckpt = _torch_load(torch, enc_path), _torch_load(torch, dec_path)
        feed_h, feed_w = int(enc_ckpt["height"]), int(enc_ckpt["width"])
        encoder = networks.LiteMono(model=self.model_name, height=feed_h, width=feed_w)
        own = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in enc_ckpt.items() if k in own})
        decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
        own = decoder.state_dict()
        decoder.load_state_dict({k: v for k, v in dec_ckpt.items() if k in own})
        encoder.to(self.device).eval(); decoder.to(self.device).eval()
        if self.device.startswith("cuda"):
            encoder.half(); decoder.half()
        self.encoder, self.decoder, self._layers = encoder, decoder, layers
        self._feed_hw = (feed_h, feed_w)

    def prepare(self, image, camera_K=None, fov_x=None):
        from PIL import Image
        from torchvision import transforms
        self._image = np.asarray(image)[:, :, :3]
        h, w = self._image.shape[:2]
        self._K = _resolve_K(camera_K, fov_x, w, h)
        if self.encoder is None:
            t0 = time.time(); self._load()
            _log("Lite-Mono loaded in %.1fs (%s)" % (time.time() - t0, self.device))
            if self._scale_is_default:
                _log("WARNING: Lite-Mono is monocular/scale-ambiguous; calibrate LITEMONO_DEPTH_SCALE before metric TSDF/VGN use")
        feed_h, feed_w = self._feed_hw
        pil = Image.fromarray(self._image.astype(np.uint8)).resize((feed_w, feed_h), Image.LANCZOS)
        tensor = transforms.ToTensor()(pil).unsqueeze(0).to(self.device)
        if self.device.startswith("cuda"):
            tensor = tensor.half()
        self._input = tensor
        return self

    def inference(self):
        import torch
        import torch.nn.functional as F
        if self._input is None:
            raise RuntimeError("LiteMonoDepth.prepare() must run before inference()")
        h, w = self._image.shape[:2]
        with torch.inference_mode():
            disp = self.decoder(self.encoder(self._input))[("disp", 0)].float()
            disp = F.interpolate(disp, (h, w), mode="bilinear", align_corners=False)
            _, depth = self._layers.disp_to_depth(disp, 0.1, 100.0)
            depth = depth.squeeze().cpu().numpy().astype(np.float32)
        depth *= self.depth_scale
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[(depth < 0.05) | (depth > 10.0)] = 0.0
        fx = float(self._K[0, 0])
        fovx = float(2.0 * np.degrees(np.arctan(w / (2.0 * max(fx, 1e-6)))))
        return {"depth": depth, "intrinsics": self._K.astype(np.float32),
                "fov_x_deg": fovx, "scale": float(self.depth_scale), "reason": None}

    def release(self):
        _free(self, "encoder", "decoder", "_input")
        self._layers = None


class VgnTensorRT:
    """VGN TensorRT engine wrapper supporting TensorRT 8.x and 10.x APIs."""

    def __init__(self, engine_path=None, threshold=None):
        self.model_path = engine_path or VGN_ENGINE
        self.threshold = VGN_QUAL_THRESHOLD if threshold is None else float(threshold)
        self.engine = self.context = self._trt = self._input = self._meta = None

    def _load(self):
        import tensorrt as trt
        if not os.path.isfile(self.model_path):
            raise RuntimeError("VGN TensorRT engine not found at %r; build it on the Jetson for its TensorRT version" % self.model_path)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as fh:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(fh.read())
        if engine is None:
            raise RuntimeError("failed to deserialize VGN TensorRT engine")
        self.engine, self.context, self._trt = engine, engine.create_execution_context(), trt

    def prepare(self, tsdf_grid, voxel_size, T_cam_volume):
        grid = np.asarray(tsdf_grid, np.float32)
        if grid.shape != (1, 40, 40, 40):
            raise ValueError("VGN expects TSDF shape (1,40,40,40), got %r" % (grid.shape,))
        if self.engine is None:
            self._load()
        self._input = np.ascontiguousarray(grid[None], dtype=np.float32)
        self._meta = (float(voxel_size), np.asarray(T_cam_volume, np.float32))
        return self

    def _execute(self):
        import torch
        trt = self._trt
        if not torch.cuda.is_available():
            raise RuntimeError("VGN TensorRT requires CUDA")
        stream = torch.cuda.current_stream()
        if hasattr(self.engine, "num_io_tensors"):
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            if len(inputs) != 1 or len(outputs) != 3:
                raise RuntimeError("VGN engine must expose 1 input and 3 outputs")
            inp_name = inputs[0]
            inp = torch.from_numpy(self._input).to(device="cuda", dtype=_torch_dtype(trt, self.engine.get_tensor_dtype(inp_name)))
            self.context.set_input_shape(inp_name, tuple(inp.shape))
            tensors = {inp_name: inp}
            for name in outputs:
                shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
                tensors[name] = torch.empty(shape, dtype=_torch_dtype(trt, self.engine.get_tensor_dtype(name)), device="cuda")
            for name, tensor in tensors.items():
                self.context.set_tensor_address(name, int(tensor.data_ptr()))
            if not self.context.execute_async_v3(stream_handle=int(stream.cuda_stream)):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            stream.synchronize()
            return {name: tensors[name].float().cpu().numpy() for name in outputs}
        nb = self.engine.num_bindings
        input_ids = [i for i in range(nb) if self.engine.binding_is_input(i)]
        output_ids = [i for i in range(nb) if not self.engine.binding_is_input(i)]
        if len(input_ids) != 1 or len(output_ids) != 3:
            raise RuntimeError("VGN engine must expose 1 input and 3 outputs")
        ii = input_ids[0]
        inp = torch.from_numpy(self._input).to(device="cuda", dtype=_torch_dtype(trt, self.engine.get_binding_dtype(ii)))
        if hasattr(self.context, "set_binding_shape"):
            self.context.set_binding_shape(ii, tuple(inp.shape))
        bindings = [0] * nb; bindings[ii] = int(inp.data_ptr()); outs = {}
        for i in output_ids:
            shape = tuple(int(x) for x in self.context.get_binding_shape(i))
            t = torch.empty(shape, dtype=_torch_dtype(trt, self.engine.get_binding_dtype(i)), device="cuda")
            bindings[i] = int(t.data_ptr()); outs[self.engine.get_binding_name(i)] = t
        if not self.context.execute_async_v2(bindings=bindings, stream_handle=int(stream.cuda_stream)):
            raise RuntimeError("TensorRT execute_async_v2 failed")
        stream.synchronize()
        return {name: tensor.float().cpu().numpy() for name, tensor in outs.items()}

    def inference(self):
        qual, rot, width = _classify_vgn_outputs(self._execute())
        voxel_size, T_cam_volume = self._meta
        gg = vgn_to_graspgroup(self._input.squeeze(0), qual, rot, width,
                               voxel_size, T_cam_volume, threshold=self.threshold)
        return {"graspgroup": gg, "reason": None}

    def release(self):
        self.engine = self.context = self._trt = self._input = self._meta = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.empty_cache()
        except Exception:
            pass


def _torch_load(torch_module, path):
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def _resolve_K(camera_K, fov_x, width, height):
    if camera_K is not None:
        K = np.asarray(camera_K, np.float64).reshape(3, 3)
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("camera_K must have positive fx/fy")
        return K
    env = os.environ.get("CAMERA_K", "").strip()
    if env:
        vals = [float(x) for x in env.replace(",", " ").split()]
        if len(vals) != 4:
            raise ValueError("CAMERA_K must be 'fx fy cx cy'")
        fx, fy, cx, cy = vals
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], np.float64)
    if fov_x is not None:
        fx = width / (2.0 * np.tan(np.radians(float(fov_x)) / 2.0))
        return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]], np.float64)
    raise ValueError("camera intrinsics are required: pass camera_K, set CAMERA_K='fx fy cx cy', or pass fov_x")


def _torch_dtype(trt, dtype):
    import torch
    np_dtype = np.dtype(trt.nptype(dtype))
    mapping = {np.dtype(np.float32): torch.float32, np.dtype(np.float16): torch.float16,
               np.dtype(np.int32): torch.int32, np.dtype(np.int8): torch.int8,
               np.dtype(np.bool_): torch.bool}
    if np_dtype not in mapping:
        raise TypeError("unsupported TensorRT dtype %s" % np_dtype)
    return mapping[np_dtype]


def _classify_vgn_outputs(outputs):
    items = [(name.lower(), np.asarray(value)) for name, value in outputs.items()]
    rot = next((v for n, v in items if "rot" in n or "quat" in n or "orient" in n), None)
    qual = next((v for n, v in items if "qual" in n or "score" in n), None)
    width = next((v for n, v in items if "width" in n), None)
    if rot is None:
        rot = next((v for _, v in items if 4 in v.shape), None)
    one_channel = [v for _, v in items if v is not rot and 1 in v.shape]
    if qual is None and one_channel: qual = one_channel.pop(0)
    if width is None and one_channel: width = one_channel.pop(0)
    if qual is None or rot is None or width is None:
        vals = [np.asarray(v) for v in outputs.values()]
        if len(vals) == 3: qual, rot, width = vals
        else: raise RuntimeError("cannot identify VGN TensorRT outputs")
    return qual.squeeze(), rot.squeeze(), width.squeeze()
