"""TensorRT VGN adapter implementing the grasp port."""

import os
import time

import numpy as np

from ..config import VGN_ENGINE, VGN_QUAL_THRESHOLD
from ..domain.types import GraspResult
from ..domain.vgn import vgn_to_graspgroup
from ..ports.grasp import GraspPort
from ..runtime import log, release_attributes


class VgnTensorRT(GraspPort):
    """Persistent TensorRT VGN engine supporting TRT 8.x and 10.x APIs."""

    def __init__(self, engine_path=None, threshold=None):
        self.model_path = engine_path or VGN_ENGINE
        self.threshold = (
            VGN_QUAL_THRESHOLD if threshold is None
            else float(threshold)
        )
        self._engine = None
        self._context = None
        self._trt = None

    def load(self):
        if self._engine is not None:
            return self

        import tensorrt as trt

        if not os.path.isfile(self.model_path):
            raise RuntimeError(
                "VGN TensorRT engine not found at %r; "
                "run prepare.sh on this Jetson" % self.model_path
            )

        started = time.time()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as handle:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(handle.read())
        if engine is None:
            raise RuntimeError(
                "failed to deserialize VGN TensorRT engine")

        self._engine = engine
        self._context = engine.create_execution_context()
        self._trt = trt
        log("VGN TensorRT engine loaded in %.1fs" % (
            time.time() - started))
        return self

    def predict(self, tsdf):
        self.load()
        grid = np.asarray(tsdf.grid, np.float32)
        if grid.shape != (1, 40, 40, 40):
            raise ValueError(
                "VGN expects TSDF shape (1,40,40,40), got %r"
                % (grid.shape,)
            )

        input_array = np.ascontiguousarray(
            grid[None], dtype=np.float32)
        quality, rotation, width = classify_vgn_outputs(
            self._execute(input_array))
        graspgroup = vgn_to_graspgroup(
            grid,
            quality,
            rotation,
            width,
            tsdf.voxel_size,
            tsdf.T_cam_volume,
            threshold=self.threshold,
        )
        return GraspResult(graspgroup=graspgroup)

    def _execute(self, input_array):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("VGN TensorRT requires CUDA")

        trt = self._trt
        engine = self._engine
        context = self._context
        stream = torch.cuda.current_stream()

        if hasattr(engine, "num_io_tensors"):
            names = [
                engine.get_tensor_name(i)
                for i in range(engine.num_io_tensors)
            ]
            inputs = [
                name for name in names
                if engine.get_tensor_mode(name)
                == trt.TensorIOMode.INPUT
            ]
            outputs = [
                name for name in names
                if engine.get_tensor_mode(name)
                == trt.TensorIOMode.OUTPUT
            ]
            if len(inputs) != 1 or len(outputs) != 3:
                raise RuntimeError(
                    "VGN engine must expose 1 input and 3 outputs")

            input_name = inputs[0]
            input_tensor = torch.from_numpy(input_array).to(
                device="cuda",
                dtype=_torch_dtype(
                    trt, engine.get_tensor_dtype(input_name)),
            )
            context.set_input_shape(
                input_name, tuple(input_tensor.shape))
            tensors = {input_name: input_tensor}

            for name in outputs:
                shape = tuple(
                    int(x)
                    for x in context.get_tensor_shape(name)
                )
                tensors[name] = torch.empty(
                    shape,
                    dtype=_torch_dtype(
                        trt, engine.get_tensor_dtype(name)),
                    device="cuda",
                )

            for name, tensor in tensors.items():
                context.set_tensor_address(
                    name, int(tensor.data_ptr()))

            if not context.execute_async_v3(
                    stream_handle=int(stream.cuda_stream)):
                raise RuntimeError(
                    "TensorRT execute_async_v3 failed")

            stream.synchronize()
            return {
                name: tensors[name].float().cpu().numpy()
                for name in outputs
            }

        binding_count = engine.num_bindings
        input_ids = [
            i for i in range(binding_count)
            if engine.binding_is_input(i)
        ]
        output_ids = [
            i for i in range(binding_count)
            if not engine.binding_is_input(i)
        ]
        if len(input_ids) != 1 or len(output_ids) != 3:
            raise RuntimeError(
                "VGN engine must expose 1 input and 3 outputs")

        input_id = input_ids[0]
        input_tensor = torch.from_numpy(input_array).to(
            device="cuda",
            dtype=_torch_dtype(
                trt, engine.get_binding_dtype(input_id)),
        )
        if hasattr(context, "set_binding_shape"):
            context.set_binding_shape(
                input_id, tuple(input_tensor.shape))

        bindings = [0] * binding_count
        bindings[input_id] = int(input_tensor.data_ptr())
        outputs = {}

        for output_id in output_ids:
            shape = tuple(
                int(x)
                for x in context.get_binding_shape(output_id)
            )
            tensor = torch.empty(
                shape,
                dtype=_torch_dtype(
                    trt, engine.get_binding_dtype(output_id)),
                device="cuda",
            )
            bindings[output_id] = int(tensor.data_ptr())
            outputs[engine.get_binding_name(output_id)] = tensor

        if not context.execute_async_v2(
                bindings=bindings,
                stream_handle=int(stream.cuda_stream)):
            raise RuntimeError(
                "TensorRT execute_async_v2 failed")

        stream.synchronize()
        return {
            name: tensor.float().cpu().numpy()
            for name, tensor in outputs.items()
        }

    def close(self):
        release_attributes(
            self, "_engine", "_context", "_trt")


def _torch_dtype(trt, dtype):
    import torch

    np_dtype = np.dtype(trt.nptype(dtype))
    mapping = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.bool_): torch.bool,
    }
    if np_dtype not in mapping:
        raise TypeError(
            "unsupported TensorRT dtype %s" % np_dtype)
    return mapping[np_dtype]


def classify_vgn_outputs(outputs):
    """Identify quality, quaternion and width tensors from TRT outputs."""
    items = [
        (name.lower(), np.asarray(value))
        for name, value in outputs.items()
    ]
    rotation = next(
        (value for name, value in items
         if "rot" in name or "quat" in name or "orient" in name),
        None,
    )
    quality = next(
        (value for name, value in items
         if "qual" in name or "score" in name),
        None,
    )
    width = next(
        (value for name, value in items if "width" in name),
        None,
    )

    if rotation is None:
        rotation = next(
            (value for _, value in items if 4 in value.shape),
            None,
        )

    one_channel = [
        value for _, value in items
        if value is not rotation and 1 in value.shape
    ]
    if quality is None and one_channel:
        quality = one_channel.pop(0)
    if width is None and one_channel:
        width = one_channel.pop(0)

    if quality is None or rotation is None or width is None:
        values = [np.asarray(value) for value in outputs.values()]
        if len(values) != 3:
            raise RuntimeError(
                "cannot identify VGN TensorRT outputs")
        quality, rotation, width = values

    return quality.squeeze(), rotation.squeeze(), width.squeeze()
