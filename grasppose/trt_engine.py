"""Small static-shape TensorRT runner using Torch only for CUDA buffers."""

import os
import time

import numpy as np

from .artifacts import verify_sha256
from .runtime import log


class TensorRTEngine:
    def __init__(self, engine_path, expected_sha256=None, label="TensorRT engine"):
        self.engine_path = engine_path
        self.expected_sha256 = expected_sha256
        self.label = label
        self._runtime = None
        self._engine = None
        self._context = None
        self._trt = None
        self._inputs = ()
        self._outputs = ()

    def load(self):
        if self._engine is not None:
            return self
        if not os.path.isfile(self.engine_path):
            raise RuntimeError("%s is missing: %s" % (self.label, self.engine_path))
        if self.expected_sha256:
            verify_sha256(self.engine_path, self.expected_sha256, self.label)
        import tensorrt as trt
        started = time.time()
        self._trt = trt
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(self.engine_path, "rb") as handle:
            self._engine = self._runtime.deserialize_cuda_engine(handle.read())
        if self._engine is None:
            raise RuntimeError("failed to deserialize %s" % self.label)
        self._context = self._engine.create_execution_context()
        if hasattr(self._engine, "num_io_tensors"):
            names = [
                self._engine.get_tensor_name(i)
                for i in range(self._engine.num_io_tensors)
            ]
            self._inputs = tuple(
                name for name in names
                if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            )
            self._outputs = tuple(
                name for name in names
                if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
            )
        else:
            names = [
                self._engine.get_binding_name(i)
                for i in range(self._engine.num_bindings)
            ]
            self._inputs = tuple(
                name for i, name in enumerate(names)
                if self._engine.binding_is_input(i)
            )
            self._outputs = tuple(
                name for i, name in enumerate(names)
                if not self._engine.binding_is_input(i)
            )
        if len(self._inputs) != 1 or not self._outputs:
            raise RuntimeError(
                "%s must have one input and at least one output"
                % self.label
            )
        log("%s loaded in %.2fs" % (self.label, time.time() - started))
        return self

    def infer_cuda(self, input_array):
        """Run with a contiguous NumPy input; return output CUDA tensors."""
        self.load()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("%s requires CUDA" % self.label)
        trt = self._trt
        engine = self._engine
        context = self._context
        name = self._inputs[0]
        np_input = np.ascontiguousarray(input_array)
        input_tensor = torch.from_numpy(np_input).to(
            device="cuda", dtype=_torch_dtype(trt, _input_dtype(engine, name))
        )
        stream = torch.cuda.current_stream()
        outputs = {}
        tensors = {name: input_tensor}
        if hasattr(engine, "num_io_tensors"):
            context.set_input_shape(name, tuple(input_tensor.shape))
            for output_name in self._outputs:
                shape = tuple(int(x) for x in context.get_tensor_shape(output_name))
                dtype = _torch_dtype(trt, engine.get_tensor_dtype(output_name))
                tensors[output_name] = torch.empty(shape, dtype=dtype, device="cuda")
            for tensor_name, tensor in tensors.items():
                context.set_tensor_address(tensor_name, int(tensor.data_ptr()))
            if not context.execute_async_v3(
                stream_handle=int(stream.cuda_stream)
            ):
                raise RuntimeError("%s execute_async_v3 failed" % self.label)
            outputs = {key: tensors[key] for key in self._outputs}
        else:
            input_index = _binding_index(engine, name)
            context.set_binding_shape(input_index, tuple(input_tensor.shape))
            bindings = [0] * engine.num_bindings
            bindings[input_index] = int(input_tensor.data_ptr())
            for output_name in self._outputs:
                output_index = _binding_index(engine, output_name)
                shape = tuple(
                    int(x) for x in context.get_binding_shape(output_index)
                )
                tensor = torch.empty(
                    shape,
                    dtype=_torch_dtype(
                        trt, engine.get_binding_dtype(output_index)
                    ),
                    device="cuda",
                )
                bindings[output_index] = int(tensor.data_ptr())
                outputs[output_name] = tensor
            if not context.execute_async_v2(
                bindings=bindings,
                stream_handle=int(stream.cuda_stream),
            ):
                raise RuntimeError("%s execute_async_v2 failed" % self.label)
        return outputs

    def infer(self, input_array):
        outputs = self.infer_cuda(input_array)
        import torch
        torch.cuda.current_stream().synchronize()
        return {name: tensor.float().cpu().numpy() for name, tensor in outputs.items()}

    def warmup(self, shape, dtype=np.float32):
        outputs = self.infer_cuda(np.zeros(shape, dtype=dtype))
        import torch
        torch.cuda.current_stream().synchronize()
        return outputs

    def close(self):
        self._context = None
        self._engine = None
        self._runtime = None
        self._trt = None
        self._inputs = ()
        self._outputs = ()


def _binding_index(engine, name):
    for index in range(engine.num_bindings):
        if engine.get_binding_name(index) == name:
            return index
    raise RuntimeError("TensorRT binding not found: %s" % name)


def _input_dtype(engine, name):
    if hasattr(engine, "num_io_tensors"):
        return engine.get_tensor_dtype(name)
    return engine.get_binding_dtype(_binding_index(engine, name))


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
        raise TypeError("unsupported TensorRT dtype %s" % np_dtype)
    return mapping[np_dtype]
