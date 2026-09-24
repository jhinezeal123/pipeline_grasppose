# Lite-Mono Tiny TensorRT native shim

This directory contains the minimal C++ runtime used by the Python depth adapter.
It intentionally does not parse ONNX at runtime: `prepare.sh` builds a serialized
FP16 TensorRT engine on the target Jetson, then this shared library only
deserializes the engine and runs `enqueueV2`.

The expected network contract is fixed and matches the pinned community export:

- input: `input`, shape `[1, 3, 192, 640]`, FP32 I/O;
- output: `disp_output`, one float disparity value per input pixel;
- TensorRT execution precision: FP16 where supported by the builder.

Build on Jetson AGX Xavier:

```bash
cmake -S native/litemono_trt -B build/litemono_trt -DCMAKE_BUILD_TYPE=Release
cmake --build build/litemono_trt -- -j2
```

Normally `bash prepare.sh` performs this build and the engine build together.
