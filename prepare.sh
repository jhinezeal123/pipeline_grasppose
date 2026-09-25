#!/usr/bin/env bash
# Prepare the Jetson Python overlay and install verified, prebuilt TRT bundles.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p model output

HOST_PYTHON="$(command -v "${PYTHON:-python3}")"
"$HOST_PYTHON" env/setup_env.py

PYTHON="$ROOT/.venv/bin/python"
export PATH="$ROOT/.venv/bin:/usr/src/tensorrt/bin:/usr/local/cuda/bin:$PATH"
"$HOST_PYTHON" env/setup_env.py --install

# The Xavier Torch/TensorRT stack comes from JetPack. These archives contain
# only model artifacts; no TensorRT engine is built by prepare.sh.
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v cmake >/dev/null || { echo "cmake is required" >&2; exit 1; }
command -v c++ >/dev/null || { echo "a C++ compiler is required" >&2; exit 1; }
export ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1
"$PYTHON" tools/fetch_prebuilt_trt.py

echo "Building the minimal Lite-Mono TensorRT C++ runtime ..."
cmake -S "$ROOT/native/litemono_trt" -B "$ROOT/build/litemono_trt" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build/litemono_trt" -- -j"${BUILD_JOBS:-2}"

YOLOE_ENGINE_PATH="${YOLOE_MODEL:-$ROOT/model/yoloe-26s-seg.engine}"
LITEMONO_ONNX_PATH="${LITEMONO_ONNX:-$ROOT/model/lite-mono-tiny_192x640_op11.onnx}"
LITEMONO_ENGINE_PATH="${LITEMONO_ENGINE:-$ROOT/model/lite-mono-tiny_192x640_op11_fp16.engine}"
VGN_ENGINE_PATH="${VGN_ENGINE:-$ROOT/model/vgn.engine}"

# Persist the selected paths for infer.sh/cold.sh, while keeping explicit
# environment overrides available for a known compatible external engine.
RUNTIME_CONFIG="$ROOT/.venv/runtime.json"
YOLOE_MODEL="$YOLOE_ENGINE_PATH" \
LITEMONO_ONNX="$LITEMONO_ONNX_PATH" \
LITEMONO_ENGINE="$LITEMONO_ENGINE_PATH" \
LITEMONO_TRT_LIBRARY="$ROOT/build/litemono_trt/liblitemono_trt.so" \
VGN_ENGINE="$VGN_ENGINE_PATH" \
"$PYTHON" - "$RUNTIME_CONFIG" <<'PY'
import json
import os
from pathlib import Path
import sys

dst = Path(sys.argv[1])
keys = (
    "YOLOE_MODEL",
    "LITEMONO_ONNX",
    "LITEMONO_ENGINE",
    "LITEMONO_TRT_LIBRARY",
    "VGN_ENGINE",
)
runtime = {
    key: str(Path(os.environ[key]).expanduser().resolve())
    for key in keys
}
dst.parent.mkdir(parents=True, exist_ok=True)
temporary = dst.with_name(dst.name + ".tmp")
temporary.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
temporary.replace(dst)
print("Runtime config:", dst)
PY

"$PYTHON" env/check_env.py
echo "Preparation complete. Start the warm worker with: bash cold.sh"
