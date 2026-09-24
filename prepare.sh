#!/usr/bin/env bash
# Prepare the Jetson runtime once: Python overlay, dependencies, model sources and artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p model output

HOST_PYTHON="$(command -v "${PYTHON:-python3}")"
"$HOST_PYTHON" env/setup_env.py

PYTHON="$ROOT/.venv/bin/python"
export PATH="$ROOT/.venv/bin:/usr/src/tensorrt/bin:/usr/local/cuda/bin:$PATH"
"$HOST_PYTHON" env/setup_env.py --install

# Do not run global "pip check" here. The overlay intentionally inherits
# JetPack/Ubuntu system packages via --system-site-packages, and unrelated
# host distributions can have pre-existing metadata conflicts. The targeted
# env/check_env.py preflight below validates the packages this pipeline uses.
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v cmake >/dev/null || { echo "cmake is required" >&2; exit 1; }
command -v c++ >/dev/null || { echo "a C++ compiler is required" >&2; exit 1; }

# YOLOE text prompting lazily installs CLIP and downloads MobileCLIP on first
# set_classes(). Do both here under the JetPack constraints so inference never
# mutates the environment at runtime.
CLIP_REV="a13192f8cb767260d7dfd98c843b0716593169e7"
CLIP_STAMP="$ROOT/.venv/ultralytics-clip-revision.txt"
if [ ! -s "$CLIP_STAMP" ] || [ "$(cat "$CLIP_STAMP")" != "$CLIP_REV" ]; then
  "$PYTHON" -m pip install \
    -c "$ROOT/.venv/host-constraints.txt" \
    "git+https://github.com/ultralytics/CLIP.git@$CLIP_REV"
  printf '%s\n' "$CLIP_REV" > "$CLIP_STAMP"
fi


if [ ! -s model/yoloe-26s-seg.pt ]; then
  "$PYTHON" - <<'PY'
from pathlib import Path
import shutil
from ultralytics import YOLOE

model = YOLOE("yoloe-26s-seg.pt")
src = Path(str(getattr(model, "ckpt_path", "yoloe-26s-seg.pt")))
dst = Path("model/yoloe-26s-seg.pt")
if not src.exists():
    raise SystemExit("Ultralytics did not resolve yoloe-26s-seg.pt")
if src.resolve() != dst.resolve():
    shutil.copy2(src, dst)
print(dst)
PY
fi

MOBILECLIP="$ROOT/mobileclip2_b.ts"
MOBILECLIP_SHA256="35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f"
if [ ! -s "$MOBILECLIP" ]; then
  TMP_CLIP="$(mktemp)"
  trap 'rm -f "$TMP_CLIP"' EXIT
  echo "Downloading YOLOE-26 MobileCLIP2 text encoder ..."
  curl -L --fail --retry 3 \
    'https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts' \
    -o "$TMP_CLIP"
  printf '%s  %s\n' "$MOBILECLIP_SHA256" "$TMP_CLIP" | sha256sum -c -
  mv "$TMP_CLIP" "$MOBILECLIP"
  trap - EXIT
fi

# From here onward all lazy YOLOE dependencies are already installed. Refuse
# runtime auto-upgrades that could replace the JetPack Torch/NumPy stack.
export ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1

echo "Validating YOLOE text-prompt path with JetPack Torch ..."
"$PYTHON" - <<'PY'
from pathlib import Path
from ultralytics import YOLOE

asset = Path("mobileclip2_b.ts")
if not asset.is_file():
    raise SystemExit("mobileclip2_b.ts is missing")
model = YOLOE("model/yoloe-26s-seg.pt")
model.set_classes(["object"])
print("YOLOE text prompt ready:", asset)
PY

# Pin the exact community-exported Lite-Mono Tiny artifact used by the
# depth-detect Jetson TensorRT benchmark. Opset 11 is intentionally selected
# for the TensorRT 8.5.x stack on JetPack 5 / Xavier.
LITEMONO_MODEL_REV="520ab0e5aaabf705c25b4f23b3316ae2c5a7bd3a"
LITEMONO_ONNX_BLOB_SHA="cbfaf3c2a0e6619d8d0ce554a35a009473d08faa"
LITEMONO_ONNX_PATH="${LITEMONO_ONNX:-$ROOT/model/lite-mono-tiny_192x640_op11.onnx}"
LITEMONO_ENGINE_PATH="${LITEMONO_ENGINE:-$ROOT/model/lite-mono-tiny_192x640_op11_fp16.engine}"

if [ ! -s "$LITEMONO_ONNX_PATH" ] || \
   [ "$(git hash-object "$LITEMONO_ONNX_PATH" 2>/dev/null || true)" != "$LITEMONO_ONNX_BLOB_SHA" ]; then
  TMP_LITEMONO="$(mktemp)"
  trap 'rm -f "$TMP_LITEMONO"' EXIT
  echo "Downloading pinned Lite-Mono Tiny ONNX (192x640, opset 11) ..."
  curl -L --fail --retry 3 \
    "https://raw.githubusercontent.com/yzfzzz/depth-detect-model/$LITEMONO_MODEL_REV/onnx/lite-mono-tiny/lite-mono-tiny_192x640_op11.onnx" \
    -o "$TMP_LITEMONO"
  ACTUAL_BLOB_SHA="$(git hash-object "$TMP_LITEMONO")"
  if [ "$ACTUAL_BLOB_SHA" != "$LITEMONO_ONNX_BLOB_SHA" ]; then
    echo "Lite-Mono ONNX Git blob mismatch: expected=$LITEMONO_ONNX_BLOB_SHA actual=$ACTUAL_BLOB_SHA" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$LITEMONO_ONNX_PATH")"
  mv "$TMP_LITEMONO" "$LITEMONO_ONNX_PATH"
  trap - EXIT
fi

TRTEXEC="$(command -v trtexec || true)"
if [ -z "$TRTEXEC" ] && [ -x /usr/src/tensorrt/bin/trtexec ]; then
  TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi
if [ -z "$TRTEXEC" ]; then
  echo "trtexec is required to build TensorRT engines" >&2
  exit 1
fi

if [ ! -s "$LITEMONO_ENGINE_PATH" ]; then
  mkdir -p "$(dirname "$LITEMONO_ENGINE_PATH")"
  echo "Building Lite-Mono Tiny FP16 TensorRT engine on this Xavier ..."
  "$TRTEXEC" \
    --onnx="$LITEMONO_ONNX_PATH" \
    --saveEngine="$LITEMONO_ENGINE_PATH" \
    --fp16 \
    --workspace=2048
fi

VGN_ENGINE_PATH="${VGN_ENGINE:-$ROOT/model/vgn.engine}"
if [ ! -s "$VGN_ENGINE_PATH" ]; then
  VGN_CHECKPOINT_PATH="${VGN_CHECKPOINT:-$ROOT/model/vgn_conv.pth}"
  if [ ! -s "$VGN_CHECKPOINT_PATH" ]; then
    TMP_VGN="$(mktemp -d)"
    trap 'rm -rf "$TMP_VGN"' EXIT
    echo "Downloading official ETH VGN data bundle ..."
    "$PYTHON" -m gdown --fuzzy       'https://drive.google.com/file/d/1MysYHve3ooWiLq12b58Nm8FWiFBMH-bJ/view?usp=sharing'       -O "$TMP_VGN/data.zip"
    "$PYTHON" - "$TMP_VGN/data.zip" "$VGN_CHECKPOINT_PATH" <<'PY'
import pathlib
import shutil
import sys
import zipfile

archive = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
with zipfile.ZipFile(archive) as zf:
    matches = [n for n in zf.namelist() if n.endswith("/models/vgn_conv.pth") or n == "data/models/vgn_conv.pth"]
    if not matches:
        matches = [n for n in zf.namelist() if pathlib.PurePosixPath(n).name == "vgn_conv.pth"]
    if not matches:
        raise SystemExit("Official VGN data bundle did not contain vgn_conv.pth")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(matches[0]) as src, dst.open("wb") as out:
        shutil.copyfileobj(src, out)
print(dst)
PY
    rm -rf "$TMP_VGN"
    trap - EXIT
  fi

  mkdir -p "$(dirname "$VGN_ENGINE_PATH")"
  ONNX_PATH="$ROOT/model/vgn.onnx"
  "$PYTHON" tools/export_vgn_onnx.py     --checkpoint "$VGN_CHECKPOINT_PATH"     --out "$ONNX_PATH"
  "$TRTEXEC"     --onnx="$ONNX_PATH"     --saveEngine="$VGN_ENGINE_PATH"     --fp16
fi

echo "Building minimal Lite-Mono TensorRT C++ runtime ..."
cmake -S "$ROOT/native/litemono_trt" -B "$ROOT/build/litemono_trt" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build/litemono_trt" -- -j"${BUILD_JOBS:-2}"

"$PYTHON" env/check_env.py
echo "Preparation complete."
