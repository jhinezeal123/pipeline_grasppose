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

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }

LITEMONO_REV="4874b35df8ed4da16159ce8be8c697028b72bf76"
if [ ! -d model/Lite-Mono/.git ]; then
  git clone https://github.com/noahzn/Lite-Mono.git model/Lite-Mono
fi
git -C model/Lite-Mono fetch --depth 1 origin "$LITEMONO_REV"
git -C model/Lite-Mono checkout --detach "$LITEMONO_REV"

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

if [ ! -s model/lite-mono/encoder.pth ] || [ ! -s model/lite-mono/depth.pth ]; then
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  echo "Downloading Lite-Mono weights ..."
  curl -L --fail --retry 3     'https://surfdrive.surf.nl/files/index.php/s/CUjiK221EFLyXDY/download'     -o "$TMP/lite-mono.bin"
  mkdir -p "$TMP/x" model/lite-mono
  "$PYTHON" - "$TMP/lite-mono.bin" "$TMP/x" <<'PY'
import pathlib
import sys
import tarfile
import zipfile

src, out = map(pathlib.Path, sys.argv[1:])
try:
    with zipfile.ZipFile(src) as zf:
        zf.extractall(out)
except zipfile.BadZipFile:
    try:
        with tarfile.open(src) as tf:
            tf.extractall(out)
    except tarfile.TarError as exc:
        raise SystemExit("Lite-Mono download is not an archive: %s" % exc)
PY
  ENC="$(find "$TMP/x" -name encoder.pth -type f | head -1 || true)"
  DEP="$(find "$TMP/x" -name depth.pth -type f | head -1 || true)"
  if [ -z "$ENC" ] || [ -z "$DEP" ]; then
    echo "Lite-Mono archive did not contain encoder.pth + depth.pth" >&2
    exit 1
  fi
  cp "$ENC" model/lite-mono/encoder.pth
  cp "$DEP" model/lite-mono/depth.pth
  rm -rf "$TMP"
  trap - EXIT
fi

VGN_ENGINE_PATH="${VGN_ENGINE:-$ROOT/model/vgn.engine}"
if [ ! -s "$VGN_ENGINE_PATH" ]; then
  VGN_CHECKPOINT_PATH="${VGN_CHECKPOINT:-$ROOT/model/vgn_conv.pth}"
  if [ ! -s "$VGN_CHECKPOINT_PATH" ]; then
    cat >&2 <<EOF
VGN TensorRT engine is missing: $VGN_ENGINE_PATH

TensorRT engines are Jetson/TensorRT-version specific, so prepare.sh builds the
engine on this device instead of downloading a foreign engine.

Place the official VGN checkpoint at:
  $ROOT/model/vgn_conv.pth

or set:
  VGN_CHECKPOINT=/path/to/vgn_conv.pth

Then rerun:
  bash prepare.sh
EOF
    exit 1
  fi

  TRTEXEC="$(command -v trtexec || true)"
  if [ -z "$TRTEXEC" ] && [ -x /usr/src/tensorrt/bin/trtexec ]; then
    TRTEXEC=/usr/src/tensorrt/bin/trtexec
  fi
  if [ -z "$TRTEXEC" ]; then
    echo "trtexec is required to build VGN TensorRT engine" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$VGN_ENGINE_PATH")"
  ONNX_PATH="$ROOT/model/vgn.onnx"
  "$PYTHON" tools/export_vgn_onnx.py     --checkpoint "$VGN_CHECKPOINT_PATH"     --out "$ONNX_PATH"
  "$TRTEXEC"     --onnx="$ONNX_PATH"     --saveEngine="$VGN_ENGINE_PATH"     --fp16
fi

"$PYTHON" env/check_env.py
echo "Preparation complete."
