#!/usr/bin/env bash
# Prepare the Jetson runtime once: Python overlay, dependencies, model sources and artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

# Fetch the prevalidated YOLOE/Lite-Mono FP32 and VGN TensorRT bundles for
# this Xavier/L4T/TensorRT stack. The installer verifies each release archive
# and its weights/engine checksums before activating checkout-local artifacts.
echo "Downloading and validating checkout-local TensorRT engines ..."
"$PYTHON" tools/fetch_prebuilt_trt.py

# Always validate the VGN files installed in this checkout. An inherited
# VGN_ENGINE path must not silently reuse an engine from another checkout.
export VGN_ENGINE="$ROOT/model/vgn.engine"
export VGN_CHECKPOINT="$ROOT/model/vgn_conv.pth"
export VGN_MANIFEST="$ROOT/model/runtime/vgn.json"
"$PYTHON" tools/vgn_manifest.py check \
  --checkpoint "$VGN_CHECKPOINT" \
  --engine "$VGN_ENGINE" \
  --manifest "$VGN_MANIFEST"

"$PYTHON" env/check_env.py
echo "Preparation complete."
