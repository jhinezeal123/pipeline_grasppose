#!/usr/bin/env bash
# Jetson Xavier setup/runner: YOLOE-26s -> Lite-Mono -> depth+K -> TSDF -> VGN TensorRT
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"; mkdir -p model output
SERVE=0; PORT=8080; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --serve) SERVE=1; shift ;;
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --port=*) PORT="${1#--port=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}
HOST_PYTHON="$(command -v "${PYTHON:-python3}")"
"$HOST_PYTHON" env/setup_env.py
PYTHON="$ROOT/.venv/bin/python"; export PATH="$ROOT/.venv/bin:/usr/src/tensorrt/bin:/usr/local/cuda/bin:$PATH"
"$HOST_PYTHON" env/setup_env.py --install

if [ ! -d model/Lite-Mono/.git ]; then git clone https://github.com/noahzn/Lite-Mono.git model/Lite-Mono; fi
git -C model/Lite-Mono fetch --depth 1 origin 4874b35df8ed4da16159ce8be8c697028b72bf76
git -C model/Lite-Mono checkout --detach 4874b35df8ed4da16159ce8be8c697028b72bf76

if [ ! -s model/yoloe-26s-seg.pt ]; then
  "$PYTHON" - <<'PY'
from pathlib import Path
import shutil
from ultralytics import YOLOE
m=YOLOE('yoloe-26s-seg.pt'); src=Path(str(getattr(m,'ckpt_path','yoloe-26s-seg.pt'))); dst=Path('model/yoloe-26s-seg.pt')
if not src.exists(): raise SystemExit('Ultralytics did not resolve yoloe-26s-seg.pt')
if src.resolve()!=dst.resolve(): shutil.copy2(src,dst)
print(dst)
PY
fi

if [ ! -s model/lite-mono/encoder.pth ] || [ ! -s model/lite-mono/depth.pth ]; then
  TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT; echo "Downloading Lite-Mono weights ..."
  curl -L --fail --retry 3 'https://surfdrive.surf.nl/files/index.php/s/CUjiK221EFLyXDY/download' -o "$TMP/lite-mono.bin"
  mkdir -p "$TMP/x" model/lite-mono
  "$PYTHON" - "$TMP/lite-mono.bin" "$TMP/x" <<'PY'
import pathlib,sys,tarfile,zipfile
src,out=map(pathlib.Path,sys.argv[1:])
try:
    with zipfile.ZipFile(src) as z: z.extractall(out)
except zipfile.BadZipFile:
    try:
        with tarfile.open(src) as t: t.extractall(out)
    except tarfile.TarError as e: raise SystemExit(f'Lite-Mono download is not an archive: {e}')
PY
  ENC="$(find "$TMP/x" -name encoder.pth -type f | head -1 || true)"; DEP="$(find "$TMP/x" -name depth.pth -type f | head -1 || true)"
  if [ -z "$ENC" ] || [ -z "$DEP" ]; then echo "Lite-Mono archive did not contain encoder.pth + depth.pth" >&2; exit 1; fi
  cp "$ENC" model/lite-mono/encoder.pth; cp "$DEP" model/lite-mono/depth.pth; rm -rf "$TMP"; trap - EXIT
fi

if [ ! -s "${VGN_ENGINE:-model/vgn.engine}" ]; then
  cat >&2 <<'EOF'
VGN TensorRT engine is missing.
1) Obtain the original VGN vgn_conv.pth checkpoint.
2) .venv/bin/python tools/export_vgn_onnx.py --checkpoint /path/vgn_conv.pth --out model/vgn.onnx
3) trtexec --onnx=model/vgn.onnx --saveEngine=model/vgn.engine --fp16
Build the engine on THIS Jetson, then rerun run.sh.
EOF
  exit 1
fi
if [ "$SERVE" = "1" ]; then exec "$PYTHON" app.py --port "$PORT"; fi
IMG="${1:-$(find img -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | head -1 || true)}"
if [ -z "$IMG" ]; then echo "No input image. Pass one, e.g. bash run.sh img/frame.png --camera-k fx fy cx cy" >&2; exit 1; fi
shift || true
exec "$PYTHON" pipeline.py --img "$IMG" --out output "$@"
