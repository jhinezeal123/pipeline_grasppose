#!/usr/bin/env bash
# Run one image through the prepared pipeline and write exactly one 4-image result set to /output.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# prepare.sh installs all YOLOE dependencies. Never let runtime auto-update
# JetPack-provided Torch/NumPy packages.
export ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1

PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo ".venv is missing. Run: bash prepare.sh" >&2
  exit 1
fi

IMG="${1:-}"
if [ -z "$IMG" ]; then
  echo "Usage: bash infer.sh /path/to/image.png [pipeline.py options]" >&2
  echo "Example: bash infer.sh img/frame.png --camera-k FX FY CX CY --prompt 'the mug'" >&2
  exit 1
fi
shift

if [ ! -f "$IMG" ]; then
  echo "Input image not found: $IMG" >&2
  exit 1
fi

OUT=/output
mkdir -p "$OUT"

exec "$PYTHON" pipeline.py   --img "$IMG"   --out "$OUT"   "$@"
