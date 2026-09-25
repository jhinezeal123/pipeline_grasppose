#!/usr/bin/env bash
# Bake the fixed prompt set into validated YOLOE TensorRT artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export VGN_ENGINE="$ROOT/model/vgn.engine"
export VGN_CHECKPOINT="$ROOT/model/vgn_conv.pth"
export VGN_MANIFEST="$ROOT/model/runtime/vgn.json"
export ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1
PYTHON="$ROOT/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo ".venv is missing. Run: bash prepare.sh" >&2
  exit 1
fi
if [ "$#" -ne 1 ]; then
  echo "Usage: bash preprocess_prompt.sh prompts.json" >&2
  exit 2
fi

if [ -z "${CAMERA_K:-}" ]; then
  echo "Set CAMERA_K='FX FY CX CY' for full-pipeline parity validation" >&2
  exit 2
fi
read -r -a CAMERA_K_VALUES <<< "$CAMERA_K"
if [ "${#CAMERA_K_VALUES[@]}" -ne 4 ]; then
  echo "CAMERA_K must contain FX FY CX CY" >&2
  exit 2
fi

CAMERA_K_SIZE_ARGS=()
if [ -n "${CAMERA_K_SIZE:-}" ]; then
  read -r -a CAMERA_K_SIZE_VALUES <<< "$CAMERA_K_SIZE"
  if [ "${#CAMERA_K_SIZE_VALUES[@]}" -ne 2 ]; then
    echo "CAMERA_K_SIZE must contain WIDTH HEIGHT" >&2
    exit 2
  fi
  CAMERA_K_SIZE_ARGS=(--camera-k-size "${CAMERA_K_SIZE_VALUES[@]}")
fi

if bash "$ROOT/cold.sh" status >/dev/null 2>&1; then
  echo "Stop the resident worker before preprocessing: bash cold.sh stop" >&2
  exit 2
fi

ARTIFACT_ROOT="${YOLOE_ARTIFACT_ROOT:-$ROOT/model/runtime/yoloe}"
CURRENT_FILE="$ARTIFACT_ROOT/CURRENT"
OLD_CURRENT=""
if [ -f "$CURRENT_FILE" ]; then
  OLD_CURRENT="$(cat "$CURRENT_FILE")"
fi

"$PYTHON" tools/preprocess_yoloe.py "$1"
if "$PYTHON" tools/validate_yoloe_fp32.py "$1" \
    --camera-k "${CAMERA_K_VALUES[@]}" \
    "${CAMERA_K_SIZE_ARGS[@]}"; then
  exit 0
fi
if [ -n "$OLD_CURRENT" ]; then
  printf '%s\n' "$OLD_CURRENT" > "$CURRENT_FILE.tmp"
  mv "$CURRENT_FILE.tmp" "$CURRENT_FILE"
else
  rm -f "$CURRENT_FILE"
fi
echo "YOLOE FP32 full-pipeline parity failed; previous prompt artifact restored." >&2
exit 1
