#!/usr/bin/env bash
# Send one image and a fixed prompt ID to the resident worker.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1

PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo ".venv is missing. Run: bash scripts/prepare.sh" >&2
  exit 1
fi

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/infer.sh IMAGE --prompt-id ID [--camera-k FX FY CX CY] [--camera-k-size WIDTH HEIGHT] [--fov-x DEG] [--render]" >&2
  echo "Inference returns grasp metadata and a RUN_ID. Rendering the four diagnostic PNGs is optional; pass --render." >&2
  echo "Start the resident models once with: bash scripts/worker.sh" >&2
  exit 2
fi

exec "$PYTHON" -S -m apps.cli.infer "$@"
