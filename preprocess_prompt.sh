#!/usr/bin/env bash
# Bake the fixed prompt set into validated YOLOE TensorRT artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
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

exec "$PYTHON" tools/preprocess_yoloe.py "$1"
