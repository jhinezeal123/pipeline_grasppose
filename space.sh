#!/usr/bin/env bash
# Start the interactive Gradio UI using the prepared resident-model pipeline.
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

bash "$ROOT/cold.sh" start
exec "$PYTHON" app.py "$@"
