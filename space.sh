#!/usr/bin/env bash
# Start Gradio as a thin client of the resident local worker.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1

PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo ".venv is missing. Run: bash prepare.sh" >&2
  exit 1
fi

bash "$ROOT/cold.sh" start
exec "$PYTHON" app.py "$@"
