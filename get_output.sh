#!/usr/bin/env bash
# Start a separate renderer for a completed inference result.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo ".venv is missing. Run: bash prepare.sh" >&2
  exit 1
fi
exec "$PYTHON" -S "$ROOT/tools/output_control.py" "$@"
