#!/usr/bin/env bash
# Start, inspect, stop, or restart the resident TensorRT model worker.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1
PYTHON="$ROOT/.venv/bin/python"
RUNTIME_DIR="$(printenv GRASP_RUNTIME_DIR || true)"
[ -n "$RUNTIME_DIR" ] || RUNTIME_DIR="$ROOT/.runtime"
SOCKET_PATH="$(printenv GRASP_WORKER_SOCKET || true)"
[ -n "$SOCKET_PATH" ] || SOCKET_PATH="$RUNTIME_DIR/worker.sock"
PID_FILE="$RUNTIME_DIR/worker.pid"
LOG_FILE="$RUNTIME_DIR/worker.log"
START_TIMEOUT="$(printenv COLD_TIMEOUT_SECONDS || true)"
[ -n "$START_TIMEOUT" ] || START_TIMEOUT=600
COMMAND=start
[ "$#" -eq 0 ] || COMMAND="$1"

if [ ! -x "$PYTHON" ]; then
  echo ".venv is missing. Run: bash prepare.sh" >&2
  exit 1
fi

mkdir -p "$RUNTIME_DIR"

worker_pid() {
  if [ -s "$PID_FILE" ]; then tr -dc '0-9' < "$PID_FILE"; fi
}

is_our_worker() {
  local pid="$1"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
  [ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)" = "$ROOT" ] || return 1
  ps -p "$pid" -o args= 2>/dev/null | grep -Fq "grasppose.worker_server"
}

rpc_status() {
  "$PYTHON" -c 'from grasppose.worker_client import request_worker; import json; print(json.dumps(request_worker({"op":"status"}, timeout=2)))' 2>/dev/null
}

show_status() {
  local state pid
  if state="$(rpc_status)"; then
    printf '%s\n' "$state"
    return 0
  fi
  pid="$(worker_pid || true)"
  if is_our_worker "$pid"; then
    printf 'worker starting (pid=%s); log: %s\n' "$pid" "$LOG_FILE"
    return 0
  fi
  echo "worker stopped"
  return 1
}

wait_for_worker() {
  local deadline=$((SECONDS + START_TIMEOUT))
  local pid state
  while [ "$SECONDS" -lt "$deadline" ]; do
    if state="$(rpc_status)"; then
      printf '%s\n' "$state"
      return 0
    fi
    pid="$(worker_pid || true)"
    if ! is_our_worker "$pid"; then
      echo "Worker failed to start. Recent log:" >&2
      tail -n 100 "$LOG_FILE" 2>/dev/null || true
      return 1
    fi
    sleep 2
  done
  echo "Worker did not become ready within $START_TIMEOUT seconds. Recent log:" >&2
  tail -n 100 "$LOG_FILE" 2>/dev/null || true
  return 1
}

start_worker() {
  local pid state
  if state="$(rpc_status)"; then
    echo "Worker already ready; reusing the resident models."
    printf '%s\n' "$state"
    return 0
  fi
  pid="$(worker_pid || true)"
  if is_our_worker "$pid"; then
    echo "Worker is already starting (pid=$pid); waiting for it."
    wait_for_worker
    return
  fi
  if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then rm -f "$PID_FILE"; fi
  : > "$LOG_FILE"
  nohup "$PYTHON" -m grasppose.worker_server serve >> "$LOG_FILE" 2>&1 < /dev/null 9>&- &
  pid="$!"
  printf '%s\n' "$pid" > "$PID_FILE"
  echo "Starting worker and warming YOLOE, Lite-Mono, and VGN ..."
  wait_for_worker
}

stop_worker() {
  local pid i
  pid="$(worker_pid || true)"
  if state="$(rpc_status)"; then
    "$PYTHON" -c 'from grasppose.worker_client import request_worker; request_worker({"op":"stop"}, timeout=3)' >/dev/null 2>&1 || true
  fi
  for i in $(seq 1 30); do
    pid="$(worker_pid || true)"
    if ! is_our_worker "$pid"; then
      rm -f "$PID_FILE"
      echo "Worker stopped."
      return 0
    fi
    sleep 1
  done
  if is_our_worker "$pid"; then
    echo "Worker did not stop through the local socket; sending SIGTERM to pid $pid."
    kill -TERM "$pid" 2>/dev/null || true
  fi
  for i in $(seq 1 10); do
    pid="$(worker_pid || true)"
    if ! is_our_worker "$pid"; then
      rm -f "$PID_FILE"
      echo "Worker stopped."
      return 0
    fi
    sleep 1
  done
  echo "Worker is still running (pid=$pid); inspect $LOG_FILE." >&2
  return 1
}

case "$COMMAND" in
  status)
    show_status
    ;;
  start|stop|restart)
    command -v flock >/dev/null || {
      echo "flock is required to serialize worker lifecycle commands" >&2
      exit 1
    }
    exec 9>"$RUNTIME_DIR/control.lock"
    flock -x 9
    case "$COMMAND" in
      start) start_worker ;;
      stop) stop_worker ;;
      restart) stop_worker; start_worker ;;
    esac
    ;;
  *)
    echo "Usage: bash cold.sh [start|status|stop|restart]" >&2
    exit 2
    ;;
esac
