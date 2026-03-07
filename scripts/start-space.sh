#!/usr/bin/env bash
set -euo pipefail

python -m trenches_env.server &
BACKEND_PID=$!

cleanup() {
  kill "${BACKEND_PID}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

exec bunx next start --hostname 0.0.0.0 --port "${PORT:-7860}"
