#!/bin/sh
set -eu

BACKEND_HOST="${HOST:-127.0.0.1}"
BACKEND_PORT="${PORT:-8000}"
NEXT_HOST="${NEXT_HOST:-127.0.0.1}"
NEXT_PORT="${FRONTEND_PORT:-3000}"

cleanup() {
  kill "${BACKEND_PID:-0}" "${FRONTEND_PID:-0}" "${NGINX_PID:-0}" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

cd /app/backend
HOST="$BACKEND_HOST" PORT="$BACKEND_PORT" python3 -m trenches_env.server &
BACKEND_PID=$!

cd /app
bun --bun run start -- --hostname "$NEXT_HOST" --port "$NEXT_PORT" &
FRONTEND_PID=$!

nginx -g 'daemon off;' &
NGINX_PID=$!

wait "$BACKEND_PID" "$FRONTEND_PID" "$NGINX_PID"
