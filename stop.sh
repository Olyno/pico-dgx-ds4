#!/usr/bin/env bash
# Stop the server ./start.sh started: stop the service inside the
# container (engine/server/service.sh stop, which stops rank 0 first so
# sessions are kept), then stop and remove the container. Frees the GPU.
#
# Usage: ./stop.sh [--forget]
#   DRY_RUN=1 ./stop.sh        # print what would happen, change nothing
# Env: CONTAINER_NAME, DSGB10_PORT, STOP_TIMEOUT (default 30)
set -euo pipefail
cd "$(dirname "$0")"

ENV_FILE="${ENV_FILE:-$PWD/.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

CONTAINER="${CONTAINER_NAME:-pico-dgx-ds4}"
PORT="${DSGB10_PORT:-18893}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
FORGET=""
[ "${1:-}" = "--forget" ] && FORGET="--forget"
DRY="${DRY_RUN:-0}"

log() { echo "[stop.sh] $*"; }
warn() { echo "[stop.sh] WARN: $*" >&2; }

exists() { docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; }
running() { [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" && exists ]; }

if ! exists; then
  log "No container named $CONTAINER: nothing to stop"
  exit 0
fi

# Requests in flight are cut off: the service does not drain them.
if running && ! curl -s --max-time 3 "http://127.0.0.1:$PORT/health" | grep -qx ok; then
  warn "the API is not answering 'ok' (ring lent, or requests held): they will be cut off"
fi

if [ "$DRY" = "1" ]; then
  log "DRY_RUN=1: would exec service.sh stop $FORGET in $CONTAINER, then docker stop -t $STOP_TIMEOUT $CONTAINER + docker rm -f"
  exit 0
fi

log "Stopping $CONTAINER (up to ${STOP_TIMEOUT}s)..."
if running; then
  # service.sh stop is the graceful path; if the container is wedged we
  # still stop it (sessions kept by the grace path only).
  docker exec "$CONTAINER" engine/server/service.sh stop $FORGET >/dev/null 2>&1 || \
    warn "service.sh stop failed inside $CONTAINER (docker logs --tail 20 $CONTAINER)"
fi
docker stop -t "$STOP_TIMEOUT" "$CONTAINER" >/dev/null 2>&1 || true
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
log "Stopped and removed $CONTAINER; its GPU memory is free again"
