#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash (the ds41f engine) in a Docker container, end to
# end: build the image from Dockerfile if needed, start the service
# (engine/server/service.sh) inside a host-networked GPU container, wait
# until the OpenAI API answers, then run a smoke test. Stop with ./stop.sh.
# Settings come from ./.env (or $ENV_FILE), then the environment.
#
#   ./start.sh                 start (no-op if already serving)
#   ./start.sh restart         stop, then start
#   ./start.sh status          exit 0 serving, 1 not
#   ./start.sh fleet           write the fleet file from TP + WORKERS (fleet/gen-fleet.sh)
#   ./start.sh --help
#   DRY_RUN=1 ./start.sh       print the docker command, change nothing
#   DSGB10_NO_AUTH=1 ./start.sh   serve without a bearer key (no key file needed)
#
# TP is the tensor-parallel degree (ranks 0..TP-1). The engine's kernels and
# ring are TP=4 specialised: dsw_convert can cut --world N files, but
# dsgb10_rank refuses TP != 4 (engine/common/tp.h) rather than compute garbage.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"

ENV_FILE="${ENV_FILE:-$REPO/.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
  echo "[start.sh] loaded env from $ENV_FILE"
fi

export TP="${TP:-${DSGB10_TP:-4}}"
export DSGB10_TP="$TP"
IMAGE="${IMAGE:-pico-dgx-ds4:latest}"
CONTAINER="${CONTAINER_NAME:-pico-dgx-ds4}"
FLEET_ENV="${FLEET:-}"  # what the user named, before the repo default
FLEET="${FLEET_ENV:-$REPO/ds41f-4xgx10-switchless/fleet/homelab.conf}"
PORT="${DSGB10_PORT:-18893}"
ADDR="${DSGB10_ADDR:-127.0.0.1}"
KEY="${DSGB10_KEY_FILE:-$HOME/dsgb10-data/dev-api-key}"
CF="${DSGB10_CF_MLP:-$HOME/cfmlp_cml.bin}"
NO_AUTH="${DSGB10_NO_AUTH:-0}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-900}"
DRY="${DRY_RUN:-0}"

die() { echo "[start.sh] ERROR: $*" >&2; exit 1; }
warn() { echo "[start.sh] WARN: $*" >&2; }
log() { echo "[start.sh] $*"; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

# A host path under $HOME is the same file at /root in the container.
in_container() {
  case "$1" in
    "$HOME"/*) printf '/root/%s' "${1#"$HOME"/}" ;;
    *) printf '%s' "$1" ;;
  esac
}

MODE=start
case "${1:-}" in
  start | restart | status | fleet) MODE=$1; shift ;;
  -h | --help | help) usage; exit 0 ;;
esac
for arg in "$@"; do
  case "$arg" in -h | --help) usage; exit 0 ;; esac
done

[[ "$TP" =~ ^[1-9][0-9]*$ ]] || die "TP='$TP' is not a positive integer (tensor-parallel degree)"

# ./start.sh fleet: write the fleet file from TP + WORKERS instead of hand-editing it.
if [ "$MODE" = "fleet" ]; then
  exec "$REPO/ds41f-4xgx10-switchless/fleet/gen-fleet.sh" \
    --out "${FLEET_ENV:-$REPO/fleet.conf}" "$@"
fi

[ -f "$FLEET" ] || die "fleet file $FLEET not found (./start.sh fleet generates one from TP and WORKERS, or set FLEET=FILE)"
for ((r = 1; r < TP; r++)); do
  grep -q "^RANK${r}_SSH=" "$FLEET" || die "fleet file $FLEET has no RANK${r}_SSH (TP=$TP needs ranks 0..$((TP - 1)))"
done
if [ "$TP" != "4" ]; then
  warn "TP=$TP: the engine is TP=4 specialised (engine/common/tp.h). dsw_convert will cut --world $TP files, but dsgb10_rank refuses TP=$TP until the kernels and ring are generalised."
fi

running() { [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]; }

if [ "$MODE" = "status" ]; then
  running || { echo "not running (container $CONTAINER)" >&2; exit 1; }
  curl -s --max-time 5 "http://127.0.0.1:$PORT/health" | grep -qx ok && { echo "serving on http://127.0.0.1:$PORT"; exit 0; }
  echo "container $CONTAINER runs but the API is not answering" >&2
  exit 1
fi

# ---------------------------------------------------------------- 1. image
if [ "${SKIP_BUILD:-0}" = "1" ]; then
  log "SKIP_BUILD=1: using existing image $IMAGE"
elif [ "$DRY" = "1" ]; then
  log "DRY_RUN: docker build -t $IMAGE -f Dockerfile . (not run)"
else
  log "Building $IMAGE (engine + front-end, sm_121a)..."
  docker build -t "$IMAGE" -f Dockerfile .
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  [ "$DRY" = "1" ] && warn "DRY_RUN: image $IMAGE missing" || die "image $IMAGE missing (build without SKIP_BUILD=1)"
}

# ---------------------------------------------------------------- 2. files
if [ "$NO_AUTH" = "1" ]; then
  log "DSGB10_NO_AUTH=1: serving without a bearer key (only safe on loopback or a trusted network)"
elif [ ! -r "$KEY" ]; then
  if [ "$DRY" = "1" ]; then
    log "DRY_RUN: would create the API key file $KEY"
  else
    mkdir -p "$(dirname "$KEY")"
    (umask 077; openssl rand -hex 24 > "$KEY")
    log "created API key file $KEY"
  fi
else
  chmod 600 "$KEY" 2>/dev/null || warn "could not chmod 600 $KEY"
fi
[ -f "$CF" ] || {
  [ "$DRY" = "1" ] && warn "DRY_RUN: $CF missing" ||
    die "$CF missing: copy ds41f-4xgx10-switchless/tools/stack4/cfmlp_cml.bin there (the engine exits at start without it)"
}
if [ -n "${CKPT:-}" ] && [ ! -d "$CKPT" ]; then
  die "checkpoint $CKPT not found (deepseek-ai/DeepSeek-V4.1-Flash or DeepSeek-V4.1-Flash-1017)"
fi

# ---------------------------------------------------------------- 3. previous
if [ "$DRY" != "1" ] && [ "$MODE" = "start" ] && running; then
  log "$CONTAINER is already running: nothing to do (./start.sh restart to restart, ./stop.sh to stop)"
  exit 0
fi
if [ "$DRY" != "1" ] && { running || docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; }; then
  log "stopping the previous container $CONTAINER..."
  ./stop.sh
fi
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  [ "$DRY" = "1" ] && log "DRY_RUN: port $PORT is in use now" ||
    die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"
fi

# ---------------------------------------------------------------- 4. launch
# Host networking keeps the fleet file's ConnectX-7 addresses; --gpus and
# /dev/infiniband give the GPU and the RDMA devices; $HOME:/root makes the
# fleet's ~/ paths (weights, Engram, spill, key, gate) resolve in-container.
# The image carries the built engine and front-end, so no repo mount.
# service.sh start returns once serving; `sleep infinity` keeps the
# container's supervisor alive.
# The production engine arguments (README "run it"); $@ appends to them, so a
# flag given twice counts once, with the command line's value. Every knob that
# the ranks must agree on goes through --setenv: workers are reached over ssh,
# which does not forward this environment.
EARGS="${DSGB10_ENGINE_ARGS:---max-ctx 1048576 --slots 16 --big-slots 4 \
--setenv DSGB10_BUDGET=1 --setenv DSGB10_OEA=4,12,9 --setenv DSGB10_DRAFT1=6 \
--setenv DSGB10_WIDE=64 --setenv DSGB10_WIDE_COST=24,2.2,57,1.1 \
--draft-head4 --draft-markov8 --setenv DSGB10_HEAD4_C1=1 \
--setenv DSGB10_CF_MLP=$(in_container "$CF") --setenv DSGB10_CF_ALPHA=1.0 \
--setenv DSGB10_CERT=1}"
# The engine exits at start without the confidence-gate weights; a custom
# DSGB10_ENGINE_ARGS that forgets them gets them here.
case "$EARGS" in
  *DSGB10_CF_MLP=*) ;;
  *) EARGS+=" --setenv DSGB10_CF_MLP=$(in_container "$CF")" ;;
esac
[ -n "${DSGB10_RING_DIRECT:-}" ] && EARGS+=" --setenv DSGB10_RING_DIRECT=$DSGB10_RING_DIRECT"
[ $# -gt 0 ] && EARGS+=" $*"
FARGS="${DSGB10_FRONT_ARGS:---tokenizer $(in_container "${CKPT:-$HOME/DeepSeek-V4.1-Flash}")/tokenizer.json --model-name deepseek-v4.1-flash}"

run_args=(
  --name "$CONTAINER" --gpus all --network host --ipc=host
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1
  --restart unless-stopped
  -v "$HOME:/root" -v /data:/data -v "$FLEET:/root/fleet.conf:ro"
  -e "TP=$TP" -e "DSGB10_TP=$TP" -e FLEET=/root/fleet.conf
  -e "DSGB10_PORT=$PORT" -e "DSGB10_ADDR=$ADDR"
  -e "DSGB10_KEY_FILE=$(in_container "$KEY")"
  -e "DSGB10_CF_MLP=$(in_container "$CF")"  # service.sh copies it to every worker
  -e "DSGB10_ENGINE_ARGS=$EARGS"
  -e "DSGB10_FRONT_ARGS=$FARGS"
)
[ "$NO_AUTH" = "1" ] && run_args+=(-e DSGB10_NO_AUTH=1)
cmd=(docker run)
[ "${FOREGROUND:-0}" = "1" ] || cmd+=(-d)
cmd+=("${run_args[@]}" "$IMAGE" bash -c 'engine/server/service.sh start && sleep infinity')

if [ "$DRY" = "1" ]; then
  log "DRY_RUN: ${cmd[*]}"
  exit 0
fi
log "starting $CONTAINER (TP=$TP, port $PORT)"
"${cmd[@]}"

# ---------------------------------------------------------------- 5. wait + smoke
log "waiting for the API (up to ${WAIT_TIMEOUT}s)..."
tries=0
until curl -s --max-time 5 "http://127.0.0.1:$PORT/health" | grep -qx ok; do
  tries=$((tries + 1))
  if [ $((tries * 5)) -ge "$WAIT_TIMEOUT" ]; then
    docker logs --tail 40 "$CONTAINER" >&2 || true
    die "the API did not answer within ${WAIT_TIMEOUT}s (docker logs -f $CONTAINER; docker exec $CONTAINER engine/server/service.sh status)"
  fi
  sleep 5
done
auth=()
[ "$NO_AUTH" = "1" ] || auth=(-H "Authorization: Bearer $(cat "$KEY")")
code=$(curl -s -o /tmp/pico-ds4-smoke.json -w '%{http_code}' --max-time 120 \
  "http://127.0.0.1:$PORT/v1/chat/completions" "${auth[@]}" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with one word: pong"}],"max_tokens":20}') || code=000
[ "$code" = "200" ] || { cat /tmp/pico-ds4-smoke.json >&2; die "smoke test failed (HTTP $code)"; }
log "DeepSeek-V4.1-Flash is now LIVE on http://127.0.0.1:$PORT/v1 (model deepseek-v4.1-flash)"
log "logs: docker logs -f $CONTAINER"
