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
#   ./start.sh prepare [ARGS]  one-time on rank 0: the checkpoint (downloaded or repaired
#                              only when it is not already complete), then every rank's .dsw
#                              and its Engram shards. ARGS go to dsw-prepare: --consume
#                              releases the checkpoint's blocks as the outputs land,
#                              --source http streams it instead of storing it,
#                              --source hybrid caches one whole shard at a time, --replace
#                              skips the "delete these?" prompt, --dest R=DIR writes a rank
#                              straight to a mounted path
#   ./start.sh clear           delete the rank files, journals, Engram shards and
#                              scratch of earlier runs, keeping the checkpoint
#                              (prepare --clear does the same, then converts)
#   ./start.sh reshard --world N   re-cut an existing rank set for another TP from
#                              the rank files alone (no checkpoint, no network, no
#                              GPU): --from DIR (default ~/dsgb10/convert), --to DIR
#                              (default ~/dsgb10/convert-tpN). The engine still
#                              serves TP=4 only, so this is for trying other splits.
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
# VISION=1 keeps the image encoder's slice of the checkpoint (prepare --vision) and
# points dsfront at it; VISION_IMAGE is the torch/PIL image that runs the encoder.
VISION="${VISION:-0}"
VISION_DIR="${VISION_DIR:-$HOME/dsgb10/vision}"
VISION_PORT=18950
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

ensure_image() {
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
}

# One-time setup on rank 0: the checkpoint (downloaded or repaired when it is not
# already complete), then every rank's .dsw and its Engram shards, by dsw-prepare
# inside the image. Serving needs those artefacts, a tokenizer and the hash map,
# not the checkpoint itself. Extra arguments go to dsw-prepare: --consume releases
# the checkpoint's blocks as the outputs land, --source http streams it instead of
# storing it, --replace skips the "delete these?" prompt, --dest R=DIR writes a
# rank straight to a mounted path.
# ./start.sh clear: the outputs of earlier runs, never the checkpoint. Offline, so
# it cannot start a download by accident.
clear_outputs() {
  local ttyarg=()
  [ -t 0 ] && ttyarg=(-it)
  log "clear: deleting the rank files, journals, Engram shards and scratch of earlier runs"
  if [ "$DRY" = "1" ]; then
    echo "    docker run --rm ${ttyarg[*]} --network none -v $HOME:/root \\"
    echo "      $IMAGE prepare/target/release/dsw-prepare clear $*"
    return 0
  fi
  docker run --rm "${ttyarg[@]}" --network none -v "$HOME:/root" \
    "$IMAGE" prepare/target/release/dsw-prepare clear "$@" || die "clear failed"
}

prepare() {
  local out="$HOME/dsgb10/convert"
  local envarg=()
  # docker run inherits none of this shell's environment: a proxy or a mirror
  # endpoint set here would be lost and the download would hang at 0 %.
  for v in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy \
    HF_ENDPOINT HF_TOKEN HF_DEBUG HF_HUB_ETAG_TIMEOUT; do
    [ -n "${!v:-}" ] && envarg+=(-e "$v")
  done
  # The shards are Xet-backed and the classic path refuses files this large.
  [ -n "${HF_NO_XET:-}" ] && envarg+=(-e HF_HUB_DISABLE_XET=1)
  # The prompt that asks before deleting existing outputs needs a terminal.
  local ttyarg=()
  [ -t 0 ] && ttyarg=(-it)

  local visionarg=()
  [ "$VISION" = "1" ] && visionarg=(--vision)
  log "prepare: checkpoint (only if not already complete), then $TP rank file(s) and Engram shards into $out"
  [ "$DRY" = "1" ] && {
    echo "    docker run --rm ${ttyarg[*]} --ipc=host --network host -v $HOME:/root -v /data:/data \\"
    echo "      $IMAGE prepare/target/release/dsw-prepare prepare --world $TP ${visionarg[*]} $*"
    return 0
  }
  mkdir -p "$out"
  docker run --rm "${ttyarg[@]}" --ipc=host --network host -v "$HOME:/root" -v /data:/data \
    -w /work/ds41f-4xgx10-switchless "${envarg[@]}" \
    "$IMAGE" prepare/target/release/dsw-prepare prepare --world "$TP" "${visionarg[@]}" "$@" ||
    die "prepare failed (HF_DEBUG=1 ./start.sh prepare for the traceback)"
}

# ./start.sh reshard: re-cut a finished rank set for another world. Offline by
# construction (`--network none`): every byte comes from the old files.
reshard() {
  local ttyarg=()
  [ -t 0 ] && ttyarg=(-it)
  log "reshard: re-cutting an existing rank set inside the container (no checkpoint, no network)"
  if [ "$DRY" = "1" ]; then
    echo "    docker run --rm ${ttyarg[*]} --network none -v $HOME:/root \\"
    echo "      $IMAGE prepare/target/release/dsw-prepare reshard $*"
    return 0
  fi
  docker run --rm "${ttyarg[@]}" --network none -v "$HOME:/root" -w /work/ds41f-4xgx10-switchless \
    "$IMAGE" prepare/target/release/dsw-prepare reshard "$@" || die "reshard failed"
}

# The image encoder is a side process on rank 0 (docs/vision.md): it imports the
# checkpoint's own inference/vision.py, so it needs torch, PIL and safetensors —
# an image of its own, named by VISION_IMAGE, not the engine image.
start_vision() {
  [ "$VISION" = "1" ] || return 0
  if [ -z "${VISION_IMAGE:-}" ]; then
    warn "VISION=1 but VISION_IMAGE is unset: start tools/vision/vision_encoder.py yourself; image requests answer 503 until it runs"
    return 0
  fi
  [ -f "$VISION_DIR/model.safetensors.index.json" ] ||
    die "no vision bundle in $VISION_DIR: run VISION=1 ./start.sh prepare first"
  if [ "$DRY" = "1" ]; then
    log "DRY_RUN: docker run $VISION_IMAGE /app/vision_encoder.py --host 127.0.0.1 --port $VISION_PORT (model $VISION_DIR)"
    return 0
  fi
  docker rm -f dsgx10-vision >/dev/null 2>&1 || true
  docker run -d --name dsgx10-vision --restart unless-stopped --gpus all --net=host --memory=6g \
    -e VISION_PRELOAD=1 -v "$VISION_DIR:/model:ro" \
    -v "$REPO/ds41f-4xgx10-switchless/tools/vision:/app:ro" \
    --entrypoint python3 "$VISION_IMAGE" /app/vision_encoder.py \
    --host 127.0.0.1 --port "$VISION_PORT" >/dev/null ||
    die "the vision encoder did not start from image $VISION_IMAGE"
  log "vision encoder on 127.0.0.1:$VISION_PORT (model $VISION_DIR)"
}

MODE=start
case "${1:-}" in
  start | restart | status | fleet | prepare | clear | reshard) MODE=$1; shift ;;
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

# A clear or a reshard only touches files: no fleet, no GPU, no checkpoint.
if [ "$MODE" != "clear" ] && [ "$MODE" != "reshard" ]; then
  [ -f "$FLEET" ] || die "fleet file $FLEET not found (./start.sh fleet generates one from TP and WORKERS, or set FLEET=FILE)"
  for ((r = 1; r < TP; r++)); do
    grep -q "^RANK${r}_SSH=" "$FLEET" || die "fleet file $FLEET has no RANK${r}_SSH (TP=$TP needs ranks 0..$((TP - 1)))"
  done
  if [ "$TP" != "4" ]; then
    warn "TP=$TP: the engine is TP=4 specialised (engine/common/tp.h). prepare cuts --world $TP files, but dsgb10_rank refuses TP=$TP until the kernels and ring are generalised."
  fi
fi

running() { [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]; }

if [ "$MODE" = "status" ]; then
  running || { echo "not running (container $CONTAINER)" >&2; exit 1; }
  curl -s --max-time 5 "http://127.0.0.1:$PORT/health" | grep -qx ok && { echo "serving on http://127.0.0.1:$PORT"; exit 0; }
  echo "container $CONTAINER runs but the API is not answering" >&2
  exit 1
fi

# ---------------------------------------------------------------- 1. image
ensure_image
if [ "$MODE" = "prepare" ]; then
  prepare "$@"
  exit 0
fi
if [ "$MODE" = "clear" ]; then
  clear_outputs "$@"
  exit 0
fi
if [ "$MODE" = "reshard" ]; then
  reshard "$@"
  exit 0
fi

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
# stack4's confidence-gate weights: the repo carries them, the ranks read a copy
# at $DSGB10_CF_MLP (README step 3). Deploy it here; without it the engine still
# serves — the shipped confidence head instead, slower drafts, the same tokens.
GATE_ON=1
if [ ! -f "$CF" ]; then
  CF_SRC=$REPO/ds41f-4xgx10-switchless/tools/stack4/cfmlp_cml.bin
  if [ ! -f "$CF_SRC" ]; then
    GATE_ON=0
    warn "$CF and $CF_SRC are both missing: serving without the learned confidence gate (slower drafts, same tokens)"
  elif [ "$DRY" = "1" ]; then
    log "DRY_RUN: would copy $CF_SRC -> $CF"
  else
    mkdir -p "$(dirname "$CF")" && cp "$CF_SRC" "$CF" &&
      log "deployed the confidence-gate weights to $CF"
  fi
fi
# Serving needs the tokenizer, not the 510 GB checkpoint: `./start.sh prepare`
# copies it beside the rank files, so it survives the checkpoint's deletion.
TOKENIZER="${TOKENIZER:-}"
if [ -z "$TOKENIZER" ]; then
  for c in "$HOME/dsgb10/convert/tokenizer.json" "$HOME/DeepSeek-V4.1-Flash-1017/tokenizer.json" \
    "$HOME/DeepSeek-V4.1-Flash/tokenizer.json" /data/models/DeepSeek-V4.1-Flash/tokenizer.json; do
    if [ -f "$c" ]; then TOKENIZER=$c && break; fi
  done
fi
if [ -z "$TOKENIZER" ]; then
  [ "$DRY" = "1" ] && warn "DRY_RUN: no tokenizer.json found (run ./start.sh prepare, or set TOKENIZER=FILE)" ||
    die "no tokenizer.json found: run ./start.sh prepare, or set TOKENIZER=FILE"
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
CFSET=""
if [ "$GATE_ON" = 1 ]; then
  CFSET="--setenv DSGB10_CF_MLP=$(in_container "$CF") --setenv DSGB10_CF_ALPHA=1.0"
fi
EARGS="${DSGB10_ENGINE_ARGS:---max-ctx 1048576 --slots 16 --big-slots 4 \
--setenv DSGB10_BUDGET=1 --setenv DSGB10_OEA=4,12,9 --setenv DSGB10_DRAFT1=6 \
--setenv DSGB10_WIDE=64 --setenv DSGB10_WIDE_COST=24,2.2,57,1.1 \
--draft-head4 --draft-markov8 --setenv DSGB10_HEAD4_C1=1 \
$CFSET --setenv DSGB10_CERT=1}"
# A custom DSGB10_ENGINE_ARGS that forgets the gate gets it here — when it is
# available; the engine serves without it, on the shipped confidence head.
if [ "$GATE_ON" = 1 ]; then
  case "$EARGS" in
    *DSGB10_CF_MLP=*) ;;
    *) EARGS+=" --setenv DSGB10_CF_MLP=$(in_container "$CF")" ;;
  esac
fi
[ -n "${DSGB10_RING_DIRECT:-}" ] && EARGS+=" --setenv DSGB10_RING_DIRECT=$DSGB10_RING_DIRECT"
[ $# -gt 0 ] && EARGS+=" $*"
FARGS="${DSGB10_FRONT_ARGS:---tokenizer $(in_container "${TOKENIZER:-$HOME/DeepSeek-V4.1-Flash/tokenizer.json}") --model-name deepseek-v4.1-flash}"
# VISION=1 accepts image parts; without the address dsfront refuses them as before.
if [ "$VISION" = "1" ]; then
  case "$FARGS" in
    *--vision-addr*) ;;
    *) FARGS+=" --vision-addr 127.0.0.1:$VISION_PORT" ;;
  esac
fi

run_args=(
  --name "$CONTAINER" --gpus all --network host --ipc=host
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1
  --restart unless-stopped
  -v "$HOME:/root" -v /data:/data -v "$FLEET:/root/fleet.conf:ro"
  -e "TP=$TP" -e "DSGB10_TP=$TP" -e FLEET=/root/fleet.conf
  -e "DSGB10_PORT=$PORT" -e "DSGB10_ADDR=$ADDR"
  -e "DSGB10_KEY_FILE=$(in_container "$KEY")"
  -e "DSGB10_ENGINE_ARGS=$EARGS"
  -e "DSGB10_FRONT_ARGS=$FARGS"
)
[ "$NO_AUTH" = "1" ] && run_args+=(-e DSGB10_NO_AUTH=1)
[ "$GATE_ON" = 1 ] && run_args+=(-e "DSGB10_CF_MLP=$(in_container "$CF")")  # service.sh copies it to workers
cmd=(docker run)
[ "${FOREGROUND:-0}" = "1" ] || cmd+=(-d)
cmd+=("${run_args[@]}" "$IMAGE" bash -c 'engine/server/service.sh start && sleep infinity')

if [ "$DRY" = "1" ]; then
  log "DRY_RUN: ${cmd[*]}"
  exit 0
fi
start_vision
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
