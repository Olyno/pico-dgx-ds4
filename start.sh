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
#   ./start.sh prepare         one-time on rank 0: download the checkpoint if missing, then
#                              cut each rank's .dsw and its Engram shards (--no-download skips the fetch)
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

# One-time setup on rank 0: the checkpoint (if missing), then each rank's .dsw and
# its Engram shards, all with the image's own binaries. Serving needs those
# artefacts and a tokenizer.json, not the checkpoint itself.
prepare() {
  local ckpt=${CKPT:-$HOME/DeepSeek-V4.1-Flash-1017}
  local out=${PREPARE_OUT:-$HOME/dsgb10/convert}
  local repo=${HF_REPO:-deepseek-ai/DeepSeek-V4.1-Flash-1017} rev=${HF_REV:-1bae0e10}
  local nodl=0 a r
  for a in "$@"; do
    case $a in
      --no-download) nodl=1 ;;
      *) die "prepare: unknown argument $a" ;;
    esac
  done
  for a in "$ckpt" "$out"; do
    case "$a" in
      "$HOME"/*|/data/*) ;;
      *) die "keep CKPT and PREPARE_OUT under \$HOME or /data (the container mounts only those): $a" ;;
    esac
  done
  [ "$DRY" = "1" ] || [ -e /dev/nvidia0 ] ||
    die "prepare needs the GB10 GPU (dsw_convert tiles on it): run it on rank 0, the head"
  local ckpt_in out_in
  ckpt_in=$(in_container "$ckpt") out_in=$(in_container "$out")

  # Xet's writer thread is what fails on some bind mounts ("Background writer
  # channel closed"); the plain HTTP path is slower and reliable. HF_XET=1 re-enables it.
  local xet=${HF_HUB_DISABLE_XET:-1}
  [ -n "${HF_XET:-}" ] && xet=0
  prep_exec() {  # what; shell in the image
    log "$1"
    [ "$DRY" = "1" ] && { echo "    $2"; return 0; }
    docker run --rm --gpus all --ipc=host --network host -v "$HOME:/root" -v /data:/data \
      -e "HF_HUB_DISABLE_XET=$xet" ${HF_TOKEN:+-e HF_TOKEN} \
      "$IMAGE" bash -lc "cd /work/ds41f-4xgx10-switchless && $2" ||
      die "prepare: '$1' failed"
  }

  if [ "$nodl" = 1 ]; then
    log "prepare: --no-download, using $ckpt"
  elif [ -f "$ckpt/config.json" ]; then
    log "prepare: checkpoint present at $ckpt"
  else
    # The checkpoint is ~510 GB: fail before a long download that cannot finish.
    local dir free_gb
    dir=$(dirname "$ckpt")
    while [ ! -d "$dir" ] && [ "$dir" != / ]; do dir=$(dirname "$dir"); done
    free_gb=$(df -Pk "$dir" | awk 'NR == 2 { printf "%d", $4 / 1048576 }')
    if [ "${free_gb:-0}" -lt "${NEED_GB:-520}" ]; then
      [ "$DRY" = "1" ] && warn "DRY_RUN: only ${free_gb:-?} GiB free on $dir (the checkpoint needs ~510)" ||
        die "only ${free_gb:-?} GiB free on $dir; the checkpoint needs ~510 GiB (NEED_GB=N to override)"
    fi
    # hf (huggingface_hub >= 0.27) or the older CLI, chosen once: a `||` chain
    # would mask a failed download as "command not found".
    prep_exec "download $repo @ $rev into $ckpt (~510 GB, resumable, xet off)" \
      "if command -v hf >/dev/null 2>&1; then hf download $repo --revision $rev --local-dir $ckpt_in; else huggingface-cli download $repo --revision $rev --local-dir $ckpt_in; fi"
  fi
  for ((r = 0; r < TP; r++)); do
    prep_exec "cut rank $r of $TP" \
      "engine/build/weights/dsw_convert --ckpt $ckpt_in --rank $r --world $TP --out $out_in/rank$r.dsw"
    prep_exec "verify rank $r" \
      "engine/build/weights/dsw_verify --ckpt $ckpt_in --file $out_in/rank$r.dsw --threads 4"
  done
  for ((r = 0; r < TP; r++)); do
    prep_exec "pack rank $r's Engram shards" \
      "engine/build/engram/engram_pack $ckpt_in $out_in/engram $r"
  done
  prep_exec "write the Engram hash map" \
    "python3 engine/engram/py/make_hashmap_stdlib.py $ckpt_in $out_in/engram/engram_hashmap.bin"

  log "done: rank files and engram/ under $out"
  log "install rank 0's files where the fleet file says, then copy the rest:"
  echo "    cp $out/rank0.dsw <RANK0_WEIGHTS>; cp -r $out/engram/* <RANK0_ENGRAM>/; cp $out/engram/engram_hashmap.bin <RANK0_HASHMAP>"
  for ((r = 1; r < TP; r++)); do
    echo "    rsync -h --progress $out/rank$r.dsw <RANK${r}_SSH>:<RANK${r}_WEIGHTS>"
    echo "    rsync -h --progress $out/engram/engram-l*-r${r}of4.bin <RANK${r}_SSH>:<RANK${r}_ENGRAM>/"
    echo "    rsync -h --progress $out/engram/engram_hashmap.bin <RANK${r}_SSH>:<RANK${r}_HASHMAP>"
  done
  log "then: ./start.sh"
}

MODE=start
case "${1:-}" in
  start | restart | status | fleet | prepare) MODE=$1; shift ;;
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
ensure_image
if [ "$MODE" = "prepare" ]; then
  prepare "$@"
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
# Serving needs the tokenizer, not the 510 GB checkpoint: the rank files and the
# Engram shards are cut from the checkpoint once (`./start.sh prepare`).
TOKENIZER="${TOKENIZER:-}"
if [ -z "$TOKENIZER" ]; then
  for c in "${CKPT:+$CKPT/tokenizer.json}" "$HOME/DeepSeek-V4.1-Flash/tokenizer.json" \
    "$HOME/DeepSeek-V4.1-Flash-1017/tokenizer.json" /data/models/DeepSeek-V4.1-Flash/tokenizer.json; do
    if [ -n "$c" ] && [ -f "$c" ]; then TOKENIZER=$c && break; fi
  done
fi
if [ -z "$TOKENIZER" ]; then
  [ "$DRY" = "1" ] && warn "DRY_RUN: no tokenizer.json found under CKPT or ~/DeepSeek-V4.1-Flash*" ||
    die "no tokenizer.json found: set CKPT to a checkpoint dir (or TOKENIZER=FILE); ./start.sh prepare downloads one"
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
