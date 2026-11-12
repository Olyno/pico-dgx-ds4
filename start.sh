#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash (the ds41f engine) in a Docker container, end to end.
# With nothing on the command line it does the whole install: build the image, use
# the checkpoint that is already here (or stream it from the Hub), cut every rank's
# .dsw, its Engram shards and the hash map, ship the other ranks' files to the boxes
# the fleet names, start the service, wait for the OpenAI API and run a smoke test.
# Run it again after an interruption: finished ranks are kept, unfinished ones
# resume, nothing is asked. Stop with ./stop.sh. Settings come from ./.env (or
# $ENV_FILE), then the environment.
#
#   ./start.sh            all of the above; nothing to do when it already serves
#   ./start.sh status     exit 0 while it serves, 1 otherwise
#   ./start.sh restart    stop, then all of the above
#   ./start.sh clear      delete earlier artifacts (never the checkpoint)
#   ./start.sh prepare    the artifacts only, with these overrides:
#       --world N         cut N ranks instead of the fleet's (1, 2 or 4; not 3)
#       --consume         release the checkpoint as its outputs verify
#       --ckpt DIR        a checkpoint somewhere other than the ones searched
#       --source S        local | hybrid | http (the default takes the local one)
#   DRY_RUN=1 ./start.sh  print every command, change nothing
#
# Settings come from ./.env: FLEET (the rank list and the paths) and, if the fleet
# is not written yet, WORKERS (the other boxes). The rank count is not a setting:
# rank 0 is the head and every ssh target in WORKERS is one more, so the fleet's
# four ranks are what `./start.sh` cuts. The engine loads world 4 only today
# (engine/common/tp.h); the cutter takes the worlds the model allows, 1, 2 and 4.
#
# The artifacts live where the fleet puts rank 0's weight file (RANK0_WEIGHTS):
# rank 0 reads them there, so nothing is copied on the head; its Engram shards and
# the hash map sit beside it, under engram/.
#
# The default run streams the checkpoint as it converts (--consume): 507 GB of
# model and 508 GB of rank files do not fit one disk. `./start.sh prepare` keeps
# it unless you pass --consume, and refuses instead of half-filling the disk.
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

# ------------------------------------------------------------ artifacts (files)
# dsw-prepare cuts every rank's weight file, Engram shards and the hash map into
# one directory, and rank 0 serves it from there: that is where the fleet already
# puts RANK0_WEIGHTS. The other ranks get their three files rsynced before the
# service starts, so one command on the head is the whole install.
fleet_raw() { awk -F= -v k="RANK$1_$2" '$1 == k { sub(/^[^=]*=/, ""); print; exit }' "$FLEET" 2>/dev/null || true; }
fleet_local() {
  local v
  v=$(fleet_raw "$1" "$2")
  case "$v" in
    "~") printf '%s' "$HOME" ;;
    "~/"*) printf '%s/%s' "$HOME" "${v#\~/}" ;;
    *) printf '%s' "$v" ;;
  esac
}
OUT="$HOME/dsgb10/convert"
if [ -n "$(fleet_raw 0 WEIGHTS)" ]; then
  OUT=$(dirname "$(fleet_local 0 WEIGHTS)")
fi

die() { echo "[start.sh] ERROR: $*" >&2; exit 1; }
warn() { echo "[start.sh] WARN: $*" >&2; }
log() { echo "[start.sh] $*"; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

# ------------------------------------------------------------- rank count (TP)
# TP is derived, never typed: rank 0 is the head (local) and each further rank is
# one more box, so the list you already maintain states it —
#   WORKERS="u@a u@b u@c"  ->  4 ranks, and ./start.sh fleet writes RANK1..3
# — and a fleet file names exactly the same ranks. With neither, the engine's own
# world, 4, is the only split that runs today.
read -r -a _WORKERS <<< "${WORKERS:-${WORKER:-}}"
fleet_ranks=0
if [ -f "$FLEET" ]; then
  last=$(grep -oE '^RANK[0-9]+_SSH=' "$FLEET" | grep -oE '[0-9]+' | sort -n | tail -1 || true)
  [ -n "${last:-}" ] && fleet_ranks=$((last + 1))
fi
derived_tp=0
if [ "${#_WORKERS[@]}" -gt 0 ]; then
  derived_tp=$((${#_WORKERS[@]} + 1))
  # A WORKERS copied from another setup overrides the fleet without a word, which
  # is how a four-box fleet ends up cut for two. Say it.
  if [ "$fleet_ranks" -gt 0 ] && [ "$fleet_ranks" != "$derived_tp" ]; then
    warn "WORKERS names $derived_tp rank(s), $FLEET names $fleet_ranks: WORKERS wins and ranks $derived_tp..$((fleet_ranks - 1)) are ignored (unset WORKERS to use the fleet's $fleet_ranks)"
  fi
elif [ "$fleet_ranks" -gt 0 ]; then
  derived_tp=$fleet_ranks
fi
[ "$derived_tp" -gt 0 ] || derived_tp=4
# There is no TP to set: the boxes you name are the rank count, and the engine is
# told the same number. Cutting another world on purpose is `prepare --world N`.
if [ -n "${TP:-${DSGB10_TP:-}}" ] && [ "${TP:-$DSGB10_TP}" != "$derived_tp" ]; then
  warn "ignoring TP=${TP:-$DSGB10_TP} from .env: the rank count is the boxes you name ($derived_tp) — remove the line"
fi
export TP="$derived_tp" DSGB10_TP="$derived_tp"

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
  local ttyarg=() outarg=(--out "$OUT")
  [ -t 0 ] && ttyarg=(-it)
  for a in "$@"; do [ "$a" = "--out" ] && outarg=(); done
  log "clear: deleting the rank files, journals, Engram shards and scratch of earlier runs in $OUT"
  if [ "$DRY" = "1" ]; then
    echo "    docker run --rm ${ttyarg[*]} --network none -v $HOME:/root -v /data:/data \\"
    echo "      $IMAGE prepare/target/release/dsw-prepare clear ${outarg[*]} $*"
    return 0
  fi
  docker run --rm "${ttyarg[@]}" --network none -v "$HOME:/root" -v /data:/data \
    "$IMAGE" prepare/target/release/dsw-prepare clear "${outarg[@]}" "$@" || die "clear failed"
}

# dsw-prepare inside the image. `auto` is the one-command run and adds what a
# fresh install needs — every rank plus the Engram shards (that is the price of
# --consume), the vision bundle when VISION=1, and --replace so that nothing
# waits for a terminal; the tool itself says which checkpoint it found, which
# source it chose and how much disk the peak takes. `explicit` passes the command
# line through untouched.
prepare() {
  local auto=$1
  shift
  local envarg=() a
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

  local args=("$@") worldarg=(--world "$TP") outarg=(--out "$OUT")
  for a in "$@"; do
    [ "$a" = "--world" ] && worldarg=()
    [ "$a" = "--out" ] && outarg=()
  done
  if [ "$auto" = "auto" ]; then
    [ "$VISION" = "1" ] && args+=(--vision)
    args+=(--replace --consume)
  fi
  log "prepare: $TP rank file(s), Engram shards and the hash map into $OUT"
  # --cpu-shares, not --cpus: the conversion takes every core while the box is
  # idle and steps aside when something else wants them (a third of the CPU).
  local cmd=(docker run --rm "${ttyarg[@]}" --ipc=host --network host --cpu-shares 512
    -v "$HOME:/root" -v /data:/data -w /work/ds41f-4xgx10-switchless "${envarg[@]}")
  if [ "$DRY" = "1" ]; then
    echo "    ${cmd[*]} $IMAGE prepare/target/release/dsw-prepare prepare ${worldarg[*]} ${outarg[*]} ${args[*]}"
    return 0
  fi
  # No mkdir here: the tool creates every directory it writes as root in the
  # container, and the head user may not be allowed to (an OUT on a root-owned
  # disk such as /data would fail before the run even started).
  "${cmd[@]}" "$IMAGE" prepare/target/release/dsw-prepare prepare "${worldarg[@]}" "${outarg[@]}" "${args[@]}" ||
    die "prepare failed (HF_DEBUG=1 ./start.sh prepare for the traceback)"
}

# How many of a rank's Engram shards a directory holds. A glob, not ls: compgen
# says "no match" with status 1, which `|| true` absorbs under pipefail.
shards_in() { compgen -G "$1/engram-l*-r$2of$TP.bin" 2>/dev/null | wc -l || true; }

# What one rank has right now: "<Engram shards> <weight file and hash map: 0|1>".
# Rank 0 is read here, the others over one ssh each (and not at all with
# DRY_RUN=1). A non-zero status means the box could not be reached — that stops
# the run rather than sending it off to download a checkpoint it does not need.
rank_probe() {
  local r=$1 w h e host files=0
  w=$(fleet_raw "$r" WEIGHTS)
  h=$(fleet_raw "$r" HASHMAP)
  e=$(fleet_raw "$r" ENGRAM)
  [ -n "$w" ] && [ -n "$h" ] && [ -n "$e" ] || return 1
  if [ "$r" = "0" ]; then
    [ -s "$(fleet_local 0 WEIGHTS)" ] && [ -s "$(fleet_local 0 HASHMAP)" ] && files=1
    printf '%s %s' "$(shards_in "$(fleet_local 0 ENGRAM)" 0)" "$files"
    return 0
  fi
  host=$(fleet_raw "$r" SSH)
  [ -n "$host" ] || return 1
  [ "$DRY" = "1" ] && { printf '0 0'; return 0; }
  # The paths are the fleet's own (`~` is left for the remote shell to expand).
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    "f=0; [ -s $w ] && [ -s $h ] && f=1; echo \$(ls $e/engram-l*-r${r}of$TP.bin 2>/dev/null | wc -l) \$f"
}

# One rank's three things: its weight file, the shared hash map and its own Engram
# shards. rsync --size-only makes a second run a few stats, and --partial makes an
# interrupted one resume.
install_rank() {
  local r=$1 w h e host
  w=$(fleet_raw "$r" WEIGHTS)
  h=$(fleet_raw "$r" HASHMAP)
  e=$(fleet_raw "$r" ENGRAM)
  host=$(fleet_raw "$r" SSH)
  [ -n "$host" ] || die "RANK${r}_SSH is missing from $FLEET"
  log "  rank $r -> $host"
  if [ "$DRY" = "1" ]; then
    echo "    ssh $host mkdir -p $(dirname "$w") $(dirname "$h") $e"
    echo "    rsync -h --progress --partial --size-only $OUT/rank$r.dsw $host:$w"
    echo "    rsync -h --progress --partial --size-only $OUT/engram/engram_hashmap.bin $host:$h"
    echo "    rsync -h --progress --partial --size-only $OUT/engram/engram-l*-r${r}of$TP.bin $host:$e/"
    return 0
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "mkdir -p $(dirname "$w") $(dirname "$h") $e" ||
    die "cannot reach $host (rank $r)"
  rsync -h --progress --partial --size-only "$OUT/rank$r.dsw" "$host:$w" ||
    die "rank $r's weight file did not arrive at $host:$w"
  rsync -h --progress --partial --size-only "$OUT/engram/engram_hashmap.bin" "$host:$h" ||
    die "the hash map did not arrive at $host:$h"
  rsync -h --progress --partial --size-only "$OUT"/engram/engram-l*-r${r}of$TP.bin "$host:$e/" ||
    die "rank $r's Engram shards did not arrive at $host:$e/"
}

# The files, cut only when they are not already where the fleet says: a second
# ./start.sh is a handful of stats and a wait.
ensure_artifacts() {
  local r probe counts=() files=() maxc=0 missing=()
  for ((r = 0; r < TP; r++)); do
    probe=$(rank_probe "$r") || die "cannot reach rank $r ($(fleet_raw "$r" SSH)): its files cannot be checked or shipped"
    counts+=("${probe% *}")
    files+=("${probe#* }")
    [ "${counts[r]}" -gt "$maxc" ] && maxc=${counts[r]}
  done
  # The rank with the most Engram shards sets the number every rank must have:
  # a directory that lost one (an interrupted ship) is not ready, and the same
  # count on all of them means the set is whole.
  [ "$maxc" -gt 0 ] || maxc=1
  for ((r = 0; r < TP; r++)); do
    if [ "${files[r]}" != "1" ] || [ "${counts[r]}" -lt "$maxc" ]; then
      missing+=("$r")
    fi
  done
  if [ ${#missing[@]} -eq 0 ]; then
    log "artifacts: all $TP rank(s) in place in $OUT ($maxc Engram shard(s) each)"
    return 0
  fi
  log "artifacts: rank(s) ${missing[*]} not in place — preparing (hours; Ctrl-C is safe, the next run resumes)"
  prepare auto
  if [ "$DRY" = "1" ]; then
    for ((r = 1; r < TP; r++)); do install_rank "$r"; done
    log "DRY_RUN: would check all $TP rank(s) once the install is done"
    return 0
  fi
  for ((r = 1; r < TP; r++)); do install_rank "$r"; done
  for ((r = 0; r < TP; r++)); do
    probe=$(rank_probe "$r") || die "rank $r stopped answering ($(fleet_raw "$r" SSH))"
    if [ "${probe% *}" -lt "$maxc" ] || [ "${probe#* }" != "1" ]; then
      die "rank $r still lacks its files ($(fleet_raw "$r" WEIGHTS), $(fleet_raw "$r" ENGRAM))"
    fi
  done
  log "artifacts: all $TP rank(s) in place in $OUT ($maxc Engram shard(s) each)"
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

MODE=run
FORCE_STOP=0
case "${1:-}" in
  -h | --help | help) usage; exit 0 ;;
  "" | -*) ;;  # no command: a run, and the rest are engine flags
  run | start) shift ;;
  restart) FORCE_STOP=1; shift ;;
  status | prepare | clear) MODE=$1; shift ;;
  *) die "'$1' is not a command (./start.sh --help): ./start.sh [status|restart|clear|prepare] [engine flags]" ;;
esac
for arg in "$@"; do
  case "$arg" in -h | --help) usage; exit 0 ;; esac
done

[[ "$TP" =~ ^[1-9][0-9]*$ ]] || die "TP='$TP' is not a positive integer (tensor-parallel degree)"

# The kernels and the ring are TP=4 by construction (engine/common/tp.h). Cutting
# another world on the way to serving would spend hours on files no build can
# load, so the default run refuses; ./start.sh prepare --world N still can, on
# purpose (and says so).
if [ "$TP" != "4" ] && [ "$MODE" != "prepare" ]; then
  die "the engine loads TP=4 only (engine/common/tp.h), so $TP rank(s) could never be served.
  The rank count follows the boxes you name: put three ssh targets in WORKERS, or let the
  fleet file name ranks 1..3. For another split on purpose: ./start.sh prepare --world N"
fi

# A clear only touches files: no fleet, no GPU, no checkpoint.
if [ "$MODE" != "clear" ]; then
  # WORKERS alone is enough: the fleet file is written from it the first time.
  if [ ! -f "$FLEET" ]; then
    [ -n "${WORKERS:-${WORKER:-}}" ] || die "no fleet file $FLEET and no WORKERS: set WORKERS=\"u@a u@b u@c\" in .env"
    "$REPO/ds41f-4xgx10-switchless/fleet/gen-fleet.sh" --out "$FLEET" >/dev/null ||
      die "could not write $FLEET from WORKERS"
    log "fleet: wrote $FLEET from WORKERS"
  fi
  for ((r = 1; r < TP; r++)); do
    grep -q "^RANK${r}_SSH=" "$FLEET" || die "fleet file $FLEET has no RANK${r}_SSH (TP=$TP needs ranks 0..$((TP - 1)))"
  done
  if [ "$TP" != "4" ]; then
    warn "TP=$TP: the engine is TP=4 specialised (engine/common/tp.h). prepare cuts --world $TP files, but dsgb10_rank refuses TP=$TP until the kernels and ring are generalised."
  fi
  # dsw-prepare cuts the weight files, the Engram shards and the hash map into one
  # directory, and rank 0 serves it from there. A fleet that puts rank 0 somewhere
  # else would serve nothing, so say what to change instead of copying 180 GB.
  if [ "$(fleet_local 0 WEIGHTS)" != "$OUT/rank0.dsw" ] ||
    [ "$(fleet_local 0 HASHMAP)" != "$OUT/engram/engram_hashmap.bin" ] ||
    [ "$(fleet_local 0 ENGRAM)" != "$OUT/engram" ]; then
    die "$FLEET puts rank 0's files outside the directory they are cut into. Set, in $FLEET:
    RANK0_WEIGHTS=$OUT/rank0.dsw
    RANK0_HASHMAP=$OUT/engram/engram_hashmap.bin
    RANK0_ENGRAM=$OUT/engram
  (it has RANK0_WEIGHTS=$(fleet_local 0 WEIGHTS), RANK0_HASHMAP=$(fleet_local 0 HASHMAP), RANK0_ENGRAM=$(fleet_local 0 ENGRAM))"
  fi
fi

running() { [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]; }

if [ "$MODE" = "status" ]; then
  running || { echo "not running (container $CONTAINER)" >&2; exit 1; }
  curl -s --max-time 5 "http://127.0.0.1:$PORT/health" | grep -qx ok && { echo "serving on http://127.0.0.1:$PORT"; exit 0; }
  echo "container $CONTAINER runs but the API is not answering" >&2
  exit 1
fi

# Serving already: no rebuild, no re-cut, no restart.
if [ "$MODE" = "run" ] && [ "$FORCE_STOP" = "0" ] && [ "$DRY" != "1" ] && running; then
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/health" | grep -qx ok; then
    log "already serving on http://127.0.0.1:$PORT: nothing to do (./start.sh restart, ./stop.sh, ./start.sh status)"
    exit 0
  fi
  log "the container runs but the API is not answering: restarting it"
  ./stop.sh
fi

# ---------------------------------------------------------------- 1. image
ensure_image
if [ "$MODE" = "prepare" ]; then
  prepare explicit "$@"
  exit 0
fi
if [ "$MODE" = "clear" ]; then
  clear_outputs "$@"
  exit 0
fi
# ---------------------------------------------------------------- 2. artifacts
# The whole install, skipped when the fleet's paths already hold it: cut the rank
# files, Engram shards and hash map (the tool decides the checkpoint and the
# source), then ship the other ranks' three files.
ensure_artifacts

# ---------------------------------------------------------------- 3. files
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
  for c in "$OUT/tokenizer.json" "$HOME/DeepSeek-V4.1-Flash-1017/tokenizer.json" \
    "$HOME/DeepSeek-V4.1-Flash/tokenizer.json" /data/models/DeepSeek-V4.1-Flash/tokenizer.json; do
    if [ -f "$c" ]; then TOKENIZER=$c && break; fi
  done
fi
if [ -z "$TOKENIZER" ]; then
  [ "$DRY" = "1" ] && warn "DRY_RUN: no tokenizer.json found (run ./start.sh prepare, or set TOKENIZER=FILE)" ||
    die "no tokenizer.json found: run ./start.sh prepare, or set TOKENIZER=FILE"
fi

# ---------------------------------------------------------------- 4. previous
if [ "$DRY" != "1" ] && { running || docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; }; then
  log "stopping the previous container $CONTAINER..."
  ./stop.sh
fi
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  [ "$DRY" = "1" ] && log "DRY_RUN: port $PORT is in use now" ||
    die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"
fi

# ---------------------------------------------------------------- 5. launch
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

# ---------------------------------------------------------------- 6. wait + smoke
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
