# Changes in pico-dgx-ds4

Fork of blaesus/dsgx10 by Andy Shu (MIT). Changes by Patrick Moorhead.
Base: upstream `a92b8f4` (master, 2026-10-18). All changes are in `ds41f-4xgx10-switchless/`.

Measured on four GB10 boxes (TP=4) over a 200G switch, serving DeepSeek-V4.1-Flash-1017, `--max-ctx 1048576`.
Speeds are decode tok/s (total across concurrent requests). "agent" = 8 concurrent sessions with
80K-300K-token prompts, 512 tokens per turn. qeval = 75-question quality eval, bar >= 70.

| # | Change | Files | Default | Lossless | Before -> after |
|---|---|---|---|---|---|
| 1 | 17-bit block index in `select_bins` (prefix split `nlow`, guard `bkeys_stride` <= 131072) | `engine/attn/indexer.cu` | on | yes | max context 512K -> 1M; needle at 947,826 tokens 3/3; prefill 9.2k tok/s; qeval 71/75; c1 unchanged |
| 2 | `DefaultBodyLimit` 64 MB (axum default 2 MB) | `frontend/src/server.rs` | on | yes | HTTP 413 above ~400K tokens -> 948K-token prompts accepted |
| 3 | 16 slots, big/small slot units (`--big-slots`), slot-aware admission, big-slot wait instead of die | `engine/model/*`, `engine/server/{main,serve}.cpp` | `--slots`/`--big-slots` | yes | code c16 172 -> 234-247, c8 170 -> 195-200; c1 unchanged; qeval 71 |
| 4 | 9..16-row normed GEMVs as rmsnorm + plain r16 GEMV (`DSGB10_FUSED16=1` reverts) | `engine/model/model.cu` | on | yes (same bits) | c16 234 -> 253, c4 187 -> 204, step 77.3 -> 74.8 ms |
| 5 | Batch-aware expert re-routing `DSGB10_OEA=K0,KP,MINROWS` | `engine/kernels/{common.cuh,moe.cu}` | off | **no** | with 4,12,9: c16 253 -> 286, agent 180.7 -> 203.4; qeval 71/72 |
| 6 | Cross-session draft row budget `DSGB10_BUDGET=1` | `engine/model/{spec.cu,ops.h,model.h}`, `engine/server/cmd.cpp` | off | yes | c16 253 -> 259, agent 180.7 -> 189.5 |

## End to end

| Build | chat c1 | code c1 | c4 | c8 | c16 | agent | qeval c4/c16 | max ctx |
|---|---|---|---|---|---|---|---|---|
| upstream, 256K, 8 slots | 80 | 140 | - | - | 177 | - | 71 | 256K |
| + 1 + 2, 1M, 4 slots | 78-82 | 134-140 | - | 170 | 172 | - | 71 | 1M |
| + 3, 16 slots (4 big) | 79 | 135-140 | 187 | 197 | 234 | 180.1 | 71/71 | 1M |
| + 4 + 6 (`DSGB10_BUDGET=1`) | 80.4 | 137.8 | 211 | 212 | 259 | 189.5 | 71/71 | 1M |
| + 5 (`DSGB10_OEA=4,12,9`) | 77.5-78.5 | 137.8-139.3 | 239 | 236 | 296 | 215.1 | 71/73 | 1M |

Engine args for the last row: `--max-ctx 1048576 --slots 16 --big-slots 4 --setenv DSGB10_BUDGET=1 --setenv DSGB10_OEA=4,12,9`.

## Wide step (branch wide64, now main)

| # | Change | Default | Lossless | Before -> after |
|---|---|---|---|---|
| 7 | Wide batch step, 17-64 rows (`DSGB10_WIDE=W`): prefill GEMMs + `moe_prefill`, indexer/compressor/head in 16-row groups | off | ~1 ulp | c16 296 -> 350, agent 215 -> 240 |
| 8 | Cost-aware wide gate (`DSGB10_WIDE_COST`) | on with WIDE | yes | wide only when drafts pay |
| 9 | One-pass drafter (`DSGB10_DRAFT1=6`) + coded-head GEMM (`DSGB10_HEADG`) | off / on | ~1 ulp / same bits | 48-row step 120 -> 109 ms |
| 10 | Gate re-fit + `WIDE=64` | - | yes | c16 350 -> 438, c8 295 -> 330, agent 240 -> 260; qeval 71/71; needle 3/3 at 948K |

## Vision

| # | Change | Files | Default | Lossless | Before -> after |
|---|---|---|---|---|---|
| 11 | Image input: side encoder (the checkpoint's `inference/vision.py` + `image_processor.py`, SDPA on contiguous 4D views: MATH -> flash, 3.1 s -> 0.39 s per 444-token image); dsfront `--vision-addr` accepts `image_url`/`image` parts, expands each placeholder into its span; SUBMIT flag 8 carries the spans' rows; prefill injects them, span rows route with `ffn.gate.bias_vl`, Engram dead on spans; decode untouched | `tools/vision/`, `frontend/src/{vision,server,protocol,main}.rs`, `engine/common/frontend_protocol.h`, `engine/model/{prefill,model,ops}.cu`, `engine/server/{serve,cmd}.cpp`, `engine/engram/hash.cpp` | off (needs `--vision-addr`) | text: same path | images refused (400) -> 15/15 image checks, ~0.26 s TTFT per image; text: chat/code c1 79.7/138.4 -> 81.9/144.4, distinct c16 367 -> 380, agent 260 -> 266, qeval 71/71 -> 71/72, needle 948K 3/3 (noise) |

## Single user (c1fast)

The served build is the exact path: chat c1 ~80-82, code c1 ~138-144 (9-run medians). Change 12 below was a lossy
experiment; the code stays in tree, off by default, and is not used.

| # | Change | Files | Default | Lossless | Before -> after |
|---|---|---|---|---|---|
| 12 | Relaxed accept for one-session greedy decode (`DSGB10_RELAX_P=P`: a draft the target gives >= P probability is kept as the row's token; `DSGB10_RELAX=G` logit-gap form) + `DSGB10_RELAX_THRESH=T` fixed verify gate for one-session greedy decodes. Batch steps and sampled requests keep exact accept and the default gate | `engine/model/sample.cu`, `engine/server/cmd.cpp` | off | **no** | **REMOVED from the build 2026-11-09 (outputs must be bit-identical; this was lossy).** Measured with P=0.003, T=0.05: chat c1 81.6 -> 102.8, code c1 140.1 -> 162.9 (our bench, 9-run medians); sparkDash prose 101 -> 145, code 166 -> 172; qeval c1/c4/c16 72/71/71; distinct c16 381, copies 434, distinct c8 311, agent 260 (vis1 same day 262), needle 948K 3/3, images 15/15 |

Method check (sparkDash 1.8.8 at 754f40a, upstream's own sdrun.mjs harness): upstream's README 95 / 151 vs our 80 / 140
was the method (one easier prompt, 256 tokens), not the engine: the exact build scores 101 / 166 on sparkDash. Full
sweep and profile: [docs/c1fast.md](ds41f-4xgx10-switchless/docs/c1fast.md).


## stack4: three small single-user levers (2026-11-10)

All three act on one-session steps only; batch (multi-user) steps run as before. Proof per lever: paired runs on the same 320
held-out prompts (same texts), GPU ms per step by rows, and the 54-prompt identity set on a test port. Design and proofs:
[docs/stack4.md](ds41f-4xgx10-switchless/docs/stack4.md).

| # | Change | Files | Default | Same output | Before -> after |
|---|---|---|---|---|---|
| 13 | `DSGB10_HEAD4_C1=1` with upstream's `--draft-head4 --draft-markov8`: MXFP4 drafter head and FP8 Markov head for one-session drafts, the FP8 head copy kept for batch draft passes | `engine/dspark/dspark.cu`, `engine/model/spec.cu` | off | yes: drafts only; 374/374 test outputs identical | draft pass -0.30..-0.38 ms; 320 prompts chat 86.19 -> 87.17, code 128.20 -> 129.29, math 133.70 -> 134.93; 16 distinct 363-370 (unchanged) |
| 14 | Confidence MLP gate `DSGB10_CF_MLP=FILE`, `DSGB10_CF_ALPHA=A` (one-session drafts): per depth the shipped logit, top-1 margin and top-1 log p (a scan kernel per rank, one all-gather, one MLP kernel), 64 hidden units; trained offline on 661K anchors | `engine/dspark/dspark.{h,cu}`, `engine/model/{spec.cu,ops.h}`, `engine/server/cmd.cpp`, `tools/stack4/` | off | every token is still the verifier's top choice; the gate changes which drafts are verified, which regroups steps and can flip a close call (alpha 1.0: 374/374 identical; alpha 0.9: 319/320 held texts) | accepted drafts per step chat 1.17 -> 1.24; chat 87.17 -> 88.17, code 129.29 -> 130.50, math 134.93 -> 136.11 |
| 15 | Certified LM head `DSGB10_CERT=1` (one-session steps whose rows are greedy without logprobs): FP8 coarse GEMV, per-row rigorous bound, cross-rank threshold, exact coded-head kernel on surviving 16-row groups (`gemv_bf16x_norm_masked`), sampler unchanged; `DSGB10_CERT_CHECK=1` runs the full head beside it | `engine/model/{model.cu,spec.cu,impl.h,model.h}`, `engine/kernels/{gemv16.cu,dsk.h}`, `engine/dspark/dspark.{h,cu}`, `engine/server/{cmd,serve}.cpp` | off | yes: same steps, same token; 1,344,416 real step rows checked on each rank, 0 mismatches, fallback exercised | head -0.18..-0.28 ms a step; chat 88.17 -> 89.45, code 130.50 -> 130.88, math 136.11 -> 137.05 |
| - | Measurement tools: `DSGB10_STEP_LOG`, `DSGB10_FORCE_NTOP`, `DSGB10_CF_DBG`, `DSGB10_CERT_FORCE`, `DSGB10_CERT_BSCALE`, `DSGB10_MH_DUMP` | `engine/server/{cmd,serve}.cpp`, `engine/model/spec.cu` | off | - | - |

Stack vs the exact build, same binary: 320 held prompts chat +3.78 %, code +2.09 %, math +2.51 %; our bench 27-run medians
79.6 / 136.6 -> 80.1 / 142.4; sparkDash c1 prose 99.1 -> 100.5, code 162.9 -> 162.8, catalogs 76.3 / 125.2 / 151.2 ->
79.5 / 130.7 / 154.0; 16 distinct 365-366, copies 446, 8 distinct 302, agent 262, prefill 10.0k, qeval 71 / 71, needle 3/3 at
948K, images 15/15.

Measured and not built: the target's output at a rejected row as the next draft (right 18 % of the time against the drafter's
65 %), a vocabulary shortlist for the drafter's head (+0.3 % simulated on top of #13), 6-7 bit coarse codes for #15 (about
0.07 ms), 7-bit packing of the FP8 dense weights (upstream's costing: no gain at the measured decode cost).

Carryover pilot (2026-11-10, offline only): carried rejected-row states with the shipped drafter frozen raise the accepted
length after a rejection by 2.2 % (3.5 % with small gated adapters) on 12,220 held anchors; realised speed with the current
confidence gate is about zero, so it is not in the engine. Numbers and code: docs/stack4.md, tools/stack4/carryover/.

Live build re-measured on our bench (27 runs, 2026-11-10, no other traffic): chat 82.7, code 142.6 (medians). README
table and charts now show 83 / 143.

Setup and bench (2026-11-10): README "run it" now lists the steps for the exact served build; tools/bench/bench.py is the
bench behind the README numbers. 16 users re-measured on the live build: 368 distinct, 436 copies (medians of 3).

Front-end fix (2026-11-11): a message that contains the image placeholder as ordinary text (a doc, a log, an error quoted
back) used to get HTTP 400 on every later turn. The placeholder is now rewritten with ASCII bars in request text and served
as plain text; image parts are unchanged; error texts no longer quote the token. `frontend/src/encoding.rs`, `server.rs`.

## stack5: sync without relays; coded weights and a 4-bit drafter body measured, not built (2026-11-11)

Three levers got a decisive measurement before any engine work; one shipped. Design, proofs and the two negative results:
[docs/stack5.md](ds41f-4xgx10-switchless/docs/stack5.md).

| # | Change | Files | Default | Same output | Before -> after |
|---|---|---|---|---|---|
| 16 | Direct ring `DSGB10_RING_DIRECT=1\|2` (`comm::Options::direct`, `ring_test --direct`) for fleets where every rank reaches every other in one hop (a switch): each rank also connects to the opposite rank and writes its contribution there itself; proxies relay nothing. Mode 2 also sends a small contribution to a neighbour as one write and one write of both flags. Buffers, flags, device kernels and the order of every sum are unchanged | `engine/comm/{ring.cpp,ring.h}`, `engine/comm/tests/ring_test.cu` | off | yes: same bits; 54/54 identity prompts, 320/320 held texts, identical steps | 10 KB all-reduce 14.45 -> 9.89 us, a step's 80 all-reduces 1.16 -> 0.79 ms (ring bench); engine GPU step -0.21 ms at one row .. -0.59 ms at six; 320 prompts chat 89.16 -> 90.28, code 131.32 -> 133.62, math 136.99 -> 139.46 |
| - | Bench only: a fixed-length lossless FP8 weight code decoded inside the GEMV (`k_gemv_fp8x`), next to upstream's fused Huffman kernel; `bench_wcomp` is now a CMake target | `engine/kernels/tests/bench_wcomp.cu`, `engine/kernels/CMakeLists.txt` | - | bit-exact (bench) | no gain: see below |

Gates with #16 on (test port; relay ring in the same window or stack4's gate run in brackets): 16 distinct 372.5, 373.9
(367.3, 361.6); copies 458.7 (439.1); 8 distinct 307.4 (302); agent 269.5 (262.4); our bench 27-run 82.4 / 143.7
(81.5 / 145.3: within its noise); sparkDash c1 prose 101.9-102.3 (100.3-100.9), code 159.9-165.3 (162.5-163.4), catalogs
80.7 / 132.4 / 156.1 (78.8 / 128.8 / 154.1); prefill 9.0k / 10.2k; qeval 71 / 71; needle 3/3 at 948K; images 15/15.

Measured and not built:
- Lossless coding of the dense FP8 weights, decoded inside the GEMV, on one Spark with our weights. Fused Huffman (upstream's
  kernel, 17 % fewer bytes): 10-48 % slower than today's kernels, as upstream found. A fixed-length 7-bit code with a
  side list for outliers (8.3-8.8 % fewer bytes, every byte exact, outputs bit-identical): -6 % to +6 % against today's
  kernels by site and row count, about par on average; it reads its stored bytes at 216-237 GB/s against 248-258 GB/s raw.
  The bar was 3 % of the dense time.
- The drafter's FP8 body as MXFP4, post hoc, in the torch mirror: accepted length -1.9 % chat, -1.3 % code, -1.4 % math for a
  pass that would be 0.30 ms shorter; first-order simulated gain 0 to +0.2 %. A 22-minute quantisation-aware pilot on
  166 held-out sequences did not close the gap (chat accepted length 1.463 -> 1.464, FP8 1.493); a full fine-tune was not run.

The chat target of 100 tok/s on the 320 held prompts was not reached: 90.3.

## prepare: conversion under a bounded disk, in Rust (2026-11-12)

`./start.sh prepare` no longer stages the whole checkpoint and every output on one disk. The work moved to
`ds41f-4xgx10-switchless/prepare` (`dsw-prepare`), CPU only, so it does not need the GPU:

| # | Change | Files | Default | Same bytes | Before -> after |
|---|---|---|---|---|---|
| 17 | Streaming conversion: the checkpoint is downloaded only when it is missing or a shard is short; batches read one source tensor, write every rank's slice, read it back, un-tile it with the inverse mapping, compare, `fsync`, journal, and only then release the source blocks (`--consume`). Sources are local files (hole punching) or HTTP ranges (`--source http`), outputs go to the staging directory or per-rank destinations (`--dest R=DIR`) | `prepare/src/{dsw,safetensors,plan,tile,convert,journal,space,source,engram}.rs`, `start.sh`, `Dockerfile`, `.env.example` | on | yes: every tensor compared with the checkpoint | peak disk for TP=4: 1026 GB (checkpoint + all ranks staged) -> 129 GB a rank, 514 GB if all four share one disk, 129 GB + one shard when streamed; the checkpoint can be released as the outputs land |
| 18 | Hybrid source (`--source hybrid`): each ordinary shard downloaded whole into `scratch/`, converted, deleted; the two 101.5 GB Engram shards read by range, so they never land on the disk. One shard of extra peak replaces one request per tensor (~96,000 of them) | `prepare/src/source.rs`, `prepare/src/convert.rs` | - | yes: same bytes | transfer count for a streamed full conversion: ~96,000 range requests -> 46 file downloads + a few thousand chunked ranges |
| 19 | `VISION=1` saves the encoder's slice of the checkpoint (`prepare/src/vision.rs`): the publisher's `inference/` scripts, an index trimmed to the tensors kept, and the shards holding `vision.`/`aligner.`/`image_`, ~2.3 GB of 510, written before any reclamation. `start.sh` then points dsfront at the encoder and runs it from `VISION_IMAGE` when set | `prepare/src/vision.rs`, `start.sh`, `.env.example` | off | yes: the shards are copied verbatim | enabling images later costs those 2.3 GB at most, not the checkpoint; `VISION=0` still refuses image parts |
| 20 | The scale-block check is `plan.cpp`'s exactly: an MXFP4 expert stores two values a byte, so its columns are doubled before the 32-element block arithmetic, and the scale's own shape must be the weight's in blocks. Without it `prepare` refused every expert weight with a scale (`cuts through its 1x32 scale blocks`) before writing a byte | `prepare/src/plan.rs`, `prepare/tests/e2e.rs` | - | yes: the plan was already right, the gate was wrong | the real geometry (w1 `[4608, 2560]` + scale `[4608, 160]`) now plans identically to `plan.cpp` at worlds 4 and 2; the fixture writes and verifies MXFP4 weights with their scales end to end |
| 21 | Admission for a consuming run is its **peak**, not the total: outputs are written shard by shard and each shard is released once its outputs are verified, so what must fit is `max(written - released)` — about one shard's outputs — plus the vision bundle. The refusal now reports that peak and the choice instead of only the total | `prepare/src/space.rs`, `prepare/src/main.rs` | - | - | a two-rank run on a disk with 95.3 GB free: `writes 409.4 GB` and refused -> `--consume fits (peak ~19 GB)` and it is accepted |
| 22 | A consuming run punches *inside* each tensor: the range is rounded inwards to whole 4 KiB blocks, so the shard's header and the bytes it shares with a neighbour survive. Without it the first tensor (whose data starts mid-block) zeroed the safetensors header, the next `prepare` could not open the checkpoint, and it started a 510 GB re-download instead of resuming. A filesystem without hole punching now keeps the checkpoint whole and says so, instead of truncating a shard | `prepare/src/convert.rs`, `prepare/tests/e2e.rs` | - | yes: outputs are unchanged | a killed `--consume` run: `implausible header length 0: downloading` -> the checkpoint reopens, the run resumes |
| 23 | A re-run of `prepare` keeps the journal, `.part` and Engram directory of an unfinished run and continues from them; only finished or unreadable outputs are offered for deletion (`--no-resume` starts over). The run prints one line per rank per shard (tensors, bytes written and released, elapsed, estimate) and Engram tables every 10 %, and `--clear` / `dsw-prepare clear` / `./start.sh clear` deletes every output of earlier runs without touching the checkpoint | `prepare/src/convert.rs`, `prepare/src/main.rs`, `start.sh`, `README.md` | - | - | re-running after Ctrl-C used to ask "delete these outputs?" and destroy the resume state; now it prints `resuming: keeping N file(s)` and continues |
| 24 | Resharding (`dsw-prepare reshard`, `./start.sh reshard`): re-cuts a finished rank set for another world from the files alone. Each old rank's bytes are un-tiled, the rectangles the new split owns are copied into the new natural buffer, re-tiled with the new geometry, written and read back; the Engram tables are re-cut by rows. The old files are first checked against the plan they claim, so a set cut by another version of the table is refused rather than re-cut | `prepare/src/reshard.rs`, `prepare/src/engram.rs`, `prepare/src/plan.rs`, `start.sh`, `README.md` | off (a command) | yes: 4 -> 2 -> 4 reproduces the original files byte for byte, and every output verifies against the checkpoint at its own world | the only way to change TP without the 510 GB checkpoint; a rebuild with no network |
| 25 | No `--max-filesize` on a range fetch: it compares the *declared* body with the cap, so a Hub URL that ignores `Range` (Xet-backed ones answer 200) made curl abort with `(63) Maximum file size exceeded` before the reader could say why — every streamed run died on the first 8-byte header probe. The reader reads one byte past the range, names the cause, and stops instead of retrying a server that will answer the same way | `prepare/src/source.rs` | - | - | `range 0+8 failed (curl: (63) Maximum file size exceeded)` x3 -> `the server sent the whole shard for bytes 0+8 (it does not honour Range …)` |
| 26 | A finished `prepare` is a no-op: a rank whose journal says done is kept and skipped (it used to refuse with "already complete"), the hash map is kept when it is there and written through `.part` + rename, the checkpoint is read for holes only when there is work to do, and it is looked for where it usually is (`$HOME/DeepSeek-V4.1-Flash[-1017]`, `/data/models/*`, the Hub cache) instead of one fixed name. A directory that holds shards but will not open is an error, not a silent 507 GB download | `prepare/src/{convert,main,checkpoint,safetensors}.rs` | - | - | a second `./start.sh prepare` used to stop at `rank0.dsw is already complete (delete it or pass --replace)`; now `rank 0: complete, kept`, `nothing to convert` |
| 27 | `./start.sh` with no arguments is the whole install: it checks each rank's three files at the fleet's paths (one ssh each), cuts what is missing (`--consume`, at `--cpu-shares 512` so the box keeps its CPU), ships every other rank's weight file, hash map and own Engram shards (`rsync --size-only --partial`), then builds, serves and waits. `./start.sh serve` keeps the old behaviour and the other commands are unchanged. Rank 0 serves the artifact directory itself (`RANK0_WEIGHTS`'s directory), so nothing is copied on the head | `start.sh`, `fleet/{homelab.conf,gen-fleet.sh}`, `.env.example` | - | - | three hand-run steps (prepare, three rsyncs per worker, start) -> `./start.sh`; the next run only stats the files |
| 28 | One entry point, five commands. `./start.sh` writes the fleet from `WORKERS` when the file is missing, cuts only what is absent, ships the other ranks' files, serves and smoke-tests; `status`, `restart`, `clear` and `prepare` (with `--world`, `--consume`, `--ckpt`, `--source`) are the rest. `TP` is gone as a setting: the rank list is the fleet file and it is the rank count, `WORKERS` only writes that file when it is missing, and a `WORKERS` that disagrees with it warns instead of halving the fleet. The settings are the API's (`DSGB10_PORT`, `DSGB10_ADDR`, `DSGB10_MODEL`) plus optional lines, and `serve`, `fleet` and `reshard` are no longer commands: an unknown first word is an error, not an engine flag | `start.sh`, `.env.example`, `README.md` | - | - | nine commands and nineteen settings lines -> five and ten (one required); a new install is `./start.sh` |
| 29 | `.env` is the API's settings and nothing else: `PORT`, `ADDR`, `MODEL`, `IS_LOCAL`, `NO_AUTH`, `RING_DIRECT`, `VISION`, `ENGINE_ARGS`, and then `WORKERS` (which only writes a missing fleet), `HF_TOKEN`, `IMAGE`, `SKIP_BUILD`, `DRY_RUN`. No `DSGB10_` prefix (an `.env` that predates it still works), `IS_LOCAL=1` refuses to download anything for the weights, and vision is on by default behind a `tools/vision/Dockerfile` the first run builds. The fleet file is the rank list; `WORKERS` never overrides it. A missing local checkpoint is an error under `--source local`, not a 507 GB download | `start.sh`, `.env.example`, `tools/vision/Dockerfile`, `prepare/src/main.rs`, `README.md` | - | - | eighteen prefixed settings -> nine plain ones, and two of them are the API's address; a fresh clone needs no `.env` at all |
| 30 | The API's port defaults to 8888, the engine's production endpoint — `service.sh` refuses 8888 unless `DSGB10_PROD=1`, which the container now passes — so `./start.sh` serves `http://127.0.0.1:8888/v1`; `PORT` in `.env` overrides it and `stop.sh` probes the same one | `start.sh`, `stop.sh`, `.env.example`, `README.md` | - | - | 18893 (the engine's test port) -> 8888, the port the fleet already serves on |
| 31 | The engine is one build per tensor-parallel world: `cmake -DDSGB10_WORLD=1|2|4` (default 4) defines `world::kWorld`, with `static_assert`s for the checkpoint's divisibility, and `model::kRanks`, `comm::kRanks`, `engram::kWorld`, `dspark::kRanks`, `attn::kHeads` and `dsk::kInter` all derive from it. Per-rank literals that bypassed the constants are gone (`kernels/moe.cu`'s `kInter = 576`, `dspark.cu`'s launch shapes, `prefill.cu`'s four-way sum); the ring's reduce-scatter and relay arithmetic are `n/kRanks` and `(r + kRanks/2) % kRanks`, its loopback mode serves world 1 and its direct mode is refused outside 4; the Engram store refuses shards named for another world; `tp::require_engine` compares `$TP` with the built world | `engine/common/world.h`, `engine/CMakeLists.txt`, `engine/{model,comm,engram,dspark,attn,kernels,server,weights}` | world 4 | yes at world 4: every derived constant is compiler-asserted to its old value (32320, 320, 128, 16, 8192, 96, 576, 1280) | worlds 2 and 1 build and are shape-checked, world 3 is refused by a static_assert; no CUDA was compiled here (no nvcc), so no world has been run |
| 32 | `./start.sh` stopped defaulting to `fleet/homelab.conf`: that file is the upstream author's fleet (10.0.0.4/.3/.2, imported with the engine on 2026-10-13) and is now only a reference. The default fleet is `fleet.conf` at the repo root, written from `WORKERS` on the first run and gitignored; `gen-fleet.sh` substitutes the `{rank}` slot in the per-rank paths (it emitted `rank{rank}.dsw` before) and keeps `~/` in the Engram paths so each box resolves them against its own home | `start.sh`, `fleet/gen-fleet.sh`, `fleet/homelab.conf`, `.gitignore`, `README.md` | - | - | a fresh clone no longer tries to ssh to the upstream author's boxes; a one-worker `WORKERS` no longer overrides the fleet |
| 33 | `prepare --repair` is the way through a local checkpoint that will not open: the directory is kept and `hf download` fetches only the files that are short or wrong, instead of the default refusal (which exists so an unreadable copy cannot trigger a blind 507 GB download). The refusal now names `--repair` first | `prepare/src/main.rs`, `start.sh`, `README.md` | - | - | a truncated header: dead end -> `pass --repair to let the download fetch what is wrong` |
| 34 | Preparation and cleanup preserve the host's absolute home path and `HOME` inside their containers; host `--ckpt`, output and destination paths no longer point outside a home bind mount at `/root`. Serving keeps its existing mapping | `start.sh`, `README.md` | on | unchanged conversion code | fixes path visibility and persistence; does not change admission or establish a 600 GB peak-total guarantee |
| 35 | A resumed run is admitted for the bytes it still has to write: the files the keep rule retains (finished ranks, `.part` files, the Engram directory) are credited at their allocated size before the space check. It used to demand the whole output set again, so a half-written 518 GB conversion on a disk with 350 GB free was refused for space that was already on it | `prepare/src/main.rs` | - | - | `writes 518.2 GB` against 350.5 GB free with 382 GB kept -> `382.0 GB of the output is already on disk, so 136.2 GB still has to be written` |
| - | Removed the `CKPT`, `PREPARE_OUT`, `HF_REPO`, `HF_REV`, `NEED_GB`, `OUT_GB` and `TP` variables: the locations have defaults, the space needed is computed from the plan, and the rank count is `1 + WORKERS` (or the fleet file's own ranks), with a `TP` left in an old `.env` ignored with a warning | `.env.example`, `start.sh`, `fleet/gen-fleet.sh` | - | - | seven knobs fewer; `prepare --world N` and `reshard --world N` are how a different world is asked for |

The partition plan is a port of `engine/weights/plan.cpp`; on a synthetic checkpoint covering every rule, including
MXFP4 experts with their scales, it prints byte-identical plans for every rank at worlds 1, 2 and 4 — and so for the
real expert geometry — and the tiling is a third implementation of `layout.h` (the kernels' `tile_fp8` / `tile_fp4`
write, `layout.h` and this crate un-tile, and the round trip is tested). Tests: 50 unit (format, permutations, rules,
scale pairs, journal replay, punching, HTTP ranges, the keep rule), 4 on the command line (admission, clear) and 12
end-to-end (four ranks plus Engram verified against the checkpoint, crash-then-resume, release-and-compare, a second
run that keeps the files, hole detection, tamper detection, 4 -> 2 -> 4 resharding).
