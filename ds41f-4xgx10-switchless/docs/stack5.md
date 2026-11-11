# stack5: sync without relays, and two levers measured and not built (2026-11-11)

Goal: more single-user speed with the same output, from three levers that had not been measured here: (A) losslessly coded
dense FP8 weights decoded inside the GEMV kernels, (B) cheaper cross-rank synchronisation, (C) a 4-bit drafter body. Each got a
cheap decisive measurement before any engine work. One shipped: on a switched fleet the ring's collectives no longer relay
(B). Chat on the 320 held-out prompts: 89.2 -> 90.3 tok/s (+1.3 %), code +1.8 %, math +1.8 %, same texts. A and C were
measured and not built. The target of 100 on chat was not reached. Every number below is measured on this cluster unless it
says simulated or estimated.

## What is live

| lever | flag | step change | gain on the 320 held prompts (chat / code / math) |
|---|---|---|---|
| direct ring: every rank writes its contribution to the opposite rank itself, nothing is relayed | `--setenv DSGB10_RING_DIRECT=2` | GPU step -0.21 ms at one row .. -0.59 ms at six | +1.26 / +1.75 / +1.81 % |

It needs a fleet where every rank reaches every other in one hop (ours: four boxes on one 200G switch, the "ring" is only
logical). On a cabled ring it must stay off. The buffers, the six arrival flags, the device-side kernels and the order of every
sum are unchanged, so results are the same bits.

## B. Synchronisation

### Where the time went (measured, `ring_test --trace`, 4 boxes on the switch)

Upstream's collectives were designed for a cabled ring: a rank writes its contribution to both neighbours, and each neighbour's
CPU proxy forwards half of it to the opposite rank. Upstream's study ([engine.md](engine.md), "The collectives' time") puts the
collectives at ~2 ms of a one-row step and names "two hops of ~4.4 us" as the floor of that algorithm; its attempts to hide the
wait under weight loads came out under 2 %. We did not redo those. On a switch the second hop is not needed.

Median microseconds after a rank's own post, 80 all-reduces of 10 KB back to back:

| event | relay (upstream) | direct |
|---|---|---|
| neighbours' data arrives | 3.7-3.9 | 3.7-3.8 |
| opposite rank's data arrives | 6.4 and 7.55 (two relayed halves) | 4.0-4.1 |
| all six flags in | 7.6 | 4.2 |
| last arrival -> next post (GPU side) | 2.5 | 2.5 |

So a 10 KB all-reduce spent 3.7 us on the first hop, 3.9 us more waiting for the relayed halves, and 2.5 us on the GPU side.
The single-box figure upstream measured for GPU -> proxy -> GPU without any network is 3.1 us, so most of what is left after
this change is the GPU and the CPU proxy signalling each other through host memory, not the wire.

### The change

`engine/comm/ring.cpp`, `ring.h`: with `DSGB10_RING_DIRECT` (or `Options::direct`), each rank opens a third pair of queue pairs,
to the opposite rank (their details travel two steps along the bootstrap sockets), and writes its contribution straight into
that rank's `opp` buffer, at the offsets the relays used, followed by the two flags the relays used to set. Proxies relay
nothing. A reduce-scatter sends each rank only its own quarter. Mode 2 also sends a small contribution to a neighbour as one
write plus one write of both flags (the two halves with a flag each existed so that a relay could start early); it measures
the same as mode 1. All ranks must agree on the mode; the bootstrap checks it. The flow-control argument of
[ring.md](ring.md) carries over: a rank posts collective s only after it has seen every rank's s - 1 arrive, which each rank
posted only after consuming s - 3, the last use of the same buffer set; the relay-source hazard is gone because nothing is
relayed.

### Measured

`ring_test` on the four boxes, microseconds per collective, back to back in a graph:

| payload | all-reduce bf16, relay -> direct | all-gather | reduce-scatter |
|---|---|---|---|
| 64 B | 10.18 -> 6.86 | 11.10 -> 8.02 | 9.57 -> 6.71 |
| 1 K | 11.59 -> 7.66 | 12.92 -> 9.01 | 10.27 -> 7.17 |
| 10 K | 14.45 -> 9.89 | 15.59 -> 11.00 | 12.23 -> 8.46 |
| 20 K | 17.88 -> 13.25 | 19.08 -> 14.45 | 13.88 -> 10.18 |
| 60 K | 26.96 -> 20.99 | 28.02 -> 21.98 | 16.78 -> 13.80 |
| 128 K | 39.90 -> 30.66 | 40.86 -> 31.57 | 21.09 -> 18.33 |
| 1 M | 207.5 -> 162.4 | 209.2 -> 164.3 | 72.9 -> 67.4 |

- A decode step's 80 all-reduces of 5120 elements: bf16 14.48 -> 9.93 us each (1.16 -> 0.79 ms), fp32 17.89 -> 13.25 us
  (1.43 -> 1.06 ms). Sampling all-gather 10.0 -> 7.1 us.
- Correctness in direct mode (both modes): every `correct` case exact; the pipelined kernels' race test exact with and without
  the 2 ms stall at caps of 1 MiB and 4 MiB; 300,000 random-size collectives with every word checked, 0 mismatches on the four
  ranks; bulk all-reduce of 32 MiB 9.98 GB/s, all-gather 25 GB/s.

In the engine, the same 320 held-out prompts through the same binary in one window, relay then direct. The steps are identical
(28,757 timed steps with the same row counts), so the comparison is exact:

| | relay | direct | change |
|---|---|---|---|
| GPU ms per step by rows 1..6 | 18.86 / 21.91 / 24.99 / 28.03 / 30.72 / 33.44 | 18.65 / 21.60 / 24.67 / 27.65 / 30.25 / 32.85 | -0.21 .. -0.59 |
| chat / code / math / all, tok/s | 89.16 / 131.32 / 136.99 / 104.21 | 90.28 / 133.62 / 139.46 / 105.70 | +1.26 / +1.75 / +1.81 / +1.43 % |
| texts | 320/320 equal to the stack4 build | 320/320 | |
| 54-prompt identity set vs the exact reference | 54/54 | 54/54 | |

The ring bench alone suggested about 4.5 us x ~145 collectives = 0.65 ms a step. The engine keeps 0.2-0.6 ms of it: part of
each wait was already hidden under the next kernel's weight loads.

### Gates (test port, binary with `DSGB10_RING_DIRECT=2`, next to the same binary on the relay ring in the same window)

| gate | direct | relay (same window) or stack4's gate run |
|---|---|---|
| 54-prompt identity set vs the exact reference | 54/54 | 54/54 |
| 320 held prompts, texts equal to the stack4 build | 320/320 | 320/320 |
| our bench, 27 runs, chat / code (medians; chat runs range 73-95) | 82.4 / 143.7 | 81.5 / 145.3 |
| sparkDash c1 prose / code (5 runs) | 101.9-102.3 / 159.9-165.3 | 100.3-100.9 / 162.5-163.4 |
| sparkDash catalogs prose / structural / code | 80.7 / 132.4 / 156.1 | 78.8 / 128.8 / 154.1 |
| 16 users, distinct prompts | 372.5, 373.9 | 367.3, 361.6 |
| 16 users, copies | 458.7 | 439.1 |
| 8 users, distinct | 307.4 | 302 (stack4) |
| agent load (8 sessions, 80-300K-token prompts) | 269.5 | 262.4 (stack4) |
| prefill 8k / 32k tok/s | 9.0k / 10.2k | 8.8k / 10.0k (stack4) |
| quality eval, 4 / 16 concurrent | 71 / 71 | 71 / 71 (stack4) |
| needle at 948K tokens | 3/3, 9.7k tok/s | 3/3 (stack4) |
| images | 15/15 | 15/15 (stack4) |

One gate run had to be repeated for a reason that is worth knowing on these boxes: CUDA on GB10 allocates only from free
pages, and the engine sizes its session store from the free device memory at start. A bench and a torch job had just read
tens of GB of files on one rank, so that rank started with 2.6 GiB free, the store came out at 0.57 GiB instead of ~8-10 GiB,
and the agent-load test re-prefilled every turn cold. Dropping those files' page cache (`posix_fadvise(DONTNEED)`, no root)
before the engine starts fixed it; the number in the table is from the clean run.

### Not tried
- SiFAR (arXiv 2607.08973) and "Every Microsecond Matters" (arXiv 2607.16100) remove barriers between collectives with dual
  buffering. The ring already has three buffer sets and posts collective s after consuming s - 2, and its kernels wait in
  place with PDL, so there is no separate barrier to remove; what remains per collective is the host-memory signalling above.
  A persistent polling kernel and merged collectives were costed by upstream at 0.05-0.15 ms a step and not built here.

## A. Losslessly coded dense FP8 weights: measured, not built

The idea: the dense FP8 tensors are 53 % of the bytes a one-row step reads, their bytes carry 6.5 bits of entropy, and the
kernels are bound by memory bandwidth, so fewer stored bytes decoded inside the GEMV could be a faster step with the same
bits. Papers: Tan et al., "Approaching Shannon Bound with Lossless LLM Weight Compression" (arXiv 2606.15789, rANS fused with
the GEMM) and ZipServ (ASPLOS 2026, fixed-length bitplanes fused with the GEMM).

Upstream had already measured the entropy-coded form on this hardware (a 12-bit Huffman decoder fused into the persistent
GEMV, [engine.md](engine.md), "Lossless weight compression"): exact, and 10-58 % slower than the raw kernels. It costed a
fixed-length code from its SASS only and never ran it. We re-ran the Huffman kernel on our weights and built and ran the
fixed-length one (`engine/kernels/tests/bench_wcomp.cu`, bench only, nothing in serving calls it).

The fixed-length code: an e4m3 byte is sign, 4-bit exponent, 3-bit mantissa. Per warp-step (512 bytes: 64 bytes of 8 rows) it
keeps 448 bytes: per lane a u64 of (sign, mantissa) nibbles and 48 bits of 3-bit exponent codes relative to the tensor's
window base. A lane chunk with an exponent outside the 8-wide window is flagged in a u32 per step and its 16 fourth bits go to
a side list found by a running popcount. Every byte value has a code, so decoding is exact for any tensor. Stored size:
8.3-8.8 % smaller (21-24 % of lane chunks flagged). The decode rebuilds the 16 bytes in registers and hands them to the same
MMAs in the same order.

One Spark, 10 layers of each site, at least 256 MB of rotating copies so no cache holds them; microseconds per GEMV call;
every byte decoded exactly on CPU and GPU, outputs bit-identical to today's kernel in 320-480 cases a site (1-8 rows, grids):

| site | rows | today | persistent | fused Huffman (17 % smaller) | fixed-length, first version | fixed-length, tuned | tuned vs best of today |
|---|---|---|---|---|---|---|---|
| wo_b, 10.5 MB | 1 | 40.66 | 42.38 | 45.07 | 40.47 | 40.92 | -0.6 % |
| wo_b | 2 / 3 / 4 / 6 | 40.7 / 40.9 / 40.9 / 40.9 | | 45.5 / 45.1 / 45.1 / 45.5 | 40.7 / 40.5 / 40.4 / 41.2 | 40.5 / 40.6 / 40.5 / 40.5 | +0.4 / +0.8 / +1.0 / +1.1 % |
| wq_b, 10.5 MB | 1 | 42.46 | 42.50 | 49.32 | 44.01 | 42.22 | +0.6 % |
| wq_b | 2 / 3 / 4 / 6 | 41.5 / 42.5 / 42.9 / 44.7 | 43.2 / 42.4 / 42.9 / 43.0 | 50.3 / 50.7 / 50.4 / 51.7 | 44.0 / 44.5 / 44.9 / 46.5 | 42.5 / 42.9 / 43.5 / 44.6 | -2.5 / -1.1 / -1.5 / -3.7 % |
| wq_a + wkv, 2.3 MB | 1 | 11.29 | 11.30 | 16.95 | 11.75 | 11.79 | -4.5 % |
| wq_a + wkv | 2 / 3 / 4 / 6 | 11.2 / 13.2 / 14.2 / 16.0 | | 17.9 / 18.8 / 19.8 / 28.4 | 12.8 / 13.9 / 15.0 / 17.3 | 11.9 / 12.4 / 13.4 / 15.3 | -5.8 / +6.1 / +5.7 / +4.3 % |

("today" and "persistent" columns are from the tuned run's job; the Huffman and first-version columns from the first job,
whose own "today" was within 1 % of these.)

- The fixed-length kernel reads its stored bytes at 216-237 GB/s; today's kernels read raw bytes at 248-258 GB/s. The 8-9 %
  fewer bytes are cancelled by the decode (about 90 more instructions and four loads a lane per step instead of one). The
  tuned version reads a run's flags into shared memory first, so the side-list load is no longer a dependent chain, and keeps
  2 or 4 steps in flight at 1, 2 or 4 CTAs per SM.
- Huffman is 10-48 % slower on our weights, as upstream found on theirs. A table-driven ANS decoder has the same serial
  per-lane structure (a table lookup and a variable refill per symbol), so it was not built.
- Entropy of the other formats (order 0, our weights): MXFP4 expert values 3.894 bits per 4-bit code (3.77 given the block
  scale), which only an entropy coder could take; their scales 0.97 bit per byte, already stored as 1 bit by upstream's scale
  code; the bf16 head's exponent 2.59 bits, already stored as 4 bits from a tile base.
- Verdict: the bar was a coded kernel at least 3 % faster than today's on the dense time. Measured: about par (-6 % to +6 %,
  mean near 0). Not built into the engine.

## C. 4-bit drafter body: measured offline, not built

The drafter only proposes, so any change to it keeps the output the model's own. Its head is already MXFP4 for one-session
drafts (stack4); its three blocks' dense tensors and `main_proj` are FP8: 141 MB of the ~336 MB a draft pass reads per rank
(the rest: 83 MB of MXFP4 head, the routed experts, the Markov head). MXFP4 for those tensors would save 70.6 MB, about
0.30 ms at 235 GB/s, 1.2 % of a chat step, if the drafts stayed as good. Related: arXiv 2607.04244 keeps its drafter on a
4-bit grid with quantisation-aware training.

Measured offline in the torch mirror of the drafter (the engine's arithmetic, drafter3's tools), on 100 chat, 30 code and 20
math held-out sequences (41,276 anchors): the FP8 tensors quantised post hoc to MXFP4 (e2m1 values, one power-of-two scale per
32 values of a row; 12 % relative weight error). "lead" is the accepted length of the 5-draft chain before the gate.

| tensors to MXFP4 | FP8 MB a rank | chat d1 / d2 / d3 | chat lead | code lead | math lead |
|---|---|---|---|---|---|
| none (shipped) | 0 | .6583 / .5352 / .4463 | 1.4931 | 3.3656 | 3.5033 |
| all dense | 141 | .6532 / .5278 / .4349 | 1.4646 (-1.9 %) | 3.3209 (-1.3 %) | 3.4535 (-1.4 %) |
| all, the better of two scales per block | 141 | .6528 / .5275 / .4372 | 1.4672 (-1.7 %) | 3.3278 (-1.1 %) | 3.4709 (-0.9 %) |
| attention (wq_a, wq_b, wkv, wo_a, wo_b) | 94 | .6553 / .5297 / .4371 | 1.4703 (-1.5 %) | 3.3405 (-0.7 %) | 3.4589 (-1.3 %) |
| wq_a, wq_b, wkv | 38 | .6574 / .5317 / .4429 | 1.4845 (-0.6 %) | 3.3584 (-0.2 %) | 3.4932 (-0.3 %) |
| wo_a, wo_b | 56 | .6560 / .5312 / .4403 | 1.4810 (-0.8 %) | 3.3508 (-0.4 %) | 3.4629 (-1.2 %) |
| shared experts | 26 | .6583 / .5334 / .4431 | 1.4900 (-0.2 %) | 3.3549 (-0.3 %) | 3.4896 (-0.4 %) |
| main_proj | 20 | .6574 / .5323 / .4434 | 1.4886 (-0.3 %) | 3.3502 (-0.5 %) | 3.5000 (-0.1 %) |

Simulated, first order (not a replay): the pass saves half the converted MB at 235 GB/s on a 24.9 ms chat step or a 29 ms code
step; the lost drafts cost the lead change times the accepted drafts' share of a step's tokens (1.24 of 2.24 on chat, 2.9 of
3.9 on code). All dense: chat +1.2 % - 1.05 % = +0.15 %, code about 0. With the better scale: chat +0.2 %, code +0.2 %.
Shared experts and main_proj alone: +0.1 %. So post-hoc 4-bit is a wash and it was not built. A quantisation-aware fine-tune
that restored the accuracy fully would be worth at most +1.2 % on chat and +1.0 % on code (simulated), and the engine would
need MXFP4 forms of the drafter's attention GEMVs.

A small quantisation-aware pilot was run (not a full fine-tune): MXFP4 rounding with a straight-through gradient on all dense
latents, 22 minutes per regime on one box each, trained on the 166 held-out sequences the evaluation does not use (about
196K anchors, four passes), scored on the same 100 / 30 / 20 sequences. Nothing was exported.

| regime | chat lead | code lead | math lead |
|---|---|---|---|
| FP8 (shipped) | 1.4931 | 3.3656 | 3.5033 |
| MXFP4 post hoc | 1.4634 | 3.3174 | 3.4665 |
| AdamW 2e-5 | 1.4640 | 3.3285 | 3.4769 |
| AdamW 1e-4 | 1.4342 | 3.3046 | 3.4863 |
| SGD with momentum, 0.03 | 1.4645 | 3.3270 | 3.4773 |

It does not close the gap: flat at the small rates, worse on chat at the larger one, which overfits the small training set.
One thing worth knowing if you try this: FP8 latents are binary fractions, so about a quarter of them sit exactly on an MXFP4
rounding tie. The first optimizer step, however small, then flips all of those at once in the gradient's direction and wrecks
the model (training accuracy at depth 1 fell from .83 to .09 within ten steps at every rate we tried, AdamW or SGD). Starting
each latent halfway between its FP8 value and its MXFP4 value removes that. A full fine-tune on millions of tokens was not
run.

## Reproduce
- Ring: `engine/comm/run4.sh <ring_test> latency graph --trace 4000 --direct 0|1|2`, then
  `correct race integrity --integrity 300000 --direct 2` and `bulk race --cap 4194304 --direct 2`.
- Weights bench: `bench_wcomp prep RANKFILE DIR 0,1,2,7,14,20,21,24,31,39`, then `bench_wcomp run DIR 1 2 3 4 6` on an idle
  GPU. Lines starting with `FIX` are the fixed-length kernel.
- Drafter sweep: [tools/stack5/body4_sweep.py](../tools/stack5/body4_sweep.py) is the block we ran inside our torch mirror of
  the drafter (the mirror and its feature dumps are not in this repository; see [stack4.md](stack4.md)).
- Page cache: [tools/stack5/evict.py](../tools/stack5/evict.py) `DIR...` drops the cache of the files under the given
  directories, without root.
