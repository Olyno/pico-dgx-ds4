// DeepSeek-V4.1-Flash on the GB10 ring: the shapes and the per-layer roles the runner needs.
// Everything here is fixed by the checkpoint (config.json) and DESIGN.md decision 1, except the world,
// which is the build's (common/world.h: DSGB10_WORLD, default 4). Every per-rank shape below derives
// from kRanks, and the kernels, the ring and the server take their per-rank geometry from these.
#pragma once
#include <cstdint>

#include "common/world.h"

namespace model {

// The tensor-parallel degree this binary is built for (one build per world; $TP must agree at startup,
// engine/common/tp.h). The ring's relay topology and the reduce-scatter's part count are comm::kRanks.
constexpr int kRanks = world::kWorld;
constexpr int kLayers = 40;
constexpr int kDim = world::kHidden;
constexpr int kHc = 4;                      // residual copies (mHC)
constexpr int kHcDim = kHc * kDim;          // 20480
constexpr int kMix = 24;                    // (2 + hc) * hc mixes per sublayer
constexpr int kVocab = world::kVocab;
constexpr int kVocabPart = kVocab / kRanks;  // 32320 head rows per rank at world 4
// Rows per step, at most (docs/batching.md, B4): one session's decode (1), verify (<= 6) or prefill-by-steps
// rows (<= kMaxSessRows), or several sessions' runs of consecutive rows (Model::launch_batch) up to 16. The
// step buffers, the StepIn / SampleOut rings, the sampler's grid and the graphs are sized by it; a graph is
// captured on first use, so unused row counts cost nothing.
constexpr int kMaxRows = 64;
// wide64 (docs/wide64.md): steps of 1..kDecodeRows rows run the decode kernels as before (bit for bit); steps of
// kDecodeRows + 1 .. kMaxRows rows (several sessions' rows, launch_batch) run the wide body: the dense projections as
// prefill GEMMs, the MoE as moe_prefill, the attention side in groups of kDecodeRows rows.
constexpr int kDecodeRows = 16;
constexpr int kMaxSlots = 16;               // session slots (Options::slots): live sessions' states, at most
// Rows of one session in a step, at most, whatever kMaxRows is: the attention kernels' contract for a
// step's slot table. attn::decode_prep's fused compressor emit and carry race across CTAs at 16 rows of
// one session (the step runs kv_write and compressor_emit, safe at 16, but the contract holds for any
// caller), and attn can check only steps without a table: the model checks every step it builds
// (next_in, launch_batch). The one-session spec graph's switch has this many bodies.
constexpr int kMaxSessRows = 8;
// Entries a drafted batch step (B3) takes at most, whatever the slots: its graph's if nodes and memory grow
// with them, and past 8 sessions a step of kMaxRows rows leaves no room for a draft each.
constexpr int kMaxSpecEnts = 8;
// Physical session units (Options::big_slots): a slot of the full context is kUnitMul units of the small
// slots' size laid end to end, so every kernel's slot * stride addressing stays uniform (a slot table holds
// units, the server's slots stay logical: cancel bits, Cmd::live).
constexpr int kMaxUnits = 64;
constexpr int kUnitMul = 8;

// attention, per rank
constexpr int kQLora = world::kQLora, kQaPart = kQLora / kRanks;  // wq_a: 320 rows per rank at world 4
constexpr int kHead = world::kLatent, kKvPart = kHead / kRanks;   // wkv: 128 rows per rank at world 4
constexpr int kCompPart = kHead / kRanks;                // compressor wkv / wgate: 128 rows at world 4
constexpr int kHeadsAll = world::kHeads;                 // query heads, the checkpoint's 64
constexpr int kHeads = kHeadsAll / kRanks;               // query heads per rank (16 at world 4)
constexpr int kQ = kHeads * kHead;                       // wq_b: 8192 rows per rank at world 4
constexpr int kOGroupHeads = 8;                          // wo_a's groups are 8 heads wide
constexpr int kOGroupRows = 1024;                        // wo_a rows per group
constexpr int kOPart = kHeads / kOGroupHeads * kOGroupRows;  // 2048 = wo_a rows = wo_b columns per rank
constexpr int kWoaK = kOGroupHeads * kHead;              // wo_a input per group (8 heads x 512)
constexpr int kIdxHeads = 32, kIdxDim = 128;             // indexer (replicated)
constexpr int kIdxQ = kIdxHeads * kIdxDim;               // 4096
constexpr int kIdxTopk = 512;
constexpr int kPoolBlocks = 2048;                        // layer 20's candidate pool (blocks of 8)

// MoE
constexpr int kExperts = world::kExperts, kGatePart = kExperts / kRanks;  // gate: 96 rows per rank at world 4
constexpr int kTopK = 6;
constexpr float kRouteScale = 1.5f;
constexpr int kInterPart = world::kExpertInter / kRanks;  // expert and shared-expert intermediate columns per rank

constexpr float kNormEps = 1e-20f;

// Engram (layers 1 and 14): 24 rows of 256 e4m3 + 8 ue8m0 per position; wkv per rank is 5 segments
// of 1280 output rows: keys 0..3 and the value, each this rank's quarter of the 5120 columns.
constexpr int kEngramCols = 24, kEngramRow = 264, kEngramK = kEngramCols * 256;  // 6144
constexpr int kEngramSeg = kDim / kRanks;                                       // 1280
constexpr int kEngramN = 5 * kEngramSeg;                                        // 6400
constexpr int kEngramPartBytes = 8 * 4 + kEngramSeg * 2;  // per row: 4 dots, 4 key sums of squares, value slice

// Layer roles (config.json: compress_ratios, kv_source_layers, index_source_layers).
constexpr int ratio_of(int L) { return L < 2 ? 0 : L < 20 ? 2 : 1; }
constexpr bool kv_src(int L) { return L == 2 || L == 8 || L == 14 || L == 20; }
constexpr int kv_source(int L) { return L < 2 ? -1 : L < 8 ? 2 : L < 14 ? 8 : L < 20 ? 14 : 20; }
constexpr bool idx_src(int L) { return kv_src(L) || (L >= 24 && L % 4 == 0); }
// The index layer whose top-k this layer attends with (itself when it is one).
constexpr int index_source(int L) { return L < 2 ? -1 : L < 20 ? kv_source(L) : 20 + (L - 20) / 4 * 4; }
constexpr bool reindex(int L) { return L > 20 && idx_src(L); }  // scores only layer 20's pool
constexpr bool engram_layer(int L) { return L == 1 || L == 14; }
constexpr int engram_index(int L) { return L == 1 ? 0 : 1; }
// Per-step index results: one slot per index source, in layer order; compressed caches: one per kv source.
constexpr int kIdxSlots = 8;
constexpr int idx_slot(int L) { return L == 2 ? 0 : L == 8 ? 1 : L == 14 ? 2 : L == 20 ? 3 : 4 + (L - 24) / 4; }
constexpr int kKvSlots = 4;
constexpr int kv_slot(int L) { return L == 2 ? 0 : L == 8 ? 1 : L == 14 ? 2 : 3; }

}  // namespace model
