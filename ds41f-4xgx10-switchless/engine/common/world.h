// The tensor-parallel world this binary is built for (DSGB10_WORLD, a CMake option, default 4).
//
// This is the one constant every per-rank shape derives from: model::kRanks, comm::kRanks,
// engram::kWorld, dspark::kRanks, attn::kHeads and dsk::kInter are all computed from world::kWorld.
// One build serves one world. $TP (common/tp.h) must agree with it at startup, and a weight file, an
// Engram shard or a fleet file cut for another world is refused rather than silently misread.
//
// The checkpoint fixes the shapes a world must divide: 129280 vocabulary rows, 5120 hidden, 64 query
// heads, 1280 q-lora rows, 384 experts, 2304 expert intermediate columns, 512 latent. Worlds 1, 2 and
// 4 divide all of them; 3 divides none, so it is refused here (the weight cutter refuses it too).
#pragma once

#ifndef DSGB10_WORLD
#error "DSGB10_WORLD is not defined: build through CMake (-DDSGB10_WORLD=1|2|4, default 4), not by calling nvcc"
#endif

namespace world {

inline constexpr int kWorld = DSGB10_WORLD;

// The checkpoint's shapes (config.json); a rank owns an equal slice of each.
inline constexpr int kVocab = 129280;
inline constexpr int kHidden = 5120;
inline constexpr int kHeads = 64;
inline constexpr int kQLora = 1280;
inline constexpr int kExperts = 384;
inline constexpr int kExpertInter = 2304;
inline constexpr int kLatent = 512;

static_assert(kWorld == 1 || kWorld == 2 || kWorld == 4,
              "DSGB10_WORLD must be 1, 2 or 4: the checkpoint's shapes (129280 vocabulary rows, 5120 hidden, "
              "64 heads, 1280 q-lora, 384 experts, 2304 expert columns, 512 latent) divide by no other world");
static_assert(kVocab % kWorld == 0 && kHidden % kWorld == 0 && kHeads % kWorld == 0 && kQLora % kWorld == 0 &&
                  kExperts % kWorld == 0 && kExpertInter % kWorld == 0 && kLatent % kWorld == 0,
              "DSGB10_WORLD does not divide the checkpoint's shapes");

}  // namespace world
