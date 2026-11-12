// Tensor-parallel degree as a decision, not a hidden constant.
#pragma once
#include <cstdlib>
#include <cstring>

#include "common/check.h"
#include "common/world.h"

namespace tp {

// The requested world size, from $TP (or $DSGB10_TP as an alias), defaulting to the world this binary was
// built for (common/world.h). Read once per process; --world/--tp flags override it where they exist
// (dsw_convert, dsgb10_rank). Anything that is not a positive integer dies.
inline int world() {
  static const int w = [] {
    const char* v = std::getenv("TP");
    if (!v || !*v) v = std::getenv("DSGB10_TP");
    if (!v || !*v) return world::kWorld;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || *end || n < 1 || n > 16) die("TP='%s': not a world size (1..16)", v);
    return (int)n;
  }();
  return w;
}

// One build serves one world (DSGB10_WORLD, common/world.h): every per-rank shape, the kernels' launch
// geometry, the ring's relay topology and the reduce-scatter's part count are that world's. $TP (or
// $DSGB10_TP) names the world the operator means to run, and this refuses a mismatch instead of
// computing another world's arithmetic. The weight cutter (dsw_convert) is generic over --world, so
// files can be cut ahead of a rebuild; the engine still refuses files of another world
// (require_file_world below).
inline void require_engine(const char* what = "engine") {
  const int w = world();
  if (w != world::kWorld)
    die("%s: TP=%d but this engine is built for TP=%d (one build per world: -DDSGB10_WORLD=%d); "
        "rebuild for TP=%d, or run with TP=%d",
        what, w, world::kWorld, world::kWorld, w, world::kWorld);
}

// A weight file must have been cut for this TP (its header's world).
inline void require_file_world(int file_world, int file_rank) {
  const int w = world();
  if (file_world != w)
    die("weights: file is rank %d of %d but TP=%d (re-cut with --world %d)",
        file_rank, file_world, w, w);
}

}  // namespace tp
