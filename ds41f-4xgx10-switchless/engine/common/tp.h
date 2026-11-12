// Tensor-parallel degree as a decision, not a hidden constant.
#pragma once
#include <cstdlib>
#include <cstring>

#include "common/check.h"

namespace tp {

// The requested world size, from $TP (or $DSGB10_TP as an alias), default 4.
// Read once per process; --world/--tp flags override it where they exist
// (dsw_convert, dsgb10_rank). Anything that is not a positive integer dies.
inline int world() {
  static const int w = [] {
    const char* v = std::getenv("TP");
    if (!v || !*v) v = std::getenv("DSGB10_TP");
    if (!v || !*v) return 4;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || *end || n < 1 || n > 16) die("TP='%s': not a world size (1..16)", v);
    return (int)n;
  }();
  return w;
}

// The GPU kernels and the ring topology are specialised to one world (4):
// per-rank splits (16 heads, 96 experts, 576 cols), the opposite-rank relay
// ((r+2)%4) and the quarter reduce-scatter (n/4) all assume it. The weight
// cutter (dsw_convert) is generic over --world, so TP!=4 files can be cut
// ahead of the kernels catching up; the engine refuses them until then,
// instead of silently computing garbage.
inline void require_engine(int expected, const char* what = "engine") {
  const int w = world();
  if (w != expected)
    die("%s: TP=%d but this build is TP=%d (kernels/ring are TP=4 specialised); "
        "re-cut weights with dsw_convert --world %d and rebuild for the new world first",
        what, w, expected, w);
}

// A weight file must have been cut for this TP (its header's world).
inline void require_file_world(int file_world, int file_rank) {
  const int w = world();
  if (file_world != w)
    die("weights: file is rank %d of %d but TP=%d (re-cut with --world %d)",
        file_rank, file_world, w, w);
}

}  // namespace tp
