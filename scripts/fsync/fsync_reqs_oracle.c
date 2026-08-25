// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Oracle for the fractal_sync (aggr, id) request encoding.
//
// The runtime must turn a virtual-cluster descriptor into a barrier request in
// O(1), with no array walk and no failure mode: our scopes are always
// pow2-ALIGNED SQUARE rectangles, which is far less general than what the
// reference generator handles. This checks that closed form against the
// reference implementation (hardware/deps/fractal_sync/sw/) over EVERY
// pow2-aligned square rectangle at every level, for every tree size we
// instantiate.
//
// Run it before writing the RTL shim, not after: the point is to pin the
// encoding contract independently, the same way scripts/vc/vc_morton.py pins
// the Morton map. Exits non-zero on any mismatch.

#include <stdio.h>

#include "fractal_sync_req_gen.h"

#define N __FSYNC_N_CU_X__

// A complete SxS subtree activates every level it climbs, and a 2-D fractal
// tree alternates h/v, so it climbs 2*log2(S) of them.
static unsigned closed_form_aggr(unsigned side) {
  unsigned levels = 0;
  for (unsigned s = side; s > 1; s >>= 1)
    levels += 2;
  return (1u << levels) - 1u;
}

int main(void) {
  unsigned checked = 0, bad = 0;

  for (unsigned side = 2; side <= N; side <<= 1) {
    for (unsigned by = 0; by < N; by += side) {
      for (unsigned bx = 0; bx < N; bx += side) {
        static fsync_cu_t cus[N * N];
        unsigned n = 0;
        for (unsigned y = by; y < by + side; y++)
          for (unsigned x = bx; x < bx + side; x++) {
            cus[n].cu_id = y * N + x;
            cus[n].x_pos = x;
            cus[n].y_pos = y;
            n++;
          }

        fsync_init_reqs(cus, n);
        if (!fsync_gen_reqs(cus, n, h_fs_dir)) {
          printf("FAIL side=%u base=(%u,%u): generator refused a legal "
                 "rectangle\n",
                 side, bx, by);
          bad++;
          continue;
        }

        const unsigned want = closed_form_aggr(side);
        // Every member must agree, or a single per-scope value cannot exist.
        for (unsigned k = 0; k < n; k++) {
          const unsigned a = cus[k].fsync_req.fs_req_aggr;
          const unsigned i = cus[k].fsync_req.fs_req_id;
          checked++;
          if (a != want) {
            if (bad < 8)
              printf("FAIL side=%u base=(%u,%u) cu=%u: aggr=0x%x want=0x%x\n",
                     side, bx, by, cus[k].cu_id, a, want);
            bad++;
          }
          if (i != cus[0].fsync_req.fs_req_id) {
            if (bad < 8)
              printf("FAIL side=%u base=(%u,%u) cu=%u: id=%u differs from %u "
                     "within one rectangle\n",
                     side, bx, by, cus[k].cu_id, i, cus[0].fsync_req.fs_req_id);
            bad++;
          }
        }
      }
    }
  }

  printf("%dx%d: %u members checked, %u mismatches\n", N, N, checked, bad);
  return bad ? 1 : 0;
}
