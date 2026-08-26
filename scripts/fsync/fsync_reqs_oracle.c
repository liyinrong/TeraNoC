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

// The RUNTIME's own implementation, not a copy of it. Checking a duplicated
// formula would only prove the duplicate is self-consistent; this makes the
// reference generator the oracle for the code that actually ships.
#include "fsync.h"

#define N __FSYNC_N_CU_X__

int main(void) {
  unsigned checked = 0, bad = 0;

  // Every pow2-aligned RECTANGLE, not only squares: rows and columns of
  // virtual clusters are rectangles, and broadcast/reduce needs them.
  for (unsigned w = 1; w <= N; w <<= 1) {
    for (unsigned h = 1; h <= N; h <<= 1) {
      if (w * h < 2)
        continue; // a single CU is not a barrier
      for (unsigned by = 0; by < N; by += h) {
        for (unsigned bx = 0; bx < N; bx += w) {
          static fsync_cu_t cus[N * N];
          unsigned n = 0;
          for (unsigned y = by; y < by + h; y++)
            for (unsigned x = bx; x < bx + w; x++) {
              cus[n].cu_id = y * N + x;
              cus[n].x_pos = x;
              cus[n].y_pos = y;
              n++;
            }

          fsync_init_reqs(cus, n);
          if (!fsync_gen_reqs(cus, n, h_fs_dir)) {
            printf("FAIL %ux%u base=(%u,%u): generator refused a legal "
                   "rectangle\n",
                   w, h, bx, by);
            bad++;
            continue;
          }

          uint32_t want_id = 0;
          const unsigned want = (unsigned)fsync_aggr_of_rect(w, h, &want_id);
          // Every member must agree, or a single per-scope value cannot exist.
          for (unsigned k = 0; k < n; k++) {
            const unsigned a = cus[k].fsync_req.fs_req_aggr;
            const unsigned i = cus[k].fsync_req.fs_req_id;
            checked++;
            if (a != want) {
              if (bad < 8)
                printf("FAIL %ux%u base=(%u,%u) cu=%u: aggr=0x%x want=0x%x\n",
                       w, h, bx, by, cus[k].cu_id, a, want);
              bad++;
            }
            // Only the 1D ids are predicted; a pair that straddles two sibling
            // nodes uses the NEIGHBOUR links (id 2/3), which the closed form
            // deliberately does not model.
            if (i != cus[0].fsync_req.fs_req_id) {
              if (bad < 8)
                printf("FAIL %ux%u base=(%u,%u) cu=%u: id=%u differs from %u "
                       "within one rectangle\n",
                       w, h, bx, by, cus[k].cu_id, i,
                       cus[0].fsync_req.fs_req_id);
              bad++;
            }
            if (i < 2 && i != want_id) {
              if (bad < 8)
                printf("FAIL %ux%u base=(%u,%u): id=%u want=%u\n", w, h, bx, by,
                       i, want_id);
              bad++;
            }
          }
        }
      }
    }
  }

  printf("%dx%d: %u members checked, %u mismatches\n", N, N, checked, bad);
  return bad ? 1 : 0;
}
