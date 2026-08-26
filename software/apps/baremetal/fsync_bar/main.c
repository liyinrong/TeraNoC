// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// End-to-end gate for the fractal_sync hardware barrier.
//
// This is the first thing that completes a barrier through real hardware: the
// tree gate (tb_fractal_sync) and the shim gate (tb_fsync_shim) both drive
// leaf ports or registers directly, with no cores, no cluster and no address
// decode in the path.
//
// The oracle is a WRITE-THEN-READ across the scope, not merely "the barrier
// returned". A barrier that returns too early is the failure that matters, and
// only a cross-tile visibility check can see it: every tile publishes its own
// slot, the barrier runs, then every tile reads every OTHER tile's slot in its
// scope. If the barrier released early, some slot is still at its initial
// value.
//
// One arrival per TILE, from core 0 of the tile -- that is the invariant the
// whole scope model rests on (a complete subtree, so aggr is uniform). The
// other cores of the tile take no part here; the intra-tile software barrier
// of the two-level design is a separate concern and is not what this gate is
// testing.

#include <stdint.h>

#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

// Separate block on purpose: clang-format sorts within a block, and
// fsync.h must come AFTER runtime.h (it calls mempool_wfi, which
// runtime.h defines static while synchronization.h declares non-static).
#include "fsync.h"

#define NUM_TILES (NUM_CORES / NUM_CORES_PER_TILE)

// MUST be initialised explicitly: `.l1` is NOBITS and the bss range is empty,
// so nothing zeroes it and every entry powers up X. An X compared here would
// not merely mis-report, it would feed a branch.
static volatile uint32_t slot[NUM_TILES] __attribute__((section(".l1")));
static volatile uint32_t errors[NUM_TILES] __attribute__((section(".l1")));

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  const uint32_t tile = core_id / NUM_CORES_PER_TILE;
  const uint32_t is_leader = (core_id % NUM_CORES_PER_TILE) == 0;

  mempool_barrier_init(core_id);

  // Zero both arrays before anything reads them.
  for (uint32_t t = core_id; t < NUM_TILES; t += num_cores) {
    slot[t] = 0;
    errors[t] = 0;
  }
  mempool_barrier(num_cores);

  if (core_id == 0)
    printf("[fsync_bar] tiles=%u leaf_side=%u aggr_global=0x%x\n",
           (unsigned)NUM_TILES, (unsigned)FSYNC_TILE_SIDE,
           (unsigned)fsync_aggr_global());

  if (is_leader) {
    // Publish. Fence so the store is visible before the barrier, not merely
    // issued -- the barrier orders arrivals, not posted writes.
    slot[tile] = tile + 1;
    __sync_synchronize();

    // The whole machine is one scope here: every tile is a member, so this is
    // the top-level square and id stays 0.
    //
    // FSYNC_BAR_NO_BARRIER is the negative control. Without it this test would
    // be worthless: if the slot writes happen to land before the check anyway,
    // a barrier that returns immediately passes just as well as a working one.
    // Build with -DFSYNC_BAR_NO_BARRIER and this MUST report errors.
#ifdef FSYNC_BAR_NO_BARRIER
    const uint32_t err = 0;
#else
    const uint32_t err = fsync_barrier_global();
#endif
    if (err) {
      errors[tile]++;
      printf("[fsync_bar] tile %u: barrier reported ERROR\n", (unsigned)tile);
    }

    // Every other tile must be visible now. If the barrier released early,
    // some slot is still 0.
    uint32_t bad = 0;
    for (uint32_t t = 0; t < NUM_TILES; t++)
      if (slot[t] != t + 1)
        bad++;
    if (bad) {
      errors[tile] += bad;
      if (tile == 0)
        printf("[fsync_bar] tile %u: %u/%u slots not visible after barrier\n",
               (unsigned)tile, (unsigned)bad, (unsigned)NUM_TILES);
    }
  }

  mempool_barrier(num_cores);

  if (core_id == 0) {
    uint32_t total = 0;
    for (uint32_t t = 0; t < NUM_TILES; t++)
      total += errors[t];
    printf("[fsync_bar] %u errors\n", (unsigned)total);
    return total ? 1 : 0;
  }
  mempool_barrier(num_cores);
  return 0;
}
