// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// A barrier baseline tight enough to judge a hardware change against.
//
// [fsync_cost] in dma_split_bench read 96, 96, 115, 118 across runs -- a 23%
// spread on the very quantity an event unit would improve by ~60 cycles. Two
// fixes: many repetitions with MIN reported (a lower bound is far less noisy
// than a mean when the noise is one-sided contention), and a decomposition, so
// the software part and the hardware part are separated rather than inferred.
//
//   A  full two-level barrier      tile rendezvous + tree
//   B  tile rendezvous only        the software wrapper, no tree
//   C  tree only                   trigger + wfi + status, one core per tile
//
// A ~ B + C, and B is what an event unit with per-core arrival aggregation
// would remove.
//
// C is only safe to measure because of the per-core wake mask: the three
// non-triggering cores of each tile are masked out, so they cannot bank the
// broadcast they would otherwise receive without ever sleeping for it.

#include <stdint.h>

#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "fsync.h"

#define COST_TILES (NUM_CORES / NUM_CORES_PER_TILE)
#define COST_REPS 64u

static fsync_tile_counter_t tile_counter[COST_TILES]
    __attribute__((aligned(FSYNC_TILE_COUNTER_ALIGNMENT), section(".l1")));
static uint32_t volatile go __attribute__((section(".l1")));
static uint32_t volatile out[8] __attribute__((section(".l1")));

// The software half of fsync_barrier_n: arrive, and the last one releases.
// No trigger, no tree.
static inline void tile_rendezvous(uint32_t tile, uint32_t participants) {
  const uint32_t expected =
      __atomic_load_n(&tile_counter[tile].done_epoch, __ATOMIC_ACQUIRE);
  if (__atomic_fetch_add(&tile_counter[tile].count, 1, __ATOMIC_RELAXED) ==
      participants - 1u) {
    tile_counter[tile].count = 0;
    __atomic_store_n(&tile_counter[tile].done_epoch, expected + 1u,
                     __ATOMIC_RELEASE);
  } else {
    while (__atomic_load_n(&tile_counter[tile].done_epoch, __ATOMIC_ACQUIRE) <=
           expected)
      mempool_wait(4u);
  }
}

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t tile = mempool_get_tile_id();
  const uint32_t lane = core % (uint32_t)NUM_CORES_PER_TILE;

  mempool_barrier_init(core);
  if (core == 0u) {
    go = 0u;
    for (uint32_t i = 0; i < 8u; ++i)
      out[i] = 0u;
  }
  for (uint32_t t = core; t < COST_TILES; t += NUM_CORES)
    fsync_tile_counter_init(&tile_counter[t]);
  mempool_barrier(NUM_CORES);

  // ---- A: full two-level barrier
  uint32_t a_min = 0xffffffffu;
  for (uint32_t r = 0; r < COST_REPS; ++r) {
    const mempool_timer_t t0 = mempool_get_timer();
    fsync_barrier_all(tile_counter, tile);
    const mempool_timer_t t1 = mempool_get_timer();
    const uint32_t d = (uint32_t)(t1 - t0);
    if (d < a_min)
      a_min = d;
  }
  if (core == 0u)
    out[0] = a_min;
  mempool_barrier(NUM_CORES);

  // ---- B: tile rendezvous only
  uint32_t b_min = 0xffffffffu;
  for (uint32_t r = 0; r < COST_REPS; ++r) {
    const mempool_timer_t t0 = mempool_get_timer();
    tile_rendezvous(tile, (uint32_t)NUM_CORES_PER_TILE);
    const mempool_timer_t t1 = mempool_get_timer();
    const uint32_t d = (uint32_t)(t1 - t0);
    if (d < b_min)
      b_min = d;
  }
  if (core == 0u)
    out[1] = b_min;
  mempool_barrier(NUM_CORES);

  // ---- C: tree only, one core per tile. The others are masked OUT of the
  // broadcast, which is the only thing that makes this safe to run in a loop.
  if (lane != 0u)
    fsync_wake_mask_exclude_self();
  mempool_barrier(NUM_CORES);

  uint32_t c_min = 0xffffffffu;
  if (lane == 0u) {
    const uint32_t aggr = fsync_aggr_global();
    for (uint32_t r = 0; r < COST_REPS; ++r) {
      const mempool_timer_t t0 = mempool_get_timer();
      // Arm before sleeping: the pulse only reaches armed cores now, so a raw
      // trigger/wfi pair without this would sleep forever.
      fsync_arm_self();
      fsync_trigger(FSYNC_ID_H, aggr);
      mempool_wfi();
      while ((fsync_status() & FSYNC_STATUS_BUSY_MASK) != 0u)
        ;
      const mempool_timer_t t1 = mempool_get_timer();
      const uint32_t d = (uint32_t)(t1 - t0);
      if (d < c_min)
        c_min = d;
    }
    if (core == 0u) {
      out[2] = c_min;
      __atomic_store_n(&go, 1u, __ATOMIC_RELEASE);
    }
  } else {
    while (__atomic_load_n(&go, __ATOMIC_ACQUIRE) == 0u)
      mempool_wait(16u);
  }
  mempool_barrier(NUM_CORES);

  // Restore, so nothing downstream inherits a narrowed mask.
  fsync_set_wake_mask((1u << NUM_CORES_PER_TILE) - 1u);
  mempool_barrier(NUM_CORES);

  if (core == 0u)
    printf("[fsync_cost_gate] reps=%u full=%u rendezvous=%u tree=%u "
           "wrapper=%d\n",
           (unsigned)COST_REPS, (unsigned)out[0], (unsigned)out[1],
           (unsigned)out[2], (int)out[0] - (int)out[2]);
  mempool_barrier(NUM_CORES);
  return 0;
}
