// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Directed gate for the per-core fsync wake mask.
//
// fs_wake is broadcast to every core of a tile. A core that receives it without
// ever sleeping for it banks a wake credit per barrier, and wake_up_q is three
// bits that wrap WITHOUT saturating -- so the eighth unconsumed wake destroys
// all eight credits and the core's next sleep never returns. That is exactly
// the shape of a tile that holds a dedicated DMA/control core.
//
// One core per tile stays out of the barriers while the others run more than
// eight of them. The oracle is the RTL detector in snitch.sv, which fires when
// a wake arrives with the counter already full:
//
//   masked   (default build)    the excluded core receives nothing -> 0 reports
//   unmasked (-DGATE_NO_MASK)   it banks one per barrier            -> reports
//
// So the negative control is a build flag, and a pass cannot be vacuous.

#include <stdint.h>

#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "fsync.h"

#define GATE_TILES (NUM_CORES / NUM_CORES_PER_TILE)
#define GATE_BARRIERS 16u // past the 8-credit wrap, twice

static fsync_tile_counter_t tile_counter[GATE_TILES]
    __attribute__((aligned(FSYNC_TILE_COUNTER_ALIGNMENT), section(".l1")));
static uint32_t volatile done_flag __attribute__((section(".l1")));
static uint32_t volatile rounds __attribute__((section(".l1")));

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t tile = mempool_get_tile_id();
  const uint32_t lane = core % (uint32_t)NUM_CORES_PER_TILE;
  // The last core of every tile stands in for the control core.
  const uint32_t excluded = (lane == (uint32_t)NUM_CORES_PER_TILE - 1u);

  mempool_barrier_init(core);
  if (core == 0u) {
    done_flag = 0u;
    rounds = 0u;
  }
  for (uint32_t t = core; t < GATE_TILES; t += NUM_CORES)
    fsync_tile_counter_init(&tile_counter[t]);
  mempool_barrier(NUM_CORES);

#ifndef GATE_NO_MASK
  // Take this core out of the tile's barrier wakes before any barrier runs.
  if (excluded)
    fsync_wake_mask_exclude_self();
#endif
  mempool_barrier(NUM_CORES);

  if (!excluded) {
    for (uint32_t r = 0; r < GATE_BARRIERS; ++r)
      fsync_barrier_n(tile_counter, tile, FSYNC_ID_H, fsync_aggr_global(),
                      (uint32_t)NUM_CORES_PER_TILE - 1u);
    if (core == 0u) {
      rounds = GATE_BARRIERS;
      __atomic_store_n(&done_flag, 1u, __ATOMIC_RELEASE);
    }
  } else {
    // Spin, deliberately: staying AWAKE is what makes an unmasked wake get
    // banked rather than consumed, which is the condition under test.
    while (__atomic_load_n(&done_flag, __ATOMIC_ACQUIRE) == 0u)
      mempool_wait(16u);
  }
  mempool_barrier(NUM_CORES);

  if (core == 0u)
    printf("[fsync_wake_mask] barriers=%u participants=%u mask=0x%x "
           "excluded_lane=%u\n",
           (unsigned)rounds, (unsigned)(NUM_CORES_PER_TILE - 1),
           (unsigned)fsync_wake_mask(), (unsigned)(NUM_CORES_PER_TILE - 1));
  mempool_barrier(NUM_CORES);
  return 0;
}
