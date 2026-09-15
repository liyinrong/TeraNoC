// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Directed gate for fsync barriers that only part of a tile takes part in.
//
// One core per tile stays out of the barriers while the others run more than
// eight of them, the shape of a tile that holds a dedicated DMA/control core.
// A core that stays subscribed to the barrier event while it never sleeps for
// it collects a pending bit from every response, and that bit wakes it early
// out of a later mempool_barrier. The non-participant therefore unsubscribes.
//
//   default                 the non-participant unsubscribes   -> passes
//   -DGATE_NO_UNSUBSCRIBE   it stays subscribed and awake      -> early_exit
//                                                                  or progress
//                                                                  errors

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
// Per-core progress: a core that left a barrier early stops short.
static uint32_t rounds_done[NUM_CORES]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")));
static uint32_t volatile progress_errors __attribute__((section(".l1")));
// Every core waits for core 0 to publish before returning: _eoc runs for every
// core and the testbench ends the run on the first nonzero EOC write, so a core
// that leaves a broken barrier early must not end the run before core 0 checks.
static uint32_t volatile published __attribute__((section(".l1")));
// Did a core leave a software barrier before every core arrived? Each core
// counts itself in and checks the total after the barrier.
static uint32_t volatile arrived __attribute__((section(".l1")));
static uint32_t volatile early_exit_errors __attribute__((section(".l1")));
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
    progress_errors = 0u;
    published = 0u;
    arrived = 0u;
    early_exit_errors = 0u;
  }
  rounds_done[core] = 0u;
  for (uint32_t t = core; t < GATE_TILES; t += NUM_CORES)
    fsync_tile_counter_init(&tile_counter[t]);
  mempool_barrier(NUM_CORES);

#ifndef GATE_NO_UNSUBSCRIBE
  if (excluded)
    fsync_unsubscribe_self();
#endif
  mempool_barrier(NUM_CORES);

  if (!excluded) {
    for (uint32_t r = 0; r < GATE_BARRIERS; ++r) {
      fsync_barrier_n(tile_counter, tile, FSYNC_ID_H, fsync_aggr_global(),
                      (uint32_t)NUM_CORES_PER_TILE - 1u);
      __atomic_store_n(&rounds_done[core], r + 1u, __ATOMIC_RELEASE);
    }
    if (core == 0u) {
      rounds = GATE_BARRIERS;
      __atomic_store_n(&done_flag, 1u, __ATOMIC_RELEASE);
    }
  } else {
    // Spin, deliberately: staying awake is the condition under test.
    while (__atomic_load_n(&done_flag, __ATOMIC_ACQUIRE) == 0u)
      mempool_wait(16u);
  }
  mempool_barrier(NUM_CORES);

  __atomic_fetch_add(&arrived, 1u, __ATOMIC_RELAXED);
  mempool_barrier(NUM_CORES);
  if (__atomic_load_n(&arrived, __ATOMIC_ACQUIRE) != (uint32_t)NUM_CORES)
    __atomic_fetch_add(&early_exit_errors, 1u, __ATOMIC_RELAXED);
  mempool_barrier(NUM_CORES);

  if (core == 0u)
    for (uint32_t c = 0; c < (uint32_t)NUM_CORES; ++c)
      if ((c % (uint32_t)NUM_CORES_PER_TILE) !=
              (uint32_t)NUM_CORES_PER_TILE - 1u &&
          __atomic_load_n(&rounds_done[c], __ATOMIC_ACQUIRE) != GATE_BARRIERS)
        ++progress_errors;
  mempool_barrier(NUM_CORES);

  if (core == 0u)
    printf("[fsync_partial_tile] barriers=%u participants=%u "
           "progress_errors=%u early_exit=%u\n",
           (unsigned)rounds, (unsigned)(NUM_CORES_PER_TILE - 1),
           (unsigned)progress_errors, (unsigned)early_exit_errors);
  if (core == 0u)
    __atomic_store_n(&published, 1u, __ATOMIC_RELEASE);
  // A spin, not a barrier: a broken barrier is what is under test.
  while (__atomic_load_n(&published, __ATOMIC_ACQUIRE) == 0u)
    mempool_wait(16u);
  return (int)(progress_errors + early_exit_errors);
}
