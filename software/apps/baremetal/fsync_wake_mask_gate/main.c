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
// Per-core progress. Under the event unit a missing unsubscribe no longer
// wraps the wake counter -- it leaves a stale pending that wakes the core out
// of a LATER mempool_barrier, so the failure is an early barrier exit, not a
// missed wake. MISSED cannot see that, and the run would otherwise report
// retval=0 with no output, which a runner reads as a pass. Check progress
// explicitly so the negative control fails loudly.
static uint32_t rounds_done[NUM_CORES]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")));
static uint32_t volatile progress_errors __attribute__((section(".l1")));
// Every core waits for core 0 to publish before any of them may return.
//
// _eoc runs after main for EVERY core (crt0.S:141) and the TB ends the run on
// the first nonzero EOC write, so a core that leaves a broken barrier early
// terminates the simulation while core 0 is still verifying -- traced exactly
// that: the excluded lane reached _eoc at cycle 3775 and killed the run at
// 3784, with core 0 stopped mid progress-check at 0x80000400. The result was a
// silent retval=0, which a runner reads as a pass.
static uint32_t volatile published __attribute__((section(".l1")));
// Direct test of the property: did this core leave a software barrier BEFORE
// every core arrived? Each core counts itself in, then checks the total after
// the barrier; a core woken by stale pending sees fewer than NUM_CORES. The
// fsync barriers themselves stay correct without the unsubscribe -- measured --
// so progress_errors alone cannot see this.
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

#ifndef GATE_NO_MASK
  // Take this core out of the tile's barrier wakes before any barrier runs.
  if (excluded)
    fsync_wake_mask_exclude_self();
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
    // Spin, deliberately: staying AWAKE is what makes an unmasked wake get
    // banked rather than consumed, which is the condition under test.
    while (__atomic_load_n(&done_flag, __ATOMIC_ACQUIRE) == 0u)
      mempool_wait(16u);
  }
  mempool_barrier(NUM_CORES);

  __atomic_fetch_add(&arrived, 1u, __ATOMIC_RELAXED);
  mempool_barrier(NUM_CORES);
  if (__atomic_load_n(&arrived, __ATOMIC_ACQUIRE) != (uint32_t)NUM_CORES)
    __atomic_fetch_add(&early_exit_errors, 1u, __ATOMIC_RELAXED);
  mempool_barrier(NUM_CORES);

  // Every participant must have completed every round. A core that left a
  // barrier early stops short, which is exactly the stale-pending failure.
  if (core == 0u)
    for (uint32_t c = 0; c < (uint32_t)NUM_CORES; ++c)
      if ((c % (uint32_t)NUM_CORES_PER_TILE) !=
              (uint32_t)NUM_CORES_PER_TILE - 1u &&
          __atomic_load_n(&rounds_done[c], __ATOMIC_ACQUIRE) != GATE_BARRIERS)
        ++progress_errors;
  mempool_barrier(NUM_CORES);

  if (core == 0u)
    printf("[fsync_wake_mask] progress_errors=%u early_exit=%u\n",
           (unsigned)progress_errors, (unsigned)early_exit_errors);
  if (core == 0u)
    printf("[fsync_wake_mask] barriers=%u participants=%u mask=0x%x "
           "excluded_lane=%u\n",
           (unsigned)rounds, (unsigned)(NUM_CORES_PER_TILE - 1),
           (unsigned)fsync_wake_mask(), (unsigned)(NUM_CORES_PER_TILE - 1));
  if (core == 0u)
    __atomic_store_n(&published, 1u, __ATOMIC_RELEASE);
  // Deliberately a spin, not a barrier: a broken barrier is what is under test,
  // so the exit path must not depend on one.
  while (__atomic_load_n(&published, __ATOMIC_ACQUIRE) == 0u)
    mempool_wait(16u);
  return (int)(progress_errors + early_exit_errors);
}
