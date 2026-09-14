// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Directed gate: nothing but its own tree response may release a core from an
// fsync barrier.
//
//   stale   (arm 1) core 0 subscribes to its group's DMA event and polls a
//           transfer to completion, so the event lands while it is awake. It
//           restores the barrier subscription and arrives FIRST at a barrier.
//   credit  (arm 2) core 1 gives core 0 a legacy wake while it is awake. Core 0
//           arrives LAST and triggers, then arrives FIRST at the next barrier.
//
// The other cores arrive late and mark themselves just before arriving, so a
// core 0 that finds a mark missing left its barrier early. Select arms with
// -DGATE_ARMS=1 or 2; the default runs both.

#include <stdint.h>

#include "dma.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "fsync.h"

#ifndef GATE_ARMS
#define GATE_ARMS 3u
#endif
#define GATE_ROUNDS 4u
#define GATE_LATE 4096u
#define GATE_TILES (NUM_CORES / NUM_CORES_PER_TILE)
#define GATE_BYTES ((uint32_t)(NUM_BANKS * 4) / (uint32_t)NUM_GROUPS)

static fsync_tile_counter_t tile_counter[GATE_TILES]
    __attribute__((aligned(FSYNC_TILE_COUNTER_ALIGNMENT), section(".l1")));
static uint32_t mark[NUM_CORES]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")));
static uint32_t finished[NUM_CORES]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")));
static uint32_t l1_buf[GATE_BYTES / 4u]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")));
static uint32_t l2_buf[GATE_BYTES / 4u]
    __attribute__((aligned(8192), section(".l2")));
static uint32_t volatile stale_errors __attribute__((section(".l1")));
static uint32_t volatile credit_errors __attribute__((section(".l1")));
static uint32_t volatile published __attribute__((section(".l1")));

// Core 0 arrives first; everyone else arrives late and marks itself first.
static uint32_t barrier_core0_first(uint32_t core, uint32_t tile,
                                    uint32_t tag) {
  if (core != 0u)
    mempool_wait(GATE_LATE);
  __atomic_store_n(&mark[core], tag, __ATOMIC_RELEASE);
  fsync_barrier_all(tile_counter, tile);
  uint32_t missing = 0u;
  if (core == 0u) {
    for (uint32_t c = 0u; c < (uint32_t)NUM_CORES; ++c)
      if (__atomic_load_n(&mark[c], __ATOMIC_ACQUIRE) < tag)
        ++missing;
    // An early exit must not also run ahead into the next barrier, where it
    // would desynchronise the tile count and hang instead of reporting.
    for (uint32_t c = 0u; c < (uint32_t)NUM_CORES; ++c)
      while (__atomic_load_n(&mark[c], __ATOMIC_ACQUIRE) < tag)
        mempool_wait(16u);
  }
  return missing;
}

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t tile = mempool_get_tile_id();

  mempool_barrier_init(core);
  mark[core] = 0u;
  finished[core] = 0u;
  for (uint32_t t = core; t < GATE_TILES; t += NUM_CORES)
    fsync_tile_counter_init(&tile_counter[t]);
  if (core == 0u) {
    stale_errors = 0u;
    credit_errors = 0u;
    published = 0u;
    for (uint32_t i = 0u; i < GATE_BYTES / 4u; ++i)
      l2_buf[i] = i + 1u;
  }
  mempool_barrier(NUM_CORES);

  for (uint32_t r = 0u; r < GATE_ROUNDS; ++r) {
    if (GATE_ARMS & 1u) {
      if (core == 0u) {
        fsync_subscribe_self(FSYNC_EVT_DMA);
        (void)dma_local_memcpy_nonblocking(0u, l1_buf, l2_buf, GATE_BYTES);
        (void)dma_local_wait(0u); // polls: the idle event lands while awake
        mempool_wait(256u);
        fsync_subscribe_self(FSYNC_EVT_FSYNC);
      }
      // Only core 0 counts: every other core would race it on the shared word.
      const uint32_t missing = barrier_core0_first(core, tile, 4u * r + 1u);
      if (core == 0u)
        stale_errors += missing;
    }
    if (GATE_ARMS & 2u) {
      if (core == 1u) {
        mempool_wait(256u);
        wake_up(0u); // core 0 is spinning below, so the wake is banked
      }
      if (core == 0u)
        mempool_wait(4u * GATE_LATE);
      __atomic_store_n(&mark[core], 4u * r + 2u, __ATOMIC_RELEASE);
      fsync_barrier_all(tile_counter, tile);
      const uint32_t missing = barrier_core0_first(core, tile, 4u * r + 3u);
      if (core == 0u)
        credit_errors += missing;
    }
  }

  // Spins, not barriers: a broken barrier is what is under test.
  __atomic_store_n(&finished[core], 1u, __ATOMIC_RELEASE);
  if (core == 0u) {
    for (uint32_t c = 0u; c < (uint32_t)NUM_CORES; ++c)
      while (__atomic_load_n(&finished[c], __ATOMIC_ACQUIRE) == 0u)
        mempool_wait(16u);
    printf("[fsync_stale] rounds=%u arms=%u stale_errors=%u credit_errors=%u\n",
           (unsigned)GATE_ROUNDS, (unsigned)GATE_ARMS, (unsigned)stale_errors,
           (unsigned)credit_errors);
    __atomic_store_n(&published, 1u, __ATOMIC_RELEASE);
  }
  while (__atomic_load_n(&published, __ATOMIC_ACQUIRE) == 0u)
    mempool_wait(16u);
  return (int)(stale_errors + credit_errors);
}
