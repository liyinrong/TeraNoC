// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "fsync.h"

#define FSYNC_EPOCH_TILES (NUM_CORES / NUM_CORES_PER_TILE)
#ifdef FSYNC_NEGATIVE_SKIP_ISSUE_EPOCH
#define FSYNC_EPOCH_ITERS 1u
#else
#define FSYNC_EPOCH_ITERS 4u
#endif

static fsync_tile_counter_t tile_counter[FSYNC_EPOCH_TILES]
    __attribute__((aligned(FSYNC_TILE_COUNTER_ALIGNMENT), section(".l1")));
static uint32_t arrival[NUM_CORES]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")));
static uint32_t total_errors __attribute__((section(".l1")));

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t tile = mempool_get_tile_id();

  mempool_barrier_init(core);
  arrival[core] = 0u;
  for (uint32_t t = core; t < FSYNC_EPOCH_TILES; t += NUM_CORES)
    fsync_tile_counter_init(&tile_counter[t]);
  if (core == 0u)
    total_errors = 0u;
  mempool_barrier(NUM_CORES);

  for (uint32_t generation = 1u; generation <= FSYNC_EPOCH_ITERS;
       ++generation) {
    if (core % NUM_CORES_PER_TILE == NUM_CORES_PER_TILE - 1u)
      mempool_wait(128u + 8u * tile);
    __atomic_store_n(&arrival[core], generation, __ATOMIC_RELEASE);
    fsync_barrier_all(tile_counter, tile);
    if (core == 0u)
      for (uint32_t peer = 0u; peer < NUM_CORES; ++peer)
        if (__atomic_load_n(&arrival[peer], __ATOMIC_ACQUIRE) < generation)
          ++total_errors;
  }

  mempool_barrier(NUM_CORES);
  for (uint32_t t = core; t < FSYNC_EPOCH_TILES; t += NUM_CORES)
    if (tile_counter[t].issue_epoch != FSYNC_EPOCH_ITERS ||
        tile_counter[t].done_epoch != FSYNC_EPOCH_ITERS ||
        tile_counter[t].status_error != 0u)
      __atomic_fetch_add(&total_errors, 1u, __ATOMIC_RELAXED);
  mempool_barrier(NUM_CORES);

  if (core == 0u)
    printf("[fsync_epoch_smoke] generations=%u issue=%u done=%u errors=%u\n",
           (unsigned)FSYNC_EPOCH_ITERS, (unsigned)tile_counter[0].issue_epoch,
           (unsigned)tile_counter[0].done_epoch, (unsigned)total_errors);
  mempool_barrier(NUM_CORES);
  return (int)total_errors;
}
