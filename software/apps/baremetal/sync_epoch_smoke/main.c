// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define SYNC_EPOCH_ITERS 8u
#define BROKEN_SPIN_LIMIT 256u

static uint32_t arrival[NUM_CORES]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")));
static uint32_t total_errors __attribute__((section(".l1")));
static uint32_t broken_count __attribute__((section(".l1")));
static uint32_t broken_timeouts __attribute__((section(".l1")));

static void generation_barrier(uint32_t generation, uint32_t core) {
  switch (generation & 3u) {
  case 0u:
    mempool_barrier(NUM_CORES);
    break;
  case 1u:
    mempool_log_barrier(2u, core);
    break;
  case 2u:
    mempool_anyradixlog_barrier(4u, core);
    break;
  default:
    mempool_linlog_barrier(4u, core);
    break;
  }
}

static void broken_count_only_barrier(uint32_t reserve_reuse) {
  const uint32_t core = mempool_get_core_id();
  if (!reserve_reuse && core == NUM_CORES - 1u) {
    while (__atomic_load_n(&broken_count, __ATOMIC_ACQUIRE) != 0u) {
    }
    return;
  }
  const uint32_t prior =
      __atomic_fetch_add(&broken_count, 1u, __ATOMIC_ACQ_REL);
  if (prior == NUM_CORES - 1u) {
    // Reserve this last core's next-generation arrival immediately. Old
    // waiters can no longer distinguish the missed zero from a still-active
    // first generation.
    __atomic_store_n(&broken_count, reserve_reuse ? 1u : 0u, __ATOMIC_RELEASE);
    return;
  }
  uint32_t spins = 0u;
  while (__atomic_load_n(&broken_count, __ATOMIC_ACQUIRE) != 0u &&
         spins < BROKEN_SPIN_LIMIT)
    ++spins;
  if (spins == BROKEN_SPIN_LIMIT)
    __atomic_fetch_add(&broken_timeouts, 1u, __ATOMIC_RELAXED);
}

int main() {
  const uint32_t core = mempool_get_core_id();

  mempool_barrier_init(core);
  arrival[core] = 0u;
  if (core == 0u) {
    total_errors = 0u;
    broken_count = 0u;
    broken_timeouts = 0u;
  }
  mempool_barrier(NUM_CORES);

  for (uint32_t generation = 1u; generation <= SYNC_EPOCH_ITERS; ++generation) {
    if (core == (generation * 17u) % NUM_CORES)
      mempool_wait(64u + generation * 8u);
    __atomic_store_n(&arrival[core], generation, __ATOMIC_RELEASE);
    generation_barrier(generation, core);
    if (core == 0u)
      for (uint32_t peer = 0u; peer < NUM_CORES; ++peer)
        if (__atomic_load_n(&arrival[peer], __ATOMIC_ACQUIRE) < generation)
          ++total_errors;
    generation_barrier(generation, core);
  }

  // Deliberate falsification: a reset-to-zero count has an ABA window. The
  // last core immediately enters reuse while some old waiters are still
  // observing the count, so those waiters miss zero and hit the bounded trip.
  if (core == NUM_CORES - 1u)
    mempool_wait(256u);
  broken_count_only_barrier(1u);
  broken_count_only_barrier(0u);
  mempool_barrier(NUM_CORES);

  if (core == 0u) {
    if (broken_timeouts == 0u)
      ++total_errors;
    printf("[sync_epoch_smoke] generations=%u broken_timeouts=%u errors=%u\n",
           (unsigned)SYNC_EPOCH_ITERS, (unsigned)broken_timeouts,
           (unsigned)total_errors);
  }
  mempool_barrier(NUM_CORES);
  return (int)total_errors;
}
