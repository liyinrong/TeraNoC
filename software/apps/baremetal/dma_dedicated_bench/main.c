// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// H3 (was A5): what a dedicated DMA core buys, measured without any RTL.
//
// RE-MEASURED against the solo collective path. The original run used
// dma_collective_all_memcpy_nonblocking + _wait_sync -- the slot path -- and
// reported 450-700 cyc/iteration. That control has since become 36% cheaper
// (solo path, f87f70de; wake-based barrier, 23f30fc2), so the old figure
// overstates what a dedicated core is worth and must not be quoted.
//
// A double-buffered L2 -> L1 stream with compute on the other buffer, run
// twice over identical machine state:
//
//   shared     every core computes; the group coordinators ALSO run the DMA
//              control, so their share of the compute finishes late and the
//              iteration barrier waits for them
//   dedicated  the coordinators do ONLY DMA control; the remaining
//              NUM_CORES - NUM_GROUPS cores carry all of the compute
//
// Both arms move the same bytes and accumulate the same golden sum, so a
// skipped iteration cannot look fast. The dedicated arm has 1/NUM_CORES_PER_
// GROUP fewer workers, so it wins only if control overhead costs more than
// that share of the compute -- which is exactly the question B5 asks.
//
// DED_REPS scales compute per iteration without touching the DMA, so a sweep
// finds the crossover instead of asserting one point.

#include <stddef.h>
#include <stdint.h>

#include "dma.h"
#include "dma_collective.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#ifndef DED_TILE_WORDS
#define DED_TILE_WORDS 4096u // must be a whole number of L1 rows
#endif
#ifndef DED_ITERS
#define DED_ITERS 4u
#endif
#ifndef DED_REPS
#define DED_REPS 1u
#endif

#define DED_BYTES ((size_t)DED_TILE_WORDS * sizeof(uint32_t))
#define DED_TOTAL (DED_ITERS * DED_TILE_WORDS)

static uint32_t l1_buf[2][DED_TILE_WORDS]
    __attribute__((section(".l1"), aligned(NUM_BANKS * 4)));
static uint32_t l2_src[DED_TOTAL]
    __attribute__((section(".l2"), aligned(8192)));
static uint32_t acc[NUM_CORES]
    __attribute__((section(".l1"), aligned(NUM_BANKS * 4)));
static uint32_t volatile t_shared __attribute__((section(".l1")));
static uint32_t volatile t_dedicated __attribute__((section(".l1")));
static uint32_t volatile sum_shared __attribute__((section(".l1")));
static uint32_t volatile sum_dedicated __attribute__((section(".l1")));

// One arm. `dedicated` moves the coordinators out of the compute set.
static uint32_t run_arm(uint32_t core, uint32_t dedicated) {
  const uint32_t coord = mempool_is_group_coordinator();
  const uint32_t worker = dedicated ? !coord : 1u;
  const uint32_t nworkers = dedicated
                                ? (uint32_t)NUM_CORES - (uint32_t)NUM_GROUPS
                                : (uint32_t)NUM_CORES;
  // Coordinators are core 0 of each group. The dense worker index drops every
  // coordinator at or below this core -- there are core/NUM_CORES_PER_GROUP + 1
  // of them, and forgetting the +1 shifts the whole range to [1, nworkers],
  // which silently never sums element 0 of each tile.
  const uint32_t wid =
      dedicated ? core - core / (uint32_t)NUM_CORES_PER_GROUP - 1u : core;

  acc[core] = 0u;
  // Prime buffer 0 so iteration 0 has something to compute on.
  if (coord)
    (void)dma_collective_all_memcpy_solo(l1_buf[0], l2_src, DED_BYTES);
  (void)dma_collective_all_wait_solo();

  const mempool_timer_t t0 = mempool_get_timer();
  uint32_t sum = 0u;
  for (uint32_t it = 0u; it < DED_ITERS; ++it) {
    const uint32_t cur = it & 1u;
    const uint32_t nxt = cur ^ 1u;
    // Prefetch the next tile, wrapping so every iteration has one outstanding
    // transfer and the two arms see an identical DMA sequence.
    if (coord)
      (void)dma_collective_all_memcpy_solo(
          l1_buf[nxt], &l2_src[((it + 1u) % DED_ITERS) * DED_TILE_WORDS],
          DED_BYTES);
    if (worker)
      for (uint32_t r = 0u; r < DED_REPS; ++r)
        for (uint32_t i = wid; i < DED_TILE_WORDS; i += nworkers)
          sum += l1_buf[cur][i];
    (void)dma_collective_all_wait_solo();
  }
  const mempool_timer_t t1 = mempool_get_timer();

  acc[core] = sum;
  mempool_barrier(NUM_CORES);
  return (uint32_t)(t1 - t0);
}

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t ncores = mempool_get_core_count();

  mempool_barrier_init(core);
  if (core == 0u) {
    t_shared = 0u;
    t_dedicated = 0u;
    sum_shared = 0u;
    sum_dedicated = 0u;
  }
  for (uint32_t i = core; i < DED_TOTAL; i += ncores)
    l2_src[i] = i + 1u;
  for (uint32_t i = core; i < ncores; i += ncores)
    acc[i] = 0u;
  mempool_barrier(ncores);

  const uint32_t cyc_shared = run_arm(core, 0u);
  if (core == 0u) {
    uint32_t s = 0u;
    for (uint32_t c = 0u; c < ncores; ++c)
      s += acc[c];
    sum_shared = s;
    t_shared = cyc_shared;
  }
  mempool_barrier(ncores);

  const uint32_t cyc_dedicated = run_arm(core, 1u);
  if (core == 0u) {
    uint32_t s = 0u;
    for (uint32_t c = 0u; c < ncores; ++c)
      s += acc[c];
    sum_dedicated = s;
    t_dedicated = cyc_dedicated;
  }
  mempool_barrier(ncores);

  if (core == 0u) {
    // Closed form: every iteration sums one whole tile of the 1..DED_TOTAL
    // ramp, DED_REPS times over.
    const uint32_t golden =
        DED_REPS *
        (uint32_t)((uint32_t)DED_TOTAL * ((uint32_t)DED_TOTAL + 1u) / 2u);
    printf("[dma_ded] words=%u iters=%u reps=%u shared=%u dedicated=%u "
           "sums=%u,%u golden=%u\n",
           (unsigned)DED_TILE_WORDS, (unsigned)DED_ITERS, (unsigned)DED_REPS,
           (unsigned)t_shared, (unsigned)t_dedicated, (unsigned)sum_shared,
           (unsigned)sum_dedicated, (unsigned)golden);
    if (sum_shared != golden || sum_dedicated != golden)
      printf("[dma_ded] SUM MISMATCH\n");
  }
  mempool_barrier(ncores);
  {
    const uint32_t golden =
        DED_REPS *
        (uint32_t)((uint32_t)DED_TOTAL * ((uint32_t)DED_TOTAL + 1u) / 2u);
    return (sum_shared == golden && sum_dedicated == golden) ? 0 : 1;
  }
}
