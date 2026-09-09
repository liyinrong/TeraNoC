// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// One producer, many consumers: an engine dependency with no software fan-out.
//
// EVERY core of every group subscribes to the engine event and sleeps. ONE core
// per group starts the engine. When it finishes, the event unit sets a pending
// bit in each subscribed core independently, so all of them wake from the
// single pulse. That is the shape of "both Spatz cores wait on RedMulE": the
// producer does not know or care how many consumers there are.
//
// The DMA is the producer here because it can be driven without configuring a
// GEMM. EVT_ENGINE (RedMulE end-of-computation) takes the identical path in the
// shim -- evt_set[bit] = source & subscribe_q[c][bit], per core -- so this
// exercises the fan-out itself, not one particular source.
//
// The gate is liveness: a core whose wfi is never answered hangs, so completing
// at all proves every subscriber was woken. The spread between the first and
// last wake shows they were released together rather than in sequence.
//
// Credits: mempool_barrier leaves every core balanced (the last arrival pulses
// all, each spends it on its own wfi), so the wfi under test genuinely sleeps
// instead of returning on a banked credit and reporting a vacuous pass.

#include <stdint.h>

#include "dma.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "fsync.h"

#define ROW_BYTES ((uint32_t)(NUM_BANKS * 4))
#define GROUP_BYTES (ROW_BYTES / (uint32_t)NUM_GROUPS)
#define FAN_ROWS 32u
#define FAN_WORDS ((ROW_BYTES * FAN_ROWS) / 4u)

static uint32_t l1_buf[FAN_WORDS]
    __attribute__((section(".l1"), aligned(NUM_BANKS * 4)));
static uint32_t l2_buf[FAN_WORDS]
    __attribute__((section(".l2"), aligned(8192)));
static uint32_t woke_at[NUM_CORES]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")));
static uint32_t volatile errs __attribute__((section(".l1")));

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t ncores = mempool_get_core_count();
  const uint32_t group = mempool_get_group_id();

  mempool_barrier_init(core);
  if (core == 0u)
    errs = 0u;
  woke_at[core] = 0u;
  for (uint32_t i = core; i < FAN_WORDS; i += ncores) {
    l2_buf[i] = i + 1u;
    l1_buf[i] = 0u;
  }
  mempool_barrier(ncores);

  // Subscribe BEFORE the producer can start: an edge landing after the
  // subscribe is held in pending, one landing before was never recorded.
  fsync_subscribe_self(FSYNC_EVT_DMA);
  mempool_barrier(ncores);

  const mempool_timer_t t0 = mempool_get_timer();
  if (mempool_is_group_coordinator()) {
    const uint32_t off = group * GROUP_BYTES;
    (void)dma_local_memcpy_2d_nonblocking(
        0u, (void *)((uint32_t)l1_buf + off),
        (const void *)((uint32_t)l2_buf + off), GROUP_BYTES, ROW_BYTES,
        ROW_BYTES, FAN_ROWS);
  }
  // Every core sleeps, consumer and producer alike. Reaching the next line at
  // all is the liveness proof.
  mempool_wfi();
  woke_at[core] = (uint32_t)(mempool_get_timer() - t0);

  fsync_subscribe_self(FSYNC_EVT_FSYNC);
  mempool_barrier(ncores);

  for (uint32_t i = core; i < FAN_WORDS; i += ncores)
    if (l1_buf[i] != i + 1u)
      __atomic_fetch_add(&errs, 1u, __ATOMIC_RELAXED);
  mempool_barrier(ncores);

  if (core == 0u) {
    uint32_t lo = 0xffffffffu, hi = 0u, zero = 0u;
    for (uint32_t c = 0; c < ncores; ++c) {
      const uint32_t w = woke_at[c];
      if (w == 0u)
        ++zero;
      if (w < lo)
        lo = w;
      if (w > hi)
        hi = w;
    }
    printf("[evt_fanout] cores=%u woke_min=%u woke_max=%u spread=%u "
           "never_woke=%u data_errs=%u\n",
           (unsigned)ncores, (unsigned)lo, (unsigned)hi, (unsigned)(hi - lo),
           (unsigned)zero, (unsigned)errs);
  }
  mempool_barrier(ncores);
  return (int)errs;
}
