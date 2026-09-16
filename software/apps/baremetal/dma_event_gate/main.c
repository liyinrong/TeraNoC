// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// The DMA completion EVENT gate. The runtime polls, so this is the only place
// the engine's event is exercised end to end.
//
// Two arms move the same bytes with the same engine and verify the same data.
// The only difference is how the issuing core waits:
//
//   poll   wait_poll()  -- spins on the status register
//   sleep  wait_event() -- subscribes to FSYNC_EVT_DMA (the iDMA frontend's
//                          completion pulse) and wfi
//
// The gate is that BOTH land the data and neither core is released early. The
// interesting number is retired instructions: a sleeping core executes almost
// none while it waits, which is what makes the wait free for the rest of the
// tile. Cycles are NOT expected to improve -- the transfer takes as long either
// way.

#include <stdint.h>

#include "dma.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "fsync.h"

#define ROW_BYTES ((uint32_t)(NUM_BANKS * 4))
#define GROUP_BYTES (ROW_BYTES / (uint32_t)NUM_GROUPS)
#define EVT_ROWS 8u
#define EVT_WORDS ((ROW_BYTES * EVT_ROWS) / 4u)

static uint32_t l1_buf[EVT_WORDS]
    __attribute__((section(".l1"), aligned(NUM_BANKS * 4)));
static uint32_t l2_buf[EVT_WORDS]
    __attribute__((section(".l2"), aligned(8192)));
static uint32_t volatile cyc[2] __attribute__((section(".l1")));
static uint32_t volatile instr[2] __attribute__((section(".l1")));
static uint32_t volatile errs[2] __attribute__((section(".l1")));

static uint32_t wait_poll(uint32_t stream) {
  while (!dma_local_idle(stream))
    mempool_wait(4u);
  return DMA_LOCAL_OK;
}

// Sleep on the engine's completion event. Subscribe BEFORE the first test and
// FENCE: an edge that lands after the subscribe is held pending and the wfi
// returns at once, but one that landed before was never recorded and the core
// would sleep forever (software/tests/baremetal/fence has the ordering proof).
// The loop re-tests because a wfi also returns for an unrelated wake. The
// barrier subscription is restored on the way out, which also drops an event
// taken while this core was awake.
static uint32_t wait_event(uint32_t stream) {
  fsync_subscribe_self(FSYNC_EVT_DMA);
  __sync_synchronize();
  while (!dma_local_idle(stream))
    mempool_wfi();
  fsync_subscribe_self(FSYNC_EVT_FSYNC);
  __sync_synchronize();
  return DMA_LOCAL_OK;
}

static void arm(uint32_t which, uint32_t core, uint32_t ncores,
                uint32_t group) {
  for (uint32_t i = core; i < EVT_WORDS; i += ncores)
    l1_buf[i] = 0u;
  mempool_barrier(ncores);

  if (mempool_is_group_coordinator()) {
    const uint32_t off = group * GROUP_BYTES;
    const uint32_t i0 = (uint32_t)read_csr(minstret);
    const mempool_timer_t t0 = mempool_get_timer();
    (void)dma_local_memcpy_2d_nonblocking(
        0u, (void *)((uint32_t)l1_buf + off),
        (const void *)((uint32_t)l2_buf + off), GROUP_BYTES, ROW_BYTES,
        ROW_BYTES, EVT_ROWS);
    const uint32_t e = which ? wait_event(0u) : wait_poll(0u);
    const mempool_timer_t t1 = mempool_get_timer();
    const uint32_t i1 = (uint32_t)read_csr(minstret);
    if (group == 0u) {
      cyc[which] = (uint32_t)(t1 - t0);
      instr[which] = i1 - i0;
      if (e)
        __atomic_fetch_add(&errs[which], 1u, __ATOMIC_RELAXED);
    }
  }
  mempool_barrier(ncores);

  for (uint32_t i = core; i < EVT_WORDS; i += ncores)
    if (l1_buf[i] != i + 1u)
      __atomic_fetch_add(&errs[which], 1u, __ATOMIC_RELAXED);
  mempool_barrier(ncores);
}

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t ncores = mempool_get_core_count();
  const uint32_t group = mempool_get_group_id();

  mempool_barrier_init(core);
  if (core == 0u)
    for (uint32_t i = 0; i < 2u; ++i) {
      cyc[i] = 0u;
      instr[i] = 0u;
      errs[i] = 0u;
    }
  for (uint32_t i = core; i < EVT_WORDS; i += ncores)
    l2_buf[i] = i + 1u;
  mempool_barrier(ncores);

  arm(0u, core, ncores, group); // poll
  arm(1u, core, ncores, group); // sleep

  if (core == 0u)
    printf("[dma_event] poll_cyc=%u poll_instr=%u sleep_cyc=%u sleep_instr=%u "
           "errs=%u,%u\n",
           (unsigned)cyc[0], (unsigned)instr[0], (unsigned)cyc[1],
           (unsigned)instr[1], (unsigned)errs[0], (unsigned)errs[1]);
  mempool_barrier(ncores);
  return (int)(errs[0] + errs[1]);
}
