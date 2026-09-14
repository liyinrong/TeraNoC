// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Directed gate for the fixed-coordinator DMA collective (dma_collective.h).
//
//   queued  SMOKE_OPERATIONS L2->L1 transfers issued back to back, completed by
//           ONE barrier-coupled wait on the last. Each starts inside a group
//           stripe and crosses several bank rows, so every group splits it into
//           first/middle/last descriptors.
//   ready   the same data copied back L1->L2, completed by the barrier-free
//           wait_ready. One core is held back before its wait; every other core
//           must leave its own wait before that core even arrives, which a
//           barrier-coupled wait cannot do.
//   scoped  one L2->L1 transfer per VC rectangle at every level, from a single
//           group to the whole cluster, all rectangles of a level at once.
//
// -DDMA_COLLECTIVE_SMOKE_NO_LAUNCH submits empty transfers: every wait must
// still return, and every data check must fail.

#include <stdint.h>

#include "dma.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define SMOKE_OPERATIONS 3u
#define SMOKE_ROW_BYTES (NUM_BANKS * sizeof(uint32_t))
#define SMOKE_GROUP_BYTES (SMOKE_ROW_BYTES / NUM_GROUPS)
#define SMOKE_OFFSET_BYTES (SMOKE_GROUP_BYTES / 2u)
#define SMOKE_BYTES (3u * SMOKE_ROW_BYTES + SMOKE_GROUP_BYTES / 2u)
#define SMOKE_OFFSET_WORDS (SMOKE_OFFSET_BYTES / sizeof(uint32_t))
#define SMOKE_WORDS (SMOKE_BYTES / sizeof(uint32_t))
#define SMOKE_L1_STRIDE_WORDS                                                  \
  (((SMOKE_OFFSET_WORDS + SMOKE_WORDS + NUM_BANKS - 1u) / NUM_BANKS) *         \
   NUM_BANKS)
#define SMOKE_L2_PERIOD_BYTES                                                  \
  ((AXI_DATA_WIDTH / 8u) * AXI_WIDTH_INTERLEAVED * L2_BANKS)
#define SMOKE_L2_STRIDE_WORDS                                                  \
  (((SMOKE_BYTES + SMOKE_L2_PERIOD_BYTES - 1u) / SMOKE_L2_PERIOD_BYTES) *      \
   (SMOKE_L2_PERIOD_BYTES / sizeof(uint32_t)))
#define SMOKE_LEVELS ((uint32_t)VC_LEVELS + 1u)
// Long enough for the ready-phase transfers to land many times over.
#define SMOKE_HOLD_CYCLES 20000u

#ifdef DMA_COLLECTIVE_SMOKE_NO_LAUNCH
#define SMOKE_XFER(n) 0u
#define SMOKE_STATUS DMA_COLLECTIVE_ERROR_ARGUMENT
#else
#define SMOKE_XFER(n) (n)
#define SMOKE_STATUS DMA_COLLECTIVE_OK
#endif

static uint32_t l1_data[SMOKE_OPERATIONS][SMOKE_L1_STRIDE_WORDS]
    __attribute__((aligned(SMOKE_ROW_BYTES), section(".l1_prio")));
static uint32_t l2_src[SMOKE_OPERATIONS][SMOKE_L2_STRIDE_WORDS]
    __attribute__((aligned(SMOKE_L2_PERIOD_BYTES), section(".l2_bss")));
static uint32_t l2_dst[SMOKE_OPERATIONS][SMOKE_L2_STRIDE_WORDS]
    __attribute__((aligned(SMOKE_L2_PERIOD_BYTES), section(".l2_bss")));
// One bank row per level, so every rectangle's range lies in one row.
static uint32_t l1_scoped[SMOKE_LEVELS][NUM_BANKS]
    __attribute__((aligned(SMOKE_ROW_BYTES), section(".l1_prio")));
static uint32_t l2_scoped[SMOKE_LEVELS][NUM_BANKS]
    __attribute__((aligned(SMOKE_L2_PERIOD_BYTES), section(".l2_bss")));
static uint32_t exit_cycle[NUM_CORES] __attribute__((section(".l1")));
static uint32_t hold_arrival __attribute__((section(".l1")));
static uint32_t data_errors[3] __attribute__((section(".l1")));
static uint32_t order_errors __attribute__((section(".l1")));
static uint32_t wait_errors __attribute__((section(".l1")));
static uint32_t status_errors __attribute__((section(".l1")));

static uint32_t expected(uint32_t operation, uint32_t word) {
  return 0x6c000000u ^ (operation << 20) ^ (word * 0x1021u);
}

static uint32_t expected_scoped(uint32_t level, uint32_t word) {
  return 0x5c000000u ^ (level << 20) ^ (word * 0x0413u);
}

static void check_status(const dma_collective_op_t *op) {
  if (op->status != SMOKE_STATUS)
    __atomic_fetch_add(&status_errors, 1u, __ATOMIC_RELAXED);
}

static void check_wait(uint32_t error) {
  if (error != DMA_COLLECTIVE_OK)
    __atomic_fetch_add(&wait_errors, 1u, __ATOMIC_RELAXED);
}

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  // The ready phase holds back the last core. It must not be a coordinator:
  // coordinators publish READY, so holding one back would hold everyone.
  const uint32_t hold_core = (uint32_t)NUM_CORES - 1u;
  const uint32_t can_hold = (uint32_t)NUM_CORES_PER_GROUP > 1u;

  mempool_barrier_init(core);
  for (uint32_t i = core; i < SMOKE_OPERATIONS * SMOKE_WORDS; i += num_cores) {
    const uint32_t operation = i / SMOKE_WORDS;
    const uint32_t word = i % SMOKE_WORDS;
    l2_src[operation][word] = expected(operation, word);
    l2_dst[operation][word] = 0xcafe0000u ^ i;
  }
  for (uint32_t i = core; i < SMOKE_OPERATIONS * SMOKE_L1_STRIDE_WORDS;
       i += num_cores)
    l1_data[i / SMOKE_L1_STRIDE_WORDS][i % SMOKE_L1_STRIDE_WORDS] =
        0xdead0000u ^ i;
  for (uint32_t i = core; i < SMOKE_LEVELS * NUM_BANKS; i += num_cores) {
    l2_scoped[i / NUM_BANKS][i % NUM_BANKS] =
        expected_scoped(i / NUM_BANKS, i % NUM_BANKS);
    l1_scoped[i / NUM_BANKS][i % NUM_BANKS] = 0xbeef0000u ^ i;
  }
  if (core == 0u) {
    for (uint32_t c = 0u; c < (uint32_t)NUM_CORES; ++c)
      exit_cycle[c] = 0u;
    hold_arrival = 0u;
    data_errors[0] = 0u;
    data_errors[1] = 0u;
    data_errors[2] = 0u;
    order_errors = 0u;
    wait_errors = 0u;
    status_errors = 0u;
  }
  mempool_barrier(num_cores);

  // ---- queued: L2 -> L1, one barrier-coupled wait ----
  dma_collective_op_t op[SMOKE_OPERATIONS];
  for (uint32_t o = 0u; o < SMOKE_OPERATIONS; ++o) {
    op[o] = dma_collective_memcpy_nonblocking(
        &l1_data[o][SMOKE_OFFSET_WORDS], l2_src[o], SMOKE_XFER(SMOKE_BYTES));
    check_status(&op[o]);
  }
  check_wait(dma_collective_wait(&op[SMOKE_OPERATIONS - 1u]));
  {
    uint32_t errors = 0u;
    for (uint32_t i = core; i < SMOKE_OPERATIONS * SMOKE_WORDS;
         i += num_cores) {
      const uint32_t operation = i / SMOKE_WORDS;
      const uint32_t word = i % SMOKE_WORDS;
      if (l1_data[operation][SMOKE_OFFSET_WORDS + word] !=
          expected(operation, word))
        ++errors;
    }
    if (errors != 0u)
      __atomic_fetch_add(&data_errors[0], errors, __ATOMIC_RELAXED);
  }
  mempool_barrier(num_cores);

  // ---- ready: L1 -> L2, barrier-free ----
  for (uint32_t o = 0u; o < SMOKE_OPERATIONS; ++o) {
    op[o] = dma_collective_memcpy_nonblocking(
        l2_dst[o], &l1_data[o][SMOKE_OFFSET_WORDS], SMOKE_XFER(SMOKE_BYTES));
    check_status(&op[o]);
  }
  if (can_hold && core == hold_core) {
    mempool_wait(SMOKE_HOLD_CYCLES);
    hold_arrival = (uint32_t)mempool_get_timer();
  }
  check_wait(dma_collective_wait_ready(&op[SMOKE_OPERATIONS - 1u]));
  exit_cycle[core] = (uint32_t)mempool_get_timer();
  mempool_barrier(num_cores);
  if (core == 0u && can_hold)
    for (uint32_t c = 0u; c < (uint32_t)NUM_CORES; ++c)
      if (c != hold_core && exit_cycle[c] >= hold_arrival)
        ++order_errors;
  {
    uint32_t errors = 0u;
    for (uint32_t i = core; i < SMOKE_OPERATIONS * SMOKE_WORDS;
         i += num_cores) {
      const uint32_t operation = i / SMOKE_WORDS;
      const uint32_t word = i % SMOKE_WORDS;
      if (l2_dst[operation][word] != expected(operation, word))
        ++errors;
    }
    if (errors != 0u)
      __atomic_fetch_add(&data_errors[1], errors, __ATOMIC_RELAXED);
  }
  mempool_barrier(num_cores);

  // ---- scoped: every VC level, every rectangle of a level at once ----
  for (uint32_t k = 0u; k < SMOKE_LEVELS; ++k) {
    const dma_collective_scope_t scope = dma_collective_scope_level(k);
    const uint32_t base = scope.base_group * (uint32_t)SMOKE_GROUP_BYTES;
    const dma_collective_op_t sop = dma_collective_submit_nonblocking(
        &scope, 0u, (void *)((uint32_t)l1_scoped[k] + base),
        (const void *)((uint32_t)l2_scoped[k] + base),
        SMOKE_XFER(scope.num_groups * (uint32_t)SMOKE_GROUP_BYTES));
    check_status(&sop);
    check_wait(dma_collective_wait(&sop));
  }
  mempool_barrier(num_cores);
  {
    uint32_t errors = 0u;
    for (uint32_t i = core; i < SMOKE_LEVELS * NUM_BANKS; i += num_cores)
      if (l1_scoped[i / NUM_BANKS][i % NUM_BANKS] !=
          expected_scoped(i / NUM_BANKS, i % NUM_BANKS))
        ++errors;
    if (errors != 0u)
      __atomic_fetch_add(&data_errors[2], errors, __ATOMIC_RELAXED);
  }
  mempool_barrier(num_cores);

  if (core == 0u)
    printf("[dma_collective_smoke] groups=%u levels=%u operations=%u "
           "bytes=%u data=%u,%u,%u order=%u wait=%u status=%u\n",
           (unsigned)NUM_GROUPS, (unsigned)SMOKE_LEVELS,
           (unsigned)SMOKE_OPERATIONS, (unsigned)SMOKE_BYTES,
           (unsigned)data_errors[0], (unsigned)data_errors[1],
           (unsigned)data_errors[2], (unsigned)order_errors,
           (unsigned)wait_errors, (unsigned)status_errors);
  mempool_barrier(num_cores);
  return (int)(data_errors[0] + data_errors[1] + data_errors[2] + order_errors +
               wait_errors + status_errors);
}
