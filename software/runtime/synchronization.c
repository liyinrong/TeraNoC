// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "async_task.h"
#include "dma.h"
#include "runtime.h"
#include "synchronization.h"

#define SYNC_SLOT_COUNT (NUM_CORES * 8u)
#define SYNC_SLOT_GLOBAL 0u
#define SYNC_SLOT_LOG 1u
#define SYNC_SLOT_ANYRADIX 2u
#define SYNC_SLOT_LINLOG 3u
#define SYNC_SLOT_STRIDED_BASE NUM_CORES
#define SYNC_SLOT_STRIDED_LOG_BASE (2u * NUM_CORES)
#define SYNC_SLOT_LOG_PARTIAL_BASE (3u * NUM_CORES)
#define SYNC_SLOT_PARTIAL_BASE (4u * NUM_CORES)

typedef struct {
  volatile uint32_t count;
  volatile uint32_t generation;
  uint32_t padding[2];
} mempool_sync_slot_t;

static mempool_sync_slot_t sync_slot[SYNC_SLOT_COUNT]
    __attribute__((aligned(NUM_CORES * 4), section(".l1")));

static inline void sync_wait(mempool_sync_slot_t *slot, uint32_t participants) {
  if (participants <= 1u) {
    __atomic_thread_fence(__ATOMIC_ACQ_REL);
    return;
  }

  const uint32_t expected =
      __atomic_load_n(&slot->generation, __ATOMIC_ACQUIRE);
  const uint32_t arrived =
      __atomic_fetch_add(&slot->count, 1u, __ATOMIC_ACQ_REL);
  if (arrived == participants - 1u) {
    __atomic_store_n(&slot->count, 0u, __ATOMIC_RELAXED);
    __atomic_store_n(&slot->generation, expected + 1u, __ATOMIC_RELEASE);
    return;
  }

  while (__atomic_load_n(&slot->generation, __ATOMIC_ACQUIRE) == expected) {
    mempool_wait(4u);
  }
}

static inline uint32_t sync_slot_index(uint32_t index) {
  return index % SYNC_SLOT_COUNT;
}

void mempool_barrier_init(uint32_t core_id) {
  // Bootstrapping is the only sleeping phase: one bounded wake source cannot
  // overflow Snitch's pending-wake counter. All steady-state barriers poll a
  // generation predicate below.
  if (core_id == 0u) {
    sync_slot[0].count = 0u;
    sync_slot[0].generation = 0u;
    __sync_synchronize();
    wake_up_all();
    mempool_wfi();
  } else {
    mempool_wfi();
  }

  for (uint32_t i = core_id; i < SYNC_SLOT_COUNT; i += NUM_CORES) {
    sync_slot[i].count = 0u;
    sync_slot[i].generation = 0u;
    sync_slot[i].padding[0] = 0u;
    sync_slot[i].padding[1] = 0u;
  }
  dma_collective_all_runtime_init();
  tnc_engine_runtime_init();
  sync_wait(&sync_slot[SYNC_SLOT_GLOBAL], NUM_CORES);
}

void mempool_barrier(uint32_t num_cores) {
  sync_wait(&sync_slot[SYNC_SLOT_GLOBAL], num_cores);
}

void mempool_strided_barrier(uint32_t *barrier, uint32_t num_cores,
                             uint32_t stride, uint32_t offset) {
  (void)barrier;
  (void)stride;
  sync_wait(&sync_slot[sync_slot_index(SYNC_SLOT_STRIDED_BASE + offset)],
            num_cores);
}

void mempool_log_barrier(uint32_t step, uint32_t core_id) {
  (void)step;
  (void)core_id;
  sync_wait(&sync_slot[SYNC_SLOT_LOG], mempool_get_core_count());
}

void mempool_anyradixlog_barrier(uint32_t radix, uint32_t core_id) {
  (void)radix;
  (void)core_id;
  sync_wait(&sync_slot[SYNC_SLOT_ANYRADIX], mempool_get_core_count());
}

void mempool_linlog_barrier(uint32_t step, uint32_t core_id) {
  (void)step;
  (void)core_id;
  sync_wait(&sync_slot[SYNC_SLOT_LINLOG], mempool_get_core_count());
}

void mempool_strided_log_barrier(uint32_t step, uint32_t core_id,
                                 uint32_t stride, uint32_t offset) {
  (void)step;
  (void)core_id;
  const uint32_t participants =
      (mempool_get_core_count() - offset + stride - 1u) / stride;
  sync_wait(&sync_slot[sync_slot_index(SYNC_SLOT_STRIDED_LOG_BASE + offset)],
            participants);
}

void mempool_log_partial_barrier(uint32_t step, uint32_t core_id,
                                 uint32_t num_cores_barrier) {
  (void)step;
  const uint32_t first = num_cores_barrier * (core_id / num_cores_barrier);
  sync_wait(&sync_slot[sync_slot_index(SYNC_SLOT_LOG_PARTIAL_BASE + first)],
            num_cores_barrier);
}

void mempool_partial_barrier(uint32_t volatile core_id,
                             uint32_t volatile core_init,
                             uint32_t volatile num_sleeping_cores,
                             uint32_t volatile memloc) {
  if (core_id < core_init || core_id >= core_init + num_sleeping_cores)
    return;
  const uint32_t index = SYNC_SLOT_PARTIAL_BASE +
                         (core_init * 4u + memloc) % (4u * (uint32_t)NUM_CORES);
  sync_wait(&sync_slot[index], num_sleeping_cores);
}
