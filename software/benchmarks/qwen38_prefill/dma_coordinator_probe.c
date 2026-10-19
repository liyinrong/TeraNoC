// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// A group coordinator issues global L1 <-> L2 copies while other cores make
// progress without entering the DMA submit/wait path. The final barrier is a
// numerical test boundary, not part of the stage-worker protocol.
#include <stdint.h>

#include "dma.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define PROBE_WORDS (NUM_BANKS * 8u)
#define PROBE_ALIGN (NUM_BANKS * sizeof(uint32_t))

static uint32_t source[PROBE_WORDS]
    __attribute__((section(".l2_bss"), aligned(PROBE_ALIGN)));
static uint32_t destination[PROBE_WORDS]
    __attribute__((section(".l1"), aligned(PROBE_ALIGN)));
static uint32_t returned[PROBE_WORDS]
    __attribute__((section(".l2_bss"), aligned(PROBE_ALIGN)));
static uint32_t ready_in[NUM_GROUPS] __attribute__((section(".l1")));
static uint32_t ready_out[NUM_GROUPS] __attribute__((section(".l1")));
static uint32_t failures[NUM_CORES] __attribute__((section(".l1")));
static uint32_t worker_progress[NUM_CORES] __attribute__((section(".l1")));

static uint32_t pattern(uint32_t index) {
  return 0x38c00000u ^ (index * 0x1021u);
}

static uint32_t coordinator_copy(void *dest, const void *src) {
  const dma_coordinator_op_t op =
      dma_coordinator_submit_linear(0u, dest, src, sizeof(source));
  if (op.status != DMA_COLLECTIVE_OK || op.id == 0u || op.descriptors == 0u)
    return 1u;
  return dma_local_wait_id(0u, op.id) != DMA_LOCAL_OK;
}

int main(void) {
  const uint32_t core = mempool_get_core_id();
  const uint32_t group = mempool_get_group_id();
  uint32_t error = 0u;
  mempool_barrier_init(core);
  for (uint32_t i = core; i < PROBE_WORDS; i += NUM_CORES) {
    source[i] = pattern(i);
    destination[i] = ~pattern(i);
    returned[i] = 0u;
  }
  if (core < NUM_GROUPS) {
    ready_in[core] = 0u;
    ready_out[core] = 0u;
  }
  worker_progress[core] = 0u;
  failures[core] = 0u;
  mempool_barrier(NUM_CORES);

  if (mempool_is_group_coordinator()) {
    error += coordinator_copy(destination, source);
    __atomic_store_n(&ready_in[group], 1u, __ATOMIC_RELEASE);
  } else {
    uint32_t progress = core;
    for (uint32_t i = 0u; i < 128u; ++i)
      progress = progress * 1664525u + 1013904223u;
    worker_progress[core] = progress;
    while (!__atomic_load_n(&ready_in[group], __ATOMIC_ACQUIRE))
      mempool_wait(4u);
  }
  mempool_barrier(NUM_CORES);
  for (uint32_t i = core; i < PROBE_WORDS; i += NUM_CORES)
    error += destination[i] != pattern(i);
  mempool_barrier(NUM_CORES);

  if (mempool_is_group_coordinator()) {
    error += coordinator_copy(returned, destination);
    __atomic_store_n(&ready_out[group], 1u, __ATOMIC_RELEASE);
  } else {
    while (!__atomic_load_n(&ready_out[group], __ATOMIC_ACQUIRE))
      mempool_wait(4u);
    error += worker_progress[core] == 0u;
  }
  mempool_barrier(NUM_CORES);
  for (uint32_t i = core; i < PROBE_WORDS; i += NUM_CORES)
    error += returned[i] != pattern(i);
  failures[core] = error;
  mempool_barrier(NUM_CORES);

  if (core == 0u) {
    uint32_t total = 0u;
    for (uint32_t i = 0u; i < NUM_CORES; ++i)
      total += failures[i];
    printf("[qwen38_dma_coordinator] bytes=%u groups=%u errors=%u\n",
           (unsigned)sizeof(source), (unsigned)NUM_GROUPS, (unsigned)total);
    failures[0] = total;
  }
  mempool_barrier(NUM_CORES);
  return (int)failures[0];
}
