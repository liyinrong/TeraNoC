// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Directed gate for the SPMD DMA collective. The transfer begins inside a
// group stripe and crosses several complete L1 bank rows, forcing the software
// splitter to emit first/middle/last descriptors. Deliberate core skew chooses
// a different dynamic issuer in each generation.

#include <stdint.h>

#include "dma_collective.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define COLLECTIVE_ROW_BYTES (NUM_BANKS * sizeof(uint32_t))
#define COLLECTIVE_GROUP_BYTES (COLLECTIVE_ROW_BYTES / NUM_GROUPS)
#define COLLECTIVE_OPERATIONS 2u
#define COLLECTIVE_OFFSET_BYTES (COLLECTIVE_GROUP_BYTES / 2u)
#define COLLECTIVE_BYTES                                                       \
  (3u * COLLECTIVE_ROW_BYTES + COLLECTIVE_GROUP_BYTES / 2u)
#define COLLECTIVE_OFFSET_WORDS (COLLECTIVE_OFFSET_BYTES / sizeof(uint32_t))
#define COLLECTIVE_WORDS (COLLECTIVE_BYTES / sizeof(uint32_t))
#define COLLECTIVE_L1_WORDS (COLLECTIVE_OFFSET_WORDS + COLLECTIVE_WORDS)
#define COLLECTIVE_L1_STRIDE_WORDS                                             \
  (((COLLECTIVE_L1_WORDS + NUM_BANKS - 1u) / NUM_BANKS) * NUM_BANKS)
#define COLLECTIVE_L2_PERIOD_BYTES                                             \
  ((AXI_DATA_WIDTH / 8u) * AXI_WIDTH_INTERLEAVED * L2_BANKS)
#define COLLECTIVE_L2_STRIDE_WORDS                                             \
  (((COLLECTIVE_BYTES + COLLECTIVE_L2_PERIOD_BYTES - 1u) /                     \
    COLLECTIVE_L2_PERIOD_BYTES) *                                              \
   (COLLECTIVE_L2_PERIOD_BYTES / sizeof(uint32_t)))

static uint32_t l1_data[COLLECTIVE_OPERATIONS][COLLECTIVE_L1_STRIDE_WORDS]
    __attribute__((aligned(COLLECTIVE_ROW_BYTES), section(".l1_prio")));
static uint32_t l2_src[COLLECTIVE_OPERATIONS][COLLECTIVE_L2_STRIDE_WORDS]
    __attribute__((aligned(COLLECTIVE_L2_PERIOD_BYTES), section(".l2_bss")));
static uint32_t l2_dst[COLLECTIVE_OPERATIONS][COLLECTIVE_L2_STRIDE_WORDS]
    __attribute__((aligned(COLLECTIVE_L2_PERIOD_BYTES), section(".l2_bss")));
static dma_collective_state_t dma_state DMA_COLLECTIVE_STORAGE;
static uint32_t total_errors __attribute__((section(".l1")));
static uint32_t data_errors[2] __attribute__((section(".l1")));
static uint32_t issuer_errors __attribute__((section(".l1")));

static uint32_t expected(uint32_t operation, uint32_t word) {
  return 0x6c000000u ^ (operation << 20) ^ (word * 0x1021u);
}

static void print_bad_ranges(const char *label, uint32_t operation,
                             const uint32_t *data) {
  uint32_t ranges = 0;
  uint32_t word = 0;
  while (word < COLLECTIVE_WORDS && ranges < 12) {
    while (word < COLLECTIVE_WORDS && data[word] == expected(operation, word))
      ++word;
    if (word == COLLECTIVE_WORDS)
      break;
    const uint32_t first = word;
    while (word < COLLECTIVE_WORDS && data[word] != expected(operation, word))
      ++word;
    printf("[dma_collective_smoke] %s[%u] bad=%u..%u got=0x%x want=0x%x\n",
           label, (unsigned)operation, (unsigned)first, (unsigned)(word - 1u),
           (unsigned)data[first], (unsigned)expected(operation, first));
    ++ranges;
  }
}

#ifndef DMA_COLLECTIVE_SMOKE_NO_LAUNCH
static uint32_t issuer_target(uint32_t group, uint32_t generation) {
  return (group + 2u * generation - 1u) % (uint32_t)NUM_CORES_PER_GROUP;
}

static uint32_t poller_target(uint32_t group, uint32_t generation) {
  return (group + 2u * generation) % (uint32_t)NUM_CORES_PER_GROUP;
}

static void skew_to_issuer(uint32_t generation) {
  const uint32_t group = mempool_get_group_id();
  const uint32_t local_core =
      mempool_get_core_id() % (uint32_t)NUM_CORES_PER_GROUP;
  const uint32_t target = issuer_target(group, generation);
  if (local_core != target)
    mempool_wait(2000u + 8u * local_core);
}

static void skew_to_poller(uint32_t generation) {
  const uint32_t group = mempool_get_group_id();
  const uint32_t local_core =
      mempool_get_core_id() % (uint32_t)NUM_CORES_PER_GROUP;
  const uint32_t target = poller_target(group, generation);
  if (local_core != target)
    mempool_wait(2000u + 8u * local_core);
}

static void check_issuer(uint32_t generation) {
  const uint32_t core = mempool_get_core_id();
  const uint32_t group = mempool_get_group_id();
  if (core % (uint32_t)NUM_CORES_PER_GROUP != 0)
    return;

  const uint32_t target = issuer_target(group, generation);
  const uint32_t expected_core = group * (uint32_t)NUM_CORES_PER_GROUP + target;
  const uint32_t slot = (generation - 1u) % DMA_COLLECTIVE_SLOTS;
  const dma_collective_slot_state_t *const state =
      &dma_state.group[group].slot[slot];
  if (state->issuer_core != expected_core || state->descriptors == 0 ||
      state->descriptors > 6) {
    __atomic_fetch_add(&issuer_errors, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&total_errors, 1, __ATOMIC_RELAXED);
  }
}

static void check_poller(uint32_t generation) {
  const uint32_t core = mempool_get_core_id();
  const uint32_t group = mempool_get_group_id();
  if (core % (uint32_t)NUM_CORES_PER_GROUP != 0)
    return;

  const uint32_t target = poller_target(group, generation);
  const uint32_t expected_core = group * (uint32_t)NUM_CORES_PER_GROUP + target;
  const uint32_t slot = (generation - 1u) % DMA_COLLECTIVE_SLOTS;
  const dma_collective_slot_state_t *const state =
      &dma_state.group[group].slot[slot];
  if (state->poller_core != expected_core) {
    __atomic_fetch_add(&issuer_errors, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&total_errors, 1, __ATOMIC_RELAXED);
  }
}
#endif

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  const dma_collective_scope_t scope = dma_collective_scope_all();

  mempool_barrier_init(core);
  for (uint32_t i = core; i < COLLECTIVE_OPERATIONS * COLLECTIVE_WORDS;
       i += num_cores) {
    const uint32_t operation = i / COLLECTIVE_WORDS;
    const uint32_t word = i % COLLECTIVE_WORDS;
    l2_src[operation][word] = expected(operation, word);
    l2_dst[operation][word] = 0xcafe0000u ^ i;
  }
  for (uint32_t i = core;
       i < COLLECTIVE_OPERATIONS * COLLECTIVE_L1_STRIDE_WORDS; i += num_cores) {
    const uint32_t operation = i / COLLECTIVE_L1_STRIDE_WORDS;
    const uint32_t word = i % COLLECTIVE_L1_STRIDE_WORDS;
    l1_data[operation][word] = 0xdead0000u ^ i;
  }
  if (core == 0) {
    total_errors = 0;
    data_errors[0] = 0;
    data_errors[1] = 0;
    issuer_errors = 0;
  }
  dma_collective_init(&dma_state, &scope);
  mempool_barrier(num_cores);

#ifndef DMA_COLLECTIVE_SMOKE_NO_LAUNCH
  dma_collective_handle_t handle[COLLECTIVE_OPERATIONS];
  for (uint32_t operation = 0; operation < COLLECTIVE_OPERATIONS; ++operation) {
    const uint32_t generation = operation + 1u;
    skew_to_issuer(generation);
    handle[operation] = dma_collective_memcpy_nonblocking(
        &dma_state, &scope, generation, 0,
        &l1_data[operation][COLLECTIVE_OFFSET_WORDS], l2_src[operation],
        COLLECTIVE_BYTES);
    check_issuer(generation);
  }
  const dma_collective_handle_t full_handle = dma_collective_memcpy_nonblocking(
      &dma_state, &scope, COLLECTIVE_OPERATIONS + 1u, 0,
      &l1_data[0][COLLECTIVE_OFFSET_WORDS], l2_src[0], COLLECTIVE_BYTES);
  if (full_handle.status != DMA_COLLECTIVE_ERROR_REUSE)
    __atomic_fetch_add(&total_errors, 1, __ATOMIC_RELAXED);
  skew_to_poller(COLLECTIVE_OPERATIONS);
  if (dma_collective_wait_through(&scope,
                                  &handle[COLLECTIVE_OPERATIONS - 1u]) != 0)
    __atomic_fetch_add(&total_errors, 1, __ATOMIC_RELAXED);
  check_poller(COLLECTIVE_OPERATIONS);
#endif
  mempool_barrier(num_cores);

  uint32_t errors = 0;
  for (uint32_t i = core; i < COLLECTIVE_OPERATIONS * COLLECTIVE_WORDS;
       i += num_cores) {
    const uint32_t operation = i / COLLECTIVE_WORDS;
    const uint32_t word = i % COLLECTIVE_WORDS;
    if (l1_data[operation][COLLECTIVE_OFFSET_WORDS + word] !=
        expected(operation, word))
      ++errors;
  }
  if (errors != 0)
    __atomic_fetch_add(&total_errors, errors, __ATOMIC_RELAXED);
  if (errors != 0)
    __atomic_fetch_add(&data_errors[0], errors, __ATOMIC_RELAXED);
  mempool_barrier(num_cores);
  if (core == 0)
    for (uint32_t operation = 0; operation < COLLECTIVE_OPERATIONS; ++operation)
      print_bad_ranges("l1", operation,
                       &l1_data[operation][COLLECTIVE_OFFSET_WORDS]);
  mempool_barrier(num_cores);

#ifndef DMA_COLLECTIVE_SMOKE_NO_LAUNCH
  for (uint32_t operation = 0; operation < COLLECTIVE_OPERATIONS; ++operation) {
    const uint32_t generation = COLLECTIVE_OPERATIONS + operation + 1u;
    skew_to_issuer(generation);
    handle[operation] = dma_collective_memcpy_nonblocking(
        &dma_state, &scope, generation, 0, l2_dst[operation],
        &l1_data[operation][COLLECTIVE_OFFSET_WORDS], COLLECTIVE_BYTES);
    check_issuer(generation);
  }
  const uint32_t last_generation = 2u * COLLECTIVE_OPERATIONS;
  skew_to_poller(last_generation);
  if (dma_collective_wait_through(&scope,
                                  &handle[COLLECTIVE_OPERATIONS - 1u]) != 0)
    __atomic_fetch_add(&total_errors, 1, __ATOMIC_RELAXED);
  check_poller(last_generation);
#endif
  mempool_barrier(num_cores);

  errors = 0;
  for (uint32_t i = core; i < COLLECTIVE_OPERATIONS * COLLECTIVE_WORDS;
       i += num_cores) {
    const uint32_t operation = i / COLLECTIVE_WORDS;
    const uint32_t word = i % COLLECTIVE_WORDS;
    if (l2_dst[operation][word] != expected(operation, word))
      ++errors;
  }
  if (errors != 0)
    __atomic_fetch_add(&total_errors, errors, __ATOMIC_RELAXED);
  if (errors != 0)
    __atomic_fetch_add(&data_errors[1], errors, __ATOMIC_RELAXED);
  mempool_barrier(num_cores);
  if (core == 0)
    for (uint32_t operation = 0; operation < COLLECTIVE_OPERATIONS; ++operation)
      print_bad_ranges("l2", operation, l2_dst[operation]);
  mempool_barrier(num_cores);

  if (core == 0) {
    printf("[dma_collective_smoke] groups=%u operations=%u bytes=%u "
           "data=%u,%u roles=%u errors=%u\n",
           (unsigned)NUM_GROUPS, (unsigned)COLLECTIVE_OPERATIONS,
           (unsigned)COLLECTIVE_BYTES, (unsigned)data_errors[0],
           (unsigned)data_errors[1], (unsigned)issuer_errors,
           (unsigned)total_errors);
#ifndef DMA_COLLECTIVE_SMOKE_NO_LAUNCH
    for (uint32_t group = 0; group < (uint32_t)NUM_GROUPS; ++group) {
      for (uint32_t slot = 0; slot < DMA_COLLECTIVE_SLOTS; ++slot) {
        const dma_collective_slot_state_t *const state =
            &dma_state.group[group].slot[slot];
        printf("[dma_collective_smoke] group=%u slot=%u generation=%u "
               "issuer=%u poller=%u descriptors=%u id=%u error=0x%x\n",
               (unsigned)group, (unsigned)slot, (unsigned)state->submit_epoch,
               (unsigned)state->issuer_core, (unsigned)state->poller_core,
               (unsigned)state->descriptors, (unsigned)state->local_id,
               (unsigned)state->error);
      }
    }
#endif
  }
  mempool_barrier(num_cores);
  return (int)total_errors;
}
