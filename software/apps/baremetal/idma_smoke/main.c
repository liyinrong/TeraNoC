// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Concurrent per-group iDMA v0.6.5 smoke test. Every group launches two ND
// transfers without waiting between them, and each group's leader sits in a
// DIFFERENT tile (group_leader below rotates it), so a group whose control port
// or backend were cross-wired to another group's engine cannot pass by
// symmetry -- the data is keyed on the group.
//
// The register context is one per group, not one per tile, so the rotation no
// longer says anything about per-tile banks; it still covers the tile-to-group
// control path, which is now an arbiter rather than a dedicated port.

#include <stdint.h>

#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define IDMA_SMOKE_ROWS 4u
#define IDMA_SMOKE_BUFFERS 2u
#ifndef IDMA_SMOKE_ACTIVE_BUFFERS
#define IDMA_SMOKE_ACTIVE_BUFFERS IDMA_SMOKE_BUFFERS
#endif
#ifndef IDMA_SMOKE_ACTIVE_GROUPS
#define IDMA_SMOKE_ACTIVE_GROUPS NUM_GROUPS
#endif
#define IDMA_GROUP_WORDS (NUM_BANKS / NUM_GROUPS)
#define IDMA_TRANSFER_WORDS ((IDMA_GROUP_WORDS < 16u) ? IDMA_GROUP_WORDS : 16u)
#define IDMA_TRANSFER_BYTES (IDMA_TRANSFER_WORDS * sizeof(uint32_t))
#define IDMA_GLOBAL_WORDS (IDMA_SMOKE_ROWS * NUM_BANKS)
#define IDMA_BUFFER_WORDS (IDMA_SMOKE_ROWS * IDMA_TRANSFER_WORDS)
#define IDMA_TOTAL_WORDS (IDMA_SMOKE_BUFFERS * IDMA_GLOBAL_WORDS)
#define IDMA_LOGICAL_WORDS (IDMA_SMOKE_BUFFERS * NUM_GROUPS * IDMA_BUFFER_WORDS)
#define IDMA_L2_BANK_STRIDE_BYTES                                              \
  ((AXI_DATA_WIDTH / 8u) * AXI_WIDTH_INTERLEAVED)
#define IDMA_L2_GROUP_WORDS (IDMA_L2_BANK_STRIDE_BYTES / 4u)
#define IDMA_L2_ALIGNMENT (IDMA_L2_BANK_STRIDE_BYTES * L2_BANKS)

#if IDMA_BUFFER_WORDS > IDMA_L2_GROUP_WORDS
#error "idma_smoke group payload exceeds one L2-bank stripe"
#endif

static uint32_t l1_data[IDMA_TOTAL_WORDS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static uint32_t l2_src[IDMA_SMOKE_BUFFERS][NUM_GROUPS][IDMA_L2_GROUP_WORDS]
    __attribute__((aligned(IDMA_L2_ALIGNMENT), section(".l2_bss")));
static uint32_t l2_dst[IDMA_SMOKE_BUFFERS][NUM_GROUPS][IDMA_L2_GROUP_WORDS]
    __attribute__((aligned(IDMA_L2_ALIGNMENT), section(".l2_bss")));
static uint32_t launch_id[NUM_GROUPS][IDMA_SMOKE_BUFFERS]
    __attribute__((section(".l1")));
static uint32_t total_errors __attribute__((section(".l1")));

static uint32_t expected(uint32_t buffer, uint32_t group, uint32_t row,
                         uint32_t word) {
  return 0x51000000u ^ (buffer << 24) ^ (group << 16) ^ (row << 8) ^ word;
}

static uint32_t group_leader(uint32_t group) {
  const uint32_t tile = group % (NUM_CORES_PER_GROUP / NUM_CORES_PER_TILE);
  return group * NUM_CORES_PER_GROUP + tile * NUM_CORES_PER_TILE;
}

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  const uint32_t group_id = mempool_get_group_id();

  mempool_barrier_init(core_id);

  for (uint32_t i = core_id; i < IDMA_LOGICAL_WORDS; i += num_cores) {
    const uint32_t word = i % IDMA_BUFFER_WORDS;
    const uint32_t group = (i / IDMA_BUFFER_WORDS) % NUM_GROUPS;
    const uint32_t buffer = i / (NUM_GROUPS * IDMA_BUFFER_WORDS);
    const uint32_t row = word / IDMA_TRANSFER_WORDS;
    const uint32_t in_row = word % IDMA_TRANSFER_WORDS;
    const uint32_t l1_index = buffer * IDMA_GLOBAL_WORDS + row * NUM_BANKS +
                              group * IDMA_GROUP_WORDS + in_row;
    l1_data[l1_index] = 0xdead0000u ^ i;
    l2_src[buffer][group][word] = expected(buffer, group, row, in_row);
    l2_dst[buffer][group][word] = 0xcafe0000u ^ i;
  }
  if (core_id == 0) {
    total_errors = 0;
    for (uint32_t g = 0; g < NUM_GROUPS; ++g)
      for (uint32_t b = 0; b < IDMA_SMOKE_BUFFERS; ++b)
        launch_id[g][b] = 0;
  }
  mempool_barrier(num_cores);

  if (group_id < IDMA_SMOKE_ACTIVE_GROUPS &&
      core_id == group_leader(group_id)) {
#ifndef IDMA_SMOKE_NO_LAUNCH
    uint32_t *const dst_2d = &l1_data[group_id * IDMA_GROUP_WORDS];
    uint32_t *const dst_3d =
        &l1_data[IDMA_GLOBAL_WORDS + group_id * IDMA_GROUP_WORDS];

    launch_id[group_id][0] = dma_local_memcpy_2d_nonblocking(
        0, dst_2d, l2_src[0][group_id], IDMA_TRANSFER_BYTES,
        NUM_BANKS * sizeof(uint32_t), IDMA_TRANSFER_BYTES, IDMA_SMOKE_ROWS);
#if IDMA_SMOKE_ACTIVE_BUFFERS > 1
    launch_id[group_id][1] = dma_local_memcpy_3d_nonblocking(
        0, dst_3d, l2_src[1][group_id], IDMA_TRANSFER_BYTES,
        NUM_BANKS * sizeof(uint32_t), IDMA_TRANSFER_BYTES, 2,
        NUM_BANKS * sizeof(uint32_t), IDMA_TRANSFER_BYTES, 2);
#endif
    if (dma_local_wait(0) != 0)
      __atomic_fetch_add(&total_errors, 1, __ATOMIC_RELAXED);
#endif
  }
  mempool_barrier(num_cores);

  if (core_id == 0)
    for (uint32_t g = 0; g < IDMA_SMOKE_ACTIVE_GROUPS; ++g)
      if (launch_id[g][0] == 0
#if IDMA_SMOKE_ACTIVE_BUFFERS > 1
          || launch_id[g][1] == 0 || launch_id[g][0] == launch_id[g][1]
#endif
      )
        ++total_errors;
  mempool_barrier(num_cores);

  uint32_t errors = 0;
  for (uint32_t i = core_id; i < IDMA_LOGICAL_WORDS; i += num_cores) {
    const uint32_t word = i % IDMA_BUFFER_WORDS;
    const uint32_t group = (i / IDMA_BUFFER_WORDS) % NUM_GROUPS;
    const uint32_t buffer = i / (NUM_GROUPS * IDMA_BUFFER_WORDS);
    const uint32_t row = word / IDMA_TRANSFER_WORDS;
    const uint32_t in_row = word % IDMA_TRANSFER_WORDS;
    const uint32_t l1_index = buffer * IDMA_GLOBAL_WORDS + row * NUM_BANKS +
                              group * IDMA_GROUP_WORDS + in_row;
    if (buffer < IDMA_SMOKE_ACTIVE_BUFFERS &&
        group < IDMA_SMOKE_ACTIVE_GROUPS &&
        l1_data[l1_index] != expected(buffer, group, row, in_row))
      ++errors;
  }
  if (errors != 0)
    __atomic_fetch_add(&total_errors, errors, __ATOMIC_RELAXED);
  mempool_barrier(num_cores);

  if (group_id < IDMA_SMOKE_ACTIVE_GROUPS &&
      core_id == group_leader(group_id)) {
#ifndef IDMA_SMOKE_NO_LAUNCH
    const dma_id_t id_2d = dma_local_memcpy_2d_nonblocking(
        0, l2_dst[0][group_id], &l1_data[group_id * IDMA_GROUP_WORDS],
        IDMA_TRANSFER_BYTES, IDMA_TRANSFER_BYTES, NUM_BANKS * sizeof(uint32_t),
        IDMA_SMOKE_ROWS);
#if IDMA_SMOKE_ACTIVE_BUFFERS > 1
    const dma_id_t id_3d = dma_local_memcpy_3d_nonblocking(
        0, l2_dst[1][group_id],
        &l1_data[IDMA_GLOBAL_WORDS + group_id * IDMA_GROUP_WORDS],
        IDMA_TRANSFER_BYTES, IDMA_TRANSFER_BYTES, NUM_BANKS * sizeof(uint32_t),
        2, IDMA_TRANSFER_BYTES, NUM_BANKS * sizeof(uint32_t), 2);
#endif
    launch_id[group_id][0] = id_2d;
#if IDMA_SMOKE_ACTIVE_BUFFERS > 1
    launch_id[group_id][1] = id_3d;
    if (dma_local_wait_id(0, id_3d) != 0)
      __atomic_fetch_add(&total_errors, 1, __ATOMIC_RELAXED);
#else
    if (dma_local_wait_id(0, id_2d) != 0)
      __atomic_fetch_add(&total_errors, 1, __ATOMIC_RELAXED);
#endif
#endif
  }
  mempool_barrier(num_cores);

  if (core_id == 0)
    for (uint32_t g = 0; g < IDMA_SMOKE_ACTIVE_GROUPS; ++g)
      if (launch_id[g][0] == 0
#if IDMA_SMOKE_ACTIVE_BUFFERS > 1
          || launch_id[g][1] == 0 || launch_id[g][0] == launch_id[g][1]
#endif
      )
        ++total_errors;
  mempool_barrier(num_cores);

  errors = 0;
  for (uint32_t i = core_id; i < IDMA_LOGICAL_WORDS; i += num_cores) {
    const uint32_t word = i % IDMA_BUFFER_WORDS;
    const uint32_t group = (i / IDMA_BUFFER_WORDS) % NUM_GROUPS;
    const uint32_t buffer = i / (NUM_GROUPS * IDMA_BUFFER_WORDS);
    const uint32_t row = word / IDMA_TRANSFER_WORDS;
    if (buffer < IDMA_SMOKE_ACTIVE_BUFFERS &&
        group < IDMA_SMOKE_ACTIVE_GROUPS &&
        l2_dst[buffer][group][word] !=
            expected(buffer, group, row, word % IDMA_TRANSFER_WORDS))
      ++errors;
  }
  if (errors != 0)
    __atomic_fetch_add(&total_errors, errors, __ATOMIC_RELAXED);
  mempool_barrier(num_cores);

  if (core_id == 0)
    printf("[idma_smoke] groups=%u active=%u buffers=%u streams/group=%u "
           "errors=%u\n",
           (unsigned)NUM_GROUPS, (unsigned)IDMA_SMOKE_ACTIVE_GROUPS,
           (unsigned)IDMA_SMOKE_ACTIVE_BUFFERS, (unsigned)NUM_DMAS_PER_GROUP,
           (unsigned)total_errors);
  mempool_barrier(num_cores);
  return (int)total_errors;
}
