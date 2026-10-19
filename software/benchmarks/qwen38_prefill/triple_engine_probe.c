// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Directed protocol gate: the group coordinator runs iDMA, the other tile
// agents run RedMulE, and independent cores run Spatz. There is no barrier
// between launching these three activities. This is not a layer benchmark.
#include <stdint.h>

#include "archi_redmule.h"
#include "async_task.h"
#include "baremetal/mempool_qwen38_vector.h"
#include "dma.h"
#include "hal_redmule.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define PROBE_M REDMULE_H
#define PROBE_N (REDMULE_H * (REDMULE_P + 1))
#define PROBE_K 512u
#define PROBE_DMA_WORDS (NUM_BANKS * 512u)
#define PROBE_VECTOR_WORDS 64u
#define PROBE_VECTOR_REPS 48u
#define HALF_ONE 0x3c00u
#define HALF_K 0x6000u

static __fp16 matrix_x[NUM_REDMULE_TILES][PROBE_M * PROBE_K]
    __attribute__((section(".l1"), aligned(4 * NUM_BANKS)));
static __fp16 matrix_w[NUM_REDMULE_TILES][PROBE_K * PROBE_N]
    __attribute__((section(".l1"), aligned(4 * NUM_BANKS)));
static __fp16 matrix_z[NUM_REDMULE_TILES][PROBE_M * PROBE_N]
    __attribute__((section(".l1"), aligned(4 * NUM_BANKS)));
static __fp16 vector_in[NUM_CORES][PROBE_VECTOR_WORDS]
    __attribute__((section(".l1"), aligned(4 * NUM_BANKS)));
static __fp16 vector_out[NUM_CORES][PROBE_VECTOR_WORDS]
    __attribute__((section(".l1"), aligned(4 * NUM_BANKS)));
static __fp16 vector_gamma[PROBE_VECTOR_WORDS]
    __attribute__((section(".l1"), aligned(4 * NUM_BANKS)));
static uint32_t dma_src[PROBE_DMA_WORDS]
    __attribute__((section(".l2_bss"), aligned(4 * NUM_BANKS)));
static uint32_t dma_dst[PROBE_DMA_WORDS]
    __attribute__((section(".l1"), aligned(4 * NUM_BANKS)));
static uint32_t dma_start[NUM_GROUPS] __attribute__((section(".l1")));
static uint32_t dma_end[NUM_GROUPS] __attribute__((section(".l1")));
static uint32_t rm_start[NUM_REDMULE_TILES] __attribute__((section(".l1")));
static uint32_t rm_end[NUM_REDMULE_TILES] __attribute__((section(".l1")));
static uint32_t spatz_start[NUM_CORES] __attribute__((section(".l1")));
static uint32_t spatz_end[NUM_CORES] __attribute__((section(".l1")));
static uint32_t failures[NUM_CORES] __attribute__((section(".l1")));

static uint32_t pattern(uint32_t index) {
  return 0xa3800000u ^ (index * 0x1021u);
}

static __attribute__((noinline)) void vector_work(uint32_t core) {
  for (uint32_t rep = 0u; rep < PROBE_VECTOR_REPS; ++rep)
    q38_norm16(vector_in[core], vector_gamma, vector_out[core],
               PROBE_VECTOR_WORDS, 0);
}

int main(void) {
  const uint32_t core = mempool_get_core_id();
  const uint32_t group = mempool_get_group_id();
  uint32_t error = 0u;
  mempool_barrier_init(core);
  for (uint32_t i = core; i < NUM_REDMULE_TILES * PROBE_M * PROBE_K;
       i += NUM_CORES)
    ((uint16_t *)matrix_x)[i] = HALF_ONE;
  for (uint32_t i = core; i < NUM_REDMULE_TILES * PROBE_K * PROBE_N;
       i += NUM_CORES)
    ((uint16_t *)matrix_w)[i] = HALF_ONE;
  for (uint32_t i = core; i < NUM_REDMULE_TILES * PROBE_M * PROBE_N;
       i += NUM_CORES)
    ((uint16_t *)matrix_z)[i] = 0u;
  for (uint32_t i = core; i < NUM_CORES * PROBE_VECTOR_WORDS; i += NUM_CORES) {
    ((uint16_t *)vector_in)[i] = HALF_ONE;
    ((uint16_t *)vector_out)[i] = 0u;
  }
  for (uint32_t i = core; i < PROBE_VECTOR_WORDS; i += NUM_CORES)
    ((uint16_t *)vector_gamma)[i] = HALF_ONE;
  for (uint32_t i = core; i < PROBE_DMA_WORDS; i += NUM_CORES) {
    dma_src[i] = pattern(i);
    dma_dst[i] = ~pattern(i);
  }
  failures[core] = 0u;
  mempool_barrier(NUM_CORES);

  if (mempool_is_group_coordinator()) {
    dma_start[group] = mempool_get_timer();
    const dma_coordinator_op_t op =
        dma_coordinator_submit_linear(0u, dma_dst, dma_src, sizeof(dma_src));
    if (op.status != DMA_COLLECTIVE_OK || op.id == 0u)
      ++error;
    else
      error += dma_local_wait_id(0u, op.id) != DMA_LOCAL_OK;
    dma_end[group] = mempool_get_timer();
  } else {
    if (tnc_is_tile_agent()) {
      const uint32_t engine = mempool_get_redmule_id();
      const int32_t target = tnc_redmule_acquire_target();
      redmule_cfg((uint32_t)matrix_x[engine], (uint32_t)matrix_w[engine],
                  (uint32_t)matrix_z[engine], PROBE_M, PROBE_K, PROBE_N, 0,
                  GEMM, Float16);
      rm_start[engine] = mempool_get_timer();
      tnc_redmule_trigger();
      spatz_start[core] = mempool_get_timer();
      vector_work(core);
      spatz_end[core] = mempool_get_timer();
      tnc_redmule_wait((uint8_t)target);
      rm_end[engine] = mempool_get_timer();
    } else {
      spatz_start[core] = mempool_get_timer();
      vector_work(core);
      spatz_end[core] = mempool_get_timer();
    }
  }
  mempool_barrier(NUM_CORES);

  for (uint32_t i = core; i < PROBE_DMA_WORDS; i += NUM_CORES)
    error += dma_dst[i] != pattern(i);
  for (uint32_t i = core; i < NUM_REDMULE_TILES * PROBE_M * PROBE_N;
       i += NUM_CORES) {
    const uint32_t engine = i / (PROBE_M * PROBE_N);
    if (engine % NUM_TILES_PER_GROUP != 0u)
      error += ((uint16_t *)matrix_z)[i] != HALF_K;
  }
  if (!mempool_is_group_coordinator())
    error += ((uint16_t *)vector_out[core])[0] == 0u;
  failures[core] = error;
  mempool_barrier(NUM_CORES);

  if (core == 0u) {
    uint32_t total = 0u;
    for (uint32_t i = 0u; i < NUM_CORES; ++i)
      total += failures[i];
    const uint32_t overlap_start =
        dma_start[0] > rm_start[1] ? dma_start[0] : rm_start[1];
    const uint32_t overlap_end =
        dma_end[0] < rm_end[1] ? dma_end[0] : rm_end[1];
    const uint32_t triple_start =
        overlap_start > spatz_start[3] ? overlap_start : spatz_start[3];
    const uint32_t triple_end =
        overlap_end < spatz_end[3] ? overlap_end : spatz_end[3];
    printf("[qwen38_triple_probe] bytes=%u dma=%u:%u rm=%u:%u "
           "spatz=%u:%u overlap=%u errors=%u\n",
           (unsigned)sizeof(dma_src), (unsigned)dma_start[0],
           (unsigned)dma_end[0], (unsigned)rm_start[1], (unsigned)rm_end[1],
           (unsigned)spatz_start[3], (unsigned)spatz_end[3],
           (unsigned)(triple_end > triple_start), (unsigned)total);
    failures[0] = total;
  }
  mempool_barrier(NUM_CORES);
  return (int)failures[0];
}
