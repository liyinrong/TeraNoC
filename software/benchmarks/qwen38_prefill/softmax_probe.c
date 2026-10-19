// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Numerical qualification component, not a complete-layer performance result.
#include "baremetal/mempool_qwen38_online_softmax.h"
#include "dma.h"
#include "mempool_checks.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "softmax_data.h"

#define LOCAL __attribute__((section(".l1"), aligned(4 * NUM_BANKS)))
static __fp16 scores[NUM_CORES * Q38_STORAGE] LOCAL;
static __fp16 probabilities[NUM_CORES * Q38_STORAGE] LOCAL;
static float statistics[NUM_CORES * 2] LOCAL;
static uint32_t errors[NUM_CORES] LOCAL;

int main(void) {
  const uint32_t core = mempool_get_core_id();
  mempool_barrier_init(core);
  errors[core] = 0;
  errors[core] +=
      dma_collective_memcpy_blocking(scores, scores_init, sizeof(scores)) != 0u;
  const uint32_t start = mempool_get_timer();
  __fp16 *result = probabilities + core * Q38_STORAGE;
  const __fp16 *input = scores + core * Q38_STORAGE;
  llm_zero_f16(result, Q38_STORAGE);
  qwen38_softmax_state_t state;
  qwen38_softmax_init(&state);
  // Zero-visible-key blocks must not change state or produce NaNs.
  const float identity = qwen38_softmax_update_f16(input, result, 0, &state);
  if (identity != 1.0f || state.denominator != 0.0f)
    errors[core]++;
  for (uint32_t offset = 0; offset < Q38_COLUMNS; offset += Q38_BLOCK) {
    const uint32_t remaining = Q38_COLUMNS - offset;
    const uint32_t count = remaining < Q38_BLOCK ? remaining : Q38_BLOCK;
    const uint32_t packed_offset = (offset / Q38_BLOCK) * Q38_STRIDE;
    const float correction = qwen38_softmax_update_f16(
        input + packed_offset, result + packed_offset, count, &state);
    // In the full attention kernel this scales the FP32 output accumulator.
    // Here retain/rescale probabilities so DPI can inspect every element.
    if (offset != 0u)
      llm_scale_f16_f32(result, packed_offset, correction);
  }
  llm_scale_f16_f32(result, Q38_STORAGE, llm_recip_pos(state.denominator));
  statistics[2 * core] = state.maximum;
  statistics[2 * core + 1] = state.denominator;
  __sync_synchronize();
  mempool_barrier(NUM_CORES);
  const uint32_t elapsed = mempool_get_timer() - start;
  errors[core] +=
      dma_collective_memcpy_blocking(probabilities_result, probabilities,
                                     sizeof(probabilities)) != 0u;
  errors[core] += dma_collective_memcpy_blocking(statistics_result, statistics,
                                                 sizeof(statistics)) != 0u;
  errors[core] += dma_collective_memcpy_blocking(scores_result, scores,
                                                 sizeof(scores)) != 0u;
  mempool_check_dpi_f16(scores_result, scores_init, NUM_CORES * Q38_STORAGE,
                        0.0f, 0);
  __sync_synchronize();
  mempool_check_dpi_f16(probabilities_result, probabilities_expected,
                        NUM_CORES * Q38_STORAGE, 0.001f, 0);
  __sync_synchronize();
  mempool_check_dpi_f32(statistics_result, statistics_expected, NUM_CORES * 2,
                        0.003f, 0);
  __sync_synchronize();
  mempool_barrier(NUM_CORES);
  if (core == 0) {
    uint32_t total = 0;
    for (uint32_t i = 0; i < NUM_CORES; ++i)
      total += errors[i];
    printf(
        "[qwen38_softmax] cores=%u columns=%u block=%u cycles=%u errors=%u\n",
        NUM_CORES, Q38_COLUMNS, Q38_BLOCK, elapsed, total);
    if (total != 0u)
      return 1;
  }
  mempool_barrier(NUM_CORES);
  return 0;
}
