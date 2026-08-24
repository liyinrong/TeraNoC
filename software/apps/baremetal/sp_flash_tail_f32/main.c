// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define LLM_FLASH_ATTENTION
#include "baremetal/mempool_llm_attention_f16.h"

enum {
  kRows = 4,
  kTile = 32,
  kHeadDim = 32,
};

static __fp16 scores[kRows * kTile]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static __fp16 context[kRows * kHeadDim]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static float running_max[kRows]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static float running_sum[kRows]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static volatile uint32_t test_result
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));

static inline uint32_t f32_bits(float value) {
  union {
    float value;
    uint32_t bits;
  } conversion = {.value = value};
  return conversion.bits;
}

static inline uint16_t f16_bits(const __fp16 *value) {
  union {
    __fp16 value;
    uint16_t bits;
  } conversion = {.value = *value};
  return conversion.bits;
}

static inline float abs_diff(float a, float b) {
  const float diff = a - b;
  return diff < 0.0f ? -diff : diff;
}

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();

  mempool_barrier_init(core_id);
  if (core_id == 0) {
    test_result = 1;
    for (uint32_t row = 0; row < kRows; ++row) {
      running_max[row] = -1e30f;
      running_sum[row] = 0.0f;
      for (uint32_t col = 0; col < kTile; ++col)
        scores[row * kTile + col] = (__fp16)(0.03125f * (float)(col + 1));
      for (uint32_t col = 0; col < kHeadDim; ++col)
        context[row * kHeadDim + col] = (__fp16)0.0f;
    }
  }
  mempool_barrier(num_cores);

  llm_attention_flash_update_f16(scores, context, running_max, running_sum,
                                 kRows, kTile, 1, kHeadDim, 0, 0, 0, 1, core_id,
                                 num_cores);
  mempool_barrier(num_cores);

  uint32_t errors = 0;
  if (core_id == 0) {
    const float expected_max[kRows] = {
        0.00552427f,
        0.01104854f,
        0.01657282f,
        0.02209709f,
    };
    const float expected_sum[kRows] = {
        1.0f,
        1.99449f,
        2.98350f,
        3.96707f,
    };
    for (uint32_t row = 0; row < kRows; ++row) {
      printf("[sp_flash_tail_f32] row=%u max=0x%08x sum=0x%08x "
             "p0=0x%04x plast=0x%04x\n",
             row, f32_bits(running_max[row]), f32_bits(running_sum[row]),
             f16_bits(&scores[row * kTile]),
             f16_bits(&scores[row * kTile + kTile - 1]));
      if (abs_diff(running_max[row], expected_max[row]) > 0.001f ||
          abs_diff(running_sum[row], expected_sum[row]) > 0.01f ||
          f16_bits(&scores[row * kTile + kTile - 1]) != 0)
        ++errors;
    }
    test_result = errors;
    printf("[sp_flash_tail_f32] %s\n", errors == 0 ? "PASS" : "FAIL");
  }
  mempool_barrier(num_cores);
  return (int)test_result;
}
