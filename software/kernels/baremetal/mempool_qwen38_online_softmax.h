// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "baremetal/mempool_llm_ffn_f16.h"

typedef struct {
  float maximum;
  float denominator;
} qwen38_softmax_state_t;

static inline void qwen38_softmax_init(qwen38_softmax_state_t *state) {
  state->maximum = -3.402823466e38f;
  state->denominator = 0.0f;
}

// Each key block starts in its own aligned, padded tile buffer. Scores contain
// the valid causal prefix; pad the allocation to a multiple of 32 FP16 elements
// with -65504 and initialize padded probability columns to zero. Do not pack
// odd-length blocks directly back-to-back. The full-layer PV consumes the
// logical columns, or includes the initialized zero probability padding.
// Returns the correction to apply to the OLD unnormalized output accumulator.
// The owner must complete that correction before adding this block's PV result.
static inline float qwen38_softmax_update_f16(const __fp16 *scores,
                                              __fp16 *probabilities,
                                              uint32_t count,
                                              qwen38_softmax_state_t *state) {
  if (count == 0u)
    return 1.0f;

  const float zero_half = llm_f16box(0.0f);
  const float zero = 0.0f;
  uint32_t capacity;
  asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
               : "=r"(capacity)
               : "r"(UINT32_MAX));
  float maximum = state->maximum;
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count));
  asm volatile("vfmv.s.f v24, %0" : : "f"(maximum));
  for (uint32_t offset = 0; offset < count;) {
    uint32_t vl;
    const uint32_t remaining = count - offset;
    const uint32_t requested = remaining < capacity ? remaining : capacity;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(requested));
    asm volatile("vle16.v v8, (%0)" : : "r"(scores + offset) : "memory");
    asm volatile("vfwadd.vf v0, v8, %0" : : "f"(zero_half));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    asm volatile("vfredmax.vs v24, v0, v24");
    offset += vl;
  }
  asm volatile("vfmv.f.s %0, v24" : "=f"(maximum));
  const float correction =
      state->denominator == 0.0f ? 0.0f : llm_expf(state->maximum - maximum);
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count));
  asm volatile("vfmv.s.f v24, %0" : : "f"(zero));
  for (uint32_t offset = 0; offset < count;) {
    uint32_t vl;
    const uint32_t remaining = count - offset;
    const uint32_t requested = remaining < capacity ? remaining : capacity;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(requested));
    asm volatile("vle16.v v8, (%0)" : : "r"(scores + offset) : "memory");
    asm volatile("vfwadd.vf v0, v8, %0" : : "f"(zero_half));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    asm volatile("vfsub.vf v0, v0, %0" : : "f"(maximum));
    llm_rvv_expf_m8();
    asm volatile("vfredusum.vs v24, v0, v24");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
    asm volatile("vfncvt.f.f.w v8, v0");
    asm volatile("vse16.v v8, (%0)" : : "r"(probabilities + offset) : "memory");
    offset += vl;
  }
  float denominator;
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count));
  asm volatile("vfmv.f.s %0, v24" : "=f"(denominator));
  state->maximum = maximum;
  state->denominator = correction * state->denominator + denominator;
  return correction;
}
