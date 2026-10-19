// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "baremetal/mempool_llm_ffn_f16.h"

static inline void q38_copy16(__fp16 *dst, const __fp16 *src, uint32_t n) {
  while (n) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(n));
    asm volatile("vle16.v v0, (%0)" : : "r"(src) : "memory");
    asm volatile("vse16.v v0, (%0)" : : "r"(dst) : "memory");
    dst += vl;
    src += vl;
    n -= vl;
  }
}

// Scalar control coefficients use integer loads, avoiding scalar FP16 memory
// instructions. Matrix/vector data paths below use Spatz loads and stores.
static inline float q38_read16(const __fp16 *p) {
  uint32_t h;
  asm volatile("lhu %0, 0(%1)" : "=r"(h) : "r"(p) : "memory");
  uint32_t fraction = h & 1023u;
  uint32_t exponent = (h >> 10) & 31u;
  uint32_t bits = (h & 32768u) << 16;
  if (exponent == 31u) {
    bits |= 0x7f800000u | (fraction << 13);
  } else if (exponent != 0u) {
    bits |= ((exponent + 112u) << 23) | (fraction << 13);
  } else if (fraction != 0u) {
    exponent = 113u;
    while ((fraction & 1024u) == 0u) {
      fraction <<= 1;
      --exponent;
    }
    bits |= (exponent << 23) | ((fraction & 1023u) << 13);
  }
  union {
    uint32_t u;
    float f;
  } value = {.u = bits};
  return value.f;
}

static inline void q38_write16(__fp16 *p, float x) {
  union {
    uint32_t u;
    float f;
  } value = {.f = llm_f16box(x)};
  asm volatile("sh %0, 0(%1)" : : "r"(value.u), "r"(p) : "memory");
}

// FP32 reduction and scaling, epsilon=1e-6. The gated-norm variant rounds
// normalized values before the learned multiplier, matching its reference.
static inline void q38_norm16(const __fp16 *x, const __fp16 *gamma, __fp16 *out,
                              uint32_t n, int gated) {
  uint32_t capacity;
  const float zero = 0.0f, zh = llm_f16box(0.0f);
  asm volatile("vsetvli %0, %1, e32, m8, tu, ma"
               : "=r"(capacity)
               : "r"(UINT32_MAX));
  asm volatile("vmv.v.i v8, 0");
  for (uint32_t offset = 0; offset < n;) {
    uint32_t vl, requested = n - offset;
    if (requested > capacity)
      requested = capacity;
    asm volatile("vsetvli %0, %1, e16, m4, tu, ma" : "=r"(vl) : "r"(requested));
    asm volatile("vle16.v v0, (%0)" : : "r"(x + offset) : "memory");
    asm volatile("vfwmacc.vv v8, v0, v0");
    offset += vl;
  }
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(capacity));
  asm volatile("vfmv.s.f v24, %0" : : "f"(zero));
  asm volatile("vfredusum.vs v24, v8, v24");
  float sum;
  asm volatile("vfmv.f.s %0, v24" : "=f"(sum));
  const float inverse = llm_rsqrt(sum * llm_recip_pos((float)n) + 1e-6f);
  for (uint32_t offset = 0; offset < n;) {
    uint32_t vl, requested = n - offset;
    if (requested > capacity)
      requested = capacity;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(requested));
    asm volatile("vle16.v v0, (%0)" : : "r"(x + offset) : "memory");
    asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
    asm volatile("vle16.v v0, (%0)" : : "r"(gamma + offset) : "memory");
    asm volatile("vfwadd.vf v16, v0, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    asm volatile("vfmul.vf v8, v8, %0" : : "f"(inverse));
    if (gated) {
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
      asm volatile("vfncvt.f.f.w v0, v8");
      asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    }
    asm volatile("vfmul.vv v8, v8, v16");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
    asm volatile("vfncvt.f.f.w v0, v8");
    asm volatile("vse16.v v0, (%0)" : : "r"(out + offset) : "memory");
    offset += vl;
  }
}

// mode 0: sigmoid; mode 1: SiLU. Both round activation to FP16 before
// multiplication. Mode 2 is the GDN gate: retain FP32 SiLU until the product.
static inline void q38_activation16(const __fp16 *gate, const __fp16 *value,
                                    __fp16 *out, uint32_t n, uint32_t mode) {
  const float one = 1.0f, zh = llm_f16box(0.0f);
  for (uint32_t offset = 0; offset < n;) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma"
                 : "=r"(vl)
                 : "r"(n - offset));
    asm volatile("vle16.v v0, (%0)" : : "r"(gate + offset) : "memory");
    asm volatile("vfwadd.vf v24, v0, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    asm volatile("vfsgnjn.vv v0, v24, v24");
    llm_rvv_expf_m8();
    asm volatile("vfadd.vf v0, v0, %0" : : "f"(one));
    llm_rvv_recip_pos_m8();
    if (mode != 0u)
      asm volatile("vfmul.vv v8, v8, v24");
    if (mode != 2u) {
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
      asm volatile("vfncvt.f.f.w v0, v8");
      asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
    }
    if (value != 0) {
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
      asm volatile("vle16.v v0, (%0)" : : "r"(value + offset) : "memory");
      asm volatile("vfwadd.vf v16, v0, %0" : : "f"(zh));
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
      asm volatile("vfmul.vv v8, v8, v16");
    }
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
    asm volatile("vfncvt.f.f.w v0, v8");
    asm volatile("vse16.v v0, (%0)" : : "r"(out + offset) : "memory");
    offset += vl;
  }
}

static inline float q38_softplus(float x) {
  const float magnitude = x < 0.0f ? -x : x;
  const float y = llm_expf(-magnitude);
  const float z = y * llm_recip_pos(2.0f + y);
  const float z2 = z * z;
  float series = 1.0f / 13.0f;
  series = 1.0f / 11.0f + z2 * series;
  series = 1.0f / 9.0f + z2 * series;
  series = 1.0f / 7.0f + z2 * series;
  series = 1.0f / 5.0f + z2 * series;
  series = 1.0f / 3.0f + z2 * series;
  series = 1.0f + z2 * series;
  return (x > 0.0f ? x : 0.0f) + 2.0f * z * series;
}
