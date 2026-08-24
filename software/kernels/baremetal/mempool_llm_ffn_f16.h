// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yinrong Li, ETH Zurich

// Spatz RVV (RV32V + Zvfh) f16 vector kernels for the dense-LLM SwiGLU FFN
// block. These implement the non-GEMM, elementwise/normalization stages that
// the research plan maps onto the vector unit, while the projection GEMMs run
// on RedMulE. Each kernel partitions its work across the active cores; the
// caller is responsible for the surrounding barriers.
//
// Op-support notes for this cluster (Spatz @ rvf=1, scalar -mno-fdiv):
//   * Tensor data lives in interleaved L1; it is touched only through VECTOR
//     loads/stores (vle16/vse16), as in the proven sp_* kernels. RMSNorm sums
//     the squares with a widening f32 MAC (vfwmacc) and an e32 reduction
//     (vfredusum), and applies the scale with vfmul.vf/.vv; the residual add is
//     vfadd.vv.
//   * The hardware has NO scalar fdiv/fsqrt or vector divide. The per-row
//     1/sqrt(rms) uses a divide/sqrt-free inverse-sqrt, while SiLU uses a
//     vector range-reduced exp and bit-seeded reciprocal with Newton
//     refinement. Spatz does not implement vfwcvt or vfrec7, so widening uses
//     vfwadd with a zero operand and reciprocal uses supported integer/FP
//     vector operations. f16 scalars handed to .vf ops are carried NaN-boxed
//     inside `float` registers (the soft-float ilp32 ABI only allows the "f"
//     inline-asm constraint on `float`).
//   * Bias-less RedMulE jobs use owner-local vector stores to clear their GEMM
//     addend. The following HWPE MMIO access waits for acknowledged Spatz
//     stores, preserving clear-before-trigger ordering.
//
// Requires a Spatz build (RVV defined). Vector length is read at runtime with
// vsetvli, so the kernels are independent of VLEN.

#pragma once
#include <stdint.h>

#ifndef RVV
#error "mempool_llm_ffn_f16.h requires a Spatz (RVV) configuration"
#endif

// f32 value -> NaN-boxed f16 held in a `float` register (usable as an e16
// scalar operand).
static inline float llm_f16box(float x) {
  float r;
  asm volatile("fcvt.h.s %0, %1" : "=f"(r) : "f"(x));
  return r;
}

// 1/sqrt(x) without hardware fsqrt/fdiv (the cluster builds with -mno-fdiv, so
// those scalar instructions trap): fast inverse-square-root bit trick plus
// three Newton-Raphson steps, using only multiply/add.
static inline float llm_rsqrt(float x) {
  float xhalf = 0.5f * x;
  union {
    float f;
    uint32_t i;
  } u;
  u.f = x;
  u.i = 0x5f3759dfu - (u.i >> 1);
  float y = u.f;
  y = y * (1.5f - xhalf * y * y);
  y = y * (1.5f - xhalf * y * y);
  y = y * (1.5f - xhalf * y * y);
  return y;
}

// Scalar f32 exp via range reduction exp(x) = 2^k * poly(r), k =
// round(x*log2e), r = x - k*ln2 in [-ln2/2, ln2/2], degree-5 minimax-ish
// Taylor; 2^k built by injecting (k+127) into the f32 exponent. Uses only
// mul/add, an f32->int convert and an integer shift (no fdiv/fsqrt).
static inline float llm_expf(float x) {
  if (x > 80.0f)
    x = 80.0f;
  if (x < -80.0f)
    x = -80.0f;
  float t = x * 1.4426950408f; // x * log2(e)
  int k = (int)(t + (t >= 0.0f ? 0.5f : -0.5f));
  float r = x - (float)k * 0.6931471806f;
  float p =
      1.0f +
      r * (1.0f + r * (0.5f + r * (0.16666667f +
                                   r * (0.041666668f + r * 0.0083333338f))));
  union {
    float f;
    uint32_t i;
  } u;
  u.i = (uint32_t)((int32_t)(k + 127) << 23); // 2^k
  return p * u.f;
}

// Reciprocal of a positive normal f32 without fdiv. The bit-level seed is
// refined to f32 precision with four Newton-Raphson steps. Current callers pass
// either a positive tensor dimension or a value in [1, 1 + exp(80)], so zero,
// subnormal, and infinity handling is unnecessary here.
static inline float llm_recip_pos(float x) {
  union {
    float f;
    uint32_t i;
  } u;
  u.f = x;
  u.i = 0x7ef311c3u - u.i;
  float y = u.f;
  y = y * (2.0f - x * y);
  y = y * (2.0f - x * y);
  y = y * (2.0f - x * y);
  y = y * (2.0f - x * y);
  return y;
}

// Owner-local vector clear for interleaved L1. RedMulE GEMM reads its addend
// from the destination buffer, so bias-less products must clear that buffer
// first. Keep the clear on Spatz because scalar stores to interleaved L1 are
// not reliable on this cluster configuration. The fence is the compiler and
// architectural barrier; HWPE MMIO waits for acknowledged Spatz stores.
static inline void llm_zero_f16(__fp16 *p, uint32_t n) {
  uint32_t avl = n;
  while (avl > 0) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vmv.v.i v0, 0");
    asm volatile("vse16.v v0, (%0)" : : "r"(p) : "memory");
    p += vl;
    avl -= vl;
  }
  __sync_synchronize();
}

// ---------------------------------------------------------------------------
// Spatz vector transcendental helpers. The e32,m8 exp helper deliberately uses
// only three aligned register groups: input/output v0, temporary/polynomial v8,
// and integer exponent v16. This leaves v24 available to callers for live data
// or a reduction accumulator. The reciprocal helper preserves the denominator
// in v0, returns the reciprocal in v8, and uses v16 as scratch. Both helpers
// assume that the caller has selected e32,m8.
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) void llm_rvv_expf_m8(void) {
  const float C80 = 80.0f, Cn80 = -80.0f;
  const float L2E = 1.4426950408f, LN2 = 0.6931471806f;
  const float A5 = 0.0083333338f, A4 = 0.041666668f, A3 = 0.16666667f;
  const float A2 = 0.5f, A1 = 1.0f;
  const int BIAS = 127;
  asm volatile("vfmin.vf v0, v0, %0" : : "f"(C80));
  asm volatile("vfmax.vf v0, v0, %0" : : "f"(Cn80));
  asm volatile("vfmul.vf v8, v0, %0" : : "f"(L2E));
  asm volatile("vfcvt.x.f.v v16, v8");
  asm volatile("vfcvt.f.x.v v8, v16");
  asm volatile("vfnmsac.vf v0, %0, v8" : : "f"(LN2));
  asm volatile("vfmv.v.f v8, %0" : : "f"(A5));
  asm volatile("vfmul.vv v8, v8, v0");
  asm volatile("vfadd.vf v8, v8, %0" : : "f"(A4));
  asm volatile("vfmul.vv v8, v8, v0");
  asm volatile("vfadd.vf v8, v8, %0" : : "f"(A3));
  asm volatile("vfmul.vv v8, v8, v0");
  asm volatile("vfadd.vf v8, v8, %0" : : "f"(A2));
  asm volatile("vfmul.vv v8, v8, v0");
  asm volatile("vfadd.vf v8, v8, %0" : : "f"(A1));
  asm volatile("vfmul.vv v8, v8, v0");
  asm volatile("vfadd.vf v8, v8, %0" : : "f"(A1));
  asm volatile("vadd.vx v16, v16, %0" : : "r"(BIAS));
  asm volatile("vsll.vi v16, v16, 23");
  asm volatile("vfmul.vv v0, v8, v16");
}

static inline __attribute__((always_inline)) void llm_rvv_recip_pos_m8(void) {
  const uint32_t SEED = 0x7ef311c3u;
  const float TWO = 2.0f;
  asm volatile("vrsub.vx v8, v0, %0" : : "r"(SEED));
  asm volatile("vfmul.vv v16, v0, v8");
  asm volatile("vfrsub.vf v16, v16, %0" : : "f"(TWO));
  asm volatile("vfmul.vv v8, v8, v16");
  asm volatile("vfmul.vv v16, v0, v8");
  asm volatile("vfrsub.vf v16, v16, %0" : : "f"(TWO));
  asm volatile("vfmul.vv v8, v8, v16");
}

// Scale an f16 vector by an f32 scalar. e16,m4 -> e32,m8 is the widest legal
// widening pair; v0 and v8 are non-overlapping source/destination groups.
static inline void llm_scale_f16_f32(__fp16 *p, uint32_t n, float scale) {
  const float ZERO_H = llm_f16box(0.0f);
  uint32_t avl = n;
  while (avl > 0) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle16.v v0, (%0)" : : "r"(p) : "memory");
    asm volatile("vfwadd.vf v8, v0, %0" : : "f"(ZERO_H));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    asm volatile("vfmul.vf v8, v8, %0" : : "f"(scale));
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
    asm volatile("vfncvt.f.f.w v0, v8");
    asm volatile("vse16.v v0, (%0)" : : "r"(p) : "memory");
    p += vl;
    avl -= vl;
  }
}

// ---------------------------------------------------------------------------
// RMSNorm: h[r,:] = x[r,:] / sqrt(mean(x[r,:]^2) + eps) * gamma[:]
// x, h are [seqlen x dmodel] row-major; gamma is [dmodel]. Rows are striped
// across cores. The sum of squares is accumulated in f32 (widening MAC) and
// reduced on the vector unit; the scale is applied with vfmul.vf/.vv.
// ---------------------------------------------------------------------------
// x and h may alias for the in-place Q/K normalization path.
static inline void llm_rmsnorm_f16(const __fp16 *x, const __fp16 *gamma,
                                   __fp16 *h, uint32_t seqlen, uint32_t dmodel,
                                   uint32_t core_id, uint32_t ncores) {
  const float eps = 1e-5f;
  const float invD = llm_recip_pos((float)dmodel);
  const float ZERO_F = 0.0f;

  for (uint32_t r = core_id; r < seqlen; r += ncores) {
    const __fp16 *xr = x + r * dmodel;
    __fp16 *hr = h + r * dmodel;

    // 1) Sum of squares via vector loads + widening f32 MAC + e32 reduction.
    //    Vector loads (vle16) are used instead of scalar flh because scalar
    //    loads from interleaved L1 stall on this cluster; the accumulation is
    //    kept in f32 for stability.
    uint32_t avl = dmodel;
    const __fp16 *p = xr;
    asm volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"(dmodel));
    asm volatile("vmv.v.i v8, 0");
    while (avl > 0) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e16, m4, tu, ma" : "=r"(vl) : "r"(avl));
      asm volatile("vle16.v v0, (%0)" : : "r"(p) : "memory");
      asm volatile("vfwmacc.vv v8, v0, v0"); // f32 += x*x
      p += vl;
      avl -= vl;
    }
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(dmodel));
    asm volatile("vfmv.s.f v24, %0" : : "f"(ZERO_F));
    asm volatile("vfredusum.vs v24, v8, v24");
    float sumsq;
    asm volatile("vfmv.f.s %0, v24" : "=f"(sumsq));

    // 2) Reciprocal rms (no hardware sqrt/div).
    float ms = sumsq * invD + eps;
    float rinv = llm_f16box(llm_rsqrt(ms)); // 1/sqrt(ms) as NaN-boxed f16

    // 3) h = x * rinv * gamma  (Spatz vector unit).
    avl = dmodel;
    const __fp16 *xp = xr;
    const __fp16 *gp = gamma;
    __fp16 *op = hr;
    while (avl > 0) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
      asm volatile("vle16.v v0, (%0)" : : "r"(xp) : "memory");
      asm volatile("vfmul.vf v0, v0, %0" : : "f"(rinv));
      asm volatile("vle16.v v8, (%0)" : : "r"(gp) : "memory");
      asm volatile("vfmul.vv v0, v0, v8");
      asm volatile("vse16.v v0, (%0)" : : "r"(op) : "memory");
      xp += vl;
      gp += vl;
      op += vl;
      avl -= vl;
    }
  }
}

// ---------------------------------------------------------------------------
// SiLU-gated multiply: a = SiLU(g) * u, elementwise over n contiguous elements.
// Computed in vector f32 using the supported exp and reciprocal approximations.
// Work is partitioned into contiguous per-core ranges.
// ---------------------------------------------------------------------------
static inline void llm_silu_mul_f16(const __fp16 *__restrict__ g,
                                    const __fp16 *__restrict__ u,
                                    __fp16 *__restrict__ a, uint32_t n,
                                    uint32_t core_id, uint32_t ncores) {
  uint32_t chunk = (n + ncores - 1) / ncores;
  uint32_t start = core_id * chunk;
  uint32_t end = start + chunk;
  if (end > n)
    end = n;

  // Four e32,m8 groups are live at peak: exp/denominator v0, reciprocal v8,
  // u/scratch v16, and the preserved g/result v24. The paired e16,m4 loads and
  // narrows keep VLMAX=128 across SEW changes without register overlap.
  const float ONE = 1.0f;
  const float ZERO_H = llm_f16box(0.0f);
  const __fp16 *gp = g + start;
  const __fp16 *up = u + start;
  __fp16 *ap = a + start;
  uint32_t avl = (start < end) ? (end - start) : 0;
  while (avl > 0) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle16.v v0, (%0)" : : "r"(gp) : "memory");
    asm volatile("vfwadd.vf v24, v0, %0" : : "f"(ZERO_H));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    asm volatile("vfsgnjn.vv v0, v24, v24");
    llm_rvv_expf_m8();
    asm volatile("vfadd.vf v0, v0, %0" : : "f"(ONE));
    llm_rvv_recip_pos_m8();

    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
    asm volatile("vle16.v v0, (%0)" : : "r"(up) : "memory");
    asm volatile("vfwadd.vf v16, v0, %0" : : "f"(ZERO_H));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    asm volatile("vfmul.vv v24, v24, v8");
    asm volatile("vfmul.vv v24, v24, v16");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
    asm volatile("vfncvt.f.f.w v0, v24");
    asm volatile("vse16.v v0, (%0)" : : "r"(ap) : "memory");
    gp += vl;
    up += vl;
    ap += vl;
    avl -= vl;
  }
}

// ---------------------------------------------------------------------------
// Residual add: out[i] += x[i] over n contiguous elements (f16, Spatz vector
// unit), partitioned into contiguous per-core ranges.
// ---------------------------------------------------------------------------
static inline void llm_residual_add_f16(__fp16 *__restrict__ out,
                                        const __fp16 *__restrict__ x,
                                        uint32_t n, uint32_t core_id,
                                        uint32_t ncores) {
  uint32_t chunk = (n + ncores - 1) / ncores;
  uint32_t start = core_id * chunk;
  uint32_t end = start + chunk;
  if (end > n)
    end = n;

  __fp16 *op = out + start;
  const __fp16 *xp = x + start;
  uint32_t avl = (start < end) ? (end - start) : 0;

  while (avl > 0) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle16.v v0, (%0)" : : "r"(op) : "memory");
    asm volatile("vle16.v v8, (%0)" : : "r"(xp) : "memory");
    asm volatile("vfadd.vv v0, v0, v8");
    asm volatile("vse16.v v0, (%0)" : : "r"(op) : "memory");
    op += vl;
    xp += vl;
    avl -= vl;
  }
}
