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
//   * The hardware has NO scalar fdiv/fsqrt (and Spatz has no vector divide),
//     so the per-row 1/sqrt(rms) uses a divide/sqrt-free inverse-sqrt and the
//     SiLU transcendental (exp + reciprocal) is computed in scalar f32
//     (range-reduced exp + divide-free reciprocal). f16 scalars handed to .vf
//     ops are carried NaN-boxed inside `float` registers (the soft-float ilp32
//     ABI only allows the "f" inline-asm constraint on `float`).
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

// SiLU(z) = z * sigmoid(z) = z / (1 + exp(-z)), scalar f32.
static inline float llm_silu_f32(float z) {
  return z * llm_recip_pos(1.0f + llm_expf(-z));
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
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vmv.v.i v0, 0");
    asm volatile("vse16.v v0, (%0)" : : "r"(p) : "memory");
    p += vl;
    avl -= vl;
  }
  __sync_synchronize();
}

// ---------------------------------------------------------------------------
// Spatz vector path (opt-in via -DLLM_SPATZ_VEC). Moves the transcendental
// stages (SiLU here; softmax/RoPE in the attention header) onto the vector unit
// per the dense_workload.md engine map. The scalar paths above stay the default
// reference; the vector numerics are host-validated (poly-exp / softmax / SiLU
// vs libm) but require a sim-clean Spatz config for full functional validation,
// hence the flag. exp uses the same degree-5 range-reduced approximation as
// llm_expf, computed on an e32,m2 vector in v16 (scratch: v20/v24/v28).
// ---------------------------------------------------------------------------
#ifdef LLM_SPATZ_VEC
static inline void llm_rvv_expf_m2(void) {
  const float C80 = 80.0f, Cn80 = -80.0f;
  const float L2E = 1.4426950408f, LN2 = 0.6931471806f;
  const float A5 = 0.0083333338f, A4 = 0.041666668f, A3 = 0.16666667f;
  const float A2 = 0.5f, A1 = 1.0f;
  const int BIAS = 127;
  asm volatile("vfmin.vf v16, v16, %0" ::"f"(C80));   // clamp hi
  asm volatile("vfmax.vf v16, v16, %0" ::"f"(Cn80));  // clamp lo
  asm volatile("vfmul.vf v20, v16, %0" ::"f"(L2E));   // t = x*log2e
  asm volatile("vfcvt.x.f.v v24, v20");               // k = round(t)
  asm volatile("vfcvt.f.x.v v20, v24");               // kf
  asm volatile("vfnmsac.vf v16, %0, v20" ::"f"(LN2)); // r = x - ln2*kf
  asm volatile("vfmv.v.f v28, %0" ::"f"(A5));         // Horner poly(r)
  asm volatile("vfmul.vv v28, v28, v16");
  asm volatile("vfadd.vf v28, v28, %0" ::"f"(A4));
  asm volatile("vfmul.vv v28, v28, v16");
  asm volatile("vfadd.vf v28, v28, %0" ::"f"(A3));
  asm volatile("vfmul.vv v28, v28, v16");
  asm volatile("vfadd.vf v28, v28, %0" ::"f"(A2));
  asm volatile("vfmul.vv v28, v28, v16");
  asm volatile("vfadd.vf v28, v28, %0" ::"f"(A1));
  asm volatile("vfmul.vv v28, v28, v16");
  asm volatile("vfadd.vf v28, v28, %0" ::"f"(A1));  // +1.0 (c0)
  asm volatile("vadd.vx v24, v24, %0" ::"r"(BIAS)); // 2^k = (k+127)<<23
  asm volatile("vsll.vi v24, v24, 23");
  asm volatile("vfmul.vv v16, v28, v24"); // exp = poly * 2^k
}
#endif // LLM_SPATZ_VEC

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

  for (uint32_t r = core_id; r < seqlen; r += ncores) {
    const __fp16 *xr = x + r * dmodel;
    __fp16 *hr = h + r * dmodel;

    // 1) Sum of squares via vector loads + widening f32 MAC + e32 reduction.
    //    Vector loads (vle16) are used instead of scalar flh because scalar
    //    loads from interleaved L1 stall on this cluster; the accumulation is
    //    kept in f32 for stability.
    uint32_t avl = dmodel;
    const __fp16 *p = xr;
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(dmodel));
    asm volatile("vmv.v.i v8, 0");
    while (avl > 0) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(avl));
      asm volatile("vle16.v v0, (%0)" ::"r"(p));
      asm volatile("vfwmacc.vv v8, v0, v0"); // f32 += x*x
      p += vl;
      avl -= vl;
    }
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(dmodel));
    asm volatile("vmv.s.x v12, zero");
    asm volatile("vfredusum.vs v12, v8, v12");
    float sumsq;
    asm volatile("vfmv.f.s %0, v12" : "=f"(sumsq));

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
      asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(avl));
      asm volatile("vle16.v v0, (%0)" ::"r"(xp));
      asm volatile("vfmul.vf v0, v0, %0" ::"f"(rinv));
      asm volatile("vle16.v v4, (%0)" ::"r"(gp));
      asm volatile("vfmul.vv v0, v0, v4");
      asm volatile("vse16.v v0, (%0)" ::"r"(op));
      xp += vl;
      gp += vl;
      op += vl;
      avl -= vl;
    }
  }
}

// ---------------------------------------------------------------------------
// SiLU-gated multiply: a = SiLU(g) * u, elementwise over n contiguous elements.
// Computed in scalar f32 (the SiLU sigmoid needs an exp and a divide, neither
// available on the Spatz vector path here). Work is partitioned into contiguous
// per-core ranges.
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

#ifdef LLM_SPATZ_VEC
  // a = g * sigmoid(g) * u, sigmoid via 1/(1+exp(-g)); exp on the vector unit,
  // reciprocal via vfrec7 + one Newton step (>=14-bit, ample for f16).
  const float ONE = 1.0f, TWO = 2.0f;
  const __fp16 *gp = g + start;
  const __fp16 *up = u + start;
  __fp16 *ap = a + start;
  uint32_t avl = (start < end) ? (end - start) : 0;
  while (avl > 0) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle16.v v0, (%0)" ::"r"(gp));
    asm volatile("vle16.v v4, (%0)" ::"r"(up));
    asm volatile("vfwcvt.f.f.v v8, v0");  // g -> f32 (m2)
    asm volatile("vfwcvt.f.f.v v12, v4"); // u -> f32 (m2)
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vl));
    asm volatile("vfsgnjn.vv v16, v8, v8");            // v16 = -g
    llm_rvv_expf_m2();                                 // v16 = exp(-g)
    asm volatile("vfadd.vf v16, v16, %0" ::"f"(ONE));  // denom = 1+exp(-g)
    asm volatile("vfrec7.v v20, v16");                 // y0 ~ 1/denom
    asm volatile("vfmul.vv v24, v16, v20");            // denom*y0
    asm volatile("vfrsub.vf v24, v24, %0" ::"f"(TWO)); // 2 - denom*y0
    asm volatile("vfmul.vv v20, v20, v24");            // y = 1/denom
    asm volatile("vfmul.vv v8, v8, v20");              // g*sigmoid(g)
    asm volatile("vfmul.vv v8, v8, v12");              // * u
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vl));
    asm volatile("vfncvt.f.f.w v0, v8"); // f32 -> f16
    asm volatile("vse16.v v0, (%0)" ::"r"(ap));
    gp += vl;
    up += vl;
    ap += vl;
    avl -= vl;
  }
#else
  for (uint32_t i = start; i < end; i++) {
    float gv = (float)g[i];
    float uv = (float)u[i];
    a[i] = (__fp16)(llm_silu_f32(gv) * uv);
  }
#endif
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
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle16.v v0, (%0)" ::"r"(op));
    asm volatile("vle16.v v4, (%0)" ::"r"(xp));
    asm volatile("vfadd.vv v0, v0, v4");
    asm volatile("vse16.v v0, (%0)" ::"r"(op));
    op += vl;
    xp += vl;
    avl -= vl;
  }
}
