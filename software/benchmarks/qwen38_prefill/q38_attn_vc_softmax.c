// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Included by q38_attention.c. Retain one block in v24..v31; exp uses
// v0/v8/v16.
#if Q38_ATTN_VC_SWEEP && Q38_AV_SWEEP_TREE_MAX
// Exact finite max, using vector pairs instead of a serial reduction. Keep
// v0's scores live for the single-pass B128 path; v24 remains untouched.
static inline __attribute__((always_inline)) float avs_tree_max(uint32_t count,
                                                                float maximum) {
  asm volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"(count));
  asm volatile("vmv.v.v v8, v0");
  for (uint32_t n = count; n > 1u;) {
    const uint32_t half = (n + 1u) / 2u, pairs = n - half;
    asm volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"(pairs));
    asm volatile("vslidedown.vx v16, v0, %0" : : "r"(half));
    asm volatile("vfmax.vv v0, v0, v16");
    n = half;
  }
  float reduced;
  asm volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"(1u));
  asm volatile("vfmv.f.s %0, v0" : "=f"(reduced));
  asm volatile("fmax.s %0, %1, %2"
               : "=f"(maximum)
               : "f"(maximum), "f"(reduced));
  asm volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"(count));
  asm volatile("vmv.v.v v0, v8");
  return maximum;
}
#endif

#if Q38_ATTN_VC_SWEEP && Q38_AV_SWEEP_REGISTER && Q38_BK > 128
#define AVS_SM_CHUNK (VLEN / 32u * 8u)
_Static_assert(VLEN == 1024u && BK <= 2u * AVS_SM_CHUNK,
               "Register softmax needs one FP16 m8 block");

// Keep the old 128-element reduction grouping, even with a 256-element chunk.
static inline __attribute__((always_inline)) float
avs_register_max(uint32_t count, float maximum) {
#if Q38_AV_SWEEP_TREE_MAX
  return avs_tree_max(count, maximum);
#else
  const uint32_t lower = Q38_MIN(128u, count);
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(lower));
  asm volatile("vfmv.s.f v8, %0" : : "f"(maximum));
  asm volatile("vfredmax.vs v8, v0, v8");
  asm volatile("vfmv.f.s %0, v8" : "=f"(maximum));
#if AVS_SM_CHUNK > 128
  if (count > 128u) {
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count - 128u));
    asm volatile("vslidedown.vx v16, v0, %0" : : "r"(128u));
    asm volatile("vfmv.s.f v8, %0" : : "f"(maximum));
    asm volatile("vfredmax.vs v8, v16, v8");
    asm volatile("vfmv.f.s %0, v8" : "=f"(maximum));
  }
#endif
  return maximum;
#endif
}

static inline __attribute__((always_inline)) float
avs_register_sum(uint32_t count, float denominator) {
  const float zero = 0.0f;
  const uint32_t lower = Q38_MIN(128u, count);
  float sum;
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(lower));
  asm volatile("vfmv.s.f v8, %0" : : "f"(zero));
  asm volatile("vfredusum.vs v8, v0, v8");
  asm volatile("vfmv.f.s %0, v8" : "=f"(sum));
  denominator += sum;
#if AVS_SM_CHUNK > 128
  if (count > 128u) {
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count - 128u));
    asm volatile("vslidedown.vx v16, v0, %0" : : "r"(128u));
    asm volatile("vfmv.s.f v8, %0" : : "f"(zero));
    asm volatile("vfredusum.vs v8, v16, v8");
    asm volatile("vfmv.f.s %0, v8" : "=f"(sum));
    denominator += sum;
  }
#endif
  return denominator;
}

static __attribute__((noinline)) float
avs_register_softmax_row(const __fp16 *score, __fp16 *prob, uint32_t visible,
                         qwen38_softmax_state_t *state) {
  if (!visible) {
    llm_zero_f16(prob, BK);
    return 1.0f;
  }
  const float zh = llm_f16box(0.0f), scale = llm_rsqrt((float)Q38_HD);
  float maximum = state->maximum;
  const uint32_t loaded = Q38_MIN(BK, (visible + 127u) / 128u * 128u);
  asm volatile("vsetvli zero, %0, e16, m8, ta, ma" : : "r"(loaded));
  asm volatile("vle16.v v24, (%0)" : : "r"(score) : "memory");
#pragma clang loop unroll(full)
  for (uint32_t off = 0; off < BK; off += AVS_SM_CHUNK) {
    if (off >= visible)
      continue;
    const uint32_t count = Q38_MIN(AVS_SM_CHUNK, visible - off);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
    if (off)
      asm volatile("vfwadd.vf v0, v28, %0" : : "f"(zh));
    else
      asm volatile("vfwadd.vf v0, v24, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count));
    asm volatile("vfmul.vf v0, v0, %0" : : "f"(scale));
    asm volatile("vsetvli zero, %0, e16, m4, tu, ma" : : "r"(count));
    if (off) {
      asm volatile("vfncvt.f.f.w v28, v0");
      asm volatile("vfwadd.vf v0, v28, %0" : : "f"(zh));
    } else {
      asm volatile("vfncvt.f.f.w v24, v0");
      asm volatile("vfwadd.vf v0, v24, %0" : : "f"(zh));
    }
    maximum = avs_register_max(count, maximum);
  }
  const float correction =
      state->denominator == 0.0f ? 0.0f : llm_expf(state->maximum - maximum);
  float denominator = 0.0f;
#pragma clang loop unroll(full)
  for (uint32_t off = 0; off < BK; off += AVS_SM_CHUNK) {
    const uint32_t width = Q38_MIN(AVS_SM_CHUNK, BK - off);
    if (off >= visible) {
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(width));
      asm volatile("vmv.v.i v8, 0");
      asm volatile("vse16.v v8, (%0)" : : "r"(prob + off) : "memory");
      continue;
    }
    const uint32_t count = Q38_MIN(AVS_SM_CHUNK, visible - off);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
    if (off)
      asm volatile("vfwadd.vf v0, v28, %0" : : "f"(zh));
    else
      asm volatile("vfwadd.vf v0, v24, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count));
    asm volatile("vfsub.vf v0, v0, %0" : : "f"(maximum));
    llm_rvv_expf_m8();
    denominator = avs_register_sum(count, denominator);
    if (count < width) {
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(width));
      asm volatile("vmv.v.i v8, 0");
      asm volatile("vse16.v v8, (%0)" : : "r"(prob + off) : "memory");
    }
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
    asm volatile("vfncvt.f.f.w v8, v0");
    asm volatile("vse16.v v8, (%0)" : : "r"(prob + off) : "memory");
  }
  state->maximum = maximum;
  state->denominator = correction * state->denominator + denominator;
  return correction;
}
#endif

#if Q38_ATTN_VC_SWEEP && Q38_AV_SWEEP_LEAN && Q38_BK > 128
// The legacy 128-element row loop with fewer vector instructions per score.
// Without EXP2 every change is bit-identical for finite scores.

// llm_rvv_expf_m8 for x <= 0: the upper clamp is dead after subtracting the
// row maximum, and A5 * r needs no splat.
static inline __attribute__((always_inline)) void avs_expf_nonpos_m8(void) {
  const float Cn80 = -80.0f, L2E = 1.4426950408f, LN2 = 0.6931471806f;
  const float A5 = 0.0083333338f, A4 = 0.041666668f, A3 = 0.16666667f;
  const float A2 = 0.5f, A1 = 1.0f;
  const int BIAS = 127;
  asm volatile("vfmax.vf v0, v0, %0" : : "f"(Cn80));
  asm volatile("vfmul.vf v8, v0, %0" : : "f"(L2E));
  asm volatile("vfcvt.x.f.v v16, v8");
  asm volatile("vfcvt.f.x.v v8, v16");
  asm volatile("vfnmsac.vf v0, %0, v8" : : "f"(LN2));
  asm volatile("vfmul.vf v8, v0, %0" : : "f"(A5));
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

#if Q38_AV_SWEEP_EXP2
// exp(x) = 2^n * p(f) for x <= 0: t = x log2(e), n = rint(t), f = t - n in
// [-1/2, 1/2], p a degree-4 relative-minimax fit (6.2e-6). 2^n is added to
// the exponent field; the clamp keeps that field normal.
static inline __attribute__((always_inline)) void avs_exp2_nonpos_m8(void) {
  const float L2E = 1.44269504f, LOW = -125.0f;
  const float C4 = 0.009570101276f, C3 = 0.0559178628f, C2 = 0.2402474433f;
  const float C1 = 0.6931217909f, C0 = 0.9999992847f;
  asm volatile("vfmul.vf v8, v0, %0" : : "f"(L2E));
  asm volatile("vfmax.vf v8, v8, %0" : : "f"(LOW));
  asm volatile("vfcvt.x.f.v v16, v8");
  asm volatile("vfcvt.f.x.v v0, v16");
  asm volatile("vfsub.vv v8, v8, v0");
  asm volatile("vfmul.vf v0, v8, %0" : : "f"(C4));
  asm volatile("vfadd.vf v0, v0, %0" : : "f"(C3));
  asm volatile("vfmul.vv v0, v0, v8");
  asm volatile("vfadd.vf v0, v0, %0" : : "f"(C2));
  asm volatile("vfmul.vv v0, v0, v8");
  asm volatile("vfadd.vf v0, v0, %0" : : "f"(C1));
  asm volatile("vfmul.vv v0, v0, v8");
  asm volatile("vfadd.vf v0, v0, %0" : : "f"(C0));
  asm volatile("vsll.vi v16, v16, 23");
  asm volatile("vadd.vv v0, v0, v16");
}
#define AVS_LEAN_EXP avs_exp2_nonpos_m8
#else
#define AVS_LEAN_EXP avs_expf_nonpos_m8
#endif

static inline __attribute__((always_inline)) float
avs_lean_softmax_row(const __fp16 *score, __fp16 *prob, uint32_t visible,
                     qwen38_softmax_state_t *state) {
  const float zh = llm_f16box(0.0f), zero = 0.0f;
  if (!visible) {
    llm_zero_f16(prob, BK);
    return 1.0f;
  }
#if !Q38_AV_SWEEP_QSCALE
  const float scale = llm_rsqrt((float)Q38_HD);
#endif
  float maximum = state->maximum;
#pragma clang loop unroll(full)
  for (uint32_t off = 0; off < BK; off += 128u) {
    if (off >= visible)
      continue;
    const uint32_t count = Q38_MIN(128u, visible - off);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(128u));
    asm volatile("vle16.v v8, (%0)" : : "r"(score + off) : "memory");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
    asm volatile("vfwadd.vf v0, v8, %0" : : "f"(zh));
#if !Q38_AV_SWEEP_QSCALE
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count));
    asm volatile("vfmul.vf v0, v0, %0" : : "f"(scale));
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
    asm volatile("vfncvt.f.f.w v8, v0");
    asm volatile("vfwadd.vf v0, v8, %0" : : "f"(zh));
#endif
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count));
#if Q38_AV_SWEEP_TREE_MAX
    maximum = avs_tree_max(count, maximum);
#else
    asm volatile("vfmv.s.f v24, %0" : : "f"(maximum));
    asm volatile("vfredmax.vs v24, v0, v24");
    asm volatile("vfmv.f.s %0, v24" : "=f"(maximum));
#endif
  }
  const float correction =
      state->denominator == 0.0f ? 0.0f : llm_expf(state->maximum - maximum);
  // Every maximum candidate is a widened FP16 score, so -maximum is exact in
  // FP16 and one widening add replaces widen + subtract.
  const float negmax = llm_f16box(-maximum);
  float denominator = 0.0f;
#pragma clang loop unroll(full)
  for (uint32_t off = 0; off < BK; off += 128u) {
    if (off >= visible) {
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(128u));
      asm volatile("vmv.v.i v16, 0");
      asm volatile("vse16.v v16, (%0)" : : "r"(prob + off) : "memory");
      continue;
    }
    const uint32_t count = Q38_MIN(128u, visible - off);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(128u));
    asm volatile("vle16.v v8, (%0)" : : "r"(score + off) : "memory");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
#if Q38_AV_SWEEP_QSCALE
    asm volatile("vfwadd.vf v0, v8, %0" : : "f"(negmax));
#else
    asm volatile("vfwadd.vf v0, v8, %0" : : "f"(zh));
#endif
    if (count < 128u) {
      // Valid scores are now in v0: an aligned clear also works in place.
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(128u));
      asm volatile("vmv.v.i v16, 0");
      asm volatile("vse16.v v16, (%0)" : : "r"(prob + off) : "memory");
    }
#if !Q38_AV_SWEEP_QSCALE
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count));
    asm volatile("vfmul.vf v0, v0, %0" : : "f"(scale));
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
    asm volatile("vfncvt.f.f.w v8, v0");
    asm volatile("vfwadd.vf v0, v8, %0" : : "f"(negmax));
#endif
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count));
    AVS_LEAN_EXP();
    asm volatile("vfmv.s.f v24, %0" : : "f"(zero));
    asm volatile("vfredusum.vs v24, v0, v24");
    float sum;
    asm volatile("vfmv.f.s %0, v24" : "=f"(sum));
    denominator += sum;
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
    asm volatile("vfncvt.f.f.w v8, v0");
    asm volatile("vse16.v v8, (%0)" : : "r"(prob + off) : "memory");
  }
  state->maximum = maximum;
  state->denominator = correction * state->denominator + denominator;
  return correction;
}
#endif

#if Q38_ATTN_VC_SWEEP && Q38_AV_SWEEP_FAST
// Base-2 online softmax in FP16 (scores carry 1/sqrt(HD) from the Q
// epilogue). The running maximum is an integer n >= max(s log2 e), so the
// block correction and alpha are exact powers of two and need no exp. A row
// is one e16 m8 group; exp2 runs on two e16 m4 halves; only the row sum is
// FP32. Constants live in registers for a whole softmax call.
#define AVS_FAST_PIECE (VLEN / 8u) // e16 m2: one memory piece
#define AVS_FAST_CHUNK (VLEN / 4u) // e16 m4: one exp2 half
_Static_assert(BK <= 2u * AVS_FAST_CHUNK && BK % AVS_FAST_PIECE == 0u,
               "FP16 softmax keeps one block row in an e16 m8 group");

typedef struct {
  float l2e, low, tiny, zh, c0, c1, c2, c3; // NaN-boxed FP16
  float zero, l2e32;
  float l2eh; // log2(e), FP16: Q38_AV_SWEEP_EXP_POLY
  int32_t bias;
} avs_fast_k_t;

static inline __attribute__((always_inline)) float avs_hbits(uint32_t bits) {
  float r;
  asm volatile("fmv.w.x %0, %1" : "=f"(r) : "r"(0xffff0000u | bits));
  return r;
}

static inline __attribute__((always_inline)) float avs_fbits(uint32_t bits) {
  float r;
  asm volatile("fmv.w.x %0, %1" : "=f"(r) : "r"(bits));
  return r;
}

// Register-only constants: no .sdata loads from L2 inside the row loop.
static inline __attribute__((always_inline)) void
avs_fast_init(avs_fast_k_t *k) {
#if Q38_AV_SWEEP_VFEXP
  k->l2e = avs_hbits(0x3c00u); // base e: t = s - o (avs_fast_ref2)
#else
  k->l2e = avs_hbits(0x3dc5u); // log2(e)
#endif
  k->low = avs_hbits(0xce40u);  // -25: below 2^-24 P is zero
  k->tiny = avs_hbits(0x1400u); // 2^-10
  k->zh = avs_hbits(0x0000u);
  // Degree-3 relative-minimax 2^f on [-1/2, 1/2], FP16 coefficients.
  k->c0 = avs_hbits(0x3c00u);
  k->c1 = avs_hbits(0x398cu);
  k->c2 = avs_hbits(0x33c3u);
  k->c3 = avs_hbits(0x2b10u);
  k->zero = avs_fbits(0u);
  k->l2e32 = avs_fbits(0x3fb8aa3bu);
  k->l2eh = avs_hbits(0x3dc5u);
  k->bias = 25;
}

// What-if: replace the exp2 polynomial by the hardware FP16 VFEXPF (e^x,
// base e), emulated by one vfmul so the timing is right and the values are not
// (2026-10-09; base e per the RTL, 2026-10-18).
#ifndef Q38_AV_SWEEP_VFEXP
#define Q38_AV_SWEEP_VFEXP 0
#endif
// 2: also fold the row's max and sum reductions into short ones, using the
// vector registers the polynomial no longer needs (full rows of 4 pieces).
// 3: also process two full rows at once (avs_fast_softmax_pair).
// 4: modes 2 + one loop over a core's full rows (avs_fast_rows): per-row
//    bookkeeping hoisted, sum issued before the stores, argument t = s - o.
_Static_assert(Q38_AV_SWEEP_VFEXP < 2 || BK == 4u * (VLEN / 8u),
               "Folded row needs BK = four e16 m2 pieces");
// Correctness only: the VFEXPF stand-ins compute the value of e^t (the 2^x
// polynomial on t log2(e)), so a numeric run checks the mode-4 paths.
#ifndef Q38_AV_SWEEP_EXP_POLY
#define Q38_AV_SWEEP_EXP_POLY 0
#endif
_Static_assert(!Q38_AV_SWEEP_EXP_POLY || Q38_AV_SWEEP_VFEXP == 4,
               "EXP_POLY emulates the mode-4 VFEXPF");
// dst = e^t on e16 m4 groups at the current vl: the VFEXPF stand-in (one
// vfmul; timing only), or with EXP_POLY its value, clobbering t, ni and fr.
// Needs the row constants l2eh, low, tiny, c0..c3 and bias in registers.
#if Q38_AV_SWEEP_EXP_POLY
#define AVS_EXPE_M4(dst, t, ni, fr)                                            \
  do {                                                                         \
    asm volatile("vfmul.vf " t ", " t ", %0" : : "f"(l2eh));                   \
    asm volatile("vfmax.vf " t ", " t ", %0" : : "f"(low));                    \
    asm volatile("vfcvt.x.f.v " ni ", " t);                                    \
    asm volatile("vfcvt.f.x.v " fr ", " ni);                                   \
    asm volatile("vfsub.vv " fr ", " t ", " fr);                               \
    asm volatile("vfmul.vf " dst ", " fr ", %0" : : "f"(c3));                  \
    asm volatile("vfadd.vf " dst ", " dst ", %0" : : "f"(c2));                 \
    asm volatile("vfmul.vv " dst ", " dst ", " fr);                            \
    asm volatile("vfadd.vf " dst ", " dst ", %0" : : "f"(c1));                 \
    asm volatile("vfmul.vv " dst ", " dst ", " fr);                            \
    asm volatile("vfadd.vf " dst ", " dst ", %0" : : "f"(c0));                 \
    asm volatile("vadd.vx " ni ", " ni ", %0" : : "r"(bias));                  \
    asm volatile("vsll.vi " ni ", " ni ", 10");                                \
    asm volatile("vfmul.vv " dst ", " dst ", " ni);                            \
    asm volatile("vfmul.vf " dst ", " dst ", %0" : : "f"(tiny));               \
  } while (0)
#else
#define AVS_EXPE_M4(dst, t, ni, fr)                                            \
  asm volatile("vfmul.vv " dst ", " t ", " t) // VFEXPF stand-in
#endif

#define AVS_LN2 0.693147181f
#define AVS_LOG2E 1.44269504f
#if Q38_AV_SWEEP_VFEXP
// fl16(n ln2) - n ln2: the FP16 rounding of a base-e reference.
static inline __attribute__((always_inline)) float avs_ln2_err(int32_t n) {
  const float x = (float)n * AVS_LN2;
  float h, back;
  asm volatile("fcvt.h.s %0, %1" : "=f"(h) : "f"(x));
  asm volatile("fcvt.s.h %0, %1" : "=f"(back) : "f"(h));
  return back - x;
}
#endif

// The row's new reference from its maximum m, and the correction of the old
// state. The running maximum is an integer n in base-2 units either way, so
// the correction is 2^(old - n). Base 2 (the polynomial) subtracts n from
// s log2(e); base e (VFEXPF) subtracts o = fl16(n ln2) from s, and the
// correction then also carries e^eps, eps = (o_old - old ln2) - (o - n ln2),
// |eps| < 2^-7, so that the rounding of o cancels between blocks.
static inline __attribute__((always_inline)) float
avs_fast_ref2(qwen38_softmax_state_t *state, float m, float l2e32,
              float *off16) {
  int32_t nrow;
  asm volatile("fcvt.w.s %0, %1, rup" : "=r"(nrow) : "f"(m * l2e32));
  int32_t n = nrow;
  float correction = 0.0f;
  if (state->denominator != 0.0f) {
    const int32_t old = (int32_t)state->maximum;
    n = old > nrow ? old : nrow;
    const int32_t shift = old - n;
    union {
      uint32_t u;
      float f;
    } c = {.u = shift < -126 ? 0u : (uint32_t)(shift + 127) << 23};
    correction = c.f;
#if Q38_AV_SWEEP_VFEXP
    const float eps = avs_ln2_err(old) - avs_ln2_err(n);
    correction *= 1.0f + eps * (1.0f + 0.5f * eps);
#endif
  }
  state->maximum = (float)n;
#if Q38_AV_SWEEP_VFEXP
  *off16 = llm_f16box((float)n * AVS_LN2);
#else
  *off16 = llm_f16box((float)n);
#endif
  return correction;
}

// Vector memory accesses use e16 m2 pieces, and loads always fetch whole
// pieces: a unit-stride e16 load with an odd vl > 1 wedges the GVSoC VLSU
// (2026-10-04); the legacy rows also only load whole chunks.
static inline __attribute__((always_inline)) void
avs_fast_load_row(const __fp16 *row, uint32_t visible) {
#pragma clang loop unroll(full)
  for (uint32_t off = 0; off < BK; off += AVS_FAST_PIECE) {
    if (off >= visible)
      continue;
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(AVS_FAST_PIECE));
    if (off == 0u)
      asm volatile("vle16.v v24, (%0)" : : "r"(row) : "memory");
    else if (off == AVS_FAST_PIECE)
      asm volatile("vle16.v v26, (%0)" : : "r"(row + off) : "memory");
    else if (off == 2u * AVS_FAST_PIECE)
      asm volatile("vle16.v v28, (%0)" : : "r"(row + off) : "memory");
    else
      asm volatile("vle16.v v30, (%0)" : : "r"(row + off) : "memory");
  }
}

// Store `count` FP16 values of v12..v15 at dst, or a chunk of zeros.
static inline __attribute__((always_inline)) void
avs_fast_store(__fp16 *dst, uint32_t count, uint32_t zero) {
  if (zero) {
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(AVS_FAST_PIECE));
    asm volatile("vmv.v.i v8, 0");
    asm volatile("vse16.v v8, (%0)" : : "r"(dst) : "memory");
    asm volatile("vse16.v v8, (%0)" : : "r"(dst + AVS_FAST_PIECE) : "memory");
    return;
  }
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma"
               :
               : "r"(Q38_MIN(AVS_FAST_PIECE, count)));
  asm volatile("vse16.v v12, (%0)" : : "r"(dst) : "memory");
  if (count > AVS_FAST_PIECE) {
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma"
                 :
                 : "r"(count - AVS_FAST_PIECE));
    asm volatile("vse16.v v14, (%0)" : : "r"(dst + AVS_FAST_PIECE) : "memory");
  }
}

static inline __attribute__((always_inline)) float
avs_fast_softmax_row(__fp16 *row, uint32_t visible,
                     qwen38_softmax_state_t *state, const avs_fast_k_t *k) {
  if (!visible) {
    llm_zero_f16(row, BK);
    return 1.0f;
  }
  // GVSoC hands a vector instruction the scalar register file as it is at
  // dispatch, without waiting for an in-flight load into its scalar operand
  // (RTL Snitch stalls the offload). Constants therefore live in their own
  // callee-saved registers, loaded at the row start, never just before a use.
  register float l2e asm("fs0") = k->l2e;
  register float low asm("fs1") = k->low;
  register float tiny asm("fs2") = k->tiny;
  register float zh asm("fs3") = k->zh;
  register float c0 asm("fs4") = k->c0;
  register float c1 asm("fs5") = k->c1;
  register float c2 asm("fs6") = k->c2;
  register float c3 asm("fs7") = k->c3;
  register float zero asm("fs8") = k->zero;
  asm volatile(""
               : "+f"(l2e), "+f"(low), "+f"(tiny), "+f"(zh), "+f"(c0), "+f"(c1),
                 "+f"(c2), "+f"(c3), "+f"(zero));
#if Q38_AV_SWEEP_EXP_POLY
  register float l2eh asm("fs9") = k->l2eh;
  asm volatile("" : "+f"(l2eh));
#endif
  int32_t bias = k->bias;
  asm volatile("" : "+r"(bias));
  float mh, m, sum;
  avs_fast_load_row(row, visible);
#if Q38_AV_SWEEP_VFEXP >= 2
  if (visible == BK) {
    // A reduction holds the VFU alone (~0.27 cycles/element + ~26): fold the
    // four pieces and the two halves element-wise, then reduce 32 elements.
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(AVS_FAST_PIECE));
    asm volatile("vfmax.vv v0, v24, v26");
    asm volatile("vfmax.vv v2, v28, v30");
    asm volatile("vfmax.vv v0, v0, v2");
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma"
                 :
                 : "r"(AVS_FAST_PIECE / 2u));
    asm volatile("vfmax.vv v0, v0, v1");
    asm volatile("vfredmax.vs v0, v0, v0");
  } else
#endif
  {
    asm volatile("vsetvli zero, %0, e16, m8, ta, ma" : : "r"(visible));
    asm volatile("vfredmax.vs v0, v24, v24");
  }
  asm volatile("vfmv.f.s %0, v0" : "=f"(mh));
  asm volatile("fcvt.s.h %0, %1" : "=f"(m) : "f"(mh));
  float n16;
  const float correction = avs_fast_ref2(state, m, k->l2e32, &n16);
#if Q38_AV_SWEEP_VFEXP >= 2
  if (visible == BK) {
    // Both halves through the (emulated) vfexp2, then the sum as one widening
    // add of the halves, an element-wise fold 128 -> 16 and a 16-element
    // reduction. FP32 sum order differs from the per-half reductions.
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(AVS_FAST_CHUNK));
    asm volatile("vfmv.v.f v4, %0" : : "f"(n16));
    asm volatile("vfmsac.vf v4, %0, v24" : : "f"(l2e));
    AVS_EXPE_M4("v8", "v4", "v16", "v20");
    asm volatile("vfmv.v.f v4, %0" : : "f"(n16));
    asm volatile("vfmsac.vf v4, %0, v28" : : "f"(l2e));
    AVS_EXPE_M4("v12", "v4", "v16", "v20");
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(AVS_FAST_PIECE));
    asm volatile("vse16.v v8, (%0)" : : "r"(row) : "memory");
    asm volatile("vse16.v v10, (%0)" : : "r"(row + AVS_FAST_PIECE) : "memory");
    asm volatile("vse16.v v12, (%0)"
                 :
                 : "r"(row + 2u * AVS_FAST_PIECE)
                 : "memory");
    asm volatile("vse16.v v14, (%0)"
                 :
                 : "r"(row + 3u * AVS_FAST_PIECE)
                 : "memory");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(AVS_FAST_CHUNK));
    asm volatile("vfwadd.vv v16, v8, v12");
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma"
                 :
                 : "r"(AVS_FAST_CHUNK / 2u));
    asm volatile("vfadd.vv v16, v16, v20");
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma"
                 :
                 : "r"(AVS_FAST_CHUNK / 4u));
    asm volatile("vfadd.vv v16, v16, v18");
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma"
                 :
                 : "r"(AVS_FAST_CHUNK / 8u));
    asm volatile("vfadd.vv v16, v16, v17");
    asm volatile("vfmv.s.f v0, %0" : : "f"(zero));
    asm volatile("vfredusum.vs v0, v16, v0");
    asm volatile("vfmv.f.s %0, v0" : "=f"(sum));
    state->denominator = correction * state->denominator + sum;
    return correction;
  }
#endif
#pragma clang loop unroll(full)
  for (uint32_t off = 0; off < BK; off += AVS_FAST_CHUNK) {
    if (off >= visible) {
      avs_fast_store(row + off, 0u, 1u);
      continue;
    }
    const uint32_t count = Q38_MIN(AVS_FAST_CHUNK, visible - off);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
    // t = s log2(e) - n with one rounding; clamp, split t = n2 + f.
    asm volatile("vfmv.v.f v0, %0" : : "f"(n16));
    if (off)
      asm volatile("vfmsac.vf v0, %0, v28" : : "f"(l2e));
    else
      asm volatile("vfmsac.vf v0, %0, v24" : : "f"(l2e));
#if Q38_AV_SWEEP_VFEXP
    // VFEXPF: e^t (stand-in timing only, unless EXP_POLY).
    AVS_EXPE_M4("v12", "v0", "v4", "v8");
#else
    asm volatile("vfmax.vf v0, v0, %0" : : "f"(low));
    asm volatile("vfcvt.x.f.v v4, v0");
    asm volatile("vfcvt.f.x.v v8, v4");
    asm volatile("vfsub.vv v8, v0, v8");
    asm volatile("vfmul.vf v12, v8, %0" : : "f"(c3));
    asm volatile("vfadd.vf v12, v12, %0" : : "f"(c2));
    asm volatile("vfmul.vv v12, v12, v8");
    asm volatile("vfadd.vf v12, v12, %0" : : "f"(c1));
    asm volatile("vfmul.vv v12, v12, v8");
    asm volatile("vfadd.vf v12, v12, %0" : : "f"(c0));
    // 2^n2 = 2^(n2+10) * 2^-10: n2 = -25 builds +0, and results below
    // 2^-14 still round to FP16 subnormals.
    asm volatile("vadd.vx v4, v4, %0" : : "r"(bias));
    asm volatile("vsll.vi v4, v4, 10");
    asm volatile("vfmul.vv v12, v12, v4");
    asm volatile("vfmul.vf v12, v12, %0" : : "f"(tiny));
#endif
    if (count < AVS_FAST_CHUNK)
      avs_fast_store(row + off, 0u, 1u);
    avs_fast_store(row + off, count, 0u);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
    asm volatile("vfwadd.vf v16, v12, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count));
    // v24 is dead once the first half is consumed; v27 carries the sum.
    if (!off)
      asm volatile("vfmv.s.f v27, %0" : : "f"(zero));
    asm volatile("vfredusum.vs v27, v16, v27");
  }
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(1u));
  asm volatile("vfmv.f.s %0, v27" : "=f"(sum));
  state->denominator = correction * state->denominator + sum;
  return correction;
}

#if Q38_AV_SWEEP_VFEXP == 3
// New per-row reference from the row maximum mh (FP16 bits in a float reg);
// returns the correction of the old state, as avs_fast_softmax_row.
static inline __attribute__((always_inline)) float
avs_fast_ref(qwen38_softmax_state_t *state, float mh, const avs_fast_k_t *k,
             float *n16) {
  float m;
  asm volatile("fcvt.s.h %0, %1" : "=f"(m) : "f"(mh));
  return avs_fast_ref2(state, m, k->l2e32, n16);
}

// Two full rows at once. Each row is one dependent chain (load, max,
// scalar reference, exp, store, sum, scalar); interleaving two chains lets
// one row's waits overlap the other's work. Row a: v16-v23, row b: v24-v31.
// The exp is the vfexp2 stand-in (timing only).
static inline __attribute__((always_inline)) void
avs_fast_softmax_pair(__fp16 *ra, __fp16 *rb, qwen38_softmax_state_t *sa,
                      qwen38_softmax_state_t *sb, const avs_fast_k_t *k,
                      float *alpha_a, float *alpha_b) {
  register float l2e asm("fs0") = k->l2e;
  register float zero asm("fs8") = k->zero;
  asm volatile("" : "+f"(l2e), "+f"(zero));
  float mha, mhb, n16a, n16b, suma, sumb;
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(AVS_FAST_PIECE));
  asm volatile("vle16.v v16, (%0)" : : "r"(ra) : "memory");
  asm volatile("vle16.v v18, (%0)" : : "r"(ra + AVS_FAST_PIECE) : "memory");
  asm volatile("vle16.v v20, (%0)"
               :
               : "r"(ra + 2u * AVS_FAST_PIECE)
               : "memory");
  asm volatile("vle16.v v22, (%0)"
               :
               : "r"(ra + 3u * AVS_FAST_PIECE)
               : "memory");
  asm volatile("vle16.v v24, (%0)" : : "r"(rb) : "memory");
  asm volatile("vle16.v v26, (%0)" : : "r"(rb + AVS_FAST_PIECE) : "memory");
  asm volatile("vle16.v v28, (%0)"
               :
               : "r"(rb + 2u * AVS_FAST_PIECE)
               : "memory");
  asm volatile("vle16.v v30, (%0)"
               :
               : "r"(rb + 3u * AVS_FAST_PIECE)
               : "memory");
  // Maxima: element-wise folds, then two 32-element reductions.
  asm volatile("vfmax.vv v0, v16, v18");
  asm volatile("vfmax.vv v4, v20, v22");
  asm volatile("vfmax.vv v2, v24, v26");
  asm volatile("vfmax.vv v6, v28, v30");
  asm volatile("vfmax.vv v0, v0, v4");
  asm volatile("vfmax.vv v2, v2, v6");
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma"
               :
               : "r"(AVS_FAST_PIECE / 2u));
  asm volatile("vfmax.vv v0, v0, v1");
  asm volatile("vfmax.vv v2, v2, v3");
  asm volatile("vfredmax.vs v0, v0, v0");
  asm volatile("vfredmax.vs v2, v2, v2");
  asm volatile("vfmv.f.s %0, v0" : "=f"(mha));
  asm volatile("vfmv.f.s %0, v2" : "=f"(mhb));
  *alpha_a = avs_fast_ref(sa, mha, k, &n16a);
  *alpha_b = avs_fast_ref(sb, mhb, k, &n16b);
  // Exponents, in place of the scores, and the stores.
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(AVS_FAST_CHUNK));
  asm volatile("vfmv.v.f v4, %0" : : "f"(n16a));
  asm volatile("vfmsac.vf v4, %0, v16" : : "f"(l2e));
  asm volatile("vfmv.v.f v8, %0" : : "f"(n16a));
  asm volatile("vfmsac.vf v8, %0, v20" : : "f"(l2e));
  asm volatile("vfmul.vv v16, v4, v4"); // vfexp2 stand-in
  asm volatile("vfmul.vv v20, v8, v8"); // vfexp2 stand-in
  asm volatile("vfmv.v.f v4, %0" : : "f"(n16b));
  asm volatile("vfmsac.vf v4, %0, v24" : : "f"(l2e));
  asm volatile("vfmv.v.f v8, %0" : : "f"(n16b));
  asm volatile("vfmsac.vf v8, %0, v28" : : "f"(l2e));
  asm volatile("vfmul.vv v24, v4, v4"); // vfexp2 stand-in
  asm volatile("vfmul.vv v28, v8, v8"); // vfexp2 stand-in
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(AVS_FAST_PIECE));
  asm volatile("vse16.v v16, (%0)" : : "r"(ra) : "memory");
  asm volatile("vse16.v v18, (%0)" : : "r"(ra + AVS_FAST_PIECE) : "memory");
  asm volatile("vse16.v v20, (%0)"
               :
               : "r"(ra + 2u * AVS_FAST_PIECE)
               : "memory");
  asm volatile("vse16.v v22, (%0)"
               :
               : "r"(ra + 3u * AVS_FAST_PIECE)
               : "memory");
  asm volatile("vse16.v v24, (%0)" : : "r"(rb) : "memory");
  asm volatile("vse16.v v26, (%0)" : : "r"(rb + AVS_FAST_PIECE) : "memory");
  asm volatile("vse16.v v28, (%0)"
               :
               : "r"(rb + 2u * AVS_FAST_PIECE)
               : "memory");
  asm volatile("vse16.v v30, (%0)"
               :
               : "r"(rb + 3u * AVS_FAST_PIECE)
               : "memory");
  // Sums: widening add of the halves, element-wise fold to 16, reduce.
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(AVS_FAST_CHUNK));
  asm volatile("vfwadd.vv v8, v16, v20");
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma"
               :
               : "r"(AVS_FAST_CHUNK / 2u));
  asm volatile("vfadd.vv v8, v8, v12");
  asm volatile("vsetvli zero, %0, e32, m2, ta, ma"
               :
               : "r"(AVS_FAST_CHUNK / 4u));
  asm volatile("vfadd.vv v8, v8, v10");
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma"
               :
               : "r"(AVS_FAST_CHUNK / 8u));
  asm volatile("vfadd.vv v8, v8, v9");
  asm volatile("vfmv.s.f v0, %0" : : "f"(zero));
  asm volatile("vfredusum.vs v0, v8, v0");
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(AVS_FAST_CHUNK));
  asm volatile("vfwadd.vv v8, v24, v28");
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma"
               :
               : "r"(AVS_FAST_CHUNK / 2u));
  asm volatile("vfadd.vv v8, v8, v12");
  asm volatile("vsetvli zero, %0, e32, m2, ta, ma"
               :
               : "r"(AVS_FAST_CHUNK / 4u));
  asm volatile("vfadd.vv v8, v8, v10");
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma"
               :
               : "r"(AVS_FAST_CHUNK / 8u));
  asm volatile("vfadd.vv v8, v8, v9");
  asm volatile("vfmv.s.f v2, %0" : : "f"(zero));
  asm volatile("vfredusum.vs v2, v8, v2");
  asm volatile("vfmv.f.s %0, v0" : "=f"(suma));
  asm volatile("vfmv.f.s %0, v2" : "=f"(sumb));
  sa->denominator = *alpha_a * sa->denominator + suma;
  sb->denominator = *alpha_b * sb->denominator + sumb;
}
#endif

// alpha is a power of two: an FP16 multiply is exact (barring underflow).
static inline __attribute__((always_inline)) void
avs_fast_rescale_row(__fp16 *z, float alpha) {
  const float a16 = llm_f16box(alpha);
#if VLEN >= 1024
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(Q38_HD));
#else
  asm volatile("vsetvli zero, %0, e16, m8, ta, ma" : : "r"(Q38_HD));
#endif
  asm volatile("vle16.v v0, (%0)" : : "r"(z) : "memory");
  asm volatile("vfmul.vf v0, v0, %0" : : "f"(a16));
  asm volatile("vse16.v v0, (%0)" : : "r"(z) : "memory");
}

#if Q38_AV_SWEEP_FINISH
// Fused gated output row: out = fp16(fp32(z * sigmoid(gate)) * inverse).
// sigmoid(x) = 1 / (1 + 2^u) for x >= 0 and 2^u / (1 + 2^u) for x < 0, with
// u = -|x| log2(e) <= 0, so the softmax row's FP16 exp2 applies. 1 / d on
// (1, 2] is two Newton steps from the seed 24/17 - 8/17 d (1.2e-5). The
// product z * sigmoid widens exactly to FP32 for the scale.
typedef struct {
  float nl2e, low, tiny, c0, c1, c2, c3, one, two, seed_a, seed_b, half;
  int32_t bias;
} avs_gate_k_t;

static inline __attribute__((always_inline)) void
avs_gate_init(avs_gate_k_t *k) {
#if Q38_AV_SWEEP_VFEXP && !Q38_AV_SWEEP_EXP_POLY
  k->nl2e = avs_hbits(0xbc00u); // -1: VFEXPF takes e^(-|x|) directly
#else
  k->nl2e = avs_hbits(0xbdc5u); // -log2(e)
#endif
  k->low = avs_hbits(0xce40u);  // -25
  k->tiny = avs_hbits(0x1400u); // 2^-10
  k->c0 = avs_hbits(0x3c00u);
  k->c1 = avs_hbits(0x398cu);
  k->c2 = avs_hbits(0x33c3u);
  k->c3 = avs_hbits(0x2b10u);
  k->one = avs_hbits(0x3c00u);
  k->two = avs_hbits(0x4000u);
  k->seed_a = avs_hbits(0x3da6u); // 24/17
  k->seed_b = avs_hbits(0xb788u); // -8/17
  k->half = avs_hbits(0x3800u);
  k->bias = 25;
}

static inline __attribute__((always_inline)) void
avs_gate_row(__fp16 *out, const __fp16 *z, const __fp16 *gate, float inverse,
             const avs_gate_k_t *k) {
  // Same model hazard as the softmax row: constants in their own registers,
  // loaded before the first vector instruction that reads them.
  register float nl2e asm("fs0") = k->nl2e;
  register float low asm("fs1") = k->low;
  register float tiny asm("fs2") = k->tiny;
  register float c0 asm("fs3") = k->c0;
  register float c1 asm("fs4") = k->c1;
  register float c2 asm("fs5") = k->c2;
  register float c3 asm("fs6") = k->c3;
  register float one asm("fs7") = k->one;
  register float two asm("fs8") = k->two;
  register float sa asm("fs9") = k->seed_a;
  register float sb asm("fs10") = k->seed_b;
  register float half asm("fs11") = k->half;
  float inv = inverse;
  int32_t bias = k->bias;
  asm volatile(""
               : "+f"(nl2e), "+f"(low), "+f"(tiny), "+f"(c0), "+f"(c1),
                 "+f"(c2), "+f"(c3), "+f"(one), "+f"(two), "+f"(sa), "+f"(sb),
                 "+f"(half), "+f"(inv), "+r"(bias));
  for (uint32_t off = 0u; off < Q38_HD;) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma"
                 : "=r"(vl)
                 : "r"(Q38_HD - off));
    asm volatile("vle16.v v24, (%0)" : : "r"(gate + off) : "memory");
    asm volatile("vle16.v v28, (%0)" : : "r"(z + off) : "memory");
    // u = -|x| log2(e), clamped; 2^u = 2^n2 p(f), f = u - n2.
    asm volatile("vfsgnjx.vv v4, v24, v24");
    asm volatile("vfmul.vf v4, v4, %0" : : "f"(nl2e));
#if Q38_AV_SWEEP_VFEXP && !Q38_AV_SWEEP_EXP_POLY
    // Timing stand-in for the FP16 VFEXPF (see the softmax row).
    asm volatile("vfmul.vv v12, v4, v4"); // e = e^u
#else
    asm volatile("vfmax.vf v4, v4, %0" : : "f"(low));
    asm volatile("vfcvt.x.f.v v8, v4");
    asm volatile("vfcvt.f.x.v v12, v8");
    asm volatile("vfsub.vv v12, v4, v12");
    asm volatile("vfmul.vf v16, v12, %0" : : "f"(c3));
    asm volatile("vfadd.vf v16, v16, %0" : : "f"(c2));
    asm volatile("vfmul.vv v16, v16, v12");
    asm volatile("vfadd.vf v16, v16, %0" : : "f"(c1));
    asm volatile("vfmul.vv v16, v16, v12");
    asm volatile("vfadd.vf v16, v16, %0" : : "f"(c0));
    asm volatile("vadd.vx v8, v8, %0" : : "r"(bias));
    asm volatile("vsll.vi v8, v8, 10");
    asm volatile("vfmul.vv v16, v16, v8");
    asm volatile("vfmul.vf v12, v16, %0" : : "f"(tiny)); // e = 2^u
#endif
    // r = 1 / (1 + e)
    asm volatile("vfadd.vf v16, v12, %0" : : "f"(one));
    asm volatile("vfmul.vf v20, v16, %0" : : "f"(sb));
    asm volatile("vfadd.vf v20, v20, %0" : : "f"(sa));
    asm volatile("vfmul.vv v4, v16, v20");
    asm volatile("vfrsub.vf v4, v4, %0" : : "f"(two));
    asm volatile("vfmul.vv v20, v20, v4");
    asm volatile("vfmul.vv v4, v16, v20");
    asm volatile("vfrsub.vf v4, v4, %0" : : "f"(two));
    asm volatile("vfmul.vv v20, v20, v4");
    // sigmoid = x < 0 ? q : r with q = e r, blended without masks (the
    // model traps on vmflt.vf): s = sgnj(1/2, x) + 1/2 is 1 or 0, and
    // sigmoid = q + s (r - q).
    asm volatile("vfmul.vv v4, v12, v20");
    asm volatile("vfsub.vv v8, v20, v4");
    asm volatile("vfmv.v.f v12, %0" : : "f"(half));
    asm volatile("vfsgnj.vv v12, v12, v24");
    asm volatile("vfadd.vf v12, v12, %0" : : "f"(half));
    asm volatile("vfmul.vv v8, v8, v12");
    asm volatile("vfadd.vv v8, v8, v4");
    asm volatile("vfwmul.vv v16, v28, v8");
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    asm volatile("vfmul.vf v16, v16, %0" : : "f"(inv));
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
    asm volatile("vfncvt.f.f.w v4, v16");
    asm volatile("vse16.v v4, (%0)" : : "r"(out + off) : "memory");
    off += vl;
  }
}
#endif
#endif
