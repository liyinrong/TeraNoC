// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __MEMPOOL_QWEN_SPATZ_OPS_H__
#define __MEMPOOL_QWEN_SPATZ_OPS_H__

#include <stdint.h>

#include "baremetal/mempool_llm_ffn_f16.h"

#define QSP_F16_LOCAL_ELEMS                                                    \
  (NUM_BANKS_PER_TILE * (sizeof(int32_t) / sizeof(__fp16)) / NUM_CORES_PER_TILE)
#define QSP_F16_ROW_ELEMS (NUM_BANKS * (sizeof(int32_t) / sizeof(__fp16)))
#define QSP_F32_LOCAL_ELEMS                                                    \
  (NUM_BANKS_PER_TILE * (sizeof(int32_t) / sizeof(float)) / NUM_CORES_PER_TILE)
#define QSP_F32_ROW_ELEMS (NUM_BANKS * (sizeof(int32_t) / sizeof(float)))

_Static_assert(QSP_F16_LOCAL_ELEMS == 64,
               "Qwen Spatz local f16 strip must be 64 elements");
_Static_assert(QSP_F32_LOCAL_ELEMS == 32,
               "Qwen Spatz local f32 strip must be 32 elements");

static inline __fp16 *qsp_f16_local(__fp16 *base, uint32_t core,
                                    uint32_t depth) {
  return base + core * QSP_F16_LOCAL_ELEMS + depth * QSP_F16_ROW_ELEMS;
}

static inline const __fp16 *qsp_f16_local_const(const __fp16 *base,
                                                uint32_t core, uint32_t depth) {
  return base + core * QSP_F16_LOCAL_ELEMS + depth * QSP_F16_ROW_ELEMS;
}

static inline float *qsp_f32_local(float *base, uint32_t core, uint32_t depth) {
  return base + core * QSP_F32_LOCAL_ELEMS + depth * QSP_F32_ROW_ELEMS;
}

// Register-blocked local residual/add. Eight m2 lanes consume all 32 vector
// registers on VLEN512; sixteen m1 lanes do so on VLEN1024.
static inline void qsp_add_local_f16(const __fp16 *a, const __fp16 *b,
                                     __fp16 *out, uint32_t core_id) {
#define QSP_ADD_LOAD(va, vb, d)                                                \
  asm volatile("vle16.v " #va ", (%0)"                                         \
               :                                                               \
               : "r"(qsp_f16_local_const(a, core_id, d))                       \
               : "memory");                                                    \
  asm volatile("vle16.v " #vb ", (%0)"                                         \
               :                                                               \
               : "r"(qsp_f16_local_const(b, core_id, d))                       \
               : "memory")
#define QSP_ADD_COMPUTE(va, vb) asm volatile("vfadd.vv " #vb ", " #va ", " #vb)
#define QSP_ADD_STORE(vb, d)                                                   \
  asm volatile("vse16.v " #vb ", (%0)"                                         \
               :                                                               \
               : "r"(qsp_f16_local(out, core_id, d))                           \
               : "memory")
#if VLEN == 512
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(64));
  QSP_ADD_LOAD(v0, v2, 0);
  QSP_ADD_LOAD(v4, v6, 1);
  QSP_ADD_LOAD(v8, v10, 2);
  QSP_ADD_LOAD(v12, v14, 3);
  QSP_ADD_LOAD(v16, v18, 4);
  QSP_ADD_LOAD(v20, v22, 5);
  QSP_ADD_LOAD(v24, v26, 6);
  QSP_ADD_LOAD(v28, v30, 7);
  QSP_ADD_COMPUTE(v0, v2);
  QSP_ADD_COMPUTE(v4, v6);
  QSP_ADD_COMPUTE(v8, v10);
  QSP_ADD_COMPUTE(v12, v14);
  QSP_ADD_COMPUTE(v16, v18);
  QSP_ADD_COMPUTE(v20, v22);
  QSP_ADD_COMPUTE(v24, v26);
  QSP_ADD_COMPUTE(v28, v30);
  QSP_ADD_STORE(v2, 0);
  QSP_ADD_STORE(v6, 1);
  QSP_ADD_STORE(v10, 2);
  QSP_ADD_STORE(v14, 3);
  QSP_ADD_STORE(v18, 4);
  QSP_ADD_STORE(v22, 5);
  QSP_ADD_STORE(v26, 6);
  QSP_ADD_STORE(v30, 7);
#elif VLEN == 1024
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(64));
  QSP_ADD_LOAD(v0, v1, 0);
  QSP_ADD_LOAD(v2, v3, 1);
  QSP_ADD_LOAD(v4, v5, 2);
  QSP_ADD_LOAD(v6, v7, 3);
  QSP_ADD_LOAD(v8, v9, 4);
  QSP_ADD_LOAD(v10, v11, 5);
  QSP_ADD_LOAD(v12, v13, 6);
  QSP_ADD_LOAD(v14, v15, 7);
  QSP_ADD_LOAD(v16, v17, 8);
  QSP_ADD_LOAD(v18, v19, 9);
  QSP_ADD_LOAD(v20, v21, 10);
  QSP_ADD_LOAD(v22, v23, 11);
  QSP_ADD_LOAD(v24, v25, 12);
  QSP_ADD_LOAD(v26, v27, 13);
  QSP_ADD_LOAD(v28, v29, 14);
  QSP_ADD_LOAD(v30, v31, 15);
  QSP_ADD_COMPUTE(v0, v1);
  QSP_ADD_COMPUTE(v2, v3);
  QSP_ADD_COMPUTE(v4, v5);
  QSP_ADD_COMPUTE(v6, v7);
  QSP_ADD_COMPUTE(v8, v9);
  QSP_ADD_COMPUTE(v10, v11);
  QSP_ADD_COMPUTE(v12, v13);
  QSP_ADD_COMPUTE(v14, v15);
  QSP_ADD_COMPUTE(v16, v17);
  QSP_ADD_COMPUTE(v18, v19);
  QSP_ADD_COMPUTE(v20, v21);
  QSP_ADD_COMPUTE(v22, v23);
  QSP_ADD_COMPUTE(v24, v25);
  QSP_ADD_COMPUTE(v26, v27);
  QSP_ADD_COMPUTE(v28, v29);
  QSP_ADD_COMPUTE(v30, v31);
  QSP_ADD_STORE(v1, 0);
  QSP_ADD_STORE(v3, 1);
  QSP_ADD_STORE(v5, 2);
  QSP_ADD_STORE(v7, 3);
  QSP_ADD_STORE(v9, 4);
  QSP_ADD_STORE(v11, 5);
  QSP_ADD_STORE(v13, 6);
  QSP_ADD_STORE(v15, 7);
  QSP_ADD_STORE(v17, 8);
  QSP_ADD_STORE(v19, 9);
  QSP_ADD_STORE(v21, 10);
  QSP_ADD_STORE(v23, 11);
  QSP_ADD_STORE(v25, 12);
  QSP_ADD_STORE(v27, 13);
  QSP_ADD_STORE(v29, 14);
  QSP_ADD_STORE(v31, 15);
#else
#error "Qwen Spatz suite supports VLEN 512 or 1024"
#endif
#undef QSP_ADD_LOAD
#undef QSP_ADD_COMPUTE
#undef QSP_ADD_STORE
}

static inline void qsp_silu_mul_local_f16(const __fp16 *g, const __fp16 *u,
                                          __fp16 *out, uint32_t core_id,
                                          uint32_t rounds) {
  for (uint32_t d = 0; d < rounds; ++d)
    llm_silu_mul_f16(qsp_f16_local_const(g, core_id, d),
                     qsp_f16_local_const(u, core_id, d),
                     qsp_f16_local(out, core_id, d), QSP_F16_LOCAL_ELEMS, 0, 1);
}

// Sigmoid, optionally multiplied by a second f16 vector. The m4->m8 widening
// path uses all four architectural m8 groups for input/result and exp/recip
// scratch while each load remains inside one tile-local 64-element strip.
static inline void qsp_sigmoid_local_f16(const __fp16 *gate, const __fp16 *mul,
                                         __fp16 *out, uint32_t core_id,
                                         uint32_t rounds) {
  const float ONE = 1.0f;
  const float ZERO_H = llm_f16box(0.0f);
  for (uint32_t d = 0; d < rounds; ++d) {
    const __fp16 *gp = qsp_f16_local_const(gate, core_id, d);
    __fp16 *op = qsp_f16_local(out, core_id, d);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(64));
    asm volatile("vle16.v v0, (%0)" : : "r"(gp) : "memory");
    asm volatile("vfwadd.vf v24, v0, %0" : : "f"(ZERO_H));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(64));
    asm volatile("vfsgnjn.vv v0, v24, v24");
    llm_rvv_expf_m8();
    asm volatile("vfadd.vf v0, v0, %0" : : "f"(ONE));
    llm_rvv_recip_pos_m8();
    if (mul != 0) {
      const __fp16 *mp = qsp_f16_local_const(mul, core_id, d);
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(64));
      asm volatile("vle16.v v0, (%0)" : : "r"(mp) : "memory");
      asm volatile("vfwadd.vf v16, v0, %0" : : "f"(ZERO_H));
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(64));
      asm volatile("vfmul.vv v8, v8, v16");
    }
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(64));
    asm volatile("vfncvt.f.f.w v0, v8");
    asm volatile("vse16.v v0, (%0)" : : "r"(op) : "memory");
  }
}

// Performance-oriented softplus approximation. y=exp(-abs(x)) is in [0,1];
// a fifth-order log1p series plus max(x,0) keeps the vector instruction and
// memory mix close to the model path. Numerical qualification is a later gate.
static inline void qsp_softplus_local_f16(const __fp16 *in, __fp16 *out,
                                          uint32_t core_id, uint32_t rounds) {
  const float ZERO_H = llm_f16box(0.0f);
  const float ZERO = 0.0f;
  const float C1 = 1.0f, C2 = -0.5f, C3 = 0.33333334f;
  const float C4 = -0.25f, C5 = 0.2f;
  for (uint32_t d = 0; d < rounds; ++d) {
    const __fp16 *ip = qsp_f16_local_const(in, core_id, d);
    __fp16 *op = qsp_f16_local(out, core_id, d);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(64));
    asm volatile("vle16.v v0, (%0)" : : "r"(ip) : "memory");
    asm volatile("vfwadd.vf v24, v0, %0" : : "f"(ZERO_H));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(64));
    asm volatile("vfsgnjx.vv v0, v24, v24");
    asm volatile("vfsgnjn.vv v0, v0, v0");
    llm_rvv_expf_m8();
    asm volatile("vfmv.v.f v8, %0" : : "f"(C5));
    asm volatile("vfmul.vv v8, v8, v0");
    asm volatile("vfadd.vf v8, v8, %0" : : "f"(C4));
    asm volatile("vfmul.vv v8, v8, v0");
    asm volatile("vfadd.vf v8, v8, %0" : : "f"(C3));
    asm volatile("vfmul.vv v8, v8, v0");
    asm volatile("vfadd.vf v8, v8, %0" : : "f"(C2));
    asm volatile("vfmul.vv v8, v8, v0");
    asm volatile("vfadd.vf v8, v8, %0" : : "f"(C1));
    asm volatile("vfmul.vv v8, v8, v0");
    asm volatile("vfmax.vf v16, v24, %0" : : "f"(ZERO));
    asm volatile("vfadd.vv v8, v8, v16");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(64));
    asm volatile("vfncvt.f.f.w v0, v8");
    asm volatile("vse16.v v0, (%0)" : : "r"(op) : "memory");
  }
}

static inline void qsp_rmsnorm_local_f16(const __fp16 *in, __fp16 *out,
                                         uint32_t core_id, uint32_t cols) {
  const uint32_t blocks = cols / QSP_F16_LOCAL_ELEMS;
  const float ZERO_H = llm_f16box(0.0f);
  const float ZERO_F = 0.0f;
  asm volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"(64));
  asm volatile("vmv.v.i v8, 0");
  for (uint32_t d = 0; d < blocks; ++d) {
    const __fp16 *ip = qsp_f16_local_const(in, core_id, d);
    asm volatile("vsetvli zero, %0, e16, m4, tu, ma" : : "r"(64));
    asm volatile("vle16.v v0, (%0)" : : "r"(ip) : "memory");
    asm volatile("vfwmacc.vv v8, v0, v0");
  }
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(64));
  asm volatile("vfmv.s.f v24, %0" : : "f"(ZERO_F));
  asm volatile("vfredusum.vs v24, v8, v24");
  float sumsq;
  asm volatile("vfmv.f.s %0, v24" : "=f"(sumsq));
  const float rinv =
      llm_f16box(llm_rsqrt(sumsq * llm_recip_pos((float)cols) + 1e-6f));
  for (uint32_t d = 0; d < blocks; ++d) {
    const __fp16 *ip = qsp_f16_local_const(in, core_id, d);
    __fp16 *op = qsp_f16_local(out, core_id, d);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(64));
    asm volatile("vle16.v v0, (%0)" : : "r"(ip) : "memory");
    asm volatile("vfmul.vf v0, v0, %0" : : "f"(rinv));
    asm volatile("vse16.v v0, (%0)" : : "r"(op) : "memory");
  }
  (void)ZERO_H;
}

static inline void qsp_softmax_local_f16(__fp16 *data, uint32_t core_id,
                                         uint32_t cols) {
  const uint32_t blocks = cols / QSP_F16_LOCAL_ELEMS;
  const float ZERO_H = llm_f16box(0.0f);
  const float NEG_BIG = -1e30f;
  const float ZERO_F = 0.0f;
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(64));
  asm volatile("vfmv.s.f v24, %0" : : "f"(NEG_BIG));
  for (uint32_t d = 0; d < blocks; ++d) {
    __fp16 *p = qsp_f16_local(data, core_id, d);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(64));
    asm volatile("vle16.v v8, (%0)" : : "r"(p) : "memory");
    asm volatile("vfwadd.vf v0, v8, %0" : : "f"(ZERO_H));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(64));
    asm volatile("vfredmax.vs v24, v0, v24");
  }
  float maxv;
  asm volatile("vfmv.f.s %0, v24" : "=f"(maxv));
  asm volatile("vfmv.s.f v24, %0" : : "f"(ZERO_F));
  for (uint32_t d = 0; d < blocks; ++d) {
    __fp16 *p = qsp_f16_local(data, core_id, d);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(64));
    asm volatile("vle16.v v8, (%0)" : : "r"(p) : "memory");
    asm volatile("vfwadd.vf v0, v8, %0" : : "f"(ZERO_H));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(64));
    asm volatile("vfsub.vf v0, v0, %0" : : "f"(maxv));
    llm_rvv_expf_m8();
    asm volatile("vfredusum.vs v24, v0, v24");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(64));
    asm volatile("vfncvt.f.f.w v8, v0");
    asm volatile("vse16.v v8, (%0)" : : "r"(p) : "memory");
  }
  float sum;
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(64));
  asm volatile("vfmv.f.s %0, v24" : "=f"(sum));
  const float inv = llm_f16box(llm_recip_pos(sum));
  for (uint32_t d = 0; d < blocks; ++d) {
    __fp16 *p = qsp_f16_local(data, core_id, d);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(64));
    asm volatile("vle16.v v0, (%0)" : : "r"(p) : "memory");
    asm volatile("vfmul.vf v0, v0, %0" : : "f"(inv));
    asm volatile("vse16.v v0, (%0)" : : "r"(p) : "memory");
  }
}

static inline void qsp_rope_local_f16(__fp16 *data, const __fp16 *cosv,
                                      const __fp16 *sinv, uint32_t core_id,
                                      uint32_t rounds) {
  const intptr_t stride = 2 * (intptr_t)sizeof(__fp16);
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(32));
  for (uint32_t d = 0; d < rounds; d += 4) {
#define QSP_ROPE_LOAD(vx0, vx1, vc, vs, lane)                                  \
  asm volatile("vlse16.v " #vx0 ", (%0), %1"                                   \
               :                                                               \
               : "r"(qsp_f16_local(data, core_id, d + lane)), "r"(stride)      \
               : "memory");                                                    \
  asm volatile("vlse16.v " #vx1 ", (%0), %1"                                   \
               :                                                               \
               : "r"(qsp_f16_local(data, core_id, d + lane) + 1), "r"(stride)  \
               : "memory");                                                    \
  asm volatile("vle16.v " #vc ", (%0)"                                         \
               :                                                               \
               : "r"(qsp_f16_local_const(cosv, core_id, d + lane))             \
               : "memory");                                                    \
  asm volatile("vle16.v " #vs ", (%0)"                                         \
               :                                                               \
               : "r"(qsp_f16_local_const(sinv, core_id, d + lane))             \
               : "memory")
#define QSP_ROPE_COMPUTE(vx0, vx1, vc, vs, vy0, vy1)                           \
  asm volatile("vfmul.vv " #vy0 ", " #vx0 ", " #vc);                           \
  asm volatile("vfnmsac.vv " #vy0 ", " #vx1 ", " #vs);                         \
  asm volatile("vfmul.vv " #vy1 ", " #vx0 ", " #vs);                           \
  asm volatile("vfmacc.vv " #vy1 ", " #vx1 ", " #vc)
#define QSP_ROPE_STORE(vy0, vy1, lane)                                         \
  asm volatile("vsse16.v " #vy0 ", (%0), %1"                                   \
               :                                                               \
               : "r"(qsp_f16_local(data, core_id, d + lane)), "r"(stride)      \
               : "memory");                                                    \
  asm volatile("vsse16.v " #vy1 ", (%0), %1"                                   \
               :                                                               \
               : "r"(qsp_f16_local(data, core_id, d + lane) + 1), "r"(stride)  \
               : "memory")
    QSP_ROPE_LOAD(v0, v1, v2, v3, 0);
    QSP_ROPE_LOAD(v4, v5, v6, v7, 1);
    QSP_ROPE_LOAD(v8, v9, v10, v11, 2);
    QSP_ROPE_LOAD(v12, v13, v14, v15, 3);
    QSP_ROPE_COMPUTE(v0, v1, v2, v3, v16, v17);
    QSP_ROPE_COMPUTE(v4, v5, v6, v7, v18, v19);
    QSP_ROPE_COMPUTE(v8, v9, v10, v11, v20, v21);
    QSP_ROPE_COMPUTE(v12, v13, v14, v15, v22, v23);
    QSP_ROPE_STORE(v16, v17, 0);
    QSP_ROPE_STORE(v18, v19, 1);
    QSP_ROPE_STORE(v20, v21, 2);
    QSP_ROPE_STORE(v22, v23, 3);
#undef QSP_ROPE_LOAD
#undef QSP_ROPE_COMPUTE
#undef QSP_ROPE_STORE
  }
}

static inline void qsp_scale_mask_local_f16(__fp16 *data, uint32_t core_id,
                                            uint32_t rounds) {
  const float scale = llm_f16box(0.0625f);
  const float mask = llm_f16box(-80.0f);
  for (uint32_t d = 0; d < rounds; ++d) {
    __fp16 *p = qsp_f16_local(data, core_id, d);
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(64));
    asm volatile("vle16.v v0, (%0)" : : "r"(p) : "memory");
    if (d + 1 == rounds)
      asm volatile("vfmv.v.f v0, %0" : : "f"(mask));
    else
      asm volatile("vfmul.vf v0, v0, %0" : : "f"(scale));
    asm volatile("vse16.v v0, (%0)" : : "r"(p) : "memory");
  }
}

static inline void qsp_conv4_local_f16(const __fp16 *in, __fp16 *out,
                                       uint32_t core_id,
                                       uint32_t output_steps) {
  const float w0 = llm_f16box(0.125f), w1 = llm_f16box(0.25f);
  const float w2 = llm_f16box(0.375f), w3 = llm_f16box(0.5f);
#if VLEN == 512
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(64));
#else
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(64));
#endif
  for (uint32_t t = 0; t < output_steps; ++t) {
    const __fp16 *p0 = qsp_f16_local_const(in, core_id, t + 0);
    const __fp16 *p1 = qsp_f16_local_const(in, core_id, t + 1);
    const __fp16 *p2 = qsp_f16_local_const(in, core_id, t + 2);
    const __fp16 *p3 = qsp_f16_local_const(in, core_id, t + 3);
    __fp16 *op = qsp_f16_local(out, core_id, t);
    asm volatile("vle16.v v0, (%0)" : : "r"(p0) : "memory");
    asm volatile("vle16.v v2, (%0)" : : "r"(p1) : "memory");
    asm volatile("vle16.v v4, (%0)" : : "r"(p2) : "memory");
    asm volatile("vle16.v v6, (%0)" : : "r"(p3) : "memory");
    asm volatile("vfmul.vf v8, v0, %0" : : "f"(w0));
    asm volatile("vfmacc.vf v8, %0, v2" : : "f"(w1));
    asm volatile("vfmacc.vf v8, %0, v4" : : "f"(w2));
    asm volatile("vfmacc.vf v8, %0, v6" : : "f"(w3));
    asm volatile("vse16.v v8, (%0)" : : "r"(op) : "memory");
  }
}

// One useful GDN recurrent update over a tile-local 32-column FP32 state
// block. Each core owns all 128 K rows for its V columns, so no communication
// occurs inside the recurrence.
static inline void qsp_delta_rule_local_f32(float *state, float *out,
                                            uint32_t core_id) {
  const float lambda = 0.984375f;
  const float kval = 0.0078125f;
  const float qval = 0.0078125f;
  const float beta = 0.5f;
  const float value = 0.25f;
#if VLEN == 512
  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(32));
#else
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(32));
#endif
  asm volatile("vmv.v.i v8, 0");
  for (uint32_t k = 0; k < 128; ++k) {
    float *sp = qsp_f32_local(state, core_id, k);
    asm volatile("vle32.v v0, (%0)" : : "r"(sp) : "memory");
    asm volatile("vfmul.vf v0, v0, %0" : : "f"(lambda));
    asm volatile("vse32.v v0, (%0)" : : "r"(sp) : "memory");
    asm volatile("vfmacc.vf v8, %0, v0" : : "f"(kval));
  }
  asm volatile("vfrsub.vf v8, v8, %0" : : "f"(value));
  asm volatile("vfmul.vf v8, v8, %0" : : "f"(beta));
  asm volatile("vmv.v.i v12, 0");
  for (uint32_t k = 0; k < 128; ++k) {
    float *sp = qsp_f32_local(state, core_id, k);
    asm volatile("vle32.v v0, (%0)" : : "r"(sp) : "memory");
    asm volatile("vfmacc.vf v0, %0, v8" : : "f"(kval));
    asm volatile("vse32.v v0, (%0)" : : "r"(sp) : "memory");
    asm volatile("vfmacc.vf v12, %0, v0" : : "f"(qval));
  }
  asm volatile("vse32.v v12, (%0)"
               :
               : "r"(qsp_f32_local(out, core_id, 0))
               : "memory");
}

#endif // __MEMPOOL_QWEN_SPATZ_OPS_H__
