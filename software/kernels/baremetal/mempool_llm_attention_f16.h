// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yinrong Li, ETH Zurich

// Dense decoder-only LLM attention block for the TeraNoC RedMulE + Spatz
// cluster. The layout follows the operation table in dense_workload.md:
//
//   h       = RMSNorm(x)
//   q/k/v   = h @ Wq/Wk/Wv
//   q,k     = optional QK RMSNorm, then RoPE
//   scores  = q @ k^T, scaled and causal-masked
//   probs   = softmax(scores)
//   ctx     = probs @ v
//   out     = ctx @ Wo + x
//
// Q, K, V, and context are stored head-panel-major:
//   [num_heads][seqlen][head_dim]
// K cache for RedMulE QK^T is stored transposed per KV head:
//   [num_kv_heads][head_dim][kv_len]

#pragma once
#include <stdint.h>

#include "dma.h"
#include "runtime.h"
#include "synchronization.h"

#include "archi_redmule.h"
#include "hal_redmule.h"

#include "baremetal/mempool_llm_ffn_f16.h"
#include "baremetal/redmule_xshift.h"

#ifndef RVV
#error "mempool_llm_attention_f16.h requires a Spatz (RVV) configuration"
#endif

#if NUM_REDMULE_TILES < 1
#error "mempool_llm_attention_f16.h requires a RedMulE configuration"
#endif

#ifndef LLM_ATTENTION_GEMM_FMT
#define LLM_ATTENTION_GEMM_FMT Float16
#endif

#if defined(LLM_FLASH_TILED) && !defined(LLM_FLASH_ATTENTION)
#error "LLM_FLASH_TILED requires LLM_FLASH_ATTENTION"
#endif

#define LLM_ATTENTION_PORT_WIDTH (REDMULE_H * (REDMULE_P + 1))

static inline uint32_t llm_attention_ne_used(uint32_t seqlen) {
  uint32_t nblk = seqlen / LLM_ATTENTION_PORT_WIDTH;
  uint32_t ne_used = NUM_REDMULE_TILES;
  if (nblk == 0)
    return 1;
  if (ne_used > nblk)
    ne_used = nblk;
  while (ne_used > 1 && (nblk % ne_used) != 0)
    ne_used--;
  return ne_used;
}

// RedMulE X contraction-start offset. Without it every engine walks the
// contraction from element 0, so after a barrier they all address the same L1
// tile and drain through it one at a time; the plan measures ~14 utilization
// points for this on a power-of-two contraction. It changes only the summation
// order, so it is always safe -- hence on by default.
//
// `n` is the contraction length. redmule_x_shift() returns 0 when the
// contraction is shorter than two PORT_WIDTH accesses, so a short-K job (e.g.
// the FFN's second GEMM, where the contraction is exactly one PORT_WIDTH) is
// left untouched automatically and costs nothing.
#ifndef LLM_RM_X_SHIFT
#define LLM_RM_X_SHIFT 1
#endif

static inline void llm_attention_redmule_job(unsigned int x, unsigned int w,
                                             unsigned int z, uint16_t m,
                                             uint16_t n, uint16_t k,
                                             uint8_t op) {
  hwpe_soft_clear();
  mempool_wait(10);
  redmule_cfg(x, w, z, m, n, k, 0, op, LLM_ATTENTION_GEMM_FMT);
#if LLM_RM_X_SHIFT
  // Derived from the calling core, so no call site has to thread an engine id.
  redmule_x_shift_set(redmule_x_shift(mempool_get_redmule_id(), (uint32_t)n,
                                      LLM_ATTENTION_PORT_WIDTH, sizeof(__fp16),
                                      NUM_BANKS_PER_TILE, NUM_BANKS));
#endif
  mempool_wait(10);
  hwpe_trigger_job();
  mempool_wfi();
}

static inline uint32_t llm_attention_min_u32(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}

static inline uint32_t llm_attention_vlmax_e16_m8(void) {
  const uint32_t max_avl = UINT32_MAX;
  uint32_t vlmax;
  asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vlmax) : "r"(max_avl));
  return vlmax;
}

static inline void llm_attention_copy_f16_m8(const __fp16 *src, __fp16 *dst,
                                             uint32_t count) {
  asm volatile("vsetvli zero, %0, e16, m8, ta, ma" : : "r"(count));
  asm volatile("vle16.v v0, (%0)" : : "r"(src) : "memory");
  asm volatile("vse16.v v0, (%0)" : : "r"(dst) : "memory");
}

// out_pm[heads][S x HEAD_DIM] = h[S x D] @ w_pm[heads][D x HEAD_DIM].
static inline void
llm_attention_proj_heads(const __fp16 *l2_w_pm, const __fp16 *l1_h,
                         __fp16 *l1_out_pm, __fp16 *l1_wbuf, uint32_t seqlen,
                         uint32_t dmodel, uint32_t head_dim, uint32_t num_heads,
                         uint32_t core_id, uint32_t num_cores,
                         uint32_t redmule_id, uint32_t ne_used) {
  const uint32_t rows_per_engine = seqlen / ne_used;
  const uint32_t panel_elems = dmodel * head_dim;
  const uint32_t out_panel_elems = seqlen * head_dim;
  __fp16 *wbuf0 = l1_wbuf;
  __fp16 *wbuf1 = l1_wbuf + panel_elems;

  if (core_id == 0)
    dma_memcpy_blocking(wbuf0, l2_w_pm, panel_elems * sizeof(__fp16));
  mempool_barrier(num_cores);

  for (uint32_t h = 0; h < num_heads; h++) {
    __fp16 *wcur = (h & 1) ? wbuf1 : wbuf0;
    __fp16 *wnxt = (h & 1) ? wbuf0 : wbuf1;

    if (core_id == 0 && (h + 1) < num_heads)
      dma_memcpy_nonblocking(wnxt, l2_w_pm + (h + 1) * panel_elems,
                             panel_elems * sizeof(__fp16));

    if (redmule_id < ne_used) {
      unsigned int x =
          (unsigned int)(l1_h + redmule_id * rows_per_engine * dmodel);
      unsigned int w = (unsigned int)wcur;
      unsigned int z = (unsigned int)(l1_out_pm + h * out_panel_elems +
                                      redmule_id * rows_per_engine * head_dim);
      llm_zero_f16((__fp16 *)z, rows_per_engine * head_dim);
      llm_attention_redmule_job(x, w, z, (uint16_t)rows_per_engine,
                                (uint16_t)dmodel, (uint16_t)head_dim, GEMM);
    }

    mempool_barrier(num_cores);
    if (core_id == 0)
      dma_wait();
    mempool_barrier(num_cores);
  }
}

// L1-constrained projection variant. A real Llama head panel is 1 MiB, so
// double buffering consumes 2 MiB before accounting for activations and other
// working buffers. Tiled attention deliberately stages one panel at a time.
static inline void llm_attention_proj_heads_single_f16(
    const __fp16 *l2_w_pm, const __fp16 *l1_h, __fp16 *l1_out_pm,
    __fp16 *l1_wbuf, uint32_t seqlen, uint32_t dmodel, uint32_t head_dim,
    uint32_t num_heads, uint32_t core_id, uint32_t num_cores,
    uint32_t redmule_id, uint32_t ne_used) {
  const uint32_t rows_per_engine = seqlen / ne_used;
  const uint32_t panel_elems = dmodel * head_dim;
  const uint32_t out_panel_elems = seqlen * head_dim;

  for (uint32_t h = 0; h < num_heads; h++) {
    if (core_id == 0)
      dma_memcpy_blocking(l1_wbuf, l2_w_pm + h * panel_elems,
                          panel_elems * sizeof(__fp16));
    mempool_barrier(num_cores);

    if (redmule_id < ne_used) {
      unsigned int x =
          (unsigned int)(l1_h + redmule_id * rows_per_engine * dmodel);
      unsigned int z = (unsigned int)(l1_out_pm + h * out_panel_elems +
                                      redmule_id * rows_per_engine * head_dim);
      llm_zero_f16((__fp16 *)z, rows_per_engine * head_dim);
      llm_attention_redmule_job(x, (unsigned int)l1_wbuf, z,
                                (uint16_t)rows_per_engine, (uint16_t)dmodel,
                                (uint16_t)head_dim, GEMM);
    }
    mempool_barrier(num_cores);
  }
}

static inline void llm_attention_qk_norm_f16(
    __fp16 *q_pm, __fp16 *k_pm, const __fp16 *q_gamma, const __fp16 *k_gamma,
    uint32_t seqlen, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t core_id, uint32_t num_cores) {
  llm_rmsnorm_f16(q_pm, q_gamma, q_pm, num_q_heads * seqlen, head_dim, core_id,
                  num_cores);
  mempool_barrier(num_cores);
  llm_rmsnorm_f16(k_pm, k_gamma, k_pm, num_kv_heads * seqlen, head_dim, core_id,
                  num_cores);
  mempool_barrier(num_cores);
}

static inline void llm_attention_rope_f16(__fp16 *x_pm, const __fp16 *cos_tbl,
                                          const __fp16 *sin_tbl,
                                          uint32_t seqlen, uint32_t num_heads,
                                          uint32_t head_dim, uint32_t core_id,
                                          uint32_t num_cores) {
  const uint32_t half_dim = head_dim >> 1;
  const uint32_t rows = num_heads * seqlen;
  const intptr_t pair_stride = 2 * sizeof(__fp16);
  const float ZERO_H = llm_f16box(0.0f);

  // Six e32,m4 groups hold x0/x1, cos/sin, and the two results. Their paired
  // e16,m2 groups process 64 rotary pairs per iteration at VLEN=512, exactly a
  // Llama head's half dimension, while preserving f32 arithmetic.
  for (uint32_t row = core_id; row < rows; row += num_cores) {
    const uint32_t token = row % seqlen;
    __fp16 *x0p = x_pm + row * head_dim;
    __fp16 *x1p = x0p + 1;
    const __fp16 *cp = cos_tbl + token * half_dim;
    const __fp16 *sp = sin_tbl + token * half_dim;
    uint32_t avl = half_dim;

    while (avl > 0) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(vl) : "r"(avl));
      asm volatile("vlse16.v v0, (%0), %1"
                   :
                   : "r"(x0p), "r"(pair_stride)
                   : "memory");
      asm volatile("vlse16.v v2, (%0), %1"
                   :
                   : "r"(x1p), "r"(pair_stride)
                   : "memory");
      asm volatile("vle16.v v4, (%0)" : : "r"(cp) : "memory");
      asm volatile("vle16.v v6, (%0)" : : "r"(sp) : "memory");
      asm volatile("vfwadd.vf v8, v0, %0" : : "f"(ZERO_H));
      asm volatile("vfwadd.vf v12, v2, %0" : : "f"(ZERO_H));
      asm volatile("vfwadd.vf v16, v4, %0" : : "f"(ZERO_H));
      asm volatile("vfwadd.vf v20, v6, %0" : : "f"(ZERO_H));

      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl));
      asm volatile("vfmul.vv v24, v12, v20");
      asm volatile("vfmsac.vv v24, v8, v16");
      asm volatile("vfmul.vv v28, v12, v16");
      asm volatile("vfmacc.vv v28, v8, v20");

      asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl));
      asm volatile("vfncvt.f.f.w v0, v24");
      asm volatile("vfncvt.f.f.w v2, v28");
      asm volatile("vsse16.v v0, (%0), %1"
                   :
                   : "r"(x0p), "r"(pair_stride)
                   : "memory");
      asm volatile("vsse16.v v2, (%0), %1"
                   :
                   : "r"(x1p), "r"(pair_stride)
                   : "memory");

      x0p += 2 * vl;
      x1p += 2 * vl;
      cp += vl;
      sp += vl;
      avl -= vl;
    }
  }
}

static inline void
llm_attention_k_to_kt_f16(const __fp16 *k_pm, __fp16 *kt_pm, uint32_t seqlen,
                          uint32_t num_kv_heads, uint32_t head_dim,
                          uint32_t core_id, uint32_t num_cores) {
  const uint32_t max_avl = UINT32_MAX;

  if (seqlen >= head_dim) {
    // Long classic key tiles: vectorize across tokens. K is strided and K^T is
    // contiguous. Split each (head,dimension) row into VLMAX-sized tasks so
    // long rows occupy all cores rather than only head_dim cores.
    uint32_t vlmax;
    asm volatile("vsetvli %0, %1, e16, m8, ta, ma"
                 : "=r"(vlmax)
                 : "r"(max_avl));
    const uint32_t blocks = (seqlen + vlmax - 1) / vlmax;
    const uint32_t tasks = num_kv_heads * head_dim * blocks;
    const intptr_t src_stride = (intptr_t)head_dim * (intptr_t)sizeof(__fp16);

    for (uint32_t task = core_id; task < tasks; task += num_cores) {
      const uint32_t block = task % blocks;
      const uint32_t hd = task / blocks;
      const uint32_t d = hd % head_dim;
      const uint32_t head = hd / head_dim;
      const uint32_t token0 = block * vlmax;
      const uint32_t count = llm_attention_min_u32(vlmax, seqlen - token0);
      const __fp16 *src = k_pm + (head * seqlen + token0) * head_dim + d;
      __fp16 *dst = kt_pm + hd * seqlen + token0;
      asm volatile("vsetvli zero, %0, e16, m8, ta, ma" : : "r"(count));
      asm volatile("vlse16.v v0, (%0), %1"
                   :
                   : "r"(src), "r"(src_stride)
                   : "memory");
      asm volatile("vse16.v v0, (%0)" : : "r"(dst) : "memory");
    }
  } else {
    // Short Llama key tiles: vectorize the contiguous head dimension and use a
    // strided store into K^T. e16,m4 holds HeadDim=128 in one vector.
    uint32_t vlmax;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma"
                 : "=r"(vlmax)
                 : "r"(max_avl));
    const uint32_t blocks = (head_dim + vlmax - 1) / vlmax;
    const uint32_t tasks = num_kv_heads * seqlen * blocks;
    const intptr_t dst_stride = (intptr_t)seqlen * (intptr_t)sizeof(__fp16);

    for (uint32_t task = core_id; task < tasks; task += num_cores) {
      const uint32_t block = task % blocks;
      const uint32_t ht = task / blocks;
      const uint32_t token = ht % seqlen;
      const uint32_t head = ht / seqlen;
      const uint32_t d0 = block * vlmax;
      const uint32_t count = llm_attention_min_u32(vlmax, head_dim - d0);
      const __fp16 *src = k_pm + (head * seqlen + token) * head_dim + d0;
      __fp16 *dst = kt_pm + (head * head_dim + d0) * seqlen + token;
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
      asm volatile("vle16.v v0, (%0)" : : "r"(src) : "memory");
      asm volatile("vsse16.v v0, (%0), %1"
                   :
                   : "r"(dst), "r"(dst_stride)
                   : "memory");
    }
  }
}

static inline void llm_attention_stage_decode_cache_f16(
    const __fp16 *l2_k_cache_t, const __fp16 *l2_v_cache, __fp16 *l1_k_cache_t,
    __fp16 *l1_v_cache, uint32_t past_len, uint32_t total_len,
    uint32_t num_kv_heads, uint32_t head_dim, uint32_t core_id,
    uint32_t num_cores) {
  (void)num_cores;
  if (core_id != 0 || past_len == 0)
    return;

  // Spatz vector ports only address TCDM; L2/SoC traffic must use the scalar
  // port or DMA. K is transposed in both caches, so each (head,dimension) row
  // is one contiguous DMA transfer despite the different row strides.
  const uint32_t kt_rows = num_kv_heads * head_dim;
  for (uint32_t hd = 0; hd < kt_rows; hd++) {
    dma_memcpy_blocking(l1_k_cache_t + hd * total_len,
                        l2_k_cache_t + hd * past_len,
                        past_len * sizeof(__fp16));
  }

  // Each V head is token-major and therefore one contiguous prefix in both
  // layouts, although consecutive destination heads have a larger stride.
  const uint32_t v_head_elems = past_len * head_dim;
  for (uint32_t head = 0; head < num_kv_heads; head++) {
    dma_memcpy_blocking(l1_v_cache + head * total_len * head_dim,
                        l2_v_cache + head * v_head_elems,
                        v_head_elems * sizeof(__fp16));
  }
}

static inline void
llm_attention_append_kv_f16(const __fp16 *k_new_pm, const __fp16 *v_new_pm,
                            __fp16 *kt_cache_pm, __fp16 *v_cache_pm,
                            uint32_t past_len, uint32_t seqlen,
                            uint32_t num_kv_heads, uint32_t head_dim,
                            uint32_t core_id, uint32_t num_cores) {
  const uint32_t total_len = past_len + seqlen;
  const uint32_t vlmax = llm_attention_vlmax_e16_m8();
  const uint32_t blocks = (head_dim + vlmax - 1) / vlmax;
  const uint32_t tasks = num_kv_heads * seqlen * blocks;
  const intptr_t kt_stride = (intptr_t)total_len * (intptr_t)sizeof(__fp16);

  // A task owns one contiguous slice of a new token. V stays contiguous while
  // K is scattered across the transposed destination with one vector store.
  for (uint32_t task = core_id; task < tasks; task += num_cores) {
    const uint32_t block = task % blocks;
    const uint32_t ht = task / blocks;
    const uint32_t token = ht % seqlen;
    const uint32_t head = ht / seqlen;
    const uint32_t d0 = block * vlmax;
    const uint32_t count = llm_attention_min_u32(vlmax, head_dim - d0);
    const uint32_t cache_token = past_len + token;
    const __fp16 *ksrc = k_new_pm + ht * head_dim + d0;
    const __fp16 *vsrc = v_new_pm + ht * head_dim + d0;
    __fp16 *kdst =
        kt_cache_pm + (head * head_dim + d0) * total_len + cache_token;
    __fp16 *vdst =
        v_cache_pm + (head * total_len + cache_token) * head_dim + d0;

    asm volatile("vsetvli zero, %0, e16, m8, ta, ma" : : "r"(count));
    asm volatile("vle16.v v0, (%0)" : : "r"(ksrc) : "memory");
    asm volatile("vle16.v v8, (%0)" : : "r"(vsrc) : "memory");
    asm volatile("vsse16.v v0, (%0), %1"
                 :
                 : "r"(kdst), "r"(kt_stride)
                 : "memory");
    asm volatile("vse16.v v8, (%0)" : : "r"(vdst) : "memory");
  }
}

static inline void llm_attention_select_window_f16(
    const __fp16 *kt_cache_pm, const __fp16 *v_cache_pm, __fp16 *kt_window_pm,
    __fp16 *v_window_pm, uint32_t total_len, uint32_t first_token,
    uint32_t attn_len, uint32_t num_kv_heads, uint32_t head_dim,
    uint32_t core_id, uint32_t num_cores) {
  if (attn_len == 0)
    return;

  const uint32_t vlmax = llm_attention_vlmax_e16_m8();

  // K and K^T are both dimension-major, so every selected window is a
  // contiguous slice within one (head,dimension) row.
  const uint32_t kt_blocks = (attn_len + vlmax - 1) / vlmax;
  const uint32_t kt_tasks = num_kv_heads * head_dim * kt_blocks;
  for (uint32_t task = core_id; task < kt_tasks; task += num_cores) {
    const uint32_t block = task % kt_blocks;
    const uint32_t hd = task / kt_blocks;
    const uint32_t col0 = block * vlmax;
    const uint32_t count = llm_attention_min_u32(vlmax, attn_len - col0);
    const __fp16 *src = kt_cache_pm + hd * total_len + first_token + col0;
    __fp16 *dst = kt_window_pm + hd * attn_len + col0;
    llm_attention_copy_f16_m8(src, dst, count);
  }

  // V is token-major, making the whole selected window contiguous per head.
  const uint32_t v_window_elems = attn_len * head_dim;
  const uint32_t v_blocks = (v_window_elems + vlmax - 1) / vlmax;
  const uint32_t v_tasks = num_kv_heads * v_blocks;
  for (uint32_t task = core_id; task < v_tasks; task += num_cores) {
    const uint32_t block = task % v_blocks;
    const uint32_t head = task / v_blocks;
    const uint32_t elem0 = block * vlmax;
    const uint32_t count = llm_attention_min_u32(vlmax, v_window_elems - elem0);
    const __fp16 *src =
        v_cache_pm + (head * total_len + first_token) * head_dim + elem0;
    __fp16 *dst = v_window_pm + head * v_window_elems + elem0;
    llm_attention_copy_f16_m8(src, dst, count);
  }
}

static inline void llm_attention_qk_matmul_f16(
    const __fp16 *q_pm, const __fp16 *kt_pm, __fp16 *scores_pm, uint32_t seqlen,
    uint32_t attn_len, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t num_cores, uint32_t redmule_id) {
  const uint32_t group = num_q_heads / num_kv_heads;
  const uint32_t q_head_elems = seqlen * head_dim;
  const uint32_t kt_head_elems = head_dim * attn_len;
  const uint32_t score_head_elems = seqlen * attn_len;
  const uint32_t num_redmules = mempool_get_redmule_count();

  if (redmule_id < num_redmules) {
    for (uint32_t qh = redmule_id; qh < num_q_heads; qh += num_redmules) {
      uint32_t kvh = qh / group;
      unsigned int q = (unsigned int)(q_pm + qh * q_head_elems);
      unsigned int kt = (unsigned int)(kt_pm + kvh * kt_head_elems);
      unsigned int s = (unsigned int)(scores_pm + qh * score_head_elems);
      llm_zero_f16((__fp16 *)s, score_head_elems);
      llm_attention_redmule_job(q, kt, s, (uint16_t)seqlen, (uint16_t)head_dim,
                                (uint16_t)attn_len, GEMM);
    }
  }
  mempool_barrier(num_cores);
}

static inline void llm_attention_scale_mask_f16(
    __fp16 *scores_pm, uint32_t seqlen, uint32_t attn_len, uint32_t num_q_heads,
    uint32_t head_dim, uint32_t first_key_pos, uint32_t past_len,
    uint32_t core_id, uint32_t num_cores) {
  const float scale = llm_rsqrt((float)head_dim);
  const float MASK_H = llm_f16box(-80.0f);
  const uint32_t rows = num_q_heads * seqlen;

  // Causality makes the visible keys a prefix of each score row. Scale that
  // prefix in f32 with the widest legal widening pair, then fill the masked
  // suffix with an e16,m8 vector rather than generating one mask per element.
  for (uint32_t r = core_id; r < rows; r += num_cores) {
    const uint32_t token = r % seqlen;
    const uint32_t last_visible = past_len + token;
    uint32_t visible = 0;
    if (last_visible >= first_key_pos) {
      visible = last_visible - first_key_pos + 1;
      if (visible > attn_len)
        visible = attn_len;
    }

    __fp16 *row = scores_pm + r * attn_len;
    llm_scale_f16_f32(row, visible, scale);

    __fp16 *p = row + visible;
    uint32_t avl = attn_len - visible;
    while (avl > 0) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
      asm volatile("vfmv.v.f v0, %0" : : "f"(MASK_H));
      asm volatile("vse16.v v0, (%0)" : : "r"(p) : "memory");
      p += vl;
      avl -= vl;
    }
  }
}

static inline void llm_attention_softmax_rows_f16(__fp16 *scores, uint32_t rows,
                                                  uint32_t cols,
                                                  uint32_t core_id,
                                                  uint32_t num_cores) {
  const float ZERO_H = llm_f16box(0.0f);
  const float ZERO_F = 0.0f;
  const float NEG_BIG = -1e30f;
  for (uint32_t r = core_id; r < rows; r += num_cores) {
    __fp16 *row = scores + r * cols;
    __fp16 *p = row;
    uint32_t avl = cols;

    // Max and exp/sum widen e16,m4 inputs into e32,m8. The exp helper consumes
    // v0/v8/v16, leaving v24 as a persistent reduction accumulator.
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(cols));
    asm volatile("vfmv.s.f v24, %0" : : "f"(NEG_BIG));
    while (avl > 0) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));
      asm volatile("vle16.v v8, (%0)" : : "r"(p) : "memory");
      asm volatile("vfwadd.vf v0, v8, %0" : : "f"(ZERO_H));
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
      asm volatile("vfredmax.vs v24, v0, v24");
      p += vl;
      avl -= vl;
    }
    float maxv;
    asm volatile("vfmv.f.s %0, v24" : "=f"(maxv));

    p = row;
    avl = cols;
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(cols));
    asm volatile("vfmv.s.f v24, %0" : : "f"(ZERO_F));
    while (avl > 0) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));
      asm volatile("vle16.v v8, (%0)" : : "r"(p) : "memory");
      asm volatile("vfwadd.vf v0, v8, %0" : : "f"(ZERO_H));
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
      asm volatile("vfsub.vf v0, v0, %0" : : "f"(maxv));
      llm_rvv_expf_m8();
      asm volatile("vfredusum.vs v24, v0, v24");
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
      asm volatile("vfncvt.f.f.w v8, v0");
      asm volatile("vse16.v v8, (%0)" : : "r"(p) : "memory");
      p += vl;
      avl -= vl;
    }
    float sum;
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(cols));
    asm volatile("vfmv.f.s %0, v24" : "=f"(sum));
    llm_scale_f16_f32(row, cols, llm_recip_pos(sum));
  }
}

static inline void
llm_attention_sv_matmul_f16(const __fp16 *scores_pm, const __fp16 *v_pm,
                            __fp16 *ctx_pm, uint32_t seqlen, uint32_t attn_len,
                            uint32_t num_q_heads, uint32_t num_kv_heads,
                            uint32_t head_dim, uint32_t num_cores,
                            uint32_t redmule_id) {
  const uint32_t group = num_q_heads / num_kv_heads;
  const uint32_t score_head_elems = seqlen * attn_len;
  const uint32_t v_head_elems = attn_len * head_dim;
  const uint32_t ctx_head_elems = seqlen * head_dim;
  const uint32_t num_redmules = mempool_get_redmule_count();

  if (redmule_id < num_redmules) {
    for (uint32_t qh = redmule_id; qh < num_q_heads; qh += num_redmules) {
      uint32_t kvh = qh / group;
      unsigned int s = (unsigned int)(scores_pm + qh * score_head_elems);
      unsigned int v = (unsigned int)(v_pm + kvh * v_head_elems);
      unsigned int c = (unsigned int)(ctx_pm + qh * ctx_head_elems);
      llm_zero_f16((__fp16 *)c, ctx_head_elems);
      llm_attention_redmule_job(s, v, c, (uint16_t)seqlen, (uint16_t)attn_len,
                                (uint16_t)head_dim, GEMM);
    }
  }
  mempool_barrier(num_cores);
}

// out[S x D] = sum_h ctx[h][S x HEAD_DIM] @ Wo[h][HEAD_DIM x D].
static inline void llm_attention_out_proj_f16(
    const __fp16 *l2_wo, const __fp16 *ctx_pm, __fp16 *out, __fp16 *wbuf,
    uint32_t seqlen, uint32_t dmodel, uint32_t head_dim, uint32_t num_q_heads,
    uint32_t core_id, uint32_t num_cores, uint32_t redmule_id,
    uint32_t ne_used) {
  const uint32_t rows_per_engine = seqlen / ne_used;
  const uint32_t panel_elems = head_dim * dmodel;
  const uint32_t ctx_head_elems = seqlen * head_dim;
  __fp16 *wbuf0 = wbuf;
  __fp16 *wbuf1 = wbuf + panel_elems;

  if (core_id == 0)
    dma_memcpy_blocking(wbuf0, l2_wo, panel_elems * sizeof(__fp16));
  mempool_barrier(num_cores);

  for (uint32_t h = 0; h < num_q_heads; h++) {
    __fp16 *wcur = (h & 1) ? wbuf1 : wbuf0;
    __fp16 *wnxt = (h & 1) ? wbuf0 : wbuf1;
    if (core_id == 0 && (h + 1) < num_q_heads)
      dma_memcpy_nonblocking(wnxt, l2_wo + (h + 1) * panel_elems,
                             panel_elems * sizeof(__fp16));

    if (redmule_id < ne_used) {
      unsigned int x = (unsigned int)(ctx_pm + h * ctx_head_elems +
                                      redmule_id * rows_per_engine * head_dim);
      unsigned int w = (unsigned int)wcur;
      unsigned int z =
          (unsigned int)(out + redmule_id * rows_per_engine * dmodel);
      if (h == 0)
        llm_zero_f16((__fp16 *)z, rows_per_engine * dmodel);
      llm_attention_redmule_job(x, w, z, (uint16_t)rows_per_engine,
                                (uint16_t)head_dim, (uint16_t)dmodel, GEMM);
    }
    mempool_barrier(num_cores);
    if (core_id == 0)
      dma_wait();
    mempool_barrier(num_cores);
  }
}

static inline void llm_attention_out_proj_single_f16(
    const __fp16 *l2_wo, const __fp16 *ctx_pm, __fp16 *out, __fp16 *wbuf,
    uint32_t seqlen, uint32_t dmodel, uint32_t head_dim, uint32_t num_q_heads,
    uint32_t core_id, uint32_t num_cores, uint32_t redmule_id,
    uint32_t ne_used) {
  const uint32_t rows_per_engine = seqlen / ne_used;
  const uint32_t panel_elems = head_dim * dmodel;
  const uint32_t ctx_head_elems = seqlen * head_dim;

  for (uint32_t h = 0; h < num_q_heads; h++) {
    if (core_id == 0)
      dma_memcpy_blocking(wbuf, l2_wo + h * panel_elems,
                          panel_elems * sizeof(__fp16));
    mempool_barrier(num_cores);

    if (redmule_id < ne_used) {
      unsigned int x = (unsigned int)(ctx_pm + h * ctx_head_elems +
                                      redmule_id * rows_per_engine * head_dim);
      unsigned int z =
          (unsigned int)(out + redmule_id * rows_per_engine * dmodel);
      if (h == 0)
        llm_zero_f16((__fp16 *)z, rows_per_engine * dmodel);
      llm_attention_redmule_job(x, (unsigned int)wbuf, z,
                                (uint16_t)rows_per_engine, (uint16_t)head_dim,
                                (uint16_t)dmodel, GEMM);
    }
    mempool_barrier(num_cores);
  }
}

// ===========================================================================
// FlashAttention (opt-in via -DLLM_FLASH_ATTENTION). Tiles the key dimension by
// LLM_FLASH_BN and keeps only O[S x HD], running max m[S] and running sum l[S]
// resident per head, so the S x attn_len score matrix is never materialized.
// Per key tile: RedMulE Sj = Q @ Ktile^T; Spatz online-softmax update
// (scale/mask, running max/sum, P = exp(Sj - m_new), rescale O by
// exp(m-m_new)); RedMulE O += P @ Vtile. Numerically identical to the
// materialized path (kept as the reference); validated against naive softmax in
// numpy.
// ===========================================================================
#ifdef LLM_FLASH_ATTENTION
#ifndef LLM_FLASH_BM
#define LLM_FLASH_BM 32
#endif
#ifndef LLM_FLASH_BN
#define LLM_FLASH_BN 64
#endif

// Gather a contiguous K^T tile [num_kv][HD][bn] from kt_full[num_kv][HD][len].
static inline void llm_attention_kt_gather_f16(const __fp16 *kt_full,
                                               __fp16 *kt_tile, uint32_t num_kv,
                                               uint32_t head_dim, uint32_t len,
                                               uint32_t j0, uint32_t bn,
                                               uint32_t core_id,
                                               uint32_t num_cores) {
  const uint32_t max_avl = UINT32_MAX;
  uint32_t vlmax;
  asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vlmax) : "r"(max_avl));
  const uint32_t blocks = (bn + vlmax - 1) / vlmax;
  const uint32_t tasks = num_kv * head_dim * blocks;

  for (uint32_t task = core_id; task < tasks; task += num_cores) {
    const uint32_t block = task % blocks;
    const uint32_t hd = task / blocks;
    const uint32_t col0 = block * vlmax;
    const uint32_t count = llm_attention_min_u32(vlmax, bn - col0);
    const __fp16 *src = kt_full + hd * len + j0 + col0;
    __fp16 *dst = kt_tile + hd * bn + col0;
    asm volatile("vsetvli zero, %0, e16, m8, ta, ma" : : "r"(count));
    asm volatile("vle16.v v0, (%0)" : : "r"(src) : "memory");
    asm volatile("vse16.v v0, (%0)" : : "r"(dst) : "memory");
  }
}

// Online-softmax update for one key tile: scale + causal mask Sj, advance the
// running max/sum, write P = exp(Sj - m_new) back into Sj, and rescale the
// running output O by exp(m_prev - m_new) (skipped on the first tile).
static inline void llm_attention_flash_update_f16(
    __fp16 *sj, __fp16 *o, float *m, float *l, uint32_t seqlen, uint32_t bn,
    uint32_t num_q, uint32_t head_dim, uint32_t j0, uint32_t first_key_pos,
    uint32_t past_len, uint32_t is_first, uint32_t core_id,
    uint32_t num_cores) {
  const float scale = llm_rsqrt((float)head_dim);
  const float ZERO_H = llm_f16box(0.0f);
  const float ZERO_F = 0.0f;
  const float MASK_RAW_H = llm_f16box(-80.0f * (float)head_dim * scale);
  const float NEG_BIG = -1e30f;
  const uint32_t rows = num_q * seqlen;
  for (uint32_t r = core_id; r < rows; r += num_cores) {
    const uint32_t i = r % seqlen;
    __fp16 *row = sj + r * bn;
    const float mprev = m[r];
    const uint32_t tile_first_key = first_key_pos + j0;
    const uint32_t last_visible = past_len + i;
    uint32_t visible = 0;
    if (last_visible >= tile_first_key) {
      visible = last_visible - tile_first_key + 1;
      if (visible > bn)
        visible = bn;
    }

    // Scale the visible prefix in f32 and reduce its maximum, but leave the
    // score tile unmodified. The full-width exponential pass below consumes
    // the raw f16 scores and fuses the scale, avoiding a tiny causal-tail
    // narrowing/store chain. e16,m4 -> e32,m8 is the widest legal widening
    // pair; v24 holds the reduction state.
    float mtile = NEG_BIG;
    if (visible > 0) {
      __fp16 *p = row;
      uint32_t avl = visible;
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(visible));
      asm volatile("vfmv.s.f v24, %0" : : "f"(NEG_BIG));
      while (avl > 0) {
        uint32_t vl;
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v8, (%0)" : : "r"(p) : "memory");
        asm volatile("vfwadd.vf v0, v8, %0" : : "f"(ZERO_H));
        asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
        asm volatile("vfmul.vf v0, v0, %0" : : "f"(scale));
        asm volatile("vfredmax.vs v24, v0, v24");
        p += vl;
        avl -= vl;
      }
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(visible));
      asm volatile("vfmv.f.s %0, v24" : "=f"(mtile));
    }

    // Masked keys are a contiguous suffix.
    __fp16 *masked = row + visible;
    uint32_t masked_avl = bn - visible;
    while (masked_avl > 0) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e16, m8, ta, ma"
                   : "=r"(vl)
                   : "r"(masked_avl));
      asm volatile("vfmv.v.f v0, %0" : : "f"(MASK_RAW_H));
      asm volatile("vse16.v v0, (%0)" : : "r"(masked) : "memory");
      masked += vl;
      masked_avl -= vl;
    }
    if (visible < bn && -80.0f > mtile)
      mtile = -80.0f;

    const float mnew = (mtile > mprev) ? mtile : mprev;

    // P = exp(scale * S_raw - mnew), retaining a full-f32 running sum in v24.
    __fp16 *p = row;
    uint32_t avl = bn;
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(bn));
    asm volatile("vfmv.s.f v24, %0" : : "f"(ZERO_F));
    while (avl > 0) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));
      asm volatile("vle16.v v8, (%0)" : : "r"(p) : "memory");
      asm volatile("vfwadd.vf v0, v8, %0" : : "f"(ZERO_H));
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
      asm volatile("vfmul.vf v0, v0, %0" : : "f"(scale));
      asm volatile("vfsub.vf v0, v0, %0" : : "f"(mnew));
      llm_rvv_expf_m8();
      asm volatile("vfredusum.vs v24, v0, v24");
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
      asm volatile("vfncvt.f.f.w v8, v0");
      asm volatile("vse16.v v8, (%0)" : : "r"(p) : "memory");
      p += vl;
      avl -= vl;
    }
    float sump;
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(bn));
    asm volatile("vfmv.f.s %0, v24" : "=f"(sump));

    const float corr = is_first ? 0.0f : llm_expf(mprev - mnew);
    l[r] = l[r] * corr + sump;
    m[r] = mnew;
    if (!is_first)
      llm_scale_f16_f32(o + r * head_dim, head_dim, corr);
  }
}

// Finalize: O[row] /= l[row]  (l > 0, reciprocal via divide-free rsqrt).
static inline void
llm_attention_flash_finalize_f16(__fp16 *o, const float *l, uint32_t seqlen,
                                 uint32_t num_q, uint32_t head_dim,
                                 uint32_t core_id, uint32_t num_cores) {
  const uint32_t rows = num_q * seqlen;
  for (uint32_t r = core_id; r < rows; r += num_cores) {
    const float inv = llm_recip_pos(l[r]);
    llm_scale_f16_f32(o + r * head_dim, head_dim, inv);
  }
}

// FlashAttention over the key range [0, attn_len): fills ctx[num_q][S x HD].
// kt_full is [num_kv][HD][attn_len]; v_pm is [num_kv][attn_len][HD].
static inline void llm_attention_flash_scores_f16(
    const __fp16 *q_pm, const __fp16 *kt_full, const __fp16 *v_pm,
    __fp16 *scores, __fp16 *ctx, float *m, float *l, __fp16 *kt_tile,
    uint32_t seqlen, uint32_t attn_len, uint32_t num_q, uint32_t num_kv,
    uint32_t head_dim, uint32_t first_key_pos, uint32_t past_len,
    uint32_t core_id, uint32_t num_cores, uint32_t redmule_id) {
  const uint32_t group = num_q / num_kv;
  const uint32_t num_redmules = mempool_get_redmule_count();
  const uint32_t bn = LLM_FLASH_BN;

  for (uint32_t r = core_id; r < num_q * seqlen; r += num_cores) {
    m[r] = -1e30f;
    l[r] = 0.0f;
  }
  mempool_barrier(num_cores);

  for (uint32_t j0 = 0; j0 < attn_len; j0 += bn) {
    uint32_t is_first = (j0 == 0);
    llm_attention_kt_gather_f16(kt_full, kt_tile, num_kv, head_dim, attn_len,
                                j0, bn, core_id, num_cores);
    mempool_barrier(num_cores);

    // Sj = Q @ Ktile^T  (head-parallel over RedMulE tiles).
    llm_attention_qk_matmul_f16(q_pm, kt_tile, scores, seqlen, bn, num_q,
                                num_kv, head_dim, num_cores, redmule_id);

    // Online-softmax update + rescale O (all cores).
    llm_attention_flash_update_f16(scores, ctx, m, l, seqlen, bn, num_q,
                                   head_dim, j0, first_key_pos, past_len,
                                   is_first, core_id, num_cores);
    mempool_barrier(num_cores);

    // O += P @ Vtile (ctx was explicitly zeroed before the first tile).
    if (redmule_id < num_redmules) {
      for (uint32_t qh = redmule_id; qh < num_q; qh += num_redmules) {
        uint32_t kvh = qh / group;
        unsigned int p = (unsigned int)(scores + qh * seqlen * bn);
        unsigned int v =
            (unsigned int)(v_pm + kvh * attn_len * head_dim + j0 * head_dim);
        unsigned int c = (unsigned int)(ctx + qh * seqlen * head_dim);
        if (is_first)
          llm_zero_f16((__fp16 *)c, seqlen * head_dim);
        llm_attention_redmule_job(p, v, c, (uint16_t)seqlen, (uint16_t)bn,
                                  (uint16_t)head_dim, GEMM);
      }
    }
    mempool_barrier(num_cores);
  }

  llm_attention_flash_finalize_f16(ctx, l, seqlen, num_q, head_dim, core_id,
                                   num_cores);
  mempool_barrier(num_cores);
}

#ifdef LLM_FLASH_TILED
// Store/load a token-major L2 KV cache:
//   [num_kv_heads][cache_len][head_dim].
// Keeping K token-major in L2 makes every key tile a contiguous DMA transfer;
// it is transposed only after reaching L1, where RedMulE consumes it.
static inline void llm_attention_store_kv_tile_f16(
    const __fp16 *k_tile, const __fp16 *v_tile, __fp16 *l2_k_cache,
    __fp16 *l2_v_cache, uint32_t cache_len, uint32_t token0, uint32_t tile_rows,
    uint32_t num_kv_heads, uint32_t head_dim, uint32_t core_id,
    uint32_t num_cores) {
  const uint32_t tile_elems = tile_rows * head_dim;
  const uint32_t head_elems = cache_len * head_dim;
  if (core_id == 0) {
    for (uint32_t h = 0; h < num_kv_heads; h++) {
      dma_memcpy_blocking(l2_k_cache + h * head_elems + token0 * head_dim,
                          k_tile + h * tile_elems, tile_elems * sizeof(__fp16));
      dma_memcpy_blocking(l2_v_cache + h * head_elems + token0 * head_dim,
                          v_tile + h * tile_elems, tile_elems * sizeof(__fp16));
    }
  }
  mempool_barrier(num_cores);
}

static inline void llm_attention_stage_kv_tile_f16(
    const __fp16 *l2_k_cache, const __fp16 *l2_v_cache, __fp16 *k_tile,
    __fp16 *v_tile, uint32_t cache_len, uint32_t token0, uint32_t tile_rows,
    uint32_t num_kv_heads, uint32_t head_dim, uint32_t core_id,
    uint32_t num_cores) {
  const uint32_t tile_elems = tile_rows * head_dim;
  const uint32_t head_elems = cache_len * head_dim;
  if (core_id == 0) {
    for (uint32_t h = 0; h < num_kv_heads; h++) {
      dma_memcpy_blocking(k_tile + h * tile_elems,
                          l2_k_cache + h * head_elems + token0 * head_dim,
                          tile_elems * sizeof(__fp16));
      dma_memcpy_blocking(v_tile + h * tile_elems,
                          l2_v_cache + h * head_elems + token0 * head_dim,
                          tile_elems * sizeof(__fp16));
    }
  }
  mempool_barrier(num_cores);
}

// True 2-D FlashAttention: q contains only BM query rows, while every BN key
// tile is streamed from the token-major L2 cache. The online max/sum state and
// context therefore scale with BM rather than the full sequence length.
static inline void llm_attention_flash_scores_tiled_f16(
    const __fp16 *q_pm, const __fp16 *l2_k_cache, const __fp16 *l2_v_cache,
    __fp16 *k_tile, __fp16 *v_tile, __fp16 *kt_tile, __fp16 *scores,
    __fp16 *ctx, float *m, float *l, uint32_t query_rows, uint32_t cache_len,
    uint32_t first_key_pos, uint32_t attn_len, uint32_t query_pos0, uint32_t bn,
    uint32_t num_q, uint32_t num_kv, uint32_t head_dim, uint32_t core_id,
    uint32_t num_cores, uint32_t redmule_id) {
  const uint32_t group = num_q / num_kv;
  const uint32_t num_redmules = mempool_get_redmule_count();

  for (uint32_t r = core_id; r < num_q * query_rows; r += num_cores) {
    m[r] = -1e30f;
    l[r] = 0.0f;
  }
  mempool_barrier(num_cores);

  for (uint32_t j0 = 0; j0 < attn_len; j0 += bn) {
    const uint32_t is_first = (j0 == 0);
    llm_attention_stage_kv_tile_f16(l2_k_cache, l2_v_cache, k_tile, v_tile,
                                    cache_len, first_key_pos + j0, bn, num_kv,
                                    head_dim, core_id, num_cores);
    llm_attention_k_to_kt_f16(k_tile, kt_tile, bn, num_kv, head_dim, core_id,
                              num_cores);
    mempool_barrier(num_cores);

    llm_attention_qk_matmul_f16(q_pm, kt_tile, scores, query_rows, bn, num_q,
                                num_kv, head_dim, num_cores, redmule_id);
    llm_attention_flash_update_f16(scores, ctx, m, l, query_rows, bn, num_q,
                                   head_dim, j0, first_key_pos, query_pos0,
                                   is_first, core_id, num_cores);
    mempool_barrier(num_cores);

    if (redmule_id < num_redmules) {
      for (uint32_t qh = redmule_id; qh < num_q; qh += num_redmules) {
        uint32_t kvh = qh / group;
        unsigned int p = (unsigned int)(scores + qh * query_rows * bn);
        unsigned int v = (unsigned int)(v_tile + kvh * bn * head_dim);
        unsigned int c = (unsigned int)(ctx + qh * query_rows * head_dim);
        if (is_first)
          llm_zero_f16((__fp16 *)c, query_rows * head_dim);
        llm_attention_redmule_job(p, v, c, (uint16_t)query_rows, (uint16_t)bn,
                                  (uint16_t)head_dim, GEMM);
      }
    }
    mempool_barrier(num_cores);
  }

  llm_attention_flash_finalize_f16(ctx, l, query_rows, num_q, head_dim, core_id,
                                   num_cores);
  mempool_barrier(num_cores);
}

static inline void llm_attention_flash_tiled_prefill_f16(
    const __fp16 *l2_x, const __fp16 *l2_gamma, const __fp16 *l2_q_gamma,
    const __fp16 *l2_k_gamma, const __fp16 *l2_cos, const __fp16 *l2_sin,
    const __fp16 *l2_wq, const __fp16 *l2_wk, const __fp16 *l2_wv,
    const __fp16 *l2_wo, __fp16 *l2_k_cache, __fp16 *l2_v_cache, __fp16 *l2_out,
    __fp16 *l1_x, __fp16 *l1_gamma, __fp16 *l1_q_gamma, __fp16 *l1_k_gamma,
    __fp16 *l1_cos, __fp16 *l1_sin, __fp16 *l1_h, __fp16 *l1_q, __fp16 *l1_k,
    __fp16 *l1_v, __fp16 *l1_kt, __fp16 *l1_scores, __fp16 *l1_ctx,
    __fp16 *l1_out, __fp16 *l1_wbuf, float *l1_m, float *l1_l, uint32_t seqlen,
    uint32_t dmodel, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t qk_norm, uint32_t bm, uint32_t bn,
    uint32_t core_id, uint32_t num_cores, uint32_t redmule_id) {
  const uint32_t ne_used = llm_attention_ne_used(bm);
  const uint32_t half_dim = head_dim >> 1;

  if (core_id == 0) {
    dma_memcpy_blocking(l1_gamma, l2_gamma, dmodel * sizeof(__fp16));
    dma_memcpy_blocking(l1_q_gamma, l2_q_gamma, head_dim * sizeof(__fp16));
    dma_memcpy_blocking(l1_k_gamma, l2_k_gamma, head_dim * sizeof(__fp16));
  }
  mempool_barrier(num_cores);

  // Build the RoPE-transformed K/V cache in L2 one query-row tile at a time.
  // K is deliberately token-major here; each later BN tile is one DMA burst.
  for (uint32_t i0 = 0; i0 < seqlen; i0 += bm) {
    if (core_id == 0) {
      dma_memcpy_blocking(l1_x, l2_x + i0 * dmodel,
                          bm * dmodel * sizeof(__fp16));
      dma_memcpy_blocking(l1_cos, l2_cos + i0 * half_dim,
                          bm * half_dim * sizeof(__fp16));
      dma_memcpy_blocking(l1_sin, l2_sin + i0 * half_dim,
                          bm * half_dim * sizeof(__fp16));
    }
    mempool_barrier(num_cores);

    llm_rmsnorm_f16(l1_x, l1_gamma, l1_h, bm, dmodel, core_id, num_cores);
    mempool_barrier(num_cores);
    llm_attention_proj_heads_single_f16(l2_wk, l1_h, l1_k, l1_wbuf, bm, dmodel,
                                        head_dim, num_kv_heads, core_id,
                                        num_cores, redmule_id, ne_used);
    llm_attention_proj_heads_single_f16(l2_wv, l1_h, l1_v, l1_wbuf, bm, dmodel,
                                        head_dim, num_kv_heads, core_id,
                                        num_cores, redmule_id, ne_used);
    if (qk_norm != 0) {
      llm_rmsnorm_f16(l1_k, l1_k_gamma, l1_k, num_kv_heads * bm, head_dim,
                      core_id, num_cores);
      mempool_barrier(num_cores);
    }
    llm_attention_rope_f16(l1_k, l1_cos, l1_sin, bm, num_kv_heads, head_dim,
                           core_id, num_cores);
    mempool_barrier(num_cores);
    llm_attention_store_kv_tile_f16(l1_k, l1_v, l2_k_cache, l2_v_cache, seqlen,
                                    i0, bm, num_kv_heads, head_dim, core_id,
                                    num_cores);
  }

  // Recompute only the small Q tile and stream every K/V tile through it.
  for (uint32_t i0 = 0; i0 < seqlen; i0 += bm) {
    if (core_id == 0) {
      dma_memcpy_blocking(l1_x, l2_x + i0 * dmodel,
                          bm * dmodel * sizeof(__fp16));
      dma_memcpy_blocking(l1_cos, l2_cos + i0 * half_dim,
                          bm * half_dim * sizeof(__fp16));
      dma_memcpy_blocking(l1_sin, l2_sin + i0 * half_dim,
                          bm * half_dim * sizeof(__fp16));
    }
    mempool_barrier(num_cores);

    llm_rmsnorm_f16(l1_x, l1_gamma, l1_h, bm, dmodel, core_id, num_cores);
    mempool_barrier(num_cores);
    llm_attention_proj_heads_single_f16(l2_wq, l1_h, l1_q, l1_wbuf, bm, dmodel,
                                        head_dim, num_q_heads, core_id,
                                        num_cores, redmule_id, ne_used);
    if (qk_norm != 0) {
      llm_rmsnorm_f16(l1_q, l1_q_gamma, l1_q, num_q_heads * bm, head_dim,
                      core_id, num_cores);
      mempool_barrier(num_cores);
    }
    llm_attention_rope_f16(l1_q, l1_cos, l1_sin, bm, num_q_heads, head_dim,
                           core_id, num_cores);
    mempool_barrier(num_cores);

    llm_attention_flash_scores_tiled_f16(
        l1_q, l2_k_cache, l2_v_cache, l1_k, l1_v, l1_kt, l1_scores, l1_ctx,
        l1_m, l1_l, bm, seqlen, 0, seqlen, i0, bn, num_q_heads, num_kv_heads,
        head_dim, core_id, num_cores, redmule_id);
    llm_attention_out_proj_single_f16(l2_wo, l1_ctx, l1_out, l1_wbuf, bm,
                                      dmodel, head_dim, num_q_heads, core_id,
                                      num_cores, redmule_id, ne_used);
    llm_residual_add_f16(l1_out, l1_x, bm * dmodel, core_id, num_cores);
    mempool_barrier(num_cores);
    if (core_id == 0)
      dma_memcpy_blocking(l2_out + i0 * dmodel, l1_out,
                          bm * dmodel * sizeof(__fp16));
    mempool_barrier(num_cores);
  }
}

static inline void llm_attention_flash_tiled_decode_f16(
    const __fp16 *l2_x, const __fp16 *l2_gamma, const __fp16 *l2_q_gamma,
    const __fp16 *l2_k_gamma, const __fp16 *l2_cos, const __fp16 *l2_sin,
    __fp16 *l2_k_cache, __fp16 *l2_v_cache, const __fp16 *l2_wq,
    const __fp16 *l2_wk, const __fp16 *l2_wv, const __fp16 *l2_wo,
    __fp16 *l2_out, __fp16 *l1_x, __fp16 *l1_gamma, __fp16 *l1_q_gamma,
    __fp16 *l1_k_gamma, __fp16 *l1_cos, __fp16 *l1_sin, __fp16 *l1_h,
    __fp16 *l1_q, __fp16 *l1_k, __fp16 *l1_v, __fp16 *l1_kt, __fp16 *l1_scores,
    __fp16 *l1_ctx, __fp16 *l1_out, __fp16 *l1_wbuf, float *l1_m, float *l1_l,
    uint32_t seqlen, uint32_t past_len, uint32_t window_len, uint32_t dmodel,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim,
    uint32_t qk_norm, uint32_t bn, uint32_t core_id, uint32_t num_cores,
    uint32_t redmule_id) {
  const uint32_t ne_used = llm_attention_ne_used(seqlen);
  const uint32_t half_dim = head_dim >> 1;
  const uint32_t total_len = past_len + seqlen;
  uint32_t attn_len = total_len;
  if (window_len != 0)
    attn_len = llm_attention_min_u32(attn_len, window_len);
  const uint32_t first_key_pos = total_len - attn_len;

  if (core_id == 0) {
    dma_memcpy_blocking(l1_x, l2_x, seqlen * dmodel * sizeof(__fp16));
    dma_memcpy_blocking(l1_gamma, l2_gamma, dmodel * sizeof(__fp16));
    dma_memcpy_blocking(l1_q_gamma, l2_q_gamma, head_dim * sizeof(__fp16));
    dma_memcpy_blocking(l1_k_gamma, l2_k_gamma, head_dim * sizeof(__fp16));
    dma_memcpy_blocking(l1_cos, l2_cos, seqlen * half_dim * sizeof(__fp16));
    dma_memcpy_blocking(l1_sin, l2_sin, seqlen * half_dim * sizeof(__fp16));
  }
  mempool_barrier(num_cores);

  llm_rmsnorm_f16(l1_x, l1_gamma, l1_h, seqlen, dmodel, core_id, num_cores);
  mempool_barrier(num_cores);
  llm_attention_proj_heads_single_f16(l2_wq, l1_h, l1_q, l1_wbuf, seqlen,
                                      dmodel, head_dim, num_q_heads, core_id,
                                      num_cores, redmule_id, ne_used);
  llm_attention_proj_heads_single_f16(l2_wk, l1_h, l1_k, l1_wbuf, seqlen,
                                      dmodel, head_dim, num_kv_heads, core_id,
                                      num_cores, redmule_id, ne_used);
  llm_attention_proj_heads_single_f16(l2_wv, l1_h, l1_v, l1_wbuf, seqlen,
                                      dmodel, head_dim, num_kv_heads, core_id,
                                      num_cores, redmule_id, ne_used);

  if (qk_norm != 0)
    llm_attention_qk_norm_f16(l1_q, l1_k, l1_q_gamma, l1_k_gamma, seqlen,
                              num_q_heads, num_kv_heads, head_dim, core_id,
                              num_cores);
  llm_attention_rope_f16(l1_q, l1_cos, l1_sin, seqlen, num_q_heads, head_dim,
                         core_id, num_cores);
  llm_attention_rope_f16(l1_k, l1_cos, l1_sin, seqlen, num_kv_heads, head_dim,
                         core_id, num_cores);
  mempool_barrier(num_cores);

  // The generated token-major cache reserves KV_LEN + S rows. Appending here
  // updates the persistent L2 cache without ever staging the past in L1.
  llm_attention_store_kv_tile_f16(l1_k, l1_v, l2_k_cache, l2_v_cache, total_len,
                                  past_len, seqlen, num_kv_heads, head_dim,
                                  core_id, num_cores);
  llm_attention_flash_scores_tiled_f16(
      l1_q, l2_k_cache, l2_v_cache, l1_k, l1_v, l1_kt, l1_scores, l1_ctx, l1_m,
      l1_l, seqlen, total_len, first_key_pos, attn_len, past_len, bn,
      num_q_heads, num_kv_heads, head_dim, core_id, num_cores, redmule_id);
  llm_attention_out_proj_single_f16(l2_wo, l1_ctx, l1_out, l1_wbuf, seqlen,
                                    dmodel, head_dim, num_q_heads, core_id,
                                    num_cores, redmule_id, ne_used);
  llm_residual_add_f16(l1_out, l1_x, seqlen * dmodel, core_id, num_cores);
  mempool_barrier(num_cores);
  if (core_id == 0)
    dma_memcpy_blocking(l2_out, l1_out, seqlen * dmodel * sizeof(__fp16));
  mempool_barrier(num_cores);
}
#endif // LLM_FLASH_TILED

// Prefill attention with the FlashAttention score path. Front (RMSNorm, Q/K/V
// proj, QK-norm, RoPE, K^T) and back (o_proj, residual) reuse the same helpers
// as llm_attention_prefill_f16; only the score/softmax/context middle differs.
static inline void llm_attention_flash_prefill_f16(
    const __fp16 *l2_x, const __fp16 *l2_gamma, const __fp16 *l2_q_gamma,
    const __fp16 *l2_k_gamma, const __fp16 *l2_cos, const __fp16 *l2_sin,
    const __fp16 *l2_wq, const __fp16 *l2_wk, const __fp16 *l2_wv,
    const __fp16 *l2_wo, __fp16 *l1_x, __fp16 *l1_gamma, __fp16 *l1_q_gamma,
    __fp16 *l1_k_gamma, __fp16 *l1_cos, __fp16 *l1_sin, __fp16 *l1_h,
    __fp16 *l1_q, __fp16 *l1_k, __fp16 *l1_v, __fp16 *l1_kt, __fp16 *l1_scores,
    __fp16 *l1_ctx, __fp16 *l1_out, __fp16 *l1_wbuf, float *l1_m, float *l1_l,
    __fp16 *l1_kt_tile, uint32_t seqlen, uint32_t dmodel, uint32_t num_q_heads,
    uint32_t num_kv_heads, uint32_t head_dim, uint32_t qk_norm,
    uint32_t core_id, uint32_t num_cores, uint32_t redmule_id) {
  const uint32_t ne_used = llm_attention_ne_used(seqlen);
  const uint32_t half_dim = head_dim >> 1;

  if (core_id == 0) {
    dma_memcpy_blocking(l1_x, l2_x, seqlen * dmodel * sizeof(__fp16));
    dma_memcpy_blocking(l1_gamma, l2_gamma, dmodel * sizeof(__fp16));
    dma_memcpy_blocking(l1_q_gamma, l2_q_gamma, head_dim * sizeof(__fp16));
    dma_memcpy_blocking(l1_k_gamma, l2_k_gamma, head_dim * sizeof(__fp16));
    dma_memcpy_blocking(l1_cos, l2_cos, seqlen * half_dim * sizeof(__fp16));
    dma_memcpy_blocking(l1_sin, l2_sin, seqlen * half_dim * sizeof(__fp16));
  }
  mempool_barrier(num_cores);

  llm_rmsnorm_f16(l1_x, l1_gamma, l1_h, seqlen, dmodel, core_id, num_cores);
  mempool_barrier(num_cores);

  llm_attention_proj_heads(l2_wq, l1_h, l1_q, l1_wbuf, seqlen, dmodel, head_dim,
                           num_q_heads, core_id, num_cores, redmule_id,
                           ne_used);
  llm_attention_proj_heads(l2_wk, l1_h, l1_k, l1_wbuf, seqlen, dmodel, head_dim,
                           num_kv_heads, core_id, num_cores, redmule_id,
                           ne_used);
  llm_attention_proj_heads(l2_wv, l1_h, l1_v, l1_wbuf, seqlen, dmodel, head_dim,
                           num_kv_heads, core_id, num_cores, redmule_id,
                           ne_used);

  if (qk_norm != 0)
    llm_attention_qk_norm_f16(l1_q, l1_k, l1_q_gamma, l1_k_gamma, seqlen,
                              num_q_heads, num_kv_heads, head_dim, core_id,
                              num_cores);

  llm_attention_rope_f16(l1_q, l1_cos, l1_sin, seqlen, num_q_heads, head_dim,
                         core_id, num_cores);
  llm_attention_rope_f16(l1_k, l1_cos, l1_sin, seqlen, num_kv_heads, head_dim,
                         core_id, num_cores);
  mempool_barrier(num_cores);

  llm_attention_k_to_kt_f16(l1_k, l1_kt, seqlen, num_kv_heads, head_dim,
                            core_id, num_cores);
  mempool_barrier(num_cores);

  llm_attention_flash_scores_f16(l1_q, l1_kt, l1_v, l1_scores, l1_ctx, l1_m,
                                 l1_l, l1_kt_tile, seqlen, seqlen, num_q_heads,
                                 num_kv_heads, head_dim, 0, 0, core_id,
                                 num_cores, redmule_id);

  llm_attention_out_proj_f16(l2_wo, l1_ctx, l1_out, l1_wbuf, seqlen, dmodel,
                             head_dim, num_q_heads, core_id, num_cores,
                             redmule_id, ne_used);

  llm_residual_add_f16(l1_out, l1_x, seqlen * dmodel, core_id, num_cores);
  mempool_barrier(num_cores);
}
#endif // LLM_FLASH_ATTENTION

static inline void llm_attention_prefill_f16(
    const __fp16 *l2_x, const __fp16 *l2_gamma, const __fp16 *l2_q_gamma,
    const __fp16 *l2_k_gamma, const __fp16 *l2_cos, const __fp16 *l2_sin,
    const __fp16 *l2_wq, const __fp16 *l2_wk, const __fp16 *l2_wv,
    const __fp16 *l2_wo, __fp16 *l1_x, __fp16 *l1_gamma, __fp16 *l1_q_gamma,
    __fp16 *l1_k_gamma, __fp16 *l1_cos, __fp16 *l1_sin, __fp16 *l1_h,
    __fp16 *l1_q, __fp16 *l1_k, __fp16 *l1_v, __fp16 *l1_kt, __fp16 *l1_scores,
    __fp16 *l1_ctx, __fp16 *l1_out, __fp16 *l1_wbuf, uint32_t seqlen,
    uint32_t dmodel, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t qk_norm, uint32_t core_id, uint32_t num_cores,
    uint32_t redmule_id) {
  const uint32_t ne_used = llm_attention_ne_used(seqlen);
  const uint32_t half_dim = head_dim >> 1;

  if (core_id == 0) {
    dma_memcpy_blocking(l1_x, l2_x, seqlen * dmodel * sizeof(__fp16));
    dma_memcpy_blocking(l1_gamma, l2_gamma, dmodel * sizeof(__fp16));
    dma_memcpy_blocking(l1_q_gamma, l2_q_gamma, head_dim * sizeof(__fp16));
    dma_memcpy_blocking(l1_k_gamma, l2_k_gamma, head_dim * sizeof(__fp16));
    dma_memcpy_blocking(l1_cos, l2_cos, seqlen * half_dim * sizeof(__fp16));
    dma_memcpy_blocking(l1_sin, l2_sin, seqlen * half_dim * sizeof(__fp16));
  }
  mempool_barrier(num_cores);

  llm_rmsnorm_f16(l1_x, l1_gamma, l1_h, seqlen, dmodel, core_id, num_cores);
  mempool_barrier(num_cores);

  llm_attention_proj_heads(l2_wq, l1_h, l1_q, l1_wbuf, seqlen, dmodel, head_dim,
                           num_q_heads, core_id, num_cores, redmule_id,
                           ne_used);
  llm_attention_proj_heads(l2_wk, l1_h, l1_k, l1_wbuf, seqlen, dmodel, head_dim,
                           num_kv_heads, core_id, num_cores, redmule_id,
                           ne_used);
  llm_attention_proj_heads(l2_wv, l1_h, l1_v, l1_wbuf, seqlen, dmodel, head_dim,
                           num_kv_heads, core_id, num_cores, redmule_id,
                           ne_used);

  if (qk_norm != 0)
    llm_attention_qk_norm_f16(l1_q, l1_k, l1_q_gamma, l1_k_gamma, seqlen,
                              num_q_heads, num_kv_heads, head_dim, core_id,
                              num_cores);

  llm_attention_rope_f16(l1_q, l1_cos, l1_sin, seqlen, num_q_heads, head_dim,
                         core_id, num_cores);
  llm_attention_rope_f16(l1_k, l1_cos, l1_sin, seqlen, num_kv_heads, head_dim,
                         core_id, num_cores);
  mempool_barrier(num_cores);

  llm_attention_k_to_kt_f16(l1_k, l1_kt, seqlen, num_kv_heads, head_dim,
                            core_id, num_cores);
  mempool_barrier(num_cores);

  llm_attention_qk_matmul_f16(l1_q, l1_kt, l1_scores, seqlen, seqlen,
                              num_q_heads, num_kv_heads, head_dim, num_cores,
                              redmule_id);
  llm_attention_scale_mask_f16(l1_scores, seqlen, seqlen, num_q_heads, head_dim,
                               0, 0, core_id, num_cores);
  mempool_barrier(num_cores);
  llm_attention_softmax_rows_f16(l1_scores, num_q_heads * seqlen, seqlen,
                                 core_id, num_cores);
  mempool_barrier(num_cores);

  llm_attention_sv_matmul_f16(l1_scores, l1_v, l1_ctx, seqlen, seqlen,
                              num_q_heads, num_kv_heads, head_dim, num_cores,
                              redmule_id);
  llm_attention_out_proj_f16(l2_wo, l1_ctx, l1_out, l1_wbuf, seqlen, dmodel,
                             head_dim, num_q_heads, core_id, num_cores,
                             redmule_id, ne_used);

  llm_residual_add_f16(l1_out, l1_x, seqlen * dmodel, core_id, num_cores);
  mempool_barrier(num_cores);
}

static inline void llm_attention_decode_f16(
    const __fp16 *l2_x, const __fp16 *l2_gamma, const __fp16 *l2_q_gamma,
    const __fp16 *l2_k_gamma, const __fp16 *l2_cos, const __fp16 *l2_sin,
    const __fp16 *l2_k_cache_t, const __fp16 *l2_v_cache, const __fp16 *l2_wq,
    const __fp16 *l2_wk, const __fp16 *l2_wv, const __fp16 *l2_wo, __fp16 *l1_x,
    __fp16 *l1_gamma, __fp16 *l1_q_gamma, __fp16 *l1_k_gamma, __fp16 *l1_cos,
    __fp16 *l1_sin, __fp16 *l1_k_cache_t, __fp16 *l1_v_cache,
    __fp16 *l1_kt_window, __fp16 *l1_v_window, __fp16 *l1_h, __fp16 *l1_q,
    __fp16 *l1_k, __fp16 *l1_v, __fp16 *l1_scores, __fp16 *l1_ctx,
    __fp16 *l1_out, __fp16 *l1_wbuf, uint32_t seqlen, uint32_t past_len,
    uint32_t window_len, uint32_t dmodel, uint32_t num_q_heads,
    uint32_t num_kv_heads, uint32_t head_dim, uint32_t qk_norm,
    uint32_t core_id, uint32_t num_cores, uint32_t redmule_id) {
  const uint32_t ne_used = llm_attention_ne_used(seqlen);
  const uint32_t half_dim = head_dim >> 1;
  const uint32_t total_len = past_len + seqlen;
  uint32_t attn_len = total_len;
  if (window_len != 0)
    attn_len = llm_attention_min_u32(attn_len, window_len);
  const uint32_t first_key_pos = total_len - attn_len;

  if (core_id == 0) {
    dma_memcpy_blocking(l1_x, l2_x, seqlen * dmodel * sizeof(__fp16));
    dma_memcpy_blocking(l1_gamma, l2_gamma, dmodel * sizeof(__fp16));
    dma_memcpy_blocking(l1_q_gamma, l2_q_gamma, head_dim * sizeof(__fp16));
    dma_memcpy_blocking(l1_k_gamma, l2_k_gamma, head_dim * sizeof(__fp16));
    dma_memcpy_blocking(l1_cos, l2_cos, seqlen * half_dim * sizeof(__fp16));
    dma_memcpy_blocking(l1_sin, l2_sin, seqlen * half_dim * sizeof(__fp16));
  }
  mempool_barrier(num_cores);

  llm_attention_stage_decode_cache_f16(
      l2_k_cache_t, l2_v_cache, l1_k_cache_t, l1_v_cache, past_len, total_len,
      num_kv_heads, head_dim, core_id, num_cores);
  mempool_barrier(num_cores);

  llm_rmsnorm_f16(l1_x, l1_gamma, l1_h, seqlen, dmodel, core_id, num_cores);
  mempool_barrier(num_cores);

  llm_attention_proj_heads(l2_wq, l1_h, l1_q, l1_wbuf, seqlen, dmodel, head_dim,
                           num_q_heads, core_id, num_cores, redmule_id,
                           ne_used);
  llm_attention_proj_heads(l2_wk, l1_h, l1_k, l1_wbuf, seqlen, dmodel, head_dim,
                           num_kv_heads, core_id, num_cores, redmule_id,
                           ne_used);
  llm_attention_proj_heads(l2_wv, l1_h, l1_v, l1_wbuf, seqlen, dmodel, head_dim,
                           num_kv_heads, core_id, num_cores, redmule_id,
                           ne_used);

  if (qk_norm != 0)
    llm_attention_qk_norm_f16(l1_q, l1_k, l1_q_gamma, l1_k_gamma, seqlen,
                              num_q_heads, num_kv_heads, head_dim, core_id,
                              num_cores);

  llm_attention_rope_f16(l1_q, l1_cos, l1_sin, seqlen, num_q_heads, head_dim,
                         core_id, num_cores);
  llm_attention_rope_f16(l1_k, l1_cos, l1_sin, seqlen, num_kv_heads, head_dim,
                         core_id, num_cores);
  mempool_barrier(num_cores);

  llm_attention_append_kv_f16(l1_k, l1_v, l1_k_cache_t, l1_v_cache, past_len,
                              seqlen, num_kv_heads, head_dim, core_id,
                              num_cores);
  mempool_barrier(num_cores);

  llm_attention_select_window_f16(
      l1_k_cache_t, l1_v_cache, l1_kt_window, l1_v_window, total_len,
      first_key_pos, attn_len, num_kv_heads, head_dim, core_id, num_cores);
  mempool_barrier(num_cores);

  llm_attention_qk_matmul_f16(l1_q, l1_kt_window, l1_scores, seqlen, attn_len,
                              num_q_heads, num_kv_heads, head_dim, num_cores,
                              redmule_id);
  llm_attention_scale_mask_f16(l1_scores, seqlen, attn_len, num_q_heads,
                               head_dim, first_key_pos, past_len, core_id,
                               num_cores);
  mempool_barrier(num_cores);
  llm_attention_softmax_rows_f16(l1_scores, num_q_heads * seqlen, attn_len,
                                 core_id, num_cores);
  mempool_barrier(num_cores);

  llm_attention_sv_matmul_f16(l1_scores, l1_v_window, l1_ctx, seqlen, attn_len,
                              num_q_heads, num_kv_heads, head_dim, num_cores,
                              redmule_id);
  llm_attention_out_proj_f16(l2_wo, l1_ctx, l1_out, l1_wbuf, seqlen, dmodel,
                             head_dim, num_q_heads, core_id, num_cores,
                             redmule_id, ne_used);

  llm_residual_add_f16(l1_out, l1_x, seqlen * dmodel, core_id, num_cores);
  mempool_barrier(num_cores);
}
