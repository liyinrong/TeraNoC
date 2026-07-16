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

static inline void llm_attention_redmule_job(unsigned int x, unsigned int w,
                                             unsigned int z, uint16_t m,
                                             uint16_t n, uint16_t k,
                                             uint8_t op) {
  hwpe_soft_clear();
  mempool_wait(10);
  redmule_cfg(x, w, z, m, n, k, 0, op, LLM_ATTENTION_GEMM_FMT);
  mempool_wait(10);
  hwpe_trigger_job();
  mempool_wfi();
}

static inline uint32_t llm_attention_min_u32(uint32_t a, uint32_t b) {
  return a < b ? a : b;
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
  const uint32_t npairs = num_heads * seqlen * half_dim;
  uint32_t chunk = (npairs + num_cores - 1) / num_cores;
  uint32_t start = core_id * chunk;
  uint32_t end = start + chunk;
  if (end > npairs)
    end = npairs;

  for (uint32_t idx = start; idx < end; idx++) {
    uint32_t pair = idx % half_dim;
    uint32_t tmp = idx / half_dim;
    uint32_t token = tmp % seqlen;
    uint32_t head = tmp / seqlen;
    uint32_t base = (head * seqlen + token) * head_dim + (pair << 1);
    float x0 = (float)x_pm[base + 0];
    float x1 = (float)x_pm[base + 1];
    float c = (float)cos_tbl[token * half_dim + pair];
    float s = (float)sin_tbl[token * half_dim + pair];
    x_pm[base + 0] = (__fp16)(x0 * c - x1 * s);
    x_pm[base + 1] = (__fp16)(x0 * s + x1 * c);
  }
}

static inline void
llm_attention_k_to_kt_f16(const __fp16 *k_pm, __fp16 *kt_pm, uint32_t seqlen,
                          uint32_t num_kv_heads, uint32_t head_dim,
                          uint32_t core_id, uint32_t num_cores) {
  const uint32_t n = num_kv_heads * seqlen * head_dim;
  uint32_t chunk = (n + num_cores - 1) / num_cores;
  uint32_t start = core_id * chunk;
  uint32_t end = start + chunk;
  if (end > n)
    end = n;

  for (uint32_t idx = start; idx < end; idx++) {
    uint32_t d = idx % head_dim;
    uint32_t tmp = idx / head_dim;
    uint32_t token = tmp % seqlen;
    uint32_t head = tmp / seqlen;
    kt_pm[(head * head_dim + d) * seqlen + token] = k_pm[idx];
  }
}

static inline void llm_attention_stage_decode_cache_f16(
    const __fp16 *l2_k_cache_t, const __fp16 *l2_v_cache, __fp16 *l1_k_cache_t,
    __fp16 *l1_v_cache, uint32_t past_len, uint32_t total_len,
    uint32_t num_kv_heads, uint32_t head_dim, uint32_t core_id,
    uint32_t num_cores) {
  const uint32_t kt_n = num_kv_heads * head_dim * past_len;
  uint32_t chunk = (kt_n + num_cores - 1) / num_cores;
  uint32_t start = core_id * chunk;
  uint32_t end = start + chunk;
  if (end > kt_n)
    end = kt_n;

  for (uint32_t idx = start; idx < end; idx++) {
    uint32_t token = idx % past_len;
    uint32_t tmp = idx / past_len;
    uint32_t d = tmp % head_dim;
    uint32_t head = tmp / head_dim;
    l1_k_cache_t[(head * head_dim + d) * total_len + token] =
        l2_k_cache_t[(head * head_dim + d) * past_len + token];
  }

  const uint32_t v_n = num_kv_heads * past_len * head_dim;
  chunk = (v_n + num_cores - 1) / num_cores;
  start = core_id * chunk;
  end = start + chunk;
  if (end > v_n)
    end = v_n;

  for (uint32_t idx = start; idx < end; idx++) {
    uint32_t d = idx % head_dim;
    uint32_t tmp = idx / head_dim;
    uint32_t token = tmp % past_len;
    uint32_t head = tmp / past_len;
    l1_v_cache[(head * total_len + token) * head_dim + d] =
        l2_v_cache[(head * past_len + token) * head_dim + d];
  }
}

static inline void
llm_attention_append_kv_f16(const __fp16 *k_new_pm, const __fp16 *v_new_pm,
                            __fp16 *kt_cache_pm, __fp16 *v_cache_pm,
                            uint32_t past_len, uint32_t seqlen,
                            uint32_t num_kv_heads, uint32_t head_dim,
                            uint32_t core_id, uint32_t num_cores) {
  const uint32_t total_len = past_len + seqlen;
  const uint32_t n = num_kv_heads * seqlen * head_dim;
  uint32_t chunk = (n + num_cores - 1) / num_cores;
  uint32_t start = core_id * chunk;
  uint32_t end = start + chunk;
  if (end > n)
    end = n;

  for (uint32_t idx = start; idx < end; idx++) {
    uint32_t d = idx % head_dim;
    uint32_t tmp = idx / head_dim;
    uint32_t token = tmp % seqlen;
    uint32_t head = tmp / seqlen;
    uint32_t cache_token = past_len + token;
    __fp16 kval = k_new_pm[(head * seqlen + token) * head_dim + d];
    __fp16 vval = v_new_pm[(head * seqlen + token) * head_dim + d];
    kt_cache_pm[(head * head_dim + d) * total_len + cache_token] = kval;
    v_cache_pm[(head * total_len + cache_token) * head_dim + d] = vval;
  }
}

static inline void llm_attention_select_window_f16(
    const __fp16 *kt_cache_pm, const __fp16 *v_cache_pm, __fp16 *kt_window_pm,
    __fp16 *v_window_pm, uint32_t total_len, uint32_t first_token,
    uint32_t attn_len, uint32_t num_kv_heads, uint32_t head_dim,
    uint32_t core_id, uint32_t num_cores) {
  const uint32_t kt_n = num_kv_heads * head_dim * attn_len;
  uint32_t chunk = (kt_n + num_cores - 1) / num_cores;
  uint32_t start = core_id * chunk;
  uint32_t end = start + chunk;
  if (end > kt_n)
    end = kt_n;
  for (uint32_t idx = start; idx < end; idx++) {
    uint32_t col = idx % attn_len;
    uint32_t tmp = idx / attn_len;
    uint32_t d = tmp % head_dim;
    uint32_t head = tmp / head_dim;
    kt_window_pm[(head * head_dim + d) * attn_len + col] =
        kt_cache_pm[(head * head_dim + d) * total_len + first_token + col];
  }

  const uint32_t v_n = num_kv_heads * attn_len * head_dim;
  chunk = (v_n + num_cores - 1) / num_cores;
  start = core_id * chunk;
  end = start + chunk;
  if (end > v_n)
    end = v_n;
  for (uint32_t idx = start; idx < end; idx++) {
    uint32_t d = idx % head_dim;
    uint32_t tmp = idx / head_dim;
    uint32_t col = tmp % attn_len;
    uint32_t head = tmp / attn_len;
    v_window_pm[(head * attn_len + col) * head_dim + d] =
        v_cache_pm[(head * total_len + first_token + col) * head_dim + d];
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
  const uint32_t n = num_q_heads * seqlen * attn_len;
  const float scale = llm_rsqrt((float)head_dim);
  uint32_t chunk = (n + num_cores - 1) / num_cores;
  uint32_t start = core_id * chunk;
  uint32_t end = start + chunk;
  if (end > n)
    end = n;

  for (uint32_t idx = start; idx < end; idx++) {
    uint32_t key_col = idx % attn_len;
    uint32_t tmp = idx / attn_len;
    uint32_t token = tmp % seqlen;
    uint32_t key_pos = first_key_pos + key_col;
    uint32_t last_visible = past_len + token;
    if (key_pos > last_visible) {
      scores_pm[idx] = (__fp16)-80.0f;
    } else {
      scores_pm[idx] = (__fp16)((float)scores_pm[idx] * scale);
    }
  }
}

static inline void llm_attention_softmax_rows_f16(__fp16 *scores, uint32_t rows,
                                                  uint32_t cols,
                                                  uint32_t core_id,
                                                  uint32_t num_cores) {
  for (uint32_t r = core_id; r < rows; r += num_cores) {
    __fp16 *row = scores + r * cols;
#ifdef LLM_SPATZ_VEC
    // Vector softmax: max -> exp(x-max) -> sum -> scale. exp on the vector
    // unit; the row sum is a reduction scalar so its reciprocal stays scalar
    // (via the divide-free llm_rsqrt of sum*sum, sum>0).
    const float NEG_BIG = -1e30f;
    const __fp16 *p = row;
    uint32_t avl = cols;
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(cols));
    asm volatile("vfmv.s.f v30, %0" ::"f"(NEG_BIG));
    while (avl > 0) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(avl));
      asm volatile("vle16.v v0, (%0)" ::"r"(p));
      asm volatile("vfwcvt.f.f.v v8, v0");
      asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vl));
      asm volatile("vfredmax.vs v30, v8, v30");
      p += vl;
      avl -= vl;
    }
    float maxv;
    asm volatile("vfmv.f.s %0, v30" : "=f"(maxv));

    p = row;
    avl = cols;
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(cols));
    asm volatile("vfmv.s.f v30, %0" ::"f"((float)0.0f));
    while (avl > 0) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(avl));
      asm volatile("vle16.v v0, (%0)" ::"r"(p));
      asm volatile("vfwcvt.f.f.v v16, v0");
      asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(vl));
      asm volatile("vfsub.vf v16, v16, %0" ::"f"(maxv)); // x - max
      llm_rvv_expf_m2();                                 // exp(x-max) in v16
      asm volatile("vfredusum.vs v30, v16, v30");        // running sum
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(vl));
      asm volatile("vfncvt.f.f.w v0, v16");
      asm volatile("vse16.v v0, (%0)" ::"r"(p));
      p += vl;
      avl -= vl;
    }
    float sum;
    asm volatile("vfmv.f.s %0, v30" : "=f"(sum));
    float inv = llm_f16box(llm_rsqrt(sum * sum)); // 1/sum, sum>0

    p = row;
    avl = cols;
    while (avl > 0) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(avl));
      asm volatile("vle16.v v0, (%0)" ::"r"(p));
      asm volatile("vfmul.vf v0, v0, %0" ::"f"(inv));
      asm volatile("vse16.v v0, (%0)" ::"r"(p));
      p += vl;
      avl -= vl;
    }
#else
    float maxv = (float)row[0];
    for (uint32_t c = 1; c < cols; c++) {
      float v = (float)row[c];
      if (v > maxv)
        maxv = v;
    }

    float sum = 0.0f;
    for (uint32_t c = 0; c < cols; c++) {
      float e = llm_expf((float)row[c] - maxv);
      row[c] = (__fp16)e;
      sum += e;
    }

    // At least one term is exp(0) = 1, so sum is positive and normal.
    float inv_sum = llm_recip_pos(sum);
    for (uint32_t c = 0; c < cols; c++)
      row[c] = (__fp16)((float)row[c] * inv_sum);
#endif
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
// Per key tile: RedMulE Sj = Q @ Ktile^T; Spatz/scalar online-softmax update
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
  const uint32_t n = num_kv * head_dim * bn;
  uint32_t chunk = (n + num_cores - 1) / num_cores;
  uint32_t start = core_id * chunk;
  uint32_t end = start + chunk;
  if (end > n)
    end = n;
  for (uint32_t idx = start; idx < end; idx++) {
    uint32_t col = idx % bn;
    uint32_t tmp = idx / bn;
    uint32_t d = tmp % head_dim;
    uint32_t hd = tmp / head_dim;
    kt_tile[(hd * head_dim + d) * bn + col] =
        kt_full[(hd * head_dim + d) * len + j0 + col];
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
  const uint32_t rows = num_q * seqlen;
  for (uint32_t r = core_id; r < rows; r += num_cores) {
    uint32_t qh = r / seqlen;
    uint32_t i = r % seqlen;
    __fp16 *row = sj + r * bn;
    float mprev = m[r];
    float mtile = -1e30f;
    for (uint32_t c = 0; c < bn; c++) {
      uint32_t key_pos = first_key_pos + j0 + c;
      float v = (key_pos > past_len + i) ? -80.0f : (float)row[c] * scale;
      row[c] = (__fp16)v;
      if (v > mtile)
        mtile = v;
    }
    float mnew = (mtile > mprev) ? mtile : mprev;
    float sump = 0.0f;
    for (uint32_t c = 0; c < bn; c++) {
      float p = llm_expf((float)row[c] - mnew);
      row[c] = (__fp16)p;
      sump += p;
    }
    float corr = is_first ? 0.0f : llm_expf(mprev - mnew);
    l[r] = l[r] * corr + sump;
    m[r] = mnew;
    if (!is_first) {
      __fp16 *orow = o + (qh * seqlen + i) * head_dim;
      for (uint32_t d = 0; d < head_dim; d++)
        orow[d] = (__fp16)((float)orow[d] * corr);
    }
  }
}

// Finalize: O[row] /= l[row]  (l > 0, reciprocal via divide-free rsqrt).
static inline void
llm_attention_flash_finalize_f16(__fp16 *o, const float *l, uint32_t seqlen,
                                 uint32_t num_q, uint32_t head_dim,
                                 uint32_t core_id, uint32_t num_cores) {
  const uint32_t rows = num_q * seqlen;
  for (uint32_t r = core_id; r < rows; r += num_cores) {
    float inv = llm_rsqrt(l[r] * l[r]);
    __fp16 *orow = o + r * head_dim;
    for (uint32_t d = 0; d < head_dim; d++)
      orow[d] = (__fp16)((float)orow[d] * inv);
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
