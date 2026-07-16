// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Classic causal multi-head attention:
//
//   Q/K/V  = x @ Wq/Wk/Wv
//   P      = softmax(mask(Q @ K^T / sqrt(HD)))
//   y      = concat(P @ V) @ Wo
//
// All heads have equal Q/K/V cardinality.  There is no RMSNorm, RoPE, GQA,
// QK-normalization, sliding window, or residual path.  The materialized path
// is the small-shape numerical reference.  LLM_FLASH_ATTENTION selects a true
// BM-by-BN path that keeps projected K/V in L2 and only query/key tiles in L1.

#pragma once

#include <stdint.h>

#include "dma.h"
#include "runtime.h"
#include "synchronization.h"

#include "baremetal/mempool_llm_attention_f16.h"

static inline void llm_classic_attention_prefill_materialized_f16(
    const __fp16 *l2_x, const __fp16 *l2_wq, const __fp16 *l2_wk,
    const __fp16 *l2_wv, const __fp16 *l2_wo, __fp16 *l2_out, __fp16 *l1_x,
    __fp16 *l1_q, __fp16 *l1_k, __fp16 *l1_v, __fp16 *l1_kt, __fp16 *l1_scores,
    __fp16 *l1_ctx, __fp16 *l1_out, __fp16 *l1_wbuf, uint32_t seqlen,
    uint32_t dmodel, uint32_t num_heads, uint32_t head_dim, uint32_t core_id,
    uint32_t num_cores, uint32_t redmule_id) {
  const uint32_t ne_used = llm_attention_ne_used(seqlen);
  if (core_id == 0)
    dma_memcpy_blocking(l1_x, l2_x, seqlen * dmodel * sizeof(__fp16));
  mempool_barrier(num_cores);

  llm_attention_proj_heads(l2_wq, l1_x, l1_q, l1_wbuf, seqlen, dmodel, head_dim,
                           num_heads, core_id, num_cores, redmule_id, ne_used);
  llm_attention_proj_heads(l2_wk, l1_x, l1_k, l1_wbuf, seqlen, dmodel, head_dim,
                           num_heads, core_id, num_cores, redmule_id, ne_used);
  llm_attention_proj_heads(l2_wv, l1_x, l1_v, l1_wbuf, seqlen, dmodel, head_dim,
                           num_heads, core_id, num_cores, redmule_id, ne_used);

  llm_attention_k_to_kt_f16(l1_k, l1_kt, seqlen, num_heads, head_dim, core_id,
                            num_cores);
  mempool_barrier(num_cores);
  llm_attention_qk_matmul_f16(l1_q, l1_kt, l1_scores, seqlen, seqlen, num_heads,
                              num_heads, head_dim, num_cores, redmule_id);
  llm_attention_scale_mask_f16(l1_scores, seqlen, seqlen, num_heads, head_dim,
                               0, 0, core_id, num_cores);
  mempool_barrier(num_cores);
  llm_attention_softmax_rows_f16(l1_scores, num_heads * seqlen, seqlen, core_id,
                                 num_cores);
  mempool_barrier(num_cores);
  llm_attention_sv_matmul_f16(l1_scores, l1_v, l1_ctx, seqlen, seqlen,
                              num_heads, num_heads, head_dim, num_cores,
                              redmule_id);
  llm_attention_out_proj_f16(l2_wo, l1_ctx, l1_out, l1_wbuf, seqlen, dmodel,
                             head_dim, num_heads, core_id, num_cores,
                             redmule_id, ne_used);

  if (core_id == 0)
    dma_memcpy_blocking(l2_out, l1_out, seqlen * dmodel * sizeof(__fp16));
  mempool_barrier(num_cores);
}

static inline void llm_classic_attention_decode_materialized_f16(
    const __fp16 *l2_x, const __fp16 *l2_k_cache, const __fp16 *l2_v_cache,
    const __fp16 *l2_wq, const __fp16 *l2_wk, const __fp16 *l2_wv,
    const __fp16 *l2_wo, __fp16 *l2_out, __fp16 *l1_x, __fp16 *l1_q,
    __fp16 *l1_k, __fp16 *l1_v, __fp16 *l1_k_full, __fp16 *l1_v_full,
    __fp16 *l1_kt, __fp16 *l1_scores, __fp16 *l1_ctx, __fp16 *l1_out,
    __fp16 *l1_wbuf, uint32_t seqlen, uint32_t past_len, uint32_t dmodel,
    uint32_t num_heads, uint32_t head_dim, uint32_t core_id, uint32_t num_cores,
    uint32_t redmule_id) {
  const uint32_t total_len = past_len + seqlen;
  const uint32_t ne_used = llm_attention_ne_used(seqlen);
  if (core_id == 0)
    dma_memcpy_blocking(l1_x, l2_x, seqlen * dmodel * sizeof(__fp16));
  mempool_barrier(num_cores);

  llm_attention_proj_heads(l2_wq, l1_x, l1_q, l1_wbuf, seqlen, dmodel, head_dim,
                           num_heads, core_id, num_cores, redmule_id, ne_used);
  llm_attention_proj_heads(l2_wk, l1_x, l1_k, l1_wbuf, seqlen, dmodel, head_dim,
                           num_heads, core_id, num_cores, redmule_id, ne_used);
  llm_attention_proj_heads(l2_wv, l1_x, l1_v, l1_wbuf, seqlen, dmodel, head_dim,
                           num_heads, core_id, num_cores, redmule_id, ne_used);

  if (core_id == 0) {
    for (uint32_t h = 0; h < num_heads; h++) {
      dma_memcpy_blocking(l1_k_full + h * total_len * head_dim,
                          l2_k_cache + h * past_len * head_dim,
                          past_len * head_dim * sizeof(__fp16));
      dma_memcpy_blocking(l1_v_full + h * total_len * head_dim,
                          l2_v_cache + h * past_len * head_dim,
                          past_len * head_dim * sizeof(__fp16));
    }
  }
  mempool_barrier(num_cores);

  const uint32_t nnew = num_heads * seqlen * head_dim;
  uint32_t chunk = (nnew + num_cores - 1) / num_cores;
  uint32_t start = core_id * chunk;
  uint32_t end = start + chunk;
  if (end > nnew)
    end = nnew;
  for (uint32_t idx = start; idx < end; idx++) {
    uint32_t d = idx % head_dim;
    uint32_t tmp = idx / head_dim;
    uint32_t token = tmp % seqlen;
    uint32_t head = tmp / seqlen;
    uint32_t dst = (head * total_len + past_len + token) * head_dim + d;
    l1_k_full[dst] = l1_k[idx];
    l1_v_full[dst] = l1_v[idx];
  }
  mempool_barrier(num_cores);

  llm_attention_k_to_kt_f16(l1_k_full, l1_kt, total_len, num_heads, head_dim,
                            core_id, num_cores);
  mempool_barrier(num_cores);
  llm_attention_qk_matmul_f16(l1_q, l1_kt, l1_scores, seqlen, total_len,
                              num_heads, num_heads, head_dim, num_cores,
                              redmule_id);
  llm_attention_scale_mask_f16(l1_scores, seqlen, total_len, num_heads,
                               head_dim, 0, past_len, core_id, num_cores);
  mempool_barrier(num_cores);
  llm_attention_softmax_rows_f16(l1_scores, num_heads * seqlen, total_len,
                                 core_id, num_cores);
  mempool_barrier(num_cores);
  llm_attention_sv_matmul_f16(l1_scores, l1_v_full, l1_ctx, seqlen, total_len,
                              num_heads, num_heads, head_dim, num_cores,
                              redmule_id);
  llm_attention_out_proj_f16(l2_wo, l1_ctx, l1_out, l1_wbuf, seqlen, dmodel,
                             head_dim, num_heads, core_id, num_cores,
                             redmule_id, ne_used);

  if (core_id == 0)
    dma_memcpy_blocking(l2_out, l1_out, seqlen * dmodel * sizeof(__fp16));
  mempool_barrier(num_cores);
}

#ifdef LLM_FLASH_ATTENTION

static inline void llm_classic_attention_project_one_f16(
    const __fp16 *l2_w, const __fp16 *l1_x, __fp16 *l1_proj, __fp16 *l1_wbuf,
    uint32_t rows, uint32_t dmodel, uint32_t head_dim, uint32_t core_id,
    uint32_t num_cores, uint32_t redmule_id) {
  const uint32_t ne_used = llm_attention_ne_used(rows);
  const uint32_t rows_per_engine = rows / ne_used;
  if (core_id == 0)
    dma_memcpy_blocking(l1_wbuf, l2_w, dmodel * head_dim * sizeof(__fp16));
  mempool_barrier(num_cores);
  if (redmule_id < ne_used) {
    unsigned int x =
        (unsigned int)(l1_x + redmule_id * rows_per_engine * dmodel);
    unsigned int z =
        (unsigned int)(l1_proj + redmule_id * rows_per_engine * head_dim);
    llm_zero_f16((__fp16 *)z, rows_per_engine * head_dim);
    llm_attention_redmule_job(x, (unsigned int)l1_wbuf, z,
                              (uint16_t)rows_per_engine, (uint16_t)dmodel,
                              (uint16_t)head_dim, GEMM);
  }
  mempool_barrier(num_cores);
}

static inline void llm_classic_attention_project_kv_f16(
    const __fp16 *l2_x, const __fp16 *l2_wk, const __fp16 *l2_wv,
    __fp16 *l2_k_new, __fp16 *l2_v_new, __fp16 *l1_x, __fp16 *l1_k_tile,
    __fp16 *l1_v_tile, __fp16 *l1_wbuf, uint32_t seqlen, uint32_t dmodel,
    uint32_t num_heads, uint32_t head_dim, uint32_t bm, uint32_t core_id,
    uint32_t num_cores, uint32_t redmule_id) {
  const uint32_t w_elems = dmodel * head_dim;
  for (uint32_t row0 = 0; row0 < seqlen; row0 += bm) {
    if (core_id == 0)
      dma_memcpy_blocking(l1_x, l2_x + row0 * dmodel,
                          bm * dmodel * sizeof(__fp16));
    mempool_barrier(num_cores);
    for (uint32_t h = 0; h < num_heads; h++) {
      llm_classic_attention_project_one_f16(
          l2_wk + h * w_elems, l1_x, l1_k_tile, l1_wbuf, bm, dmodel, head_dim,
          core_id, num_cores, redmule_id);
      if (core_id == 0)
        dma_memcpy_blocking(l2_k_new + (h * seqlen + row0) * head_dim,
                            l1_k_tile, bm * head_dim * sizeof(__fp16));
      mempool_barrier(num_cores);

      llm_classic_attention_project_one_f16(
          l2_wv + h * w_elems, l1_x, l1_v_tile, l1_wbuf, bm, dmodel, head_dim,
          core_id, num_cores, redmule_id);
      if (core_id == 0)
        dma_memcpy_blocking(l2_v_new + (h * seqlen + row0) * head_dim,
                            l1_v_tile, bm * head_dim * sizeof(__fp16));
      mempool_barrier(num_cores);
    }
  }
}

static inline void llm_classic_attention_stage_kv_tile_f16(
    const __fp16 *l2_k_cache, const __fp16 *l2_v_cache, const __fp16 *l2_k_new,
    const __fp16 *l2_v_new, __fp16 *l1_k, __fp16 *l1_v, uint32_t head,
    uint32_t j0, uint32_t bn, uint32_t past_len, uint32_t seqlen,
    uint32_t head_dim, uint32_t core_id, uint32_t num_cores) {
  const uint32_t n = bn * head_dim;
  uint32_t chunk = (n + num_cores - 1) / num_cores;
  uint32_t start = core_id * chunk;
  uint32_t end = start + chunk;
  if (end > n)
    end = n;
  for (uint32_t idx = start; idx < end; idx++) {
    uint32_t token = idx / head_dim;
    uint32_t d = idx % head_dim;
    uint32_t pos = j0 + token;
    if (pos < past_len) {
      uint32_t src = (head * past_len + pos) * head_dim + d;
      l1_k[idx] = l2_k_cache[src];
      l1_v[idx] = l2_v_cache[src];
    } else {
      uint32_t src = (head * seqlen + pos - past_len) * head_dim + d;
      l1_k[idx] = l2_k_new[src];
      l1_v[idx] = l2_v_new[src];
    }
  }
}

static inline void
llm_classic_attention_gemm_rows_f16(const __fp16 *x, const __fp16 *w, __fp16 *z,
                                    uint32_t rows, uint32_t contraction,
                                    uint32_t columns, uint32_t clear,
                                    uint32_t num_cores, uint32_t redmule_id) {
  const uint32_t ne_used = llm_attention_ne_used(rows);
  const uint32_t rows_per_engine = rows / ne_used;
  if (redmule_id < ne_used) {
    unsigned int xe =
        (unsigned int)(x + redmule_id * rows_per_engine * contraction);
    unsigned int ze =
        (unsigned int)(z + redmule_id * rows_per_engine * columns);
    if (clear)
      llm_zero_f16((__fp16 *)ze, rows_per_engine * columns);
    llm_attention_redmule_job(xe, (unsigned int)w, ze,
                              (uint16_t)rows_per_engine, (uint16_t)contraction,
                              (uint16_t)columns, GEMM);
  }
  mempool_barrier(num_cores);
}

static inline void llm_classic_attention_flash_f16(
    const __fp16 *l2_x, const __fp16 *l2_k_cache, const __fp16 *l2_v_cache,
    const __fp16 *l2_wq, const __fp16 *l2_wk, const __fp16 *l2_wv,
    const __fp16 *l2_wo, __fp16 *l2_k_new, __fp16 *l2_v_new, __fp16 *l2_out,
    __fp16 *l1_x, __fp16 *l1_q, __fp16 *l1_k, __fp16 *l1_v, __fp16 *l1_kt,
    __fp16 *l1_scores, __fp16 *l1_ctx, __fp16 *l1_out, __fp16 *l1_wbuf,
    float *l1_m, float *l1_l, uint32_t seqlen, uint32_t past_len,
    uint32_t dmodel, uint32_t num_heads, uint32_t head_dim, uint32_t bm,
    uint32_t bn, uint32_t core_id, uint32_t num_cores, uint32_t redmule_id) {
  const uint32_t total_len = past_len + seqlen;
  const uint32_t w_elems = dmodel * head_dim;

  llm_classic_attention_project_kv_f16(
      l2_x, l2_wk, l2_wv, l2_k_new, l2_v_new, l1_x, l1_k, l1_v, l1_wbuf, seqlen,
      dmodel, num_heads, head_dim, bm, core_id, num_cores, redmule_id);

  for (uint32_t q0 = 0; q0 < seqlen; q0 += bm) {
    if (core_id == 0)
      dma_memcpy_blocking(l1_x, l2_x + q0 * dmodel,
                          bm * dmodel * sizeof(__fp16));
    mempool_barrier(num_cores);

    for (uint32_t h = 0; h < num_heads; h++) {
      llm_classic_attention_project_one_f16(l2_wq + h * w_elems, l1_x, l1_q,
                                            l1_wbuf, bm, dmodel, head_dim,
                                            core_id, num_cores, redmule_id);

      for (uint32_t r = core_id; r < bm; r += num_cores) {
        l1_m[r] = -1e30f;
        l1_l[r] = 0.0f;
      }
      mempool_barrier(num_cores);

      for (uint32_t j0 = 0; j0 < total_len; j0 += bn) {
        const uint32_t first = (j0 == 0);
        llm_classic_attention_stage_kv_tile_f16(
            l2_k_cache, l2_v_cache, l2_k_new, l2_v_new, l1_k, l1_v, h, j0, bn,
            past_len, seqlen, head_dim, core_id, num_cores);
        mempool_barrier(num_cores);
        llm_attention_k_to_kt_f16(l1_k, l1_kt, bn, 1, head_dim, core_id,
                                  num_cores);
        mempool_barrier(num_cores);

        llm_classic_attention_gemm_rows_f16(
            l1_q, l1_kt, l1_scores, bm, head_dim, bn, 1, num_cores, redmule_id);
        llm_attention_flash_update_f16(l1_scores, l1_ctx, l1_m, l1_l, bm, bn, 1,
                                       head_dim, j0, 0, past_len + q0, first,
                                       core_id, num_cores);
        mempool_barrier(num_cores);
        llm_classic_attention_gemm_rows_f16(l1_scores, l1_v, l1_ctx, bm, bn,
                                            head_dim, first, num_cores,
                                            redmule_id);
      }

      llm_attention_flash_finalize_f16(l1_ctx, l1_l, bm, 1, head_dim, core_id,
                                       num_cores);
      mempool_barrier(num_cores);

      if (core_id == 0)
        dma_memcpy_blocking(l1_wbuf, l2_wo + h * w_elems,
                            w_elems * sizeof(__fp16));
      mempool_barrier(num_cores);
      llm_classic_attention_gemm_rows_f16(l1_ctx, l1_wbuf, l1_out, bm, head_dim,
                                          dmodel, h == 0, num_cores,
                                          redmule_id);
    }

    if (core_id == 0)
      dma_memcpy_blocking(l2_out + q0 * dmodel, l1_out,
                          bm * dmodel * sizeof(__fp16));
    mempool_barrier(num_cores);
  }
}

#endif // LLM_FLASH_ATTENTION
