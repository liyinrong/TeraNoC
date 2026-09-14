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
#include "baremetal/mempool_llm_perf.h"

typedef struct {
  llm_stage_cycles_t input_dma;
  llm_stage_cycles_t q_projection;
  llm_stage_cycles_t k_projection;
  llm_stage_cycles_t v_projection;
  llm_stage_cycles_t kv_cache_load;
  llm_stage_cycles_t kv_cache_append;
  llm_stage_cycles_t k_transpose;
  llm_stage_cycles_t qk_gemm;
  llm_stage_cycles_t scale_mask;
  llm_stage_cycles_t softmax;
  llm_stage_cycles_t pv_gemm;
  llm_stage_cycles_t output_projection;
  llm_stage_cycles_t output_dma;
} llm_classic_attention_perf_t;

// Fields are totals over repeated tiles and heads. In the Flash path,
// input_dma includes both K/V- and Q-input tile loads, kv_cache_append is the
// new K/V spill to L2, kv_cache_load stages mixed old/new K/V tiles, and
// softmax includes online initialization, update, and finalization.

static inline void llm_classic_attention_prefill_materialized_f16(
    const __fp16 *l2_x, const __fp16 *l2_wq, const __fp16 *l2_wk,
    const __fp16 *l2_wv, const __fp16 *l2_wo, __fp16 *l2_out, __fp16 *l1_x,
    __fp16 *l1_q, __fp16 *l1_k, __fp16 *l1_v, __fp16 *l1_kt, __fp16 *l1_scores,
    __fp16 *l1_ctx, __fp16 *l1_out, __fp16 *l1_wbuf, uint32_t seqlen,
    uint32_t dmodel, uint32_t num_heads, uint32_t head_dim, uint32_t core_id,
    uint32_t num_cores, uint32_t redmule_id,
    llm_classic_attention_perf_t *perf) {
  const uint32_t ne_used = llm_attention_ne_used(seqlen);
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    dma_collective_memcpy_blocking(l1_x, l2_x,
                                   seqlen * dmodel * sizeof(__fp16));
    llm_stage_timer_stop(stage_start, &perf->input_dma);
  }

  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_proj_heads(l2_wq, l1_x, l1_q, l1_wbuf, seqlen, dmodel,
                             head_dim, num_heads, core_id, num_cores,
                             redmule_id, ne_used);
    llm_stage_timer_stop(stage_start, &perf->q_projection);
  }
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_proj_heads(l2_wk, l1_x, l1_k, l1_wbuf, seqlen, dmodel,
                             head_dim, num_heads, core_id, num_cores,
                             redmule_id, ne_used);
    llm_stage_timer_stop(stage_start, &perf->k_projection);
  }
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_proj_heads(l2_wv, l1_x, l1_v, l1_wbuf, seqlen, dmodel,
                             head_dim, num_heads, core_id, num_cores,
                             redmule_id, ne_used);
    llm_stage_timer_stop(stage_start, &perf->v_projection);
  }

  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_k_to_kt_f16(l1_k, l1_kt, seqlen, num_heads, head_dim, core_id,
                              num_cores);
    mempool_barrier(num_cores);
    llm_stage_timer_stop(stage_start, &perf->k_transpose);
  }
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_qk_matmul_f16(l1_q, l1_kt, l1_scores, seqlen, seqlen,
                                num_heads, num_heads, head_dim, num_cores,
                                redmule_id);
    llm_stage_timer_stop(stage_start, &perf->qk_gemm);
  }
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_scale_mask_f16(l1_scores, seqlen, seqlen, num_heads, head_dim,
                                 0, 0, core_id, num_cores);
    mempool_barrier(num_cores);
    llm_stage_timer_stop(stage_start, &perf->scale_mask);
  }
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_softmax_rows_f16(l1_scores, num_heads * seqlen, seqlen,
                                   core_id, num_cores);
    mempool_barrier(num_cores);
    llm_stage_timer_stop(stage_start, &perf->softmax);
  }
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_sv_matmul_f16(l1_scores, l1_v, l1_ctx, seqlen, seqlen,
                                num_heads, num_heads, head_dim, num_cores,
                                redmule_id);
    llm_stage_timer_stop(stage_start, &perf->pv_gemm);
  }
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_out_proj_f16(l2_wo, l1_ctx, l1_out, l1_wbuf, seqlen, dmodel,
                               head_dim, num_heads, core_id, num_cores,
                               redmule_id, ne_used);
    llm_stage_timer_stop(stage_start, &perf->output_projection);
  }

  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    dma_collective_memcpy_blocking(l2_out, l1_out,
                                   seqlen * dmodel * sizeof(__fp16));
    llm_stage_timer_stop(stage_start, &perf->output_dma);
  }
}

static inline void llm_classic_attention_decode_materialized_f16(
    const __fp16 *l2_x, const __fp16 *l2_k_cache, const __fp16 *l2_v_cache,
    const __fp16 *l2_wq, const __fp16 *l2_wk, const __fp16 *l2_wv,
    const __fp16 *l2_wo, __fp16 *l2_out, __fp16 *l1_x, __fp16 *l1_q,
    __fp16 *l1_k, __fp16 *l1_v, __fp16 *l1_k_full, __fp16 *l1_v_full,
    __fp16 *l1_kt, __fp16 *l1_scores, __fp16 *l1_ctx, __fp16 *l1_out,
    __fp16 *l1_wbuf, uint32_t seqlen, uint32_t past_len, uint32_t dmodel,
    uint32_t num_heads, uint32_t head_dim, uint32_t core_id, uint32_t num_cores,
    uint32_t redmule_id, llm_classic_attention_perf_t *perf) {
  const uint32_t total_len = past_len + seqlen;
  const uint32_t ne_used = llm_attention_ne_used(seqlen);
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    dma_collective_memcpy_blocking(l1_x, l2_x,
                                   seqlen * dmodel * sizeof(__fp16));
    llm_stage_timer_stop(stage_start, &perf->input_dma);
  }

  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_proj_heads(l2_wq, l1_x, l1_q, l1_wbuf, seqlen, dmodel,
                             head_dim, num_heads, core_id, num_cores,
                             redmule_id, ne_used);
    llm_stage_timer_stop(stage_start, &perf->q_projection);
  }
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_proj_heads(l2_wk, l1_x, l1_k, l1_wbuf, seqlen, dmodel,
                             head_dim, num_heads, core_id, num_cores,
                             redmule_id, ne_used);
    llm_stage_timer_stop(stage_start, &perf->k_projection);
  }
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_proj_heads(l2_wv, l1_x, l1_v, l1_wbuf, seqlen, dmodel,
                             head_dim, num_heads, core_id, num_cores,
                             redmule_id, ne_used);
    llm_stage_timer_stop(stage_start, &perf->v_projection);
  }

  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    dma_collective_op_t op = {.seq = 0};
    for (uint32_t h = 0; h < num_heads; h++) {
      dma_collective_memcpy_nonblocking(l1_k_full + h * total_len * head_dim,
                                        l2_k_cache + h * past_len * head_dim,
                                        past_len * head_dim * sizeof(__fp16));
      op = dma_collective_memcpy_nonblocking(
          l1_v_full + h * total_len * head_dim,
          l2_v_cache + h * past_len * head_dim,
          past_len * head_dim * sizeof(__fp16));
    }
    if (num_heads != 0u)
      dma_collective_wait(&op);
    else
      mempool_barrier(num_cores);
    llm_stage_timer_stop(stage_start, &perf->kv_cache_load);
  }

  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    const uint32_t vlmax = llm_attention_vlmax_e16_m8();
    const uint32_t head_elems = seqlen * head_dim;
    const uint32_t blocks = (head_elems + vlmax - 1) / vlmax;
    const uint32_t tasks = num_heads * blocks;
    for (uint32_t task = core_id; task < tasks; task += num_cores) {
      const uint32_t block = task % blocks;
      const uint32_t head = task / blocks;
      const uint32_t elem0 = block * vlmax;
      const uint32_t count = llm_attention_min_u32(vlmax, head_elems - elem0);
      const __fp16 *ksrc = l1_k + head * head_elems + elem0;
      const __fp16 *vsrc = l1_v + head * head_elems + elem0;
      __fp16 *kdst =
          l1_k_full + (head * total_len + past_len) * head_dim + elem0;
      __fp16 *vdst =
          l1_v_full + (head * total_len + past_len) * head_dim + elem0;
      asm volatile("vsetvli zero, %0, e16, m8, ta, ma" : : "r"(count));
      asm volatile("vle16.v v0, (%0)" : : "r"(ksrc) : "memory");
      asm volatile("vle16.v v8, (%0)" : : "r"(vsrc) : "memory");
      asm volatile("vse16.v v0, (%0)" : : "r"(kdst) : "memory");
      asm volatile("vse16.v v8, (%0)" : : "r"(vdst) : "memory");
    }
    mempool_barrier(num_cores);
    llm_stage_timer_stop(stage_start, &perf->kv_cache_append);
  }

  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_k_to_kt_f16(l1_k_full, l1_kt, total_len, num_heads, head_dim,
                              core_id, num_cores);
    mempool_barrier(num_cores);
    llm_stage_timer_stop(stage_start, &perf->k_transpose);
  }
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_qk_matmul_f16(l1_q, l1_kt, l1_scores, seqlen, total_len,
                                num_heads, num_heads, head_dim, num_cores,
                                redmule_id);
    llm_stage_timer_stop(stage_start, &perf->qk_gemm);
  }
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_scale_mask_f16(l1_scores, seqlen, total_len, num_heads,
                                 head_dim, 0, past_len, core_id, num_cores);
    mempool_barrier(num_cores);
    llm_stage_timer_stop(stage_start, &perf->scale_mask);
  }
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_softmax_rows_f16(l1_scores, num_heads * seqlen, total_len,
                                   core_id, num_cores);
    mempool_barrier(num_cores);
    llm_stage_timer_stop(stage_start, &perf->softmax);
  }
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_sv_matmul_f16(l1_scores, l1_v_full, l1_ctx, seqlen, total_len,
                                num_heads, num_heads, head_dim, num_cores,
                                redmule_id);
    llm_stage_timer_stop(stage_start, &perf->pv_gemm);
  }
  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    llm_attention_out_proj_f16(l2_wo, l1_ctx, l1_out, l1_wbuf, seqlen, dmodel,
                               head_dim, num_heads, core_id, num_cores,
                               redmule_id, ne_used);
    llm_stage_timer_stop(stage_start, &perf->output_projection);
  }

  {
    const mempool_timer_t stage_start = llm_stage_timer_start();
    dma_collective_memcpy_blocking(l2_out, l1_out,
                                   seqlen * dmodel * sizeof(__fp16));
    llm_stage_timer_stop(stage_start, &perf->output_dma);
  }
}

#ifdef LLM_FLASH_ATTENTION

static inline void llm_classic_attention_project_one_f16(
    const __fp16 *l2_w, const __fp16 *l1_x, __fp16 *l1_proj, __fp16 *l1_wbuf,
    uint32_t rows, uint32_t dmodel, uint32_t head_dim, uint32_t core_id,
    uint32_t num_cores, uint32_t redmule_id) {
  const uint32_t ne_used = llm_attention_ne_used(rows);
  const uint32_t rows_per_engine = rows / ne_used;
  dma_collective_memcpy_blocking(l1_wbuf, l2_w,
                                 dmodel * head_dim * sizeof(__fp16));
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
    uint32_t num_cores, uint32_t redmule_id,
    llm_classic_attention_perf_t *perf) {
  const uint32_t w_elems = dmodel * head_dim;
  for (uint32_t row0 = 0; row0 < seqlen; row0 += bm) {
    {
      const mempool_timer_t stage_start = llm_stage_timer_start();
      dma_collective_memcpy_blocking(l1_x, l2_x + row0 * dmodel,
                                     bm * dmodel * sizeof(__fp16));
      llm_stage_timer_stop(stage_start, &perf->input_dma);
    }
    for (uint32_t h = 0; h < num_heads; h++) {
      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        llm_classic_attention_project_one_f16(
            l2_wk + h * w_elems, l1_x, l1_k_tile, l1_wbuf, bm, dmodel, head_dim,
            core_id, num_cores, redmule_id);
        llm_stage_timer_stop(stage_start, &perf->k_projection);
      }
      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        dma_collective_memcpy_blocking(
            l2_k_new + (h * seqlen + row0) * head_dim, l1_k_tile,
            bm * head_dim * sizeof(__fp16));
        llm_stage_timer_stop(stage_start, &perf->kv_cache_append);
      }

      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        llm_classic_attention_project_one_f16(
            l2_wv + h * w_elems, l1_x, l1_v_tile, l1_wbuf, bm, dmodel, head_dim,
            core_id, num_cores, redmule_id);
        llm_stage_timer_stop(stage_start, &perf->v_projection);
      }
      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        dma_collective_memcpy_blocking(
            l2_v_new + (h * seqlen + row0) * head_dim, l1_v_tile,
            bm * head_dim * sizeof(__fp16));
        llm_stage_timer_stop(stage_start, &perf->kv_cache_append);
      }
    }
  }
}

static inline void llm_classic_attention_stage_kv_tile_f16(
    const __fp16 *l2_k_cache, const __fp16 *l2_v_cache, const __fp16 *l2_k_new,
    const __fp16 *l2_v_new, __fp16 *l1_k, __fp16 *l1_v, uint32_t head,
    uint32_t j0, uint32_t bn, uint32_t past_len, uint32_t seqlen,
    uint32_t head_dim, uint32_t core_id, uint32_t num_cores) {
  (void)num_cores;
  // Every core calls; each group's coordinator issues its own stripe.
  uint32_t past_tokens = 0;
  if (j0 < past_len)
    past_tokens = llm_attention_min_u32(bn, past_len - j0);

  if (past_tokens > 0) {
    const uint32_t src = (head * past_len + j0) * head_dim;
    const uint32_t elems = past_tokens * head_dim;
    dma_collective_memcpy_nonblocking(l1_k, l2_k_cache + src,
                                      elems * sizeof(__fp16));
    dma_collective_memcpy_blocking(l1_v, l2_v_cache + src,
                                   elems * sizeof(__fp16));
  }

  const uint32_t new_tokens = bn - past_tokens;
  if (new_tokens > 0) {
    const uint32_t first_new = j0 + past_tokens - past_len;
    const uint32_t src = (head * seqlen + first_new) * head_dim;
    const uint32_t dst = past_tokens * head_dim;
    const uint32_t elems = new_tokens * head_dim;
    dma_collective_memcpy_nonblocking(l1_k + dst, l2_k_new + src,
                                      elems * sizeof(__fp16));
    dma_collective_memcpy_blocking(l1_v + dst, l2_v_new + src,
                                   elems * sizeof(__fp16));
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
    uint32_t bn, uint32_t core_id, uint32_t num_cores, uint32_t redmule_id,
    llm_classic_attention_perf_t *perf) {
  const uint32_t total_len = past_len + seqlen;
  const uint32_t w_elems = dmodel * head_dim;

  llm_classic_attention_project_kv_f16(
      l2_x, l2_wk, l2_wv, l2_k_new, l2_v_new, l1_x, l1_k, l1_v, l1_wbuf, seqlen,
      dmodel, num_heads, head_dim, bm, core_id, num_cores, redmule_id, perf);

  for (uint32_t q0 = 0; q0 < seqlen; q0 += bm) {
    {
      const mempool_timer_t stage_start = llm_stage_timer_start();
      dma_collective_memcpy_blocking(l1_x, l2_x + q0 * dmodel,
                                     bm * dmodel * sizeof(__fp16));
      llm_stage_timer_stop(stage_start, &perf->input_dma);
    }

    for (uint32_t h = 0; h < num_heads; h++) {
      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        llm_classic_attention_project_one_f16(l2_wq + h * w_elems, l1_x, l1_q,
                                              l1_wbuf, bm, dmodel, head_dim,
                                              core_id, num_cores, redmule_id);
        llm_stage_timer_stop(stage_start, &perf->q_projection);
      }

      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        for (uint32_t r = core_id; r < bm; r += num_cores) {
          l1_m[r] = -1e30f;
          l1_l[r] = 0.0f;
        }
        mempool_barrier(num_cores);
        llm_stage_timer_stop(stage_start, &perf->softmax);
      }

      for (uint32_t j0 = 0; j0 < total_len; j0 += bn) {
        const uint32_t first = (j0 == 0);
        {
          const mempool_timer_t stage_start = llm_stage_timer_start();
          llm_classic_attention_stage_kv_tile_f16(
              l2_k_cache, l2_v_cache, l2_k_new, l2_v_new, l1_k, l1_v, h, j0, bn,
              past_len, seqlen, head_dim, core_id, num_cores);
          mempool_barrier(num_cores);
          llm_stage_timer_stop(stage_start, &perf->kv_cache_load);
        }
        {
          const mempool_timer_t stage_start = llm_stage_timer_start();
          llm_attention_k_to_kt_f16(l1_k, l1_kt, bn, 1, head_dim, core_id,
                                    num_cores);
          mempool_barrier(num_cores);
          llm_stage_timer_stop(stage_start, &perf->k_transpose);
        }

        {
          const mempool_timer_t stage_start = llm_stage_timer_start();
          llm_classic_attention_gemm_rows_f16(l1_q, l1_kt, l1_scores, bm,
                                              head_dim, bn, 1, num_cores,
                                              redmule_id);
          llm_stage_timer_stop(stage_start, &perf->qk_gemm);
        }
        {
          const mempool_timer_t stage_start = llm_stage_timer_start();
          llm_attention_flash_update_f16(l1_scores, l1_ctx, l1_m, l1_l, bm, bn,
                                         1, head_dim, j0, 0, past_len + q0,
                                         first, core_id, num_cores);
          mempool_barrier(num_cores);
          llm_stage_timer_stop(stage_start, &perf->softmax);
        }
        {
          const mempool_timer_t stage_start = llm_stage_timer_start();
          llm_classic_attention_gemm_rows_f16(l1_scores, l1_v, l1_ctx, bm, bn,
                                              head_dim, first, num_cores,
                                              redmule_id);
          llm_stage_timer_stop(stage_start, &perf->pv_gemm);
        }
      }

      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        llm_attention_flash_finalize_f16(l1_ctx, l1_l, bm, 1, head_dim, core_id,
                                         num_cores);
        mempool_barrier(num_cores);
        llm_stage_timer_stop(stage_start, &perf->softmax);
      }

      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        dma_collective_memcpy_blocking(l1_wbuf, l2_wo + h * w_elems,
                                       w_elems * sizeof(__fp16));
        llm_classic_attention_gemm_rows_f16(l1_ctx, l1_wbuf, l1_out, bm,
                                            head_dim, dmodel, h == 0, num_cores,
                                            redmule_id);
        llm_stage_timer_stop(stage_start, &perf->output_projection);
      }
    }

    {
      const mempool_timer_t stage_start = llm_stage_timer_start();
      dma_collective_memcpy_blocking(l2_out + q0 * dmodel, l1_out,
                                     bm * dmodel * sizeof(__fp16));
      llm_stage_timer_stop(stage_start, &perf->output_dma);
    }
  }
}

#endif // LLM_FLASH_ATTENTION
