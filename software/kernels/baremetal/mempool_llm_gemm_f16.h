// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yinrong Li, ETH Zurich

// Pure-GEMM dense-LLM benchmarks: the representative GEMM workloads of the
// classic Transformer FFN and attention layers, with every non-GEMM layer stage
// removed.
//
//   FFN        hidden = x @ W1 ; y = hidden @ W2
//              (no ReLU, no normalization, no gating, no residual)
//   attention  q/k/v = x @ Wq/Wk/Wv ; scores = q @ k^T ;
//              ctx = scores @ v ; y = concat_h(ctx_h) @ Wo_h
//              (no scale, no causal mask, no softmax, no residual)
//
// Shapes, L2 operand layouts and tiling are identical to the classic layer
// benchmarks (mempool_llm_classic_ffn_f16.h /
// mempool_llm_classic_attention_f16.h), so the two tiers are directly
// comparable: the only difference is which stages execute. Reusing the classic
// tiling also keeps every operand tile a contiguous 1-D DMA, and makes the FFN
// W1 GEMM's panel-major output exactly the W2 GEMM's panel-major contraction
// input.
//
// The K -> K^T marshalling is kept: it is BN*HEAD_DIM element moves against
// BM*HEAD_DIM*BN MACs, and it is GEMM operand staging rather than a layer
// stage. Its cost, and that of every tile/weight DMA, lands in the `stage`
// counter so it can be subtracted from the wall-clock total.
//
// Each distinct GEMM shape is a separately timed segment. Because dropping
// softmax leaves the key-tile loop a plain accumulation, ctx = sum_j
// scores_j @ v_j is exactly scores @ v over the full key length, so the golden
// model stays a chained GEMM sequence.

#pragma once

#include <stdint.h>

#include "dma.h"
#include "runtime.h"
#include "synchronization.h"

// Per-GEMM segment timing is this benchmark's deliverable, so - unlike the
// classic layer benchmarks - it stays enabled for the real (LLM_PERF_ONLY)
// shapes: there are only two (FFN) or four (attention) coarse segments to mark.
// Override with -DLLM_CLASSIC_STAGE_PERF=0 for a single whole-kernel region.
#ifndef LLM_CLASSIC_STAGE_PERF
#define LLM_CLASSIC_STAGE_PERF 1
#endif

#include "baremetal/mempool_llm_attention_f16.h"
#include "baremetal/mempool_llm_perf.h"

#define LLM_GEMM_PORT_WIDTH LLM_ATTENTION_PORT_WIDTH

// Keep the shipped row-only schedule as the default.  The stage-1 DSE enables
// the packed row-by-column schedule through the generated data header so the
// operand layout and the kernel can never silently disagree.
#ifndef LLM_GEMM_2D
#define LLM_GEMM_2D 0
#endif

#ifndef NE_R
#define NE_R 1
#endif

#ifndef NE_C
#define NE_C 1
#endif

static inline void llm_gemm_copy_f16(const __fp16 *src, __fp16 *dst,
                                     uint32_t count) {
  const uint32_t vlmax = llm_attention_vlmax_e16_m8();
  while (count > 0) {
    const uint32_t chunk = llm_attention_min_u32(count, vlmax);
    llm_attention_copy_f16_m8(src, dst, chunk);
    src += chunk;
    dst += chunk;
    count -= chunk;
  }
}

// ---------------------------------------------------------------------------
// FFN: y[S x D] = (x[S x D] @ W1[D x I]) @ W2[I x D]
// ---------------------------------------------------------------------------

typedef struct {
  llm_stage_cycles_t stage;   // token/output tile DMA + W1|W2 panel streaming
  llm_stage_cycles_t gemm_w1; // hidden = x @ W1_p  (ROW_TILE x D x PANEL_I)
  llm_stage_cycles_t gemm_w2; // out   += hidden @ W2_p (ROW_TILE x PANEL_I x D)
} llm_gemm_ffn_perf_t;

// Token-tiled over S and intermediate-tiled over I. W1 is panel-major
// ([NUM_IP][D x PANEL_I]) so panel p is one contiguous DMA; the row-only path
// keeps W2 row-major [I x D], so contraction panel p is one contiguous block.
// Both stream from L2 double-buffered through l1_wbuf, which holds two sets of
// {W1_p, W2_p}.  The shipped schedule row-splits every GEMM.  LLM_GEMM_2D
// instead maps the engines as NE_R row groups by NE_C output-column groups.
// Its generated operands are packed as W1[p][c][D][PANEL_I/NE_C] and
// W2[p][cin][cout][PANEL_I/NE_C][D/NE_C].  Hidden and output shards are packed
// by (row group, column group), so every RedMulE job still sees dense matrices.
//
// l1_x / l1_out: ROW_TILE*D, l1_hidden: ROW_TILE*PANEL_I,
// l1_wbuf: 2 * (D*PANEL_I + PANEL_I*D).
static inline void llm_gemm_ffn_f16(
    const __fp16 *l2_x, const __fp16 *l2_w1, const __fp16 *l2_w2,
    __fp16 *l2_out, __fp16 *l1_x, __fp16 *l1_hidden, __fp16 *l1_out,
    __fp16 *l1_wbuf, uint32_t seqlen, uint32_t dmodel, uint32_t inter,
    uint32_t row_tile, uint32_t panel, uint32_t npanel, uint32_t core_id,
    uint32_t num_cores, uint32_t redmule_id, llm_gemm_ffn_perf_t *perf) {
  (void)inter;
  (void)core_id;
  const uint32_t w1_elems = dmodel * panel;
  const uint32_t w2_elems = panel * dmodel;
  const uint32_t set_elems = w1_elems + w2_elems;
#if LLM_GEMM_2D
  const uint32_t ne_used = NE_R * NE_C;
  const uint32_t rows_per_engine = row_tile / NE_R;
  const uint32_t panel_per_engine = panel / NE_C;
  const uint32_t dmodel_per_engine = dmodel / NE_C;
  const uint32_t cores_per_engine = num_cores / NUM_REDMULE_TILES;
  const uint32_t row_team_cores = NE_C * cores_per_engine;
#else
  const uint32_t ne_used = llm_attention_ne_used(row_tile);
  const uint32_t rows_per_engine = row_tile / ne_used;
#endif

  for (uint32_t row0 = 0; row0 < seqlen; row0 += row_tile) {
    {
      const mempool_timer_t stage_start = llm_stage_timer_start();
      dma_collective_memcpy_nonblocking(l1_x, l2_x + row0 * dmodel,
                                        row_tile * dmodel * sizeof(__fp16));
      dma_collective_memcpy_nonblocking(l1_wbuf, l2_w1,
                                        w1_elems * sizeof(__fp16));
      dma_collective_memcpy_blocking(l1_wbuf + w1_elems, l2_w2,
                                     w2_elems * sizeof(__fp16));
      llm_stage_timer_stop(stage_start, &perf->stage);
    }

    for (uint32_t p = 0; p < npanel; p++) {
      __fp16 *cur = l1_wbuf + ((p & 1) ? set_elems : 0);
      __fp16 *nxt = l1_wbuf + ((p & 1) ? 0 : set_elems);
      __fp16 *w1 = cur;
      __fp16 *w2 = cur + w1_elems;
      const uint32_t prefetch = p + 1 < npanel;
      dma_collective_op_t prefetch_op = {.seq = 0};

      // Issue the next panel's weight prefetch before the timed GEMMs so it
      // overlaps them; the completion wait is charged back to `stage`.
      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        if (prefetch) {
          dma_collective_memcpy_nonblocking(nxt, l2_w1 + (p + 1) * w1_elems,
                                            w1_elems * sizeof(__fp16));
          prefetch_op = dma_collective_memcpy_nonblocking(
              nxt + w1_elems, l2_w2 + (p + 1) * w2_elems,
              w2_elems * sizeof(__fp16));
        }
        mempool_barrier(num_cores);
        llm_stage_timer_stop(stage_start, &perf->stage);
      }

      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        if (redmule_id < ne_used) {
#if LLM_GEMM_2D
          const uint32_t engine = redmule_id;
          __fp16 *hidden =
              l1_hidden + engine * rows_per_engine * panel_per_engine;
          __fp16 *out = l1_out + engine * rows_per_engine * dmodel_per_engine;
          llm_zero_f16(hidden, rows_per_engine * panel_per_engine);
          if (p == 0)
            llm_zero_f16(out, rows_per_engine * dmodel_per_engine);
#else
          __fp16 *hidden = l1_hidden + redmule_id * rows_per_engine * panel;
          __fp16 *out = l1_out + redmule_id * rows_per_engine * dmodel;
          llm_zero_f16(hidden, rows_per_engine * panel);
          if (p == 0)
            llm_zero_f16(out, rows_per_engine * dmodel);
#endif
        }
        mempool_barrier(num_cores);
        llm_stage_timer_stop(stage_start, &perf->stage);
      }

      {
        const mempool_timer_t gemm_start = llm_stage_timer_start();
        if (redmule_id < ne_used) {
#if LLM_GEMM_2D
          const uint32_t row_group = redmule_id / NE_C;
          const uint32_t col_group = redmule_id % NE_C;
          unsigned int x =
              (unsigned int)(l1_x + row_group * rows_per_engine * dmodel);
          unsigned int hidden =
              (unsigned int)(l1_hidden +
                             redmule_id * rows_per_engine * panel_per_engine);
          unsigned int w =
              (unsigned int)(w1 + col_group * dmodel * panel_per_engine);
          llm_attention_redmule_job(x, w, hidden, (uint16_t)rows_per_engine,
                                    (uint16_t)dmodel,
                                    (uint16_t)panel_per_engine, GEMM);
#else
          unsigned int x =
              (unsigned int)(l1_x + redmule_id * rows_per_engine * dmodel);
          unsigned int hidden =
              (unsigned int)(l1_hidden + redmule_id * rows_per_engine * panel);
          llm_attention_redmule_job(x, (unsigned int)w1, hidden,
                                    (uint16_t)rows_per_engine, (uint16_t)dmodel,
                                    (uint16_t)panel, GEMM);
#endif
        }
#if LLM_GEMM_2D
        const uint32_t engine_slot = core_id / cores_per_engine;
        if (engine_slot < ne_used && row_team_cores > 1)
          mempool_log_partial_barrier(2, core_id, row_team_cores);
#else
        mempool_barrier(num_cores);
#endif
        llm_stage_timer_stop(gemm_start, &perf->gemm_w1);
      }

      {
        const mempool_timer_t gemm_start = llm_stage_timer_start();
        if (redmule_id < ne_used) {
#if LLM_GEMM_2D
          const uint32_t row_group = redmule_id / NE_C;
          const uint32_t out_col_group = redmule_id % NE_C;
          unsigned int out =
              (unsigned int)(l1_out +
                             redmule_id * rows_per_engine * dmodel_per_engine);
          for (uint32_t in_col_group = 0; in_col_group < NE_C; in_col_group++) {
            unsigned int hidden =
                (unsigned int)(l1_hidden + (row_group * NE_C + in_col_group) *
                                               rows_per_engine *
                                               panel_per_engine);
            unsigned int w =
                (unsigned int)(w2 + (in_col_group * NE_C + out_col_group) *
                                        panel_per_engine * dmodel_per_engine);
            llm_attention_redmule_job(hidden, w, out, (uint16_t)rows_per_engine,
                                      (uint16_t)panel_per_engine,
                                      (uint16_t)dmodel_per_engine, GEMM);
          }
#else
          unsigned int hidden =
              (unsigned int)(l1_hidden + redmule_id * rows_per_engine * panel);
          unsigned int out =
              (unsigned int)(l1_out + redmule_id * rows_per_engine * dmodel);
          llm_attention_redmule_job(hidden, (unsigned int)w2, out,
                                    (uint16_t)rows_per_engine, (uint16_t)panel,
                                    (uint16_t)dmodel, GEMM);
#endif
        }
        mempool_barrier(num_cores);
        llm_stage_timer_stop(gemm_start, &perf->gemm_w2);
      }

      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        if (prefetch)
          dma_collective_wait(&prefetch_op);
        else
          mempool_barrier(num_cores);
        llm_stage_timer_stop(stage_start, &perf->stage);
      }
    }

    {
      const mempool_timer_t stage_start = llm_stage_timer_start();
#if LLM_GEMM_2D
      if (redmule_id < ne_used) {
        const uint32_t row_group = redmule_id / NE_C;
        const uint32_t col_group = redmule_id % NE_C;
        const __fp16 *src =
            l1_out + redmule_id * rows_per_engine * dmodel_per_engine;
        __fp16 *dst = l1_x + row_group * rows_per_engine * dmodel +
                      col_group * dmodel_per_engine;
        for (uint32_t row = 0; row < rows_per_engine; row++)
          llm_gemm_copy_f16(src + row * dmodel_per_engine, dst + row * dmodel,
                            dmodel_per_engine);
      }
      mempool_barrier(num_cores);
      dma_collective_memcpy_blocking(l2_out + row0 * dmodel, l1_x,
                                     row_tile * dmodel * sizeof(__fp16));
#else
      dma_collective_memcpy_blocking(l2_out + row0 * dmodel, l1_out,
                                     row_tile * dmodel * sizeof(__fp16));
#endif
      mempool_barrier(num_cores);
      llm_stage_timer_stop(stage_start, &perf->stage);
    }
  }
}

// ---------------------------------------------------------------------------
// Attention: y[S x D] = concat_h(scores_h @ v_h) @ Wo, scores_h = q_h @ k_h^T
// ---------------------------------------------------------------------------

typedef struct {
  llm_stage_cycles_t stage;    // tile/weight DMA + K^T marshalling
  llm_stage_cycles_t gemm_qkv; // q/k/v = x @ Wq/Wk/Wv (BM x D x HEAD_DIM)
  llm_stage_cycles_t gemm_qk;  // scores = q @ k^T     (BM x HEAD_DIM x BN)
  llm_stage_cycles_t gemm_pv;  // ctx   += scores @ v  (BM x BN x HEAD_DIM)
  llm_stage_cycles_t gemm_o;   // out   += ctx @ Wo_h  (BM x HEAD_DIM x D)
} llm_gemm_attention_perf_t;

// One projection GEMM tile: proj[rows x head_dim] = x[rows x dmodel] @ w, with
// the weight already resident in L1. Row-split across the RedMulE engines.
static inline void llm_gemm_proj_tile_f16(const __fp16 *l1_x,
                                          const __fp16 *l1_w, __fp16 *l1_proj,
                                          uint32_t rows, uint32_t dmodel,
                                          uint32_t head_dim, uint32_t num_cores,
                                          uint32_t redmule_id) {
  const uint32_t ne_used = llm_attention_ne_used(rows);
  const uint32_t rows_per_engine = rows / ne_used;
  if (redmule_id < ne_used) {
    unsigned int x =
        (unsigned int)(l1_x + redmule_id * rows_per_engine * dmodel);
    unsigned int z =
        (unsigned int)(l1_proj + redmule_id * rows_per_engine * head_dim);
    llm_zero_f16((__fp16 *)z, rows_per_engine * head_dim);
    llm_attention_redmule_job(x, (unsigned int)l1_w, z,
                              (uint16_t)rows_per_engine, (uint16_t)dmodel,
                              (uint16_t)head_dim, GEMM);
  }
  mempool_barrier(num_cores);
}

// One row-split GEMM tile with an optional destination clear:
// z[rows x columns] (+)= x[rows x contraction] @ w[contraction x columns].
static inline void llm_gemm_rows_f16(const __fp16 *x, const __fp16 *w,
                                     __fp16 *z, uint32_t rows,
                                     uint32_t contraction, uint32_t columns,
                                     uint32_t clear, uint32_t num_cores,
                                     uint32_t redmule_id) {
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

// Stage one BN-token key/value tile of head `head` into L1, splicing the L2
// past cache ([heads][past_len][head_dim]) and the freshly projected new
// tokens ([heads][seqlen][head_dim]) at j0 == past_len.
static inline void llm_gemm_stage_kv_tile_f16(
    const __fp16 *l2_k_cache, const __fp16 *l2_v_cache, const __fp16 *l2_k_new,
    const __fp16 *l2_v_new, __fp16 *l1_k, __fp16 *l1_v, uint32_t head,
    uint32_t j0, uint32_t bn, uint32_t past_len, uint32_t seqlen,
    uint32_t head_dim, uint32_t core_id) {
  (void)core_id;
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

// Project every new token's K and V for all heads and spill them to L2, so the
// BM-by-BN attention loop can stage arbitrary key tiles. Weight panels are
// [heads][dmodel x head_dim]; the L2 spills are [heads][seqlen x head_dim].
static inline void llm_gemm_attention_project_kv_f16(
    const __fp16 *l2_x, const __fp16 *l2_wk, const __fp16 *l2_wv,
    __fp16 *l2_k_new, __fp16 *l2_v_new, __fp16 *l1_x, __fp16 *l1_k,
    __fp16 *l1_v, __fp16 *l1_wbuf, uint32_t seqlen, uint32_t dmodel,
    uint32_t num_heads, uint32_t head_dim, uint32_t bm, uint32_t core_id,
    uint32_t num_cores, uint32_t redmule_id, llm_gemm_attention_perf_t *perf) {
  (void)core_id;
  const uint32_t w_elems = dmodel * head_dim;

  for (uint32_t row0 = 0; row0 < seqlen; row0 += bm) {
    {
      const mempool_timer_t stage_start = llm_stage_timer_start();
      dma_collective_memcpy_blocking(l1_x, l2_x + row0 * dmodel,
                                     bm * dmodel * sizeof(__fp16));
      llm_stage_timer_stop(stage_start, &perf->stage);
    }

    for (uint32_t h = 0; h < num_heads; h++) {
      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        dma_collective_memcpy_blocking(l1_wbuf, l2_wk + h * w_elems,
                                       w_elems * sizeof(__fp16));
        llm_stage_timer_stop(stage_start, &perf->stage);
      }
      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        llm_gemm_proj_tile_f16(l1_x, l1_wbuf, l1_k, bm, dmodel, head_dim,
                               num_cores, redmule_id);
        llm_stage_timer_stop(stage_start, &perf->gemm_qkv);
      }

      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        dma_collective_memcpy_nonblocking(l2_k_new +
                                              (h * seqlen + row0) * head_dim,
                                          l1_k, bm * head_dim * sizeof(__fp16));
        dma_collective_memcpy_blocking(l1_wbuf, l2_wv + h * w_elems,
                                       w_elems * sizeof(__fp16));
        llm_stage_timer_stop(stage_start, &perf->stage);
      }
      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        llm_gemm_proj_tile_f16(l1_x, l1_wbuf, l1_v, bm, dmodel, head_dim,
                               num_cores, redmule_id);
        llm_stage_timer_stop(stage_start, &perf->gemm_qkv);
      }
      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        dma_collective_memcpy_blocking(l2_v_new +
                                           (h * seqlen + row0) * head_dim,
                                       l1_v, bm * head_dim * sizeof(__fp16));
        llm_stage_timer_stop(stage_start, &perf->stage);
      }
    }
  }
}

// Full GEMM-only multi-head attention. past_len == 0 is the prefill geometry
// (l2_k_cache / l2_v_cache unused); a non-zero past_len is the decode geometry
// with token-major L2 caches.
//
// L1 scratch (f16 elements):
//   l1_x, l1_out : BM*D          l1_q, l1_ctx : BM*HEAD_DIM
//   l1_k, l1_v   : BN*HEAD_DIM   l1_kt        : HEAD_DIM*BN
//   l1_scores    : BM*BN         l1_wbuf      : D*HEAD_DIM
static inline void llm_gemm_attention_f16(
    const __fp16 *l2_x, const __fp16 *l2_k_cache, const __fp16 *l2_v_cache,
    const __fp16 *l2_wq, const __fp16 *l2_wk, const __fp16 *l2_wv,
    const __fp16 *l2_wo, __fp16 *l2_k_new, __fp16 *l2_v_new, __fp16 *l2_out,
    __fp16 *l1_x, __fp16 *l1_q, __fp16 *l1_k, __fp16 *l1_v, __fp16 *l1_kt,
    __fp16 *l1_scores, __fp16 *l1_ctx, __fp16 *l1_out, __fp16 *l1_wbuf,
    uint32_t seqlen, uint32_t past_len, uint32_t dmodel, uint32_t num_heads,
    uint32_t head_dim, uint32_t bm, uint32_t bn, uint32_t core_id,
    uint32_t num_cores, uint32_t redmule_id, llm_gemm_attention_perf_t *perf) {
  const uint32_t total_len = past_len + seqlen;
  const uint32_t w_elems = dmodel * head_dim;

  llm_gemm_attention_project_kv_f16(
      l2_x, l2_wk, l2_wv, l2_k_new, l2_v_new, l1_x, l1_k, l1_v, l1_wbuf, seqlen,
      dmodel, num_heads, head_dim, bm, core_id, num_cores, redmule_id, perf);

  for (uint32_t q0 = 0; q0 < seqlen; q0 += bm) {
    {
      const mempool_timer_t stage_start = llm_stage_timer_start();
      dma_collective_memcpy_blocking(l1_x, l2_x + q0 * dmodel,
                                     bm * dmodel * sizeof(__fp16));
      llm_stage_timer_stop(stage_start, &perf->stage);
    }

    for (uint32_t h = 0; h < num_heads; h++) {
      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        dma_collective_memcpy_blocking(l1_wbuf, l2_wq + h * w_elems,
                                       w_elems * sizeof(__fp16));
        llm_stage_timer_stop(stage_start, &perf->stage);
      }
      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        llm_gemm_proj_tile_f16(l1_x, l1_wbuf, l1_q, bm, dmodel, head_dim,
                               num_cores, redmule_id);
        llm_stage_timer_stop(stage_start, &perf->gemm_qkv);
      }

      for (uint32_t j0 = 0; j0 < total_len; j0 += bn) {
        const uint32_t first = (j0 == 0);
        {
          const mempool_timer_t stage_start = llm_stage_timer_start();
          llm_gemm_stage_kv_tile_f16(l2_k_cache, l2_v_cache, l2_k_new, l2_v_new,
                                     l1_k, l1_v, h, j0, bn, past_len, seqlen,
                                     head_dim, core_id);
          mempool_barrier(num_cores);
          llm_attention_k_to_kt_f16(l1_k, l1_kt, bn, 1, head_dim, core_id,
                                    num_cores);
          mempool_barrier(num_cores);
          llm_stage_timer_stop(stage_start, &perf->stage);
        }

        {
          const mempool_timer_t stage_start = llm_stage_timer_start();
          llm_gemm_rows_f16(l1_q, l1_kt, l1_scores, bm, head_dim, bn, 1,
                            num_cores, redmule_id);
          llm_stage_timer_stop(stage_start, &perf->gemm_qk);
        }
        {
          const mempool_timer_t stage_start = llm_stage_timer_start();
          llm_gemm_rows_f16(l1_scores, l1_v, l1_ctx, bm, bn, head_dim, first,
                            num_cores, redmule_id);
          llm_stage_timer_stop(stage_start, &perf->gemm_pv);
        }
      }

      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        dma_collective_memcpy_blocking(l1_wbuf, l2_wo + h * w_elems,
                                       w_elems * sizeof(__fp16));
        llm_stage_timer_stop(stage_start, &perf->stage);
      }
      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        llm_gemm_rows_f16(l1_ctx, l1_wbuf, l1_out, bm, head_dim, dmodel, h == 0,
                          num_cores, redmule_id);
        llm_stage_timer_stop(stage_start, &perf->gemm_o);
      }
    }

    {
      const mempool_timer_t stage_start = llm_stage_timer_start();
      dma_collective_memcpy_blocking(l2_out + q0 * dmodel, l1_out,
                                     bm * dmodel * sizeof(__fp16));
      llm_stage_timer_stop(stage_start, &perf->stage);
    }
  }
}
