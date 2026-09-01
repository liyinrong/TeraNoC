// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yinrong Li, ETH Zurich

// Dense-LLM attention block, PREFILL phase. This is the Llama-style attention
// path from dense_workload.md: RMSNorm, Q/K/V projections, optional QK-norm,
// RoPE, causal QK^T/softmax/scores*V, output projection, and residual.

#include <stdint.h>

#include "printf.h"

#include "data_llm_attention_prefill_f16.h"
#include "mempool_checks.h"

#include "baremetal/mempool_llm_attention_f16.h"
#include "baremetal/mempool_llm_tiling.h"

#define KV_D (N_KV_HEADS * HEAD_DIM)
#define HALF_HEAD_DIM (HEAD_DIM / 2)

#ifdef LLM_FLASH_TILED
_Static_assert(D == N_Q_HEADS * HEAD_DIM && N_Q_HEADS % N_KV_HEADS == 0,
               "tiled attention requires consistent GQA geometry");
_Static_assert(S % LLM_FLASH_BM == 0 && S % LLM_FLASH_BN == 0,
               "S must be divisible by both FlashAttention tile sizes");
_Static_assert(
    LLM_FLASH_BN <= LLM_FLASH_BM &&
        LLM_FLASH_BM % LLM_ATTENTION_PORT_WIDTH == 0 &&
        LLM_FLASH_BN % LLM_ATTENTION_PORT_WIDTH == 0,
    "FlashAttention tiles must be RedMulE-port aligned and BN <= BM");

#define LLM_TILED_F16_ELEMS                                                    \
  (5L * LLM_FLASH_BM * D + D + 2L * HEAD_DIM +                                 \
   2L * LLM_FLASH_BM * HALF_HEAD_DIM +                                         \
   2L * N_KV_HEADS * LLM_FLASH_BM * HEAD_DIM +                                 \
   N_KV_HEADS * HEAD_DIM * LLM_FLASH_BN +                                      \
   N_Q_HEADS * LLM_FLASH_BM * LLM_FLASH_BN + D * HEAD_DIM)
#define LLM_TILED_ALIGN_SLACK (17L * NUM_BANKS * sizeof(int32_t))
_Static_assert(2L * LLM_TILED_F16_ELEMS + 8L * N_Q_HEADS * LLM_FLASH_BM +
                       LLM_TILED_ALIGN_SLACK <=
                   L1_TILE_BUDGET_BYTES,
               "tiled attention working set exceeds the L1 budget");

__fp16 l1_x[LLM_FLASH_BM * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_gamma[D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_q_gamma[HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_k_gamma[HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_cos[LLM_FLASH_BM * HALF_HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_sin[LLM_FLASH_BM * HALF_HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_h[LLM_FLASH_BM * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_q[N_Q_HEADS * LLM_FLASH_BM * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_k[N_KV_HEADS * LLM_FLASH_BM * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_v[N_KV_HEADS * LLM_FLASH_BM * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_kt[N_KV_HEADS * HEAD_DIM * LLM_FLASH_BN]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_scores[N_Q_HEADS * LLM_FLASH_BM * LLM_FLASH_BN]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_ctx[N_Q_HEADS * LLM_FLASH_BM * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_out[LLM_FLASH_BM * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_wbuf[D * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
float l1_m[N_Q_HEADS * LLM_FLASH_BM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
float l1_l[N_Q_HEADS * LLM_FLASH_BM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));

__fp16 l2_k_stream[N_KV_HEADS * S * HEAD_DIM]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
__fp16 l2_v_stream[N_KV_HEADS * S * HEAD_DIM]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
#else
// FlashAttention keeps only an S x LLM_FLASH_BN score tile resident; the
// materialized path holds the full S x S.
#ifdef LLM_FLASH_ATTENTION
#define LLM_SCORES_COLS LLM_FLASH_BN
#else
#define LLM_SCORES_COLS S
#endif

_Static_assert(
    LLM_F16_FITS_L1(3 * (S * D) + D + 2 * HEAD_DIM + 2 * (S * HALF_HEAD_DIM) +
                    2 * (N_Q_HEADS * S * HEAD_DIM) +
                    3 * (N_KV_HEADS * S * HEAD_DIM) +
                    (N_Q_HEADS * S * LLM_SCORES_COLS) + 2 * (D * HEAD_DIM)),
    "llm_attention_prefill: resident L1 working set exceeds the L1 budget");

__fp16 l1_x[S * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_gamma[D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_q_gamma[HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_k_gamma[HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_cos[S * HALF_HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_sin[S * HALF_HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_h[S * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_q[N_Q_HEADS * S * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_k[N_KV_HEADS * S * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_v[N_KV_HEADS * S * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_kt[N_KV_HEADS * HEAD_DIM * S]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_scores[N_Q_HEADS * S * LLM_SCORES_COLS]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_ctx[N_Q_HEADS * S * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_out[S * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_wbuf[2 * D * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));

#ifdef LLM_FLASH_ATTENTION
_Static_assert(S % LLM_FLASH_BN == 0 &&
                   LLM_FLASH_BN % LLM_ATTENTION_PORT_WIDTH == 0,
               "LLM_FLASH_BN must divide S and be a multiple of PORT_WIDTH");
float l1_m[N_Q_HEADS * S]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
float l1_l[N_Q_HEADS * S]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_kt_tile[N_KV_HEADS * HEAD_DIM * LLM_FLASH_BN]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
#endif
#endif

__fp16 l2_check[S * D]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  uint32_t redmule_id = mempool_get_redmule_id();
  mempool_barrier_init(core_id);

  if (core_id == 0) {
    printf("\n[llm_attention_prefill_f16] S=%d D=%d Hq=%d Hkv=%d Hd=%d "
           "QK_NORM=%d | redmule=%d cores=%d engines_used=%d\n",
           S, D, N_Q_HEADS, N_KV_HEADS, HEAD_DIM, QK_NORM, NUM_REDMULE_TILES,
           num_cores,
#ifdef LLM_FLASH_TILED
           llm_attention_ne_used(LLM_FLASH_BM));
#else
           llm_attention_ne_used(S));
#endif
  }
  mempool_barrier(num_cores);

  uint32_t time_init = mempool_get_timer();
  mempool_start_benchmark();

#ifdef LLM_FLASH_TILED
  llm_attention_flash_tiled_prefill_f16(
      l2_x, l2_gamma, l2_q_gamma, l2_k_gamma, l2_cos, l2_sin, l2_wq, l2_wk,
      l2_wv, l2_wo, l2_k_stream, l2_v_stream, l2_check, l1_x, l1_gamma,
      l1_q_gamma, l1_k_gamma, l1_cos, l1_sin, l1_h, l1_q, l1_k, l1_v, l1_kt,
      l1_scores, l1_ctx, l1_out, l1_wbuf, l1_m, l1_l, S, D, N_Q_HEADS,
      N_KV_HEADS, HEAD_DIM, QK_NORM, LLM_FLASH_BM, LLM_FLASH_BN, core_id,
      num_cores, redmule_id);
#elif defined(LLM_FLASH_ATTENTION)
  llm_attention_flash_prefill_f16(
      l2_x, l2_gamma, l2_q_gamma, l2_k_gamma, l2_cos, l2_sin, l2_wq, l2_wk,
      l2_wv, l2_wo, l1_x, l1_gamma, l1_q_gamma, l1_k_gamma, l1_cos, l1_sin,
      l1_h, l1_q, l1_k, l1_v, l1_kt, l1_scores, l1_ctx, l1_out, l1_wbuf, l1_m,
      l1_l, l1_kt_tile, S, D, N_Q_HEADS, N_KV_HEADS, HEAD_DIM, QK_NORM, core_id,
      num_cores, redmule_id);
#else
  llm_attention_prefill_f16(
      l2_x, l2_gamma, l2_q_gamma, l2_k_gamma, l2_cos, l2_sin, l2_wq, l2_wk,
      l2_wv, l2_wo, l1_x, l1_gamma, l1_q_gamma, l1_k_gamma, l1_cos, l1_sin,
      l1_h, l1_q, l1_k, l1_v, l1_kt, l1_scores, l1_ctx, l1_out, l1_wbuf, S, D,
      N_Q_HEADS, N_KV_HEADS, HEAD_DIM, QK_NORM, core_id, num_cores, redmule_id);
#endif

  mempool_stop_benchmark();
  uint32_t time_end = mempool_get_timer();

  if (mempool_is_group_coordinator()) {
    if (core_id == 0)
      printf("[llm_attention_prefill_f16] kernel takes %d cycles\n",
             time_end - time_init);
#ifndef LLM_FLASH_TILED
    dma_collective_all_memcpy_blocking(l2_check, l1_out,
                                       S * D * sizeof(__fp16));
#endif
  }
  mempool_barrier(num_cores);

#if !LLM_PERF_ONLY
  if (core_id == 0)
    mempool_check_dpi_f16(l2_check, l2_y, S * D, 0.2f, 0);
  mempool_barrier(num_cores);
#endif

  return 0;
}
