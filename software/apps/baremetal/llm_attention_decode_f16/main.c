// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yinrong Li, ETH Zurich

// Dense-LLM attention block, DECODE phase. The query batch is intentionally
// small; the benchmark stresses weight streaming for projections and KV-cache
// streaming for QK^T / scores*V, including the KV append path.

#include <stdint.h>

#include "printf.h"

#include "data_llm_attention_decode_f16.h"
#include "mempool_checks.h"

#include "baremetal/mempool_llm_attention_f16.h"
#include "baremetal/mempool_llm_tiling.h"

#define KV_D (N_KV_HEADS * HEAD_DIM)
#define HALF_HEAD_DIM (HEAD_DIM / 2)
#define KV_TOTAL (KV_LEN + S)

#ifdef LLM_FLASH_TILED
#define LLM_KV_TILE_ROWS ((S > LLM_FLASH_BN) ? S : LLM_FLASH_BN)
#define LLM_DECODE_ATTN_LEN                                                    \
  ((WINDOW_LEN == 0 || WINDOW_LEN > KV_TOTAL) ? KV_TOTAL : WINDOW_LEN)
_Static_assert(KV_CACHE_TOKEN_MAJOR == 1,
               "tiled decode requires a token-major L2 KV cache");
_Static_assert(D == N_Q_HEADS * HEAD_DIM && N_Q_HEADS % N_KV_HEADS == 0,
               "tiled attention requires consistent GQA geometry");
_Static_assert(S % LLM_ATTENTION_PORT_WIDTH == 0 &&
                   LLM_FLASH_BN % LLM_ATTENTION_PORT_WIDTH == 0 &&
                   LLM_DECODE_ATTN_LEN % LLM_FLASH_BN == 0,
               "decode rows, attention length, and BN must be port aligned");

#define LLM_TILED_DECODE_F16_ELEMS                                             \
  (5L * S * D + D + 2L * HEAD_DIM + 2L * S * HALF_HEAD_DIM +                   \
   2L * N_KV_HEADS * LLM_KV_TILE_ROWS * HEAD_DIM +                             \
   N_KV_HEADS * HEAD_DIM * LLM_FLASH_BN + N_Q_HEADS * S * LLM_FLASH_BN +       \
   D * HEAD_DIM)
#define LLM_TILED_DECODE_ALIGN_SLACK (17L * NUM_BANKS * sizeof(int32_t))
_Static_assert(2L * LLM_TILED_DECODE_F16_ELEMS + 8L * N_Q_HEADS * S +
                       LLM_TILED_DECODE_ALIGN_SLACK <=
                   L1_TILE_BUDGET_BYTES,
               "tiled decode working set exceeds the L1 budget");

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
__fp16 l1_k[N_KV_HEADS * LLM_KV_TILE_ROWS * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_v[N_KV_HEADS * LLM_KV_TILE_ROWS * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_kt[N_KV_HEADS * HEAD_DIM * LLM_FLASH_BN]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_scores[N_Q_HEADS * S * LLM_FLASH_BN]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_ctx[N_Q_HEADS * S * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_out[S * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_wbuf[D * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
float l1_m[N_Q_HEADS * S]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
float l1_l[N_Q_HEADS * S]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
#else
// Materialized decode stages the full cache/window and score matrix in L1.
_Static_assert(
    LLM_F16_FITS_L1(3 * (S * D) + D + 2 * HEAD_DIM + 2 * (S * HALF_HEAD_DIM) +
                    4 * (N_KV_HEADS * HEAD_DIM * KV_TOTAL) +
                    2 * (N_Q_HEADS * S * HEAD_DIM) +
                    2 * (N_KV_HEADS * S * HEAD_DIM) +
                    (N_Q_HEADS * S * KV_TOTAL) + 2 * (D * HEAD_DIM)),
    "llm_attention_decode: resident L1 working set exceeds the L1 budget");

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
__fp16 l1_k_cache_t[N_KV_HEADS * HEAD_DIM * KV_TOTAL]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_v_cache[N_KV_HEADS * KV_TOTAL * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_kt_window[N_KV_HEADS * HEAD_DIM * KV_TOTAL]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_v_window[N_KV_HEADS * KV_TOTAL * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_h[S * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_q[N_Q_HEADS * S * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_k[N_KV_HEADS * S * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_v[N_KV_HEADS * S * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_scores[N_Q_HEADS * S * KV_TOTAL]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_ctx[N_Q_HEADS * S * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_out[S * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_wbuf[2 * D * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
#endif

__fp16 l2_check[S * D]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  uint32_t redmule_id = mempool_get_redmule_id();
  mempool_barrier_init(core_id);

  if (core_id == 0) {
    printf("\n[llm_attention_decode_f16] S=%d L=%d W=%d D=%d Hq=%d Hkv=%d "
           "Hd=%d QK_NORM=%d | redmule=%d cores=%d engines_used=%d\n",
           S, KV_LEN, WINDOW_LEN, D, N_Q_HEADS, N_KV_HEADS, HEAD_DIM, QK_NORM,
           NUM_REDMULE_TILES, num_cores, llm_attention_ne_used(S));
  }
  mempool_barrier(num_cores);

  uint32_t time_init = mempool_get_timer();
  mempool_start_benchmark();

#ifdef LLM_FLASH_TILED
  llm_attention_flash_tiled_decode_f16(
      l2_x, l2_gamma, l2_q_gamma, l2_k_gamma, l2_cos, l2_sin, l2_k_cache_t,
      l2_v_cache, l2_wq, l2_wk, l2_wv, l2_wo, l2_check, l1_x, l1_gamma,
      l1_q_gamma, l1_k_gamma, l1_cos, l1_sin, l1_h, l1_q, l1_k, l1_v, l1_kt,
      l1_scores, l1_ctx, l1_out, l1_wbuf, l1_m, l1_l, S, KV_LEN, WINDOW_LEN, D,
      N_Q_HEADS, N_KV_HEADS, HEAD_DIM, QK_NORM, LLM_FLASH_BN, core_id,
      num_cores, redmule_id);
#else
  llm_attention_decode_f16(
      l2_x, l2_gamma, l2_q_gamma, l2_k_gamma, l2_cos, l2_sin, l2_k_cache_t,
      l2_v_cache, l2_wq, l2_wk, l2_wv, l2_wo, l1_x, l1_gamma, l1_q_gamma,
      l1_k_gamma, l1_cos, l1_sin, l1_k_cache_t, l1_v_cache, l1_kt_window,
      l1_v_window, l1_h, l1_q, l1_k, l1_v, l1_scores, l1_ctx, l1_out, l1_wbuf,
      S, KV_LEN, WINDOW_LEN, D, N_Q_HEADS, N_KV_HEADS, HEAD_DIM, QK_NORM,
      core_id, num_cores, redmule_id);
#endif

  mempool_stop_benchmark();
  uint32_t time_end = mempool_get_timer();

  if (core_id == 0) {
    printf("[llm_attention_decode_f16] kernel takes %d cycles\n",
           time_end - time_init);
#ifndef LLM_FLASH_TILED
    dma_memcpy_blocking(l2_check, l1_out, S * D * sizeof(__fp16));
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
