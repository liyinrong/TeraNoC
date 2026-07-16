// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Classic Transformer causal MHA, decode phase with a token-major KV cache.

#include <stdint.h>

#include "printf.h"

#include "data_llm_classic_attention_decode_f16.h"
#include "mempool_checks.h"

#include "baremetal/mempool_llm_classic_attention_f16.h"
#include "baremetal/mempool_llm_tiling.h"

#define KV_TOTAL (KV_LEN + S)

_Static_assert(D == N_HEADS * HEAD_DIM,
               "classic MHA requires D == N_HEADS*HEAD_DIM");
_Static_assert(S % LLM_ATTENTION_PORT_WIDTH == 0 &&
                   D % LLM_ATTENTION_PORT_WIDTH == 0 &&
                   HEAD_DIM % LLM_ATTENTION_PORT_WIDTH == 0,
               "classic MHA dimensions must be RedMulE multiples");

#ifdef LLM_FLASH_ATTENTION
_Static_assert(S % LLM_FLASH_BM == 0 && KV_TOTAL % LLM_FLASH_BN == 0,
               "Flash BM must divide S and BN must divide KV_LEN+S");
_Static_assert(LLM_FLASH_BN >= LLM_FLASH_BM,
               "Flash BN must hold one projected BM-row K/V tile");
_Static_assert(LLM_FLASH_BM % LLM_ATTENTION_PORT_WIDTH == 0 &&
                   LLM_FLASH_BN % LLM_ATTENTION_PORT_WIDTH == 0,
               "Flash BM and BN must be RedMulE multiples");
_Static_assert(LLM_F16_FITS_L1(2 * (LLM_FLASH_BM * D) + D * HEAD_DIM +
                               2 * (LLM_FLASH_BM * HEAD_DIM) +
                               3 * (LLM_FLASH_BN * HEAD_DIM) +
                               LLM_FLASH_BM * LLM_FLASH_BN + 4 * LLM_FLASH_BM),
               "classic FlashAttention tiles exceed the L1 budget");

__fp16 l1_x[LLM_FLASH_BM * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_q[LLM_FLASH_BM * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_k[LLM_FLASH_BN * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_v[LLM_FLASH_BN * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_kt[HEAD_DIM * LLM_FLASH_BN]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_scores[LLM_FLASH_BM * LLM_FLASH_BN]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_ctx[LLM_FLASH_BM * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_out[LLM_FLASH_BM * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_wbuf[D * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
float l1_m[LLM_FLASH_BM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
float l1_l[LLM_FLASH_BM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));

__fp16 l2_k_new[N_HEADS * S * HEAD_DIM]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
__fp16 l2_v_new[N_HEADS * S * HEAD_DIM]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
#else
_Static_assert(KV_TOTAL % LLM_ATTENTION_PORT_WIDTH == 0,
               "KV_LEN+S must be a RedMulE multiple");
_Static_assert(LLM_F16_FITS_L1(6 * (S * D) + 3 * (KV_TOTAL * D) +
                               N_HEADS * S * KV_TOTAL + 2 * (D * HEAD_DIM)),
               "materialized classic decode exceeds the L1 budget");

__fp16 l1_x[S * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_q[N_HEADS * S * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_k[N_HEADS * S * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_v[N_HEADS * S * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_k_full[N_HEADS * KV_TOTAL * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_v_full[N_HEADS * KV_TOTAL * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_kt[N_HEADS * HEAD_DIM * KV_TOTAL]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_scores[N_HEADS * S * KV_TOTAL]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_ctx[N_HEADS * S * HEAD_DIM]
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
#ifdef LLM_FLASH_ATTENTION
    printf("\n[llm_classic_attention_decode_f16] S=%d KV=%d D=%d H=%d "
           "Hd=%d Flash BM=%d BN=%d | redmule=%d cores=%d\n",
           S, KV_LEN, D, N_HEADS, HEAD_DIM, LLM_FLASH_BM, LLM_FLASH_BN,
           NUM_REDMULE_TILES, num_cores);
#else
    printf("\n[llm_classic_attention_decode_f16] S=%d KV=%d D=%d H=%d "
           "Hd=%d materialized | redmule=%d cores=%d\n",
           S, KV_LEN, D, N_HEADS, HEAD_DIM, NUM_REDMULE_TILES, num_cores);
#endif
  }
  mempool_barrier(num_cores);

  uint32_t time_init = mempool_get_timer();
  mempool_start_benchmark();
#ifdef LLM_FLASH_ATTENTION
  llm_classic_attention_flash_f16(
      l2_x, l2_k_cache, l2_v_cache, l2_wq, l2_wk, l2_wv, l2_wo, l2_k_new,
      l2_v_new, l2_check, l1_x, l1_q, l1_k, l1_v, l1_kt, l1_scores, l1_ctx,
      l1_out, l1_wbuf, l1_m, l1_l, S, KV_LEN, D, N_HEADS, HEAD_DIM,
      LLM_FLASH_BM, LLM_FLASH_BN, core_id, num_cores, redmule_id);
#else
  llm_classic_attention_decode_materialized_f16(
      l2_x, l2_k_cache, l2_v_cache, l2_wq, l2_wk, l2_wv, l2_wo, l2_check, l1_x,
      l1_q, l1_k, l1_v, l1_k_full, l1_v_full, l1_kt, l1_scores, l1_ctx, l1_out,
      l1_wbuf, S, KV_LEN, D, N_HEADS, HEAD_DIM, core_id, num_cores, redmule_id);
#endif
  mempool_stop_benchmark();
  uint32_t time_end = mempool_get_timer();

  if (core_id == 0)
    printf("[llm_classic_attention_decode_f16] kernel takes %d cycles\n",
           time_end - time_init);
  mempool_barrier(num_cores);

#if !LLM_PERF_ONLY
  if (core_id == 0)
    mempool_check_dpi_f16(l2_check, l2_y, S * D, 0.2f, 0);
  mempool_barrier(num_cores);
#endif

  return 0;
}
