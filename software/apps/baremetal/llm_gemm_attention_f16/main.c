// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Pure-GEMM multi-head attention layer: the four representative GEMMs of a
// Transformer attention block - Q/K/V projections (S x D x HEAD_DIM per head),
// scores = q @ k^T (BM x HEAD_DIM x BN), ctx = scores @ v (BM x BN x HEAD_DIM)
// and the head-accumulated output projection (BM x HEAD_DIM x D). No scale,
// causal mask, softmax or residual - see mempool_llm_gemm_f16.h.

#include <stdint.h>

#include "printf.h"

#include "data_llm_gemm_attention_f16.h"
#include "mempool_checks.h"

#include "baremetal/mempool_llm_gemm_f16.h"
#include "baremetal/mempool_llm_tiling.h"

_Static_assert(D == N_HEADS * HEAD_DIM, "MHA requires D == N_HEADS*HEAD_DIM");
_Static_assert(S % LLM_GEMM_PORT_WIDTH == 0 && D % LLM_GEMM_PORT_WIDTH == 0 &&
                   HEAD_DIM % LLM_GEMM_PORT_WIDTH == 0,
               "MHA dimensions must be RedMulE multiples");
_Static_assert(S % LLM_GEMM_BM == 0 && S % LLM_GEMM_BN == 0,
               "BM and BN must divide S");
_Static_assert(LLM_GEMM_BN >= LLM_GEMM_BM,
               "BN must hold one projected BM-row K/V tile");
_Static_assert(LLM_GEMM_BM % LLM_GEMM_PORT_WIDTH == 0 &&
                   LLM_GEMM_BN % LLM_GEMM_PORT_WIDTH == 0,
               "BM and BN must be RedMulE multiples");
_Static_assert(LLM_F16_FITS_L1(2 * (LLM_GEMM_BM * D) + D * HEAD_DIM +
                               2 * (LLM_GEMM_BM * HEAD_DIM) +
                               3 * (LLM_GEMM_BN * HEAD_DIM) +
                               LLM_GEMM_BM * LLM_GEMM_BN),
               "attention GEMM tiles exceed the L1 budget");

__fp16 l1_x[LLM_GEMM_BM * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_q[LLM_GEMM_BM * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_k[LLM_GEMM_BN * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_v[LLM_GEMM_BN * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_kt[HEAD_DIM * LLM_GEMM_BN]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_scores[LLM_GEMM_BM * LLM_GEMM_BN]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_ctx[LLM_GEMM_BM * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_out[LLM_GEMM_BM * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_wbuf[D * HEAD_DIM]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));

__fp16 l2_k_new[N_HEADS * S * HEAD_DIM]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
__fp16 l2_v_new[N_HEADS * S * HEAD_DIM]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
__fp16 l2_check[S * D]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  uint32_t redmule_id = mempool_get_redmule_id();
  mempool_barrier_init(core_id);

  if (core_id == 0) {
    printf("\n[llm_gemm_attention_f16] S=%d D=%d H=%d Hd=%d BM=%d BN=%d "
           "| redmule=%d cores=%d\n",
           S, D, N_HEADS, HEAD_DIM, LLM_GEMM_BM, LLM_GEMM_BN, NUM_REDMULE_TILES,
           num_cores);
  }
  mempool_barrier(num_cores);

  llm_gemm_attention_perf_t perf = {0};
  uint32_t time_init = mempool_get_timer();
#if !LLM_CLASSIC_STAGE_PERF
  mempool_start_benchmark();
#endif
  llm_gemm_attention_f16(
      (const __fp16 *)l2_x, (const __fp16 *)0, (const __fp16 *)0, l2_wq, l2_wk,
      l2_wv, l2_wo, l2_k_new, l2_v_new, l2_check, l1_x, l1_q, l1_k, l1_v, l1_kt,
      l1_scores, l1_ctx, l1_out, l1_wbuf, S, 0, D, N_HEADS, HEAD_DIM,
      LLM_GEMM_BM, LLM_GEMM_BN, core_id, num_cores, redmule_id, &perf);
#if !LLM_CLASSIC_STAGE_PERF
  mempool_stop_benchmark();
#endif
  uint32_t time_end = mempool_get_timer();
  mempool_barrier(num_cores);

  if (core_id == 0) {
#if LLM_CLASSIC_STAGE_PERF
    printf("[llm_gemm_attention_f16] stage=gemm_qkv cycles=%u\n",
           perf.gemm_qkv);
    printf("[llm_gemm_attention_f16] stage=gemm_qk cycles=%u\n", perf.gemm_qk);
    printf("[llm_gemm_attention_f16] stage=gemm_pv cycles=%u\n", perf.gemm_pv);
    printf("[llm_gemm_attention_f16] stage=gemm_o cycles=%u\n", perf.gemm_o);
    printf("[llm_gemm_attention_f16] stage=stage cycles=%u\n", perf.stage);
#endif
    printf("[llm_gemm_attention_f16] kernel takes %u clock cycles\n",
           time_end - time_init);
  }
  mempool_barrier(num_cores);

#if !LLM_PERF_ONLY
  if (core_id == 0)
    mempool_check_dpi_f16(l2_check, l2_y, S * D, 0.2f, 0);
  mempool_barrier(num_cores);
#endif

  return 0;
}
