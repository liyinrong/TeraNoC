// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Pure-GEMM FFN layer: the two representative GEMMs of a Transformer FFN,
// hidden = x @ W1 (S x D x I) followed by y = hidden @ W2 (S x I x D). No ReLU,
// normalization, gating or residual - see mempool_llm_gemm_f16.h.

#include <stdint.h>

#include "printf.h"

#include "data_llm_gemm_ffn_f16.h"
#include "mempool_checks.h"

#include "baremetal/mempool_llm_gemm_f16.h"
#include "baremetal/mempool_llm_tiling.h"

_Static_assert(LLM_GEMM_2D == 0 || LLM_GEMM_2D == 1,
               "LLM_GEMM_2D must be 0 or 1");
_Static_assert(S % ROW_TILE == 0, "ROW_TILE must divide S");
_Static_assert(ROW_TILE % LLM_GEMM_PORT_WIDTH == 0,
               "ROW_TILE must be a RedMulE row multiple");
_Static_assert(D % LLM_GEMM_PORT_WIDTH == 0,
               "D must be a RedMulE contraction/column multiple");
_Static_assert(I % PANEL_I == 0 && PANEL_I % LLM_GEMM_PORT_WIDTH == 0,
               "PANEL_I must divide I and be a RedMulE multiple");
_Static_assert(NUM_IP == I / PANEL_I,
               "NUM_IP must cover every intermediate panel exactly once");
_Static_assert(ROW_TILE <= UINT16_MAX && D <= UINT16_MAX &&
                   PANEL_I <= UINT16_MAX,
               "FFN GEMM dimensions must fit the RedMulE configuration");
#if LLM_GEMM_2D
#if NE_R < 1 || NE_C < 1
#error "NE_R and NE_C must be positive"
#endif
_Static_assert(NE_R *NE_C <= NUM_REDMULE_TILES,
               "2-D map requests more engines than the configuration has");
_Static_assert(NUM_CORES % NUM_REDMULE_TILES == 0,
               "RedMulE core teams must partition the cores exactly");
_Static_assert(NUM_CORES >= NUM_REDMULE_TILES,
               "each RedMulE needs at least one owner core");
_Static_assert(ROW_TILE % NE_R == 0 &&
                   (ROW_TILE / NE_R) % LLM_GEMM_PORT_WIDTH == 0,
               "2-D row shard must be a RedMulE row multiple");
_Static_assert(PANEL_I % NE_C == 0 &&
                   (PANEL_I / NE_C) % LLM_GEMM_PORT_WIDTH == 0,
               "2-D W1 column shard must be a RedMulE column multiple");
_Static_assert(D % NE_C == 0 && (D / NE_C) % LLM_GEMM_PORT_WIDTH == 0,
               "2-D W2 column shard must be a RedMulE column multiple");
_Static_assert((NE_C * (NUM_CORES / NUM_REDMULE_TILES) &
                (NE_C * (NUM_CORES / NUM_REDMULE_TILES) - 1)) == 0,
               "row-group core team must be a power of two");
#endif
_Static_assert(LLM_F16_FITS_L1(2 * (ROW_TILE * D) + ROW_TILE * PANEL_I +
                               4 * (D * PANEL_I)),
               "FFN GEMM tiles exceed the L1 budget");

__fp16 l1_x[ROW_TILE * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_hidden[ROW_TILE * PANEL_I]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_out[ROW_TILE * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_wbuf[4 * D * PANEL_I]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));

__fp16 l2_check[S * D]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  uint32_t redmule_id = mempool_get_redmule_id();
  mempool_barrier_init(core_id);

  if (core_id == 0) {
#if LLM_GEMM_2D
    printf("\n[llm_gemm_ffn_f16] S=%d D=%d I=%d "
           "ROW_TILE=%d PANEL_I=%d NUM_IP=%d | mapping=2d NE_R=%d NE_C=%d "
           "redmule=%d cores=%d\n",
           S, D, I, ROW_TILE, PANEL_I, NUM_IP, NE_R, NE_C, NUM_REDMULE_TILES,
           num_cores);
#else
    printf("\n[llm_gemm_ffn_f16] S=%d D=%d I=%d "
           "ROW_TILE=%d PANEL_I=%d NUM_IP=%d | mapping=row redmule=%d "
           "cores=%d\n",
           S, D, I, ROW_TILE, PANEL_I, NUM_IP, NUM_REDMULE_TILES, num_cores);
#endif
  }
  mempool_barrier(num_cores);

  llm_gemm_ffn_perf_t perf = {0};
  uint32_t time_init = mempool_get_timer();
#if !LLM_CLASSIC_STAGE_PERF
  mempool_start_benchmark();
#endif
  llm_gemm_ffn_f16(l2_x, l2_w1, l2_w2, l2_check, l1_x, l1_hidden, l1_out,
                   l1_wbuf, S, D, I, ROW_TILE, PANEL_I, NUM_IP, core_id,
                   num_cores, redmule_id, &perf);
#if !LLM_CLASSIC_STAGE_PERF
  mempool_stop_benchmark();
#endif
  uint32_t time_end = mempool_get_timer();
  mempool_barrier(num_cores);

  if (core_id == 0) {
#if LLM_CLASSIC_STAGE_PERF
    printf("[llm_gemm_ffn_f16] stage=gemm_w1 cycles=%u\n", perf.gemm_w1);
    printf("[llm_gemm_ffn_f16] stage=gemm_w2 cycles=%u\n", perf.gemm_w2);
    printf("[llm_gemm_ffn_f16] stage=stage cycles=%u\n", perf.stage);
#endif
    printf("[llm_gemm_ffn_f16] kernel takes %u clock cycles\n",
           time_end - time_init);
  }
  mempool_barrier(num_cores);

#if !LLM_PERF_ONLY
  if (core_id == 0)
    mempool_check_dpi_f16(l2_check, l2_y, S * D, 0.05f, 0);
  mempool_barrier(num_cores);
#endif

  return 0;
}
