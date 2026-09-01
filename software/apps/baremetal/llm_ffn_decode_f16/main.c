// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yinrong Li, ETH Zurich

// Dense-LLM SwiGLU FFN block, DECODE phase (small token batch). Same pipeline
// as the prefill app but with a small S so the GEMMs are weight-bandwidth
// bound (AI ~ 1): the benchmark stresses the double-buffered weight streaming
// from L2 through TeraNoC rather than RedMulE array utilization. Workload shape
// is set in the data header (gendata_params.hjson :: llm_ffn_decode_f16).

#include <stdint.h>

#include "printf.h"

#include "data_llm_ffn_decode_f16.h"
#include "mempool_checks.h"

#include "baremetal/mempool_llm_ffn_pipeline_f16.h"
#include "baremetal/mempool_llm_tiling.h"

_Static_assert(S % ROW_TILE == 0,
               "llm_ffn: ROW_TILE must divide the sequence length");
_Static_assert(ROW_TILE % LLM_PORT_WIDTH == 0,
               "llm_ffn: ROW_TILE must be a RedMulE row-block multiple");

#if ROW_TILE == S
_Static_assert(LLM_F16_FITS_L1(3 * (S * D) + D + 2 * (S * PANEL_I) +
                               6 * (D * PANEL_I)),
               "llm_ffn: resident L1 working set exceeds the L1 budget");

__fp16 l1_x[S * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_gamma[D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_h[S * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_g[S * PANEL_I]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_u[S * PANEL_I]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_out[S * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_wbuf[6 * D * PANEL_I]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
#else
_Static_assert(LLM_F16_FITS_L1(ROW_TILE *D + D + 2 * (ROW_TILE * PANEL_I) +
                               3 * (D * PANEL_I)),
               "llm_ffn: tiled L1 working set exceeds the L1 budget");

__fp16 l1_gamma[D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_row[ROW_TILE * D]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_g[ROW_TILE * PANEL_I]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_u[ROW_TILE * PANEL_I]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_wbuf[3 * D * PANEL_I]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));

__fp16 l2_h[S * D] __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
#endif

__fp16 l2_check[S * D]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  uint32_t redmule_id = mempool_get_redmule_id();
  mempool_barrier_init(core_id);

  if (core_id == 0) {
    printf("\n[llm_ffn_decode_f16] S=%d ROW_TILE=%d D=%d I=%d PANEL_I=%d "
           "NUM_IP=%d | redmule=%d cores=%d engines_used=%d\n",
           S, ROW_TILE, D, I, PANEL_I, NUM_IP, NUM_REDMULE_TILES, num_cores,
           llm_ffn_ne_used(ROW_TILE));
  }
  mempool_barrier(num_cores);

#ifdef LLM_FFN_NEGATIVE_CORRUPT_WEIGHT
  // Deliberate G2.7 falsification: the DPI golden must reject this panel.
  if (core_id == 0)
    ((uint16_t *)l2_wg)[0] ^= 0x7bffu;
  mempool_barrier(num_cores);
#endif

  uint32_t time_init = mempool_get_timer();
#ifndef LLM_FFN_DISABLE_TRACE
  mempool_start_benchmark();
#endif

#if ROW_TILE == S
  llm_ffn_swiglu_f16(l2_x, l2_gamma, l2_wg, l2_wu, l2_wd, l1_x, l1_gamma, l1_h,
                     l1_g, l1_u, l1_out, l1_wbuf, S, D, I, PANEL_I, NUM_IP,
                     core_id, num_cores, redmule_id);
#else
  llm_ffn_swiglu_row_tiled_f16(l2_x, l2_gamma, l2_wg, l2_wu, l2_wd, l2_h,
                               l2_check, l1_gamma, l1_row, l1_g, l1_u, l1_wbuf,
                               S, ROW_TILE, D, I, PANEL_I, NUM_IP, core_id,
                               num_cores, redmule_id);
#endif

#ifndef LLM_FFN_DISABLE_TRACE
  mempool_stop_benchmark();
#endif
  uint32_t time_end = mempool_get_timer();

  if (mempool_is_group_coordinator()) {
    if (core_id == 0)
      printf("[llm_ffn_decode_f16] kernel takes %d cycles\n",
             time_end - time_init);
#if ROW_TILE == S
    dma_collective_all_memcpy_blocking(l2_check, l1_out,
                                       S * D * sizeof(__fp16));
#endif
  }
  mempool_barrier(num_cores);

#if !LLM_PERF_ONLY
  if (core_id == 0)
    mempool_check_dpi_f16(l2_check, l2_y, S * D, 0.05f, 0);
  mempool_barrier(num_cores);
#endif

  return 0;
}
