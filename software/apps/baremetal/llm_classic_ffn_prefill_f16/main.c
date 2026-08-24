// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Classic Transformer FFN prefill: W1 -> ReLU -> W2.

#include <stdint.h>

#include "printf.h"

#include "data_llm_classic_ffn_prefill_f16.h"
#include "mempool_checks.h"

#include "baremetal/mempool_llm_classic_ffn_f16.h"
#include "baremetal/mempool_llm_tiling.h"

_Static_assert(S % ROW_TILE == 0, "ROW_TILE must divide S");
_Static_assert(ROW_TILE % LLM_PORT_WIDTH == 0,
               "ROW_TILE must be a RedMulE row multiple");
_Static_assert(D % LLM_PORT_WIDTH == 0,
               "D must be a RedMulE contraction/column multiple");
_Static_assert(I % PANEL_I == 0 && PANEL_I % LLM_PORT_WIDTH == 0,
               "PANEL_I must divide I and be a RedMulE multiple");
_Static_assert(LLM_F16_FITS_L1(2 * (ROW_TILE * D) + ROW_TILE * PANEL_I +
                               4 * (D * PANEL_I)),
               "classic FFN tiles exceed the L1 budget");

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
    printf("\n[llm_classic_ffn_prefill_f16] S=%d D=%d I=%d "
           "ROW_TILE=%d PANEL_I=%d NUM_IP=%d | redmule=%d cores=%d\n",
           S, D, I, ROW_TILE, PANEL_I, NUM_IP, NUM_REDMULE_TILES, num_cores);
  }
  mempool_barrier(num_cores);

  llm_classic_ffn_perf_t perf = {0};
  uint32_t time_init = mempool_get_timer();
#if !LLM_CLASSIC_STAGE_PERF
  mempool_start_benchmark();
#endif
  llm_classic_ffn_f16(l2_x, l2_w1, l2_w2, l2_check, l1_x, l1_hidden, l1_out,
                      l1_wbuf, S, D, I, ROW_TILE, PANEL_I, NUM_IP, core_id,
                      num_cores, redmule_id, &perf);
#if !LLM_CLASSIC_STAGE_PERF
  mempool_stop_benchmark();
#endif
  uint32_t time_end = mempool_get_timer();
  mempool_barrier(num_cores);

  if (core_id == 0) {
#if LLM_CLASSIC_STAGE_PERF
    printf("[llm_classic_ffn_prefill_f16] stage=load_x_w1_w2 cycles=%u\n",
           perf.load_x_w1_w2);
    printf("[llm_classic_ffn_prefill_f16] stage=w1_and_prefetch cycles=%u\n",
           perf.w1_and_prefetch);
    printf("[llm_classic_ffn_prefill_f16] stage=relu cycles=%u\n", perf.relu);
    printf("[llm_classic_ffn_prefill_f16] stage=w2 cycles=%u\n", perf.w2);
    printf("[llm_classic_ffn_prefill_f16] stage=prefetch_wait cycles=%u\n",
           perf.prefetch_wait);
    printf("[llm_classic_ffn_prefill_f16] stage=store_y cycles=%u\n",
           perf.store_y);
#endif
    printf("[llm_classic_ffn_prefill_f16] kernel takes %u clock cycles\n",
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
