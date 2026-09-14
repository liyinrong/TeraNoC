// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Token- and intermediate-tiled classic Transformer FFN:
//
//   hidden = ReLU(x @ W1)
//   y      = hidden @ W2
//
// There is no normalization, gating, or residual addition.  W1 is stored
// panel-major; W2 is row-major.  Both are double-buffered from L2.

#pragma once

#include <stdint.h>

#include "dma.h"
#include "runtime.h"
#include "synchronization.h"

#include "baremetal/mempool_llm_ffn_pipeline_f16.h"
#include "baremetal/mempool_llm_perf.h"

typedef struct {
  llm_stage_cycles_t load_x_w1_w2;
  llm_stage_cycles_t w1_and_prefetch;
  llm_stage_cycles_t relu;
  llm_stage_cycles_t w2;
  llm_stage_cycles_t prefetch_wait;
  llm_stage_cycles_t store_y;
} llm_classic_ffn_perf_t;

static inline void llm_classic_relu_f16(__fp16 *data, uint32_t count,
                                        uint32_t core_id, uint32_t num_cores) {
  uint32_t chunk = (count + num_cores - 1) / num_cores;
  uint32_t start = core_id * chunk;
  uint32_t end = start + chunk;
  if (end > count)
    end = count;

  const float ZERO_H = llm_f16box(0.0f);
  __fp16 *p = data + start;
  uint32_t avl = (start < end) ? end - start : 0;
  while (avl > 0) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle16.v v0, (%0)" : : "r"(p) : "memory");
    asm volatile("vfmax.vf v0, v0, %0" : : "f"(ZERO_H));
    asm volatile("vse16.v v0, (%0)" : : "r"(p) : "memory");
    p += vl;
    avl -= vl;
  }
}

// l1_x/l1_out: ROW_TILE*D, l1_hidden: ROW_TILE*PANEL_I,
// l1_wbuf: two sets of {W1[D*PANEL_I], W2[PANEL_I*D]}.
static inline void llm_classic_ffn_f16(
    const __fp16 *l2_x, const __fp16 *l2_w1, const __fp16 *l2_w2,
    __fp16 *l2_out, __fp16 *l1_x, __fp16 *l1_hidden, __fp16 *l1_out,
    __fp16 *l1_wbuf, uint32_t seqlen, uint32_t dmodel, uint32_t inter,
    uint32_t row_tile, uint32_t panel, uint32_t npanel, uint32_t core_id,
    uint32_t num_cores, uint32_t redmule_id, llm_classic_ffn_perf_t *perf) {
  (void)inter;
  const uint32_t w1_elems = dmodel * panel;
  const uint32_t w2_elems = panel * dmodel;
  const uint32_t set_elems = w1_elems + w2_elems;

  for (uint32_t row0 = 0; row0 < seqlen; row0 += row_tile) {
    const uint32_t ne_used = llm_ffn_ne_used(row_tile);
    const uint32_t rows_per_engine = row_tile / ne_used;

    {
      const mempool_timer_t stage_start = llm_stage_timer_start();
      dma_collective_memcpy_nonblocking(l1_x, l2_x + row0 * dmodel,
                                        row_tile * dmodel * sizeof(__fp16));
      dma_collective_memcpy_nonblocking(l1_wbuf, l2_w1,
                                        w1_elems * sizeof(__fp16));
      dma_collective_memcpy_blocking(l1_wbuf + w1_elems, l2_w2,
                                     w2_elems * sizeof(__fp16));
      llm_stage_timer_stop(stage_start, &perf->load_x_w1_w2);
    }

    for (uint32_t p = 0; p < npanel; p++) {
      __fp16 *cur = l1_wbuf + ((p & 1) ? set_elems : 0);
      __fp16 *nxt = l1_wbuf + ((p & 1) ? 0 : set_elems);
      __fp16 *w1 = cur;
      __fp16 *w2 = cur + w1_elems;
      const uint32_t prefetch = p + 1 < npanel;
      dma_collective_op_t prefetch_op = {.seq = 0};

      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        if (prefetch) {
          dma_collective_memcpy_nonblocking(nxt, l2_w1 + (p + 1) * w1_elems,
                                            w1_elems * sizeof(__fp16));
          prefetch_op = dma_collective_memcpy_nonblocking(
              nxt + w1_elems, l2_w2 + (p + 1) * w2_elems,
              w2_elems * sizeof(__fp16));
        }

        if (redmule_id < ne_used) {
          unsigned int x =
              (unsigned int)(l1_x + redmule_id * rows_per_engine * dmodel);
          unsigned int hidden =
              (unsigned int)(l1_hidden + redmule_id * rows_per_engine * panel);
          llm_zero_f16((__fp16 *)hidden, rows_per_engine * panel);
          llm_redmule_job(x, (unsigned int)w1, hidden,
                          (uint16_t)rows_per_engine, (uint16_t)dmodel,
                          (uint16_t)panel, GEMM);
        }
        mempool_barrier(num_cores);
        llm_stage_timer_stop(stage_start, &perf->w1_and_prefetch);
      }

      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        llm_classic_relu_f16(l1_hidden, row_tile * panel, core_id, num_cores);
        mempool_barrier(num_cores);
        llm_stage_timer_stop(stage_start, &perf->relu);
      }

      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        if (redmule_id < ne_used) {
          unsigned int hidden =
              (unsigned int)(l1_hidden + redmule_id * rows_per_engine * panel);
          unsigned int out =
              (unsigned int)(l1_out + redmule_id * rows_per_engine * dmodel);
          if (p == 0)
            llm_zero_f16((__fp16 *)out, rows_per_engine * dmodel);
          llm_redmule_job(hidden, (unsigned int)w2, out,
                          (uint16_t)rows_per_engine, (uint16_t)panel,
                          (uint16_t)dmodel, GEMM);
        }
        mempool_barrier(num_cores);
        llm_stage_timer_stop(stage_start, &perf->w2);
      }

      {
        const mempool_timer_t stage_start = llm_stage_timer_start();
        if (prefetch)
          dma_collective_wait(&prefetch_op);
        else
          mempool_barrier(num_cores);
        llm_stage_timer_stop(stage_start, &perf->prefetch_wait);
      }
    }

    {
      const mempool_timer_t stage_start = llm_stage_timer_start();
      dma_collective_memcpy_blocking(l2_out + row0 * dmodel, l1_out,
                                     row_tile * dmodel * sizeof(__fp16));
      llm_stage_timer_stop(stage_start, &perf->store_y);
    }
  }
}
