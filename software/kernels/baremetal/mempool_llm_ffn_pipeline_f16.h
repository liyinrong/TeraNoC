// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yinrong Li, ETH Zurich

// Dense-LLM SwiGLU FFN block pipeline for the TeraNoC RedMulE + Spatz cluster.
//
//   h = RMSNorm(x) * gamma         (Spatz RVV, per-row)
//   for each intermediate panel p of width PANEL_I:
//     g_p = h @ Wg_p               (RedMulE, gate_proj panel)
//     u_p = h @ Wu_p               (RedMulE, up_proj panel)
//     a_p = SiLU(g_p) * u_p        (Spatz RVV, elementwise)
//     out += a_p @ Wd_p            (RedMulE, down_proj, accumulated)
//   out += x                       (Spatz RVV, residual)
//
// FUSED intermediate tiling: the block streams the I dimension one PANEL_I-wide
// panel at a time, so only ONE panel of the gate/up intermediate (S x PANEL_I)
// is ever resident instead of the full S x I. This keeps the L1 working set
// bounded regardless of the (large) intermediate width, which is what lets real
// Llama-3.1-8B geometry (I = 14336) run: the full 352 MB of weights live in L2
// and are streamed panel-by-panel.
//
// DOUBLE BUFFERING: while the RedMulE engines compute panel p, one coordinator
// in every group prefetches that group's stripe of the three p+1 weight panels
// with non-blocking DMA. l1_wbuf holds two sets of {Wg_p, Wu_p, Wd_p}
// (6 * D * PANEL_I elements).
//
// Weight layouts: Wg/Wu are PANEL-MAJOR ([NUM_IP][D x PANEL_I]) so panel p is a
// contiguous DMA; Wd is NATURAL [I x D] so contraction panel p is the
// contiguous [PANEL_I x D] row-block. Each GEMM is row-split across
// min(NUM_REDMULE_TILES, S/PORT_WIDTH) engines; every tile dim is a multiple of
// the RedMulE PORT_WIDTH.

#pragma once
#include <stdint.h>

#include "async_task.h"
#include "dma.h"
#include "runtime.h"
#include "synchronization.h"

#include "archi_redmule.h"
#include "hal_redmule.h"

#include "baremetal/mempool_llm_ffn_f16.h"

#ifndef RVV
#error "mempool_llm_ffn_pipeline_f16.h requires a Spatz (RVV) configuration"
#endif

#if NUM_REDMULE_TILES < 1
#error "mempool_llm_ffn_pipeline_f16.h requires a RedMulE configuration"
#endif

// RedMulE GEMM input/output format (override to Float8 for the FP8 path once
// the FP8 golden-data generator is in place).
#ifndef LLM_GEMM_FMT
#define LLM_GEMM_FMT Float16
#endif

#define LLM_PORT_WIDTH (REDMULE_H * (REDMULE_P + 1))

// Number of RedMulE engines used for an S-row GEMM, and rows per engine. The
// row count is forced to divide evenly and to stay a multiple of PORT_WIDTH.
static inline uint32_t llm_ffn_ne_used(uint32_t seqlen) {
  uint32_t nblk = seqlen / LLM_PORT_WIDTH; // PORT_WIDTH-row blocks
  uint32_t ne = NUM_REDMULE_TILES;
  uint32_t ne_used = (ne < nblk) ? ne : nblk;
  if (ne_used < 1)
    ne_used = 1;
  while (nblk % ne_used != 0)
    ne_used--;
  return ne_used;
}

// One RedMulE job on the calling owner core's tile-local engine.
static inline void llm_redmule_job(unsigned int x, unsigned int w,
                                   unsigned int z, uint16_t m, uint16_t n,
                                   uint16_t k, uint8_t op) {
  const int32_t target = tnc_redmule_acquire_target();
  redmule_cfg(x, w, z, m, n, k, 0, op, LLM_GEMM_FMT);
  mempool_wait(10);
  tnc_redmule_trigger();
  tnc_redmule_wait((uint8_t)target);
}

// Full SwiGLU FFN block, fused over intermediate panels with double-buffered
// weight streaming. Inputs in L2, scratch + result in L1. Leaves the result in
// l1_out for the caller to copy back to L2 (and check, for the toy shapes).
//
// L1 scratch sizes (elements):
//   l1_x, l1_h, l1_out : S * D
//   l1_gamma           : D
//   l1_g, l1_u         : S * PANEL_I     (one intermediate panel)
//   l1_wbuf            : 6 * D * PANEL_I  (double-buffered Wg_p|Wu_p|Wd_p)
static inline void
llm_ffn_swiglu_f16(const __fp16 *l2_x, const __fp16 *l2_gamma,
                   const __fp16 *l2_wg, const __fp16 *l2_wu,
                   const __fp16 *l2_wd, __fp16 *l1_x, __fp16 *l1_gamma,
                   __fp16 *l1_h, __fp16 *l1_g, __fp16 *l1_u, __fp16 *l1_out,
                   __fp16 *l1_wbuf, uint32_t seqlen, uint32_t dmodel,
                   uint32_t inter, uint32_t panel, uint32_t npanel,
                   uint32_t core_id, uint32_t num_cores, uint32_t redmule_id) {
  (void)inter;
  const uint32_t ne_used = llm_ffn_ne_used(seqlen);
  const uint32_t Sk = seqlen / ne_used;               // rows per engine
  const uint32_t gu_elems = dmodel * panel;           // Wg_p / Wu_p panel size
  const uint32_t wd_elems = panel * dmodel;           // Wd_p panel size
  const uint32_t set_elems = 2 * gu_elems + wd_elems; // one double-buffer set

  // Stage inputs into L1 (activations resident; weights stream per panel).
  dma_collective_memcpy_nonblocking(l1_x, l2_x,
                                    seqlen * dmodel * sizeof(__fp16));
  dma_collective_memcpy_blocking(l1_gamma, l2_gamma, dmodel * sizeof(__fp16));

  // h = RMSNorm(x) * gamma
  llm_rmsnorm_f16(l1_x, l1_gamma, l1_h, seqlen, dmodel, core_id, num_cores);
  mempool_barrier(num_cores);

  // Prime panel 0's three weight panels.
  dma_collective_memcpy_nonblocking(l1_wbuf, l2_wg, gu_elems * sizeof(__fp16));
  dma_collective_memcpy_nonblocking(l1_wbuf + gu_elems, l2_wu,
                                    gu_elems * sizeof(__fp16));
  dma_collective_memcpy_blocking(l1_wbuf + 2 * gu_elems, l2_wd,
                                 wd_elems * sizeof(__fp16));

  for (uint32_t p = 0; p < npanel; p++) {
    __fp16 *cur = l1_wbuf + ((p & 1) ? set_elems : 0);
    __fp16 *nxt = l1_wbuf + ((p & 1) ? 0 : set_elems);
    __fp16 *wg = cur;
    __fp16 *wu = cur + gu_elems;
    __fp16 *wd = cur + 2 * gu_elems;

    // Prefetch panel p+1's weights (overlaps this panel's compute).
    const uint32_t prefetch = (p + 1) < npanel;
    dma_collective_op_t prefetch_op = {.seq = 0};
    if (prefetch) {
      dma_collective_memcpy_nonblocking(nxt, l2_wg + (p + 1) * gu_elems,
                                        gu_elems * sizeof(__fp16));
      dma_collective_memcpy_nonblocking(nxt + gu_elems,
                                        l2_wu + (p + 1) * gu_elems,
                                        gu_elems * sizeof(__fp16));
      prefetch_op = dma_collective_memcpy_nonblocking(
          nxt + 2 * gu_elems, l2_wd + (p + 1) * wd_elems,
          wd_elems * sizeof(__fp16));
    }

    // g_p = h @ Wg_p ; u_p = h @ Wu_p  (row-split across engines).
    if (redmule_id < ne_used) {
      unsigned int X = (unsigned int)(l1_h + redmule_id * Sk * dmodel);
      unsigned int Zg = (unsigned int)(l1_g + redmule_id * Sk * panel);
      unsigned int Zu = (unsigned int)(l1_u + redmule_id * Sk * panel);
      // Keep the pre-fix zero-addend GEMM sequence for compatibility; current
      // RTL MATMUL also starts each traversal from zero.
      llm_zero_f16((__fp16 *)Zg, Sk * panel);
      llm_zero_f16((__fp16 *)Zu, Sk * panel);
      llm_redmule_job(X, (unsigned int)wg, Zg, (uint16_t)Sk, (uint16_t)dmodel,
                      (uint16_t)panel, GEMM);
      llm_redmule_job(X, (unsigned int)wu, Zu, (uint16_t)Sk, (uint16_t)dmodel,
                      (uint16_t)panel, GEMM);
    }
    mempool_barrier(num_cores);

    // a_p = SiLU(g_p) * u_p  (in place into l1_g).
    llm_silu_mul_f16(l1_g, l1_u, l1_g, seqlen * panel, core_id, num_cores);
    mempool_barrier(num_cores);

    // out += a_p @ Wd_p (l1_out was explicitly zeroed before panel 0).
    if (redmule_id < ne_used) {
      unsigned int X = (unsigned int)(l1_g + redmule_id * Sk * panel);
      unsigned int Z = (unsigned int)(l1_out + redmule_id * Sk * dmodel);
      if (p == 0)
        llm_zero_f16((__fp16 *)Z, Sk * dmodel);
      llm_redmule_job(X, (unsigned int)wd, Z, (uint16_t)Sk, (uint16_t)panel,
                      (uint16_t)dmodel, GEMM);
    }
    mempool_barrier(num_cores);
    if (prefetch)
      dma_collective_wait(&prefetch_op);
    else
      mempool_barrier(num_cores);
  }

  // out += x  (residual)
  llm_residual_add_f16(l1_out, l1_x, seqlen * dmodel, core_id, num_cores);
  mempool_barrier(num_cores);
}

// L1-constrained variant. RMS-normalized rows and partial outputs spill to L2,
// while one ROW_TILE x D activation/output buffer and one set of three weight
// panels remain in L1. The panel loop is outermost, so every model weight is
// transferred from L2 once; partial outputs are streamed once per I panel.
// l1_row is reused for h and out, and l1_g is reused as residual-DMA scratch
// after the final down projection.
//
// L1 scratch sizes (f16 elements):
//   l1_gamma : D
//   l1_row   : ROW_TILE * D
//   l1_g/u   : ROW_TILE * PANEL_I each
//   l1_wbuf  : 3 * D * PANEL_I
static inline void llm_ffn_swiglu_row_tiled_f16(
    const __fp16 *l2_x, const __fp16 *l2_gamma, const __fp16 *l2_wg,
    const __fp16 *l2_wu, const __fp16 *l2_wd, __fp16 *l2_h, __fp16 *l2_out,
    __fp16 *l1_gamma, __fp16 *l1_row, __fp16 *l1_g, __fp16 *l1_u,
    __fp16 *l1_wbuf, uint32_t seqlen, uint32_t row_tile, uint32_t dmodel,
    uint32_t inter, uint32_t panel, uint32_t npanel, uint32_t core_id,
    uint32_t num_cores, uint32_t redmule_id) {
  (void)inter;
  const uint32_t nrow_tiles = seqlen / row_tile;
  const uint32_t ne_used = llm_ffn_ne_used(row_tile);
  const uint32_t rows_per_engine = row_tile / ne_used;
  const uint32_t gu_elems = dmodel * panel;
  const uint32_t wd_elems = panel * dmodel;
  const uint32_t row_elems = row_tile * dmodel;
  const uint32_t activation_elems = row_tile * panel;

  dma_collective_memcpy_blocking(l1_gamma, l2_gamma, dmodel * sizeof(__fp16));

  // Normalize every row once. The persistent normalized activation lives in
  // L2 because a real S x D tensor is larger than the constrained L1.
  for (uint32_t rt = 0; rt < nrow_tiles; rt++) {
    const uint32_t row0 = rt * row_tile;
    dma_collective_memcpy_blocking(l1_row, l2_x + row0 * dmodel,
                                   row_elems * sizeof(__fp16));
    llm_rmsnorm_f16(l1_row, l1_gamma, l1_row, row_tile, dmodel, core_id,
                    num_cores);
    mempool_barrier(num_cores);
    dma_collective_memcpy_blocking(l2_h + row0 * dmodel, l1_row,
                                   row_elems * sizeof(__fp16));
  }

  for (uint32_t p = 0; p < npanel; p++) {
    __fp16 *wg = l1_wbuf;
    __fp16 *wu = l1_wbuf + gu_elems;
    __fp16 *wd = l1_wbuf + 2 * gu_elems;

    dma_collective_memcpy_nonblocking(wg, l2_wg + p * gu_elems,
                                      gu_elems * sizeof(__fp16));
    dma_collective_memcpy_nonblocking(wu, l2_wu + p * gu_elems,
                                      gu_elems * sizeof(__fp16));
    dma_collective_memcpy_blocking(wd, l2_wd + p * wd_elems,
                                   wd_elems * sizeof(__fp16));

    for (uint32_t rt = 0; rt < nrow_tiles; rt++) {
      const uint32_t row0 = rt * row_tile;
      dma_collective_memcpy_blocking(l1_row, l2_h + row0 * dmodel,
                                     row_elems * sizeof(__fp16));

      if (redmule_id < ne_used) {
        unsigned int x =
            (unsigned int)(l1_row + redmule_id * rows_per_engine * dmodel);
        unsigned int zg =
            (unsigned int)(l1_g + redmule_id * rows_per_engine * panel);
        unsigned int zu =
            (unsigned int)(l1_u + redmule_id * rows_per_engine * panel);
        llm_zero_f16((__fp16 *)zg, rows_per_engine * panel);
        llm_zero_f16((__fp16 *)zu, rows_per_engine * panel);
        llm_redmule_job(x, (unsigned int)wg, zg, (uint16_t)rows_per_engine,
                        (uint16_t)dmodel, (uint16_t)panel, GEMM);
        llm_redmule_job(x, (unsigned int)wu, zu, (uint16_t)rows_per_engine,
                        (uint16_t)dmodel, (uint16_t)panel, GEMM);
      }
      mempool_barrier(num_cores);

      llm_silu_mul_f16(l1_g, l1_u, l1_g, activation_elems, core_id, num_cores);
      mempool_barrier(num_cores);

      // h is no longer needed for this panel, so reuse l1_row for the partial
      // output. Panel zero starts from +0; later panels load the FP16 feedback.
      if (p != 0)
        dma_collective_memcpy_blocking(l1_row, l2_out + row0 * dmodel,
                                       row_elems * sizeof(__fp16));
      else
        mempool_barrier(num_cores);
      if (redmule_id < ne_used) {
        unsigned int x =
            (unsigned int)(l1_g + redmule_id * rows_per_engine * panel);
        unsigned int z =
            (unsigned int)(l1_row + redmule_id * rows_per_engine * dmodel);
        if (p == 0)
          llm_zero_f16((__fp16 *)z, rows_per_engine * dmodel);
        llm_redmule_job(x, (unsigned int)wd, z, (uint16_t)rows_per_engine,
                        (uint16_t)panel, (uint16_t)dmodel, GEMM);
      }
      mempool_barrier(num_cores);

      // Add the residual on the final panel. Stream x through the now-dead gate
      // buffer in bounded chunks so a second ROW_TILE x D buffer is
      // unnecessary.
      if ((p + 1) == npanel) {
        for (uint32_t off = 0; off < row_elems; off += activation_elems) {
          uint32_t n = row_elems - off;
          if (n > activation_elems)
            n = activation_elems;
          dma_collective_memcpy_blocking(l1_g, l2_x + row0 * dmodel + off,
                                         n * sizeof(__fp16));
          llm_residual_add_f16(l1_row + off, l1_g, n, core_id, num_cores);
          mempool_barrier(num_cores);
        }
      }

      dma_collective_memcpy_blocking(l2_out + row0 * dmodel, l1_row,
                                     row_elems * sizeof(__fp16));
    }
  }
}
