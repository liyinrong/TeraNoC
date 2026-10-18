// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Engine concurrency: RedMulE GEMMs, a double-buffered L2->L1 weight prefetch
// and a Spatz AXPY run at the same time, as in a layer. Each panel, the group
// coordinators prefetch the next weight panel, every tile agent runs one GEMM
// on the current panel, and the RVV cores of every tile stream an interleaved
// (mostly remote) AXPY. CONC_RM / CONC_DMA / CONC_SP switch each engine off to
// measure it alone. Performance only: no numeric check.

#include <stdint.h>

#include "dma_collective.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "baremetal/mempool_llm_attention_f16.h"
#include "baremetal/mempool_sp_axpy_f32.h"

#ifndef CONC_RM
#define CONC_RM 1
#endif
#ifndef CONC_DMA
#define CONC_DMA 1
#endif
#ifndef CONC_SP
#define CONC_SP 1
#endif
#ifndef CONC_PANELS
#define CONC_PANELS 4
#endif
#ifndef CONC_K
#define CONC_K 512 // GEMM depth (N of the RedMulE job)
#endif
#ifndef CONC_SP_REPS
#define CONC_SP_REPS 1 // AXPY sweeps over the core's chunk per panel
#endif

#define CONC_NE NUM_REDMULE_TILES
#define CONC_NE_C 4u
#define CONC_NE_R (CONC_NE / CONC_NE_C)
#define CONC_RPE 32u // rows per engine
#define CONC_CPE 64u // columns per engine
#define CONC_PANEL_ELEMS (CONC_NE_C * CONC_K * CONC_CPE)
#define CONC_SP_PER_TILE (NUM_CORES_PER_TILE - NUM_SCALAR_CORES_PER_TILE)
#define CONC_SP_CORES (NUM_CORES / NUM_CORES_PER_TILE * CONC_SP_PER_TILE)
#define CONC_SP_CHUNK 1024u

__fp16 conc_x[CONC_NE_R * CONC_RPE * CONC_K]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1")));
__fp16 conc_w[2][CONC_PANEL_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1")));
__fp16 conc_z[CONC_NE * CONC_RPE * CONC_CPE]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1")));
float conc_sx[CONC_SP_CORES * CONC_SP_CHUNK]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1")));
float conc_sy[CONC_SP_CORES * CONC_SP_CHUNK]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1")));
__fp16 conc_l2_w[CONC_PANELS * CONC_PANEL_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2")));

// The scalar cores of a 1+3 tile have no FP. Keep every FP instruction in here:
// inlined, the compiler hoists the 1.0f operand above the RVV-core branch and
// the scalar cores trap.
static void __attribute__((noinline)) conc_axpy(float *x, float *y,
                                                 uint32_t vl) {
  for (uint32_t r = 0; r < CONC_SP_REPS; r++)
    faxpy_v32b(1.0f, x, y, CONC_SP_CHUNK / vl, vl);
}

// Integer stores only: the scalar cores of a 1+3 tile have no FP.
static void fill_words(void *p, uint32_t bytes, uint32_t word, uint32_t core,
                       uint32_t num_cores) {
  uint32_t *w = (uint32_t *)p;
  for (uint32_t i = core; i < bytes / 4; i += num_cores)
    w[i] = word;
}

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  const uint32_t redmule_id = mempool_get_redmule_id();
  const uint32_t lane = core_id % NUM_CORES_PER_TILE;
  const uint32_t is_sp = lane >= NUM_SCALAR_CORES_PER_TILE;
  const uint32_t sp_idx = core_id / NUM_CORES_PER_TILE * CONC_SP_PER_TILE +
                          (lane - NUM_SCALAR_CORES_PER_TILE);

  mempool_barrier_init(core_id);
  fill_words(conc_x, sizeof(conc_x), 0x3c003c00u, core_id, num_cores);
  fill_words(conc_w, sizeof(conc_w), 0x3c003c00u, core_id, num_cores);
  fill_words(conc_z, sizeof(conc_z), 0u, core_id, num_cores);
  fill_words(conc_sx, sizeof(conc_sx), 0x3f800000u, core_id, num_cores);
  fill_words(conc_sy, sizeof(conc_sy), 0u, core_id, num_cores);
  fill_words(conc_l2_w, sizeof(conc_l2_w), 0x3c003c00u, core_id, num_cores);
  mempool_barrier(num_cores);

  uint32_t vl = 0;
  if (is_sp)
    asm volatile("vsetvli %0, %1, e32, m4, ta, ma"
                 : "=r"(vl)
                 : "r"(CONC_SP_CHUNK));
  const uint32_t t0 = mempool_get_timer();
  for (uint32_t p = 0; p < CONC_PANELS; p++) {
    const __fp16 *wcur = conc_w[p & 1];
#if CONC_DMA
    const uint32_t prefetch = p + 1 < CONC_PANELS;
    dma_collective_op_t op = {.seq = 0};
    if (prefetch)
      op = dma_collective_memcpy_nonblocking(
          conc_w[(p + 1) & 1], conc_l2_w + (p + 1) * CONC_PANEL_ELEMS,
          CONC_PANEL_ELEMS * sizeof(__fp16));
#endif
#if CONC_RM
    if (redmule_id < CONC_NE) {
      const uint32_t rg = redmule_id / CONC_NE_C, cg = redmule_id % CONC_NE_C;
      llm_attention_redmule_job(
          (unsigned int)(conc_x + rg * CONC_RPE * CONC_K),
          (unsigned int)(wcur + cg * CONC_K * CONC_CPE),
          (unsigned int)(conc_z + redmule_id * CONC_RPE * CONC_CPE),
          (uint16_t)CONC_RPE, (uint16_t)CONC_K, (uint16_t)CONC_CPE, GEMM);
    }
#else
    (void)wcur;
#endif
#if CONC_SP
    if (is_sp)
      conc_axpy(conc_sx + sp_idx * CONC_SP_CHUNK,
                conc_sy + sp_idx * CONC_SP_CHUNK, vl);
#endif
#if CONC_DMA
    if (prefetch)
      dma_collective_wait(&op);
#endif
    mempool_barrier(num_cores);
  }
  const uint32_t t1 = mempool_get_timer();

  if (core_id == 0) {
    printf("[dac_concurrency] rm=%d dma=%d sp=%d panels=%d k=%d sp_reps=%d "
           "cycles=%u\n",
           CONC_RM, CONC_DMA, CONC_SP, CONC_PANELS, CONC_K, CONC_SP_REPS,
           t1 - t0);
    printf("[dac_concurrency] kernel takes %u clock cycles\n", t1 - t0);
  }
  mempool_barrier(num_cores);
  return 0;
}
