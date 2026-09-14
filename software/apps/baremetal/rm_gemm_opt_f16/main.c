// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Marco Bertuletti, ETH Zurich

#include <stdint.h>
#include <string.h>

#include "async_task.h"
#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "archi_redmule.h"
#include "hal_redmule.h"

#include "data_rm_gemm_opt_f16.h"
#include "mempool_checks.h"

#define ELEMENTS_PER_ROW (NUM_BANKS * sizeof(int32_t) / sizeof(int16_t))
#define PORT_WIDTH (REDMULE_H * (REDMULE_P + 1))

// Engine mode: default SINGLE. PARALLEL preserves the shipped row split;
// RM_GEMM_TILED_2D selects the packed row-by-column DSE schedule from the data
// header so the operand layout and kernel cannot silently disagree.
#ifndef RM_GEMM_TILED_2D
#define RM_GEMM_TILED_2D 0
#endif
#ifndef LLM_PERF_ONLY
#define LLM_PERF_ONLY 0
#endif
#if RM_GEMM_TILED_2D != 0 && RM_GEMM_TILED_2D != 1
#error "RM_GEMM_TILED_2D must be 0 or 1"
#endif
#if defined(SINGLE) && defined(PARALLEL)
#error "Select only one of SINGLE and PARALLEL"
#endif
#if RM_GEMM_TILED_2D && (defined(SINGLE) || defined(PARALLEL))
#error "RM_GEMM_TILED_2D is exclusive with SINGLE and PARALLEL"
#endif
#if !RM_GEMM_TILED_2D && !defined(SINGLE) && !defined(PARALLEL)
#define SINGLE
#endif
// Stagger each engine's X panel in L1 / its starting W column, so the engines
// do not all hit the same banks at the same time. Overridable from the build.
#ifndef XSHIFT
#define XSHIFT 0
#endif
#ifndef WSHIFT
#define WSHIFT 0
#endif

#if RM_GEMM_TILED_2D

#include "baremetal/mempool_llm_tiling.h"

#if NE_R < 1 || NE_C < 1
#error "NE_R and NE_C must be positive"
#endif

_Static_assert(matrix_M > 0 && matrix_N > 0 && matrix_P > 0,
               "GEMM dimensions must be positive");
_Static_assert(matrix_M % ROW_TILE == 0, "ROW_TILE must divide matrix_M");
_Static_assert(matrix_P % COL_TILE == 0, "COL_TILE must divide matrix_P");
_Static_assert(ROW_TILE % NE_R == 0, "NE_R must divide ROW_TILE");
_Static_assert(COL_TILE % NE_C == 0, "NE_C must divide COL_TILE");
_Static_assert(NE_R *NE_C == NUM_REDMULE_TILES,
               "the 2-D map must use every RedMulE");
_Static_assert(NUM_CORES % NUM_REDMULE_TILES == 0,
               "each RedMulE needs an integral core-owner stride");
_Static_assert((ROW_TILE / NE_R) % PORT_WIDTH == 0,
               "each engine row shard must tile the RedMulE port");
_Static_assert(matrix_N % PORT_WIDTH == 0,
               "the contraction must tile the RedMulE port");
_Static_assert((COL_TILE / NE_C) % PORT_WIDTH == 0,
               "each engine column shard must tile the RedMulE port");
_Static_assert((ROW_TILE / NE_R) <= UINT16_MAX && matrix_N <= UINT16_MAX &&
                   (COL_TILE / NE_C) <= UINT16_MAX,
               "RedMulE job dimensions must fit uint16_t");
_Static_assert((((long)ROW_TILE * matrix_N + (long)matrix_N * COL_TILE +
                 (long)ROW_TILE * COL_TILE) *
                (long)sizeof(__fp16)) <= L1_TILE_BUDGET_BYTES,
               "tiled GEMM buffers exceed the L1 budget");

__fp16 l1_X[ROW_TILE * matrix_N]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_W[matrix_N * COL_TILE]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_Y[ROW_TILE * COL_TILE]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));

#else

__fp16 l1_X[(matrix_M * matrix_N) +
            PORT_WIDTH * NUM_REDMULE_TILES * (NUM_REDMULE_TILES + 1)]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_W[matrix_N * matrix_P]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_Y[matrix_M * matrix_P]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));

#endif

__fp16 l2_Z_check[matrix_M * matrix_P]
#if RM_GEMM_TILED_2D
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
#else
    __attribute__((aligned(4 * NUM_BANKS), section(".l2")));
#endif

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  uint32_t redmule_id = mempool_get_redmule_id();
  uint32_t time_init, time_end;
  mempool_barrier_init(core_id);

#if RM_GEMM_TILED_2D
  const uint32_t rows_per_engine = ROW_TILE / NE_R;
  const uint32_t cols_per_engine = COL_TILE / NE_C;
  const uint32_t tile_elems = ROW_TILE * COL_TILE;
  const uint32_t num_col_panels = matrix_P / COL_TILE;
  uint32_t stage_cycles = 0;
  uint32_t gemm_cycles = 0;

  if (core_id == 0) {
    printf("\n[rm_gemm_opt_f16] M=%d N=%d P=%d LLM_PERF_ONLY=%d "
           "mapping=tiled_2d NE_R=%d NE_C=%d ROW_TILE=%d COL_TILE=%d "
           "local=%dx%dx%d redmule=%d cores=%d\n",
           matrix_M, matrix_N, matrix_P, LLM_PERF_ONLY, NE_R, NE_C, ROW_TILE,
           COL_TILE, rows_per_engine, matrix_N, cols_per_engine,
           NUM_REDMULE_TILES, num_cores);
  }
  mempool_barrier(num_cores);

  time_init = mempool_get_timer();
  mempool_start_benchmark();
  for (uint32_t col0 = 0; col0 < matrix_P; col0 += COL_TILE) {
    const uint32_t col_panel = col0 / COL_TILE;
    uint32_t stage_start = mempool_get_timer();
    dma_collective_memcpy_blocking(l1_W, l2_W + col_panel * matrix_N * COL_TILE,
                                   matrix_N * COL_TILE * sizeof(__fp16));
    stage_cycles += mempool_get_timer() - stage_start;

    for (uint32_t row0 = 0; row0 < matrix_M; row0 += ROW_TILE) {
      const uint32_t row_panel = row0 / ROW_TILE;
      const uint32_t output_tile =
          (row_panel * num_col_panels + col_panel) * tile_elems;

      stage_start = mempool_get_timer();
      dma_collective_memcpy_nonblocking(l1_X, l2_X + row0 * matrix_N,
                                        ROW_TILE * matrix_N * sizeof(__fp16));
      dma_collective_memcpy_blocking(l1_Y, l2_Y + output_tile,
                                     tile_elems * sizeof(__fp16));
      stage_cycles += mempool_get_timer() - stage_start;

      uint32_t gemm_start = mempool_get_timer();
      if (redmule_id < NUM_REDMULE_TILES) {
        const uint32_t row_group = redmule_id / NE_C;
        const uint32_t col_group = redmule_id % NE_C;
        unsigned int X_ptr =
            (unsigned int)(l1_X + row_group * rows_per_engine * matrix_N);
        unsigned int W_ptr =
            (unsigned int)(l1_W + col_group * matrix_N * cols_per_engine);
        unsigned int Y_ptr =
            (unsigned int)(l1_Y +
                           redmule_id * rows_per_engine * cols_per_engine);
        const int32_t target = tnc_redmule_acquire_target();
        redmule_cfg(X_ptr, W_ptr, Y_ptr, (uint16_t)rows_per_engine,
                    (uint16_t)matrix_N, (uint16_t)cols_per_engine, 0, GEMM,
                    Float16);
        mempool_wait(10);
        tnc_redmule_trigger();
        tnc_redmule_wait((uint8_t)target);
      }
      mempool_log_barrier(2, core_id);
      gemm_cycles += mempool_get_timer() - gemm_start;

      stage_start = mempool_get_timer();
      dma_collective_memcpy_blocking(l2_Z_check + output_tile, l1_Y,
                                     tile_elems * sizeof(__fp16));
      stage_cycles += mempool_get_timer() - stage_start;
    }
  }
  mempool_stop_benchmark();
  time_end = mempool_get_timer();
#endif

#ifdef SINGLE
  // Transfer
  dma_collective_memcpy_nonblocking(l1_X, l2_X,
                                    (matrix_M * matrix_N) * sizeof(int16_t));
  dma_collective_memcpy_nonblocking(l1_W, l2_W,
                                    (matrix_N * matrix_P) * sizeof(int16_t));
  dma_collective_memcpy_blocking(l1_Y, l2_Y,
                                 (matrix_M * matrix_P) * sizeof(int16_t));
  // Compute
  time_init = mempool_get_timer();
  mempool_start_benchmark();
  if (redmule_id == 0) {
    unsigned int X_ptr = (unsigned int)(l1_X);
    unsigned int Y_ptr = (unsigned int)(l1_Y);
    unsigned int W_ptr = (unsigned int)(l1_W);
    const int32_t target = tnc_redmule_acquire_target();
    redmule_cfg(X_ptr, W_ptr, Y_ptr, matrix_M, matrix_N, matrix_P, 0, GEMM,
                Float16);
    mempool_wait(10);
    tnc_redmule_trigger();
    tnc_redmule_wait((uint8_t)target);
  }
  mempool_log_barrier(2, core_id);
  mempool_stop_benchmark();
  time_end = mempool_get_timer();
#endif

#ifdef PARALLEL

  uint32_t num_redmules = mempool_get_redmule_count();
  uint32_t X_shift;
  uint32_t W_shift;

  // Transfer
  for (uint32_t i = 0; i < num_redmules; i++) {
    X_shift = (XSHIFT == 1) ? (i * PORT_WIDTH) % matrix_N : 0;
    dma_collective_memcpy_nonblocking(
        l1_X + i * (matrix_M * matrix_N / num_redmules) + X_shift,
        l2_X + i * (matrix_M * matrix_N / num_redmules),
        (matrix_M * matrix_N / num_redmules) * sizeof(int16_t));
  }
  dma_collective_memcpy_nonblocking(l1_W, l2_W,
                                    (matrix_N * matrix_P) * sizeof(int16_t));
  dma_collective_memcpy_blocking(l1_Y, l2_Y,
                                 (matrix_M * matrix_P) * sizeof(int16_t));

  // Compute
  time_init = mempool_get_timer();
  mempool_start_benchmark();
  if (redmule_id < num_redmules) {
    X_shift = (XSHIFT == 1) ? (redmule_id * PORT_WIDTH) % matrix_N : 0;
    W_shift = (WSHIFT == 1) ? (redmule_id * PORT_WIDTH) % matrix_P : 0;
    unsigned int X_ptr =
        (unsigned int)(l1_X +
                       redmule_id * (matrix_M * matrix_N / num_redmules) +
                       X_shift);
    unsigned int Y_ptr =
        (unsigned int)(l1_Y +
                       redmule_id * (matrix_M * matrix_P / num_redmules));
    unsigned int W_ptr = (unsigned int)(l1_W);
    const int32_t target = tnc_redmule_acquire_target();
    redmule_cfg(X_ptr, W_ptr, Y_ptr, (matrix_M / num_redmules), matrix_N,
                matrix_P, W_shift, GEMM, Float16);
    mempool_wait(10);
    tnc_redmule_trigger();
    tnc_redmule_wait((uint8_t)target);
  }
  mempool_log_barrier(2, core_id);
  mempool_stop_benchmark();
  time_end = mempool_get_timer();
#endif

  if (core_id == 0) {
    uint32_t clock_cycles = (time_end - time_init);
#if RM_GEMM_TILED_2D
    printf("[rm_gemm_opt_f16] stage=gemm cycles=%u\n", gemm_cycles);
    printf("[rm_gemm_opt_f16] stage=stage cycles=%u\n", stage_cycles);
    printf("[rm_gemm_opt_f16] kernel takes %u clock cycles\n", clock_cycles);
#else
    printf("\nKernel execution takes %d clock cycles\n", clock_cycles);
#endif
  }
#if !RM_GEMM_TILED_2D
  dma_collective_memcpy_blocking(l2_Z_check, l1_Y,
                                 matrix_M * matrix_P * sizeof(int16_t));
#endif
#if !LLM_PERF_ONLY
  mempool_check_dpi_f16(l2_Z_check, l2_Z, matrix_M * matrix_P, 0.05f, 0);
#endif
  mempool_barrier(num_cores);
  return 0;
}
