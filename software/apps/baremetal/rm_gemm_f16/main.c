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

#include "data_rm_gemm_f16.h"
#include "mempool_checks.h"

#define ELEMENTS_PER_ROW (NUM_BANKS * sizeof(int32_t) / sizeof(int16_t))
#define PORT_WIDTH (REDMULE_H * (REDMULE_P + 1))
// Engine mode: default SINGLE; the build plan selects the multi-engine
// throughput-optimized path with -DPARALLEL.
#if !defined(SINGLE) && !defined(PARALLEL)
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

__fp16 l1_X[(matrix_M * matrix_N) +
            PORT_WIDTH * NUM_REDMULE_TILES * (NUM_REDMULE_TILES + 1)]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_W[matrix_N * matrix_P]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 l1_Y[matrix_M * matrix_P]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));

__fp16 l2_Z_check[matrix_M * matrix_P]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  uint32_t redmule_id = mempool_get_redmule_id();
  uint32_t time_init, time_end;
  mempool_barrier_init(core_id);

#ifdef SINGLE
  // Transfer
  if (tnc_is_group_coordinator()) {
    dma_collective_all_memcpy_blocking(l1_X, l2_X,
                                       (matrix_M * matrix_N) * sizeof(int16_t));
    dma_collective_all_memcpy_blocking(l1_W, l2_W,
                                       (matrix_N * matrix_P) * sizeof(int16_t));
    dma_collective_all_memcpy_blocking(l1_Y, l2_Y,
                                       (matrix_M * matrix_P) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);
  // Compute
  mempool_barrier(num_cores);
  time_init = mempool_get_timer();
  mempool_start_benchmark();
  if (redmule_id == 0 && tnc_is_tile_agent()) {
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
  if (tnc_is_group_coordinator()) {
    for (uint32_t i = 0; i < num_redmules; i++) {
      X_shift = (XSHIFT == 1) ? (i * PORT_WIDTH) % matrix_N : 0;
      dma_collective_all_memcpy_blocking(
          l1_X + i * (matrix_M * matrix_N / num_redmules) + X_shift,
          l2_X + i * (matrix_M * matrix_N / num_redmules),
          (matrix_M * matrix_N / num_redmules) * sizeof(int16_t));
    }
    dma_collective_all_memcpy_blocking(l1_W, l2_W,
                                       (matrix_N * matrix_P) * sizeof(int16_t));
    dma_collective_all_memcpy_blocking(l1_Y, l2_Y,
                                       (matrix_M * matrix_P) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

  // Compute
  mempool_barrier(num_cores);
  time_init = mempool_get_timer();
  mempool_start_benchmark();
  if (redmule_id < num_redmules && tnc_is_tile_agent()) {
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
    printf("\nKernel execution takes %d clock cycles\n", clock_cycles);
  }
  if (mempool_is_group_coordinator()) {
    dma_collective_all_memcpy_blocking(l2_Z_check, l1_Y,
                                       matrix_M * matrix_P * sizeof(int16_t));
  }
  mempool_check_dpi_f16(l2_Z_check, l2_Z, matrix_M * matrix_P, 0.05f, 0);
  mempool_barrier(num_cores);
  return 0;
}
