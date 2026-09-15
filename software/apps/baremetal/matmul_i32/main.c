// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Samuel Riedel, ETH Zurich
//         Yinrong Li, ETH Zurich

#include <stdint.h>
#include <string.h>

#include "dma.h"
#include "encoding.h"
#include "runtime.h"
#include "synchronization.h"

#include "data_matmul_i32.h"

#include "baremetal/mempool_matmul_i32p.h"
#include "mempool_checks.h"

// Barrier selection. Build with -DUSE_FSYNC_BARRIER to run every barrier on
// the fractal_sync H-tree instead of the software barrier; the default is
// unchanged, so previously recorded numbers stay comparable. The two must not
// be mixed inside one phase -- mempool_barrier's sleeping path treats any wake
// as its own -- so the switch covers ALL of this app's barriers at once.
#ifdef USE_FSYNC_BARRIER
#include "fsync_app.h"
#define APP_BARRIER_INIT(core) mempool_barrier_init(core)
#define APP_BARRIER(n) fsync_app_barrier()
#define APP_KERNEL_BARRIER(core) fsync_app_barrier()
#else
#define APP_BARRIER_INIT(core) mempool_barrier_init(core)
#define APP_BARRIER(n) mempool_barrier(n)
#define APP_KERNEL_BARRIER(core) mempool_log_barrier(2, core)
#endif

#define NOC_OPT

int32_t matrix_a[matrix_M * matrix_N]
    __attribute__((aligned(NUM_BANKS * 4), section(".l1_prio")));
int32_t matrix_b[matrix_N * matrix_P]
    __attribute__((aligned(NUM_BANKS * 4), section(".l1_prio")));
int32_t matrix_c[matrix_M * matrix_P]
    __attribute__((aligned(NUM_BANKS * 4), section(".l1_prio")));

int32_t l2_C_check[matrix_M * matrix_P]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  uint32_t time_init, time_end;
  APP_BARRIER_INIT(core_id);

  // Initialize data
  dma_collective_memcpy_nonblocking(matrix_a, l2_A,
                                    matrix_M * matrix_N * sizeof(int32_t));
  dma_collective_memcpy_blocking(matrix_b, l2_B,
                                 matrix_N * matrix_P * sizeof(int32_t));
  APP_BARRIER(num_cores);

  // Benchmark
  time_init = mempool_get_timer();
#if defined(NOC_OPT)
  mempool_start_benchmark();
  mat_mul_unrolled_4x4_conflict_nocopt_parallel_asm(
      matrix_a, matrix_b, matrix_c, matrix_M, matrix_N, matrix_P, core_id,
      num_cores);
  APP_KERNEL_BARRIER(core_id);
  mempool_stop_benchmark();
#elif defined(XBAR_OPT)
  mempool_start_benchmark();
  mat_mul_unrolled_4x4_conflict_opt_parallel_asm(matrix_a, matrix_b, matrix_c,
                                                 matrix_M, matrix_N, matrix_P,
                                                 core_id, num_cores);
  APP_KERNEL_BARRIER(core_id);
  mempool_stop_benchmark();
#elif defined(NO_OPT)
  mempool_start_benchmark();
  mat_mul_unrolled_4x4_parallel_asm(matrix_a, matrix_b, matrix_c, matrix_M,
                                    matrix_N, matrix_P, core_id, num_cores);
  APP_KERNEL_BARRIER(core_id);
  mempool_stop_benchmark();
#elif defined(__XPULPIMG)
  mempool_start_benchmark();
  matmul_unrolled_2x2_parallel_i32_xpulpv2(matrix_a, matrix_b, matrix_c,
                                           matrix_M, matrix_N, matrix_P,
                                           core_id, num_cores);
  APP_KERNEL_BARRIER(core_id);
  mempool_stop_benchmark();
#else
  mempool_start_benchmark();
  matmul_unrolled_2x2_parallel_i32_rv32im(matrix_a, matrix_b, matrix_c,
                                          matrix_M, matrix_N, matrix_P, core_id,
                                          num_cores);
  APP_KERNEL_BARRIER(core_id);
  mempool_stop_benchmark();
#endif
  time_end = mempool_get_timer();

  if (core_id == 0) {
    uint32_t clock_cycles = (time_end - time_init);
    printf("\nKernel execution takes %d clock cycles\n", clock_cycles);
  }

  // Verify results
  dma_collective_memcpy_blocking(l2_C_check, matrix_c,
                                 matrix_M * matrix_P * sizeof(int32_t));
  mempool_check_dpi_i32(l2_C_check, l2_C, matrix_M * matrix_P, 0, 0);
  APP_BARRIER(num_cores);
  return 0;
}
