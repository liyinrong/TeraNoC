// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yichao Zhang, ETH Zurich

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "data_axpy_i32.h"

#include "baremetal/mempool_axpy_i32.h"
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

int32_t l1_X[array_N]
    __attribute__((aligned(NUM_BANKS * 4), section(".l1_prio")));
int32_t l1_Y[array_N]
    __attribute__((aligned(NUM_BANKS * 4), section(".l1_prio")));
int volatile error __attribute__((section(".l1_prio")));

int32_t l2_Z_check[array_N]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2")));

int main() {

  uint32_t const core_id = mempool_get_core_id();
  uint32_t const num_cores = mempool_get_core_count();
  uint32_t time_init, time_end;
  APP_BARRIER_INIT(core_id);

  // Initialize data
  time_init = 0;
  time_end = 0;
  dma_collective_memcpy_nonblocking(l1_X, l2_X, array_N * sizeof(int32_t));
  dma_collective_memcpy_blocking(l1_Y, l2_Y, array_N * sizeof(int32_t));
  if (mempool_is_group_coordinator())
    error = 0;
  register volatile int32_t a = l2_A;
  APP_BARRIER(num_cores);

  // Benchmark
  time_init = mempool_get_timer();
  mempool_start_benchmark();
  calc_axpy_unloop_x4_localbank(l1_X, l1_Y, a, array_N, core_id, num_cores);
  APP_KERNEL_BARRIER(core_id);
  mempool_stop_benchmark();
  time_end = mempool_get_timer();

  if (core_id == 0) {
    uint32_t clock_cycles = (time_end - time_init);
    printf("\nKernel execution takes %d clock cycles\n", clock_cycles);
  }

  // Verify results
  dma_collective_memcpy_blocking(l2_Z_check, l1_Y, array_N * sizeof(int32_t));
  mempool_check_dpi_i32(l2_Z_check, l2_Z, array_N, 0, 0);
  APP_BARRIER(num_cores);

  return 0;
}
