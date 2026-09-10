// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Marco Bertuletti, ETH Zurich

#include <stdint.h>
#include <string.h>

#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "data_dotp_f16.h"
uint32_t red_barrier[NUM_BANKS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));

#include "baremetal/mempool_dotp_f16.h"

// Barrier selection. Build with -DUSE_FSYNC_BARRIER to run every barrier on
// the fractal_sync H-tree instead of the software barrier; the default is
// unchanged, so previously recorded numbers stay comparable. The two must not
// be mixed inside one phase -- mempool_barrier's sleeping path treats any wake
// as its own -- so the switch covers ALL of this app's barriers at once.
#ifdef USE_FSYNC_BARRIER
#include "fsync_app.h"
#define APP_BARRIER_INIT(core) fsync_app_barrier_init(core)
#define APP_BARRIER(n) fsync_app_barrier()
#define APP_KERNEL_BARRIER(core) fsync_app_barrier()
#else
#define APP_BARRIER_INIT(core) mempool_barrier_init(core)
#define APP_BARRIER(n) mempool_barrier(n)
#define APP_KERNEL_BARRIER(core) mempool_log_barrier(2, core)
#endif

__fp16 l1_X[array_N]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
__fp16 l1_Y[array_N]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
__fp16 sum[2 * NUM_BANKS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));

int main() {

  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  uint32_t time_init, time_end;
  APP_BARRIER_INIT(core_id);

  time_init = 0;
  time_end = 0;
  if (mempool_is_group_coordinator()) {
    dma_collective_all_memcpy_blocking(l1_X, l2_X, array_N * sizeof(int16_t));
    dma_collective_all_memcpy_blocking(l1_Y, l2_Y, array_N * sizeof(int16_t));
  }
  for (uint32_t k = core_id; k < NUM_BANKS; k += num_cores) {
    sum[k] = 0;
    red_barrier[k] = 0;
  }
  APP_BARRIER(num_cores);

  // PARALLEL, LOCAL ACCESSES
  time_init = mempool_get_timer();
  mempool_start_benchmark();
  dotp_f16vecp_local_unrolled4(l1_X, l1_Y, sum, array_N);
  mempool_stop_benchmark();
  time_end = mempool_get_timer();

  // Check results
  if (core_id == 0) {
    uint32_t clock_cycles = (time_end - time_init);
    printf("\nKernel execution takes %d clock cycles\n", clock_cycles);
    printf("Result ==> %x\n", *(uint32_t *)&sum[0]);
    printf("Check  ==> %x\n\n", *(uint32_t *)&l2_Z);
  }
  APP_BARRIER(num_cores);

  return 0;
}
