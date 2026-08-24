// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

enum { kLanes = 4 };

static uint32_t initial[kLanes]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static uint32_t poison[kLanes]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static uint32_t zero_result[kLanes]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static uint32_t nonzero_result[kLanes]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static volatile uint32_t test_result
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();

  mempool_barrier_init(core_id);

  if (core_id == 0) {
    const uint32_t scalar = 0xa5c39e71u;
    uint32_t errors = 0;

    initial[0] = 0x10203040u;
    initial[1] = 0x11223344u;
    initial[2] = 0x55667788u;
    initial[3] = 0x99aabbccu;
    poison[0] = 0xdeadbeefu;
    poison[1] = 0xcafef00du;
    poison[2] = 0x13579bdfu;
    poison[3] = 0x2468ace0u;

    asm volatile("vsetivli zero, 4, e32, m1, tu, mu");
    asm volatile("vle32.v v0, (%0)" : : "r"(poison) : "memory");
    asm volatile("vle32.v v8, (%0)" : : "r"(initial) : "memory");
    asm volatile("vmv.s.x v8, zero");
    asm volatile("vse32.v v8, (%0)" : : "r"(zero_result) : "memory");

    asm volatile("vle32.v v8, (%0)" : : "r"(initial) : "memory");
    asm volatile("vmv.s.x v8, %0" : : "r"(scalar));
    asm volatile("vse32.v v8, (%0)" : : "r"(nonzero_result) : "memory");

    if (zero_result[0] != 0u)
      ++errors;
    if (nonzero_result[0] != scalar)
      ++errors;
    for (uint32_t lane = 1; lane < kLanes; ++lane) {
      if (zero_result[lane] != initial[lane])
        ++errors;
      if (nonzero_result[lane] != initial[lane])
        ++errors;
    }

    test_result = errors;
    printf("[sp_vmv_s_x_i32] zero=0x%08x nonzero=0x%08x "
           "upper=%08x/%08x/%08x errors=%u\n",
           zero_result[0], nonzero_result[0], nonzero_result[1],
           nonzero_result[2], nonzero_result[3], errors);
    printf("[sp_vmv_s_x_i32] %s\n", errors == 0 ? "PASS" : "FAIL");
  }

  mempool_barrier(num_cores);
  return (int)test_result;
}
