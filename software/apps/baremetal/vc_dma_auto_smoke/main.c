// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Semantic gate for the address-derived VC DMA entry point.
//
// vc_dma_memcpy_auto_blocking takes only (dest, src, len) and reconstructs the
// level, the rectangle, the group scope and BOTH strides from the addresses.
// The property that has to hold is the one the window exists to provide:
//
//     VC-local byte o holds flat byte o
//
// so the check reads back through the window, not through a global alias. A
// wrong stride still lands the right number of bytes -- it just puts them at
// the wrong rows -- which is exactly what a byte-count check would miss.
//
// Negative control: the same transfer submitted with a deliberately shifted
// source must NOT reproduce the pattern, so a vacuous pass is detectable.

#include <stdint.h>

#include "printf.h"
#include "runtime.h"
#include "synchronization.h"
#include "vc.h"

#define AUTO_LEVEL 1
// One whole rectangle row at AUTO_LEVEL, in bytes.
#define AUTO_ROW (VC_ROW_BYTES * (1u << AUTO_LEVEL) * (1u << AUTO_LEVEL))
#define AUTO_ROWS 8u
#define AUTO_BYTES (AUTO_ROW * AUTO_ROWS)
#define AUTO_WORDS (AUTO_BYTES / 4u)

static uint32_t l2_src[AUTO_WORDS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
static uint32_t total_errors __attribute__((section(".l1")));
static uint32_t control_errors __attribute__((section(".l1")));

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();

  mempool_barrier_init(core_id);
  vc_init();
  mempool_barrier(num_cores);

  if (vc_reserve(AUTO_LEVEL, AUTO_BYTES) < 0) {
    if (core_id == 0)
      printf("[vc_dma_auto] reserve of %u B failed\n", (unsigned)AUTO_BYTES);
    return 1;
  }
  uint32_t *const dst = (uint32_t *)vc_malloc(AUTO_LEVEL, AUTO_BYTES);

  if (core_id == 0) {
    total_errors = 0;
    control_errors = 0;
  }
  // Distinct per word, so a row landing at the wrong stride is visible.
  for (uint32_t i = core_id; i < AUTO_WORDS; i += num_cores)
    l2_src[i] = i + 1u;
  for (uint32_t i = core_id; i < AUTO_WORDS; i += num_cores)
    dst[i] = 0u;
  mempool_barrier(num_cores);

  if (core_id == 0)
    printf("[vc_dma_auto] level=%u row=%u rows=%u bytes=%u dst=0x%x\n",
           (unsigned)AUTO_LEVEL, (unsigned)AUTO_ROW, (unsigned)AUTO_ROWS,
           (unsigned)AUTO_BYTES, (unsigned)(uint32_t)dst);

  // Every core of the rectangle calls; each group's coordinator issues.
  const uint32_t error = vc_dma_memcpy_auto_blocking(dst, l2_src, AUTO_BYTES);
  mempool_barrier(num_cores);

  {
    uint32_t bad = 0u;
    for (uint32_t i = core_id; i < AUTO_WORDS; i += num_cores)
      if (dst[i] != i + 1u)
        ++bad;
    if (bad)
      __atomic_fetch_add(&total_errors, bad, __ATOMIC_RELAXED);
  }
  mempool_barrier(num_cores);

  // Negative control: one rectangle row of source shift. If the readback above
  // were insensitive to placement this would pass too.
  for (uint32_t i = core_id; i < AUTO_WORDS; i += num_cores)
    dst[i] = 0u;
  mempool_barrier(num_cores);
  const uint32_t control_error = vc_dma_memcpy_auto_blocking(
      dst, l2_src + (AUTO_ROW / 4u), AUTO_BYTES - AUTO_ROW);
  mempool_barrier(num_cores);
  {
    uint32_t same = 0u;
    for (uint32_t i = core_id; i < AUTO_WORDS - (AUTO_ROW / 4u); i += num_cores)
      if (dst[i] == i + 1u)
        ++same;
    if (same)
      __atomic_fetch_add(&control_errors, same, __ATOMIC_RELAXED);
  }
  mempool_barrier(num_cores);

  if (core_id == 0)
    printf("[vc_dma_auto] status=%u,%u errors=%u control_matches=%u\n",
           (unsigned)error, (unsigned)control_error, (unsigned)total_errors,
           (unsigned)control_errors);
  mempool_barrier(num_cores);
  return (int)total_errors;
}
