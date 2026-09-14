// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Minimal repro for the VC DMA path.
//
// vc_smoke builds descriptors but never submits a VC DMA transfer, so the
// whole L2 -> virtual-cluster DMA is uncovered by the WP3 gate even though the
// gate is green. The qwen VC arm is the only thing that exercises it, and it
// wedges right after its banner -- 77 minutes with no output and no EOC. This
// app strips that path down to the DMA alone: no RedMulE, no GEMM, no tiling.
//
// Escalating sizes with a print on BOTH sides of every transfer, so a wedge
// names the exact stage instead of leaving a silent transcript.

#include <stdint.h>

#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"
#include "vc.h"

#define REPRO_LEVEL 1
// One whole rectangle row at REPRO_LEVEL, in bytes.
#define REPRO_ROW (VC_ROW_BYTES * (1u << REPRO_LEVEL) * (1u << REPRO_LEVEL))
#define REPRO_ROWS_MAX 160
#define REPRO_BYTES (REPRO_ROW * REPRO_ROWS_MAX)

static uint16_t l2_src[REPRO_BYTES / 2]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
// One slot per group: all rectangle leaders issue concurrently.
static uint32_t cyc[NUM_GROUPS][3] __attribute__((section(".l1")));
static uint32_t total_errors __attribute__((section(".l1")));

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();

  mempool_barrier_init(core_id);
  vc_init();
  mempool_barrier(num_cores);

  if (core_id == 0)
    printf("[vc_dma_repro] row=%u rows_max=%u bytes=%u\n", (unsigned)REPRO_ROW,
           (unsigned)REPRO_ROWS_MAX, (unsigned)REPRO_BYTES);

  if (vc_reserve(REPRO_LEVEL, REPRO_BYTES) < 0) {
    if (core_id == 0)
      printf("[vc_dma_repro] reserve of %u B failed\n", (unsigned)REPRO_BYTES);
    return 1;
  }
  uint16_t *const dst = (uint16_t *)vc_malloc(REPRO_LEVEL, REPRO_BYTES);
  const uint32_t off = (uint32_t)dst - vc_window_base(REPRO_LEVEL);
  const vc_rect_t rect = vc_rect(REPRO_LEVEL);
  const uint32_t group_id = mempool_get_group_id();
  const uint32_t tile = group_id % NUM_TILES_PER_GROUP;
  const uint32_t leader =
      group_id * NUM_CORES_PER_GROUP + tile * NUM_CORES_PER_TILE;

  // Seed one word per row so a landed transfer is distinguishable from the
  // zero-filled .l2_bss it came from.
  if (core_id == 0) {
    total_errors = 0;
    for (uint32_t g = 0; g < NUM_GROUPS; ++g)
      for (uint32_t s = 0; s < 3; ++s)
        cyc[g][s] = 0;
    for (uint32_t r = 0; r < REPRO_ROWS_MAX; r++)
      l2_src[r * (REPRO_ROW / 2)] = (uint16_t)(0xA000u + r);
  }
  if (core_id == leader && group_id == rect.base_gid)
    for (uint32_t r = 0; r < REPRO_ROWS_MAX; ++r)
      dst[r * (REPRO_ROW / 2)] = (uint16_t)(0x5A00u + r);
  mempool_barrier(num_cores);

  const uint32_t stages[3] = {1, 4, REPRO_ROWS_MAX};
  for (uint32_t s = 0; s < 3; s++) {
    const uint32_t rows = stages[s];
    const uint32_t bytes = rows * REPRO_ROW;
    if (core_id == 0)
      printf("[vc_dma_repro] stage %u: %u rows / %u B -- issuing\n",
             (unsigned)s, (unsigned)rows, (unsigned)bytes);
    mempool_barrier(num_cores);

    const vc_desc_t d = vc_descriptor(&rect, off, bytes);
    uint32_t transfer_error = 0;
    if (core_id == leader) {
      // Report the shape BEFORE issuing: if the transfer wedges, the last line
      // in the transcript still says which descriptor did it.
      if (core_id == 0)
        printf("[vc_dma_repro] stage %u: dst=0x%x len=%u n=%ux%u str=%u,%u\n",
               (unsigned)s, (unsigned)d.base, (unsigned)d.len,
               (unsigned)d.count[0], (unsigned)d.count[1],
               (unsigned)d.stride[0], (unsigned)d.stride[1]);
    }
    const mempool_timer_t t0 = mempool_get_timer();
#ifndef VC_DMA_REPRO_NO_LAUNCH
    const vc_desc_t sd = {.base = (uint32_t)l2_src,
                          .len = bytes,
                          .count = {1, 1},
                          .stride = {0, 0}};
    transfer_error = vc_dma_memcpy_blocking(&d, &sd);
#endif
    const mempool_timer_t t1 = mempool_get_timer();
    if (core_id == leader) {
      cyc[group_id][s] = (uint32_t)(t1 - t0);
      if (group_id == rect.base_gid && transfer_error != 0)
        __atomic_fetch_add(&total_errors, 1, __ATOMIC_RELAXED);
    }
    mempool_barrier(num_cores);

    if (core_id == leader && group_id == rect.base_gid) {
      uint32_t bad = 0;
      for (uint32_t r = 0; r < rows; r++)
        if (*(volatile uint16_t *)(dst + r * (REPRO_ROW / 2)) !=
            (uint16_t)(0xA000u + r))
          bad++;
      if (bad != 0)
        __atomic_fetch_add(&total_errors, bad, __ATOMIC_RELAXED);
      if (core_id == 0)
        printf("[vc_dma_repro] stage %u: landed, %u/%u rows wrong, %u cyc "
               "(%u cyc/run)\n",
               (unsigned)s, (unsigned)bad, (unsigned)rows, (unsigned)cyc[0][s],
               (unsigned)(cyc[0][s] / rows));
    }
    mempool_barrier(num_cores);
  }

  if (core_id == 0)
    printf("[vc_dma_repro] done, errors=%u\n", (unsigned)total_errors);
  mempool_barrier(num_cores);
  return (int)total_errors;
}
