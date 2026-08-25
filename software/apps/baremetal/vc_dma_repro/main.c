// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Minimal repro for the VC DMA path.
//
// vc_smoke builds descriptors but never calls vc_dma_memcpy_blocking, so the
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
// Shared so the leader's measurement is visible to the core that prints.
static uint32_t cyc[3] __attribute__((section(".l1")));

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
  // ONE leader for the whole chip, not one per rectangle. The frontend is a
  // single central engine whose descriptor lives in shared MMIO registers with
  // no lock (software/runtime/dma.h:48-77), so four rectangle leaders issuing
  // at once overwrite each other's SRC/DST/LEN and race the done poll. That is
  // a property of the PRE-WP4 central DMA, not of the VC map, so isolating the
  // map means one transfer at a time.
  const uint32_t leader = 0;

  // Seed one word per row so a landed transfer is distinguishable from the
  // zero-filled .l2_bss it came from.
  if (core_id == 0)
    for (uint32_t r = 0; r < REPRO_ROWS_MAX; r++)
      l2_src[r * (REPRO_ROW / 2)] = (uint16_t)(0xA000u + r);
  mempool_barrier(num_cores);

  const uint32_t stages[3] = {1, 4, REPRO_ROWS_MAX};
  for (uint32_t s = 0; s < 3; s++) {
    const uint32_t rows = stages[s];
    const uint32_t bytes = rows * REPRO_ROW;
    if (core_id == 0)
      printf("[vc_dma_repro] stage %u: %u rows / %u B -- issuing\n",
             (unsigned)s, (unsigned)rows, (unsigned)bytes);
    mempool_barrier(num_cores);

    if (core_id == leader * NUM_CORES_PER_TILE) {
      const vc_desc_t d = vc_descriptor(&rect, off, bytes);
      const vc_desc_t sd = {.base = (uint32_t)l2_src,
                            .len = bytes,
                            .count = {1, 1},
                            .stride = {0, 0}};
      // Report the shape BEFORE issuing: if the transfer wedges, the last line
      // in the transcript still says which descriptor did it.
      printf("[vc_dma_repro] stage %u: dst=0x%x len=%u n=%ux%u str=%u,%u\n",
             (unsigned)s, (unsigned)d.base, (unsigned)d.len,
             (unsigned)d.count[0], (unsigned)d.count[1], (unsigned)d.stride[0],
             (unsigned)d.stride[1]);
      const mempool_timer_t t0 = mempool_get_timer();
      vc_dma_memcpy_blocking(&d, &sd);
      const mempool_timer_t t1 = mempool_get_timer();
      cyc[s] = (uint32_t)(t1 - t0);
    }
    mempool_barrier(num_cores);

    if (core_id == 0) {
      uint32_t bad = 0;
      for (uint32_t r = 0; r < rows; r++)
        if (*(volatile uint16_t *)(dst + r * (REPRO_ROW / 2)) !=
            (uint16_t)(0xA000u + r))
          bad++;
      printf("[vc_dma_repro] stage %u: landed, %u/%u rows wrong, %u cyc "
             "(%u cyc/run)\n",
             (unsigned)s, (unsigned)bad, (unsigned)rows, (unsigned)cyc[s],
             (unsigned)(cyc[s] / rows));
    }
    mempool_barrier(num_cores);
  }

  if (core_id == 0)
    printf("[vc_dma_repro] done\n");
  mempool_barrier(num_cores);
  return 0;
}
