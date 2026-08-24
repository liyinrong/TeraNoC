// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// L2 <-> L1 DMA bandwidth, the one thing the HBM channel placement governs.
//
// The Qwen RedMulE kernels stage their operands with a DMA *before* the timed
// region, so their cycle counts say nothing about the L2 attach. This does the
// opposite: it times nothing but the transfer.
//
// The distributed DMA forks a transfer so group g owns L1 stripe g, and the L2
// fold puts that stripe in bank g/(num_groups/l2_banks). Every group therefore
// moves DMA_BW_BYTES/num_groups bytes over its own path to its own channel, and
// the aggregate bytes/cycle is what the placement is trying to maximise.

#include <stdint.h>

#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

// 4 MiB: a whole number of L1 periods at every mesh size (the period is
// num_groups x per-group L1 width, at most 64 KiB here) and comfortably inside
// the smallest L1 in the comparison (16 MiB at S1 4x4).
#define DMA_BW_BYTES (4 * 1024 * 1024)
#define DMA_BW_WORDS (DMA_BW_BYTES / 4)
// One period, to pay the descriptor and first-touch cost outside the timer.
#define DMA_WARM_BYTES 65536

uint32_t dma_l1_buf[DMA_BW_WORDS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
uint32_t dma_l2_buf[DMA_BW_WORDS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  mempool_barrier_init(core_id);

  if (core_id == 0) {
    printf("\n[dma_l2_l1_bw] groups=%d cores=%d banks=%d bytes=%d\n", NUM_GROUPS,
           NUM_CORES, NUM_BANKS, DMA_BW_BYTES);

    dma_memcpy_blocking(dma_l1_buf, dma_l2_buf, DMA_WARM_BYTES);

    mempool_start_benchmark();
    const uint32_t t0 = mempool_get_timer();
    dma_memcpy_blocking(dma_l1_buf, dma_l2_buf, DMA_BW_BYTES);
    const uint32_t t1 = mempool_get_timer();
    mempool_stop_benchmark();

    mempool_start_benchmark();
    const uint32_t t2 = mempool_get_timer();
    dma_memcpy_blocking(dma_l2_buf, dma_l1_buf, DMA_BW_BYTES);
    const uint32_t t3 = mempool_get_timer();
    mempool_stop_benchmark();

    printf("[dma_l2_l1_bw] stage=read cycles=%u\n", t1 - t0);
    printf("[dma_l2_l1_bw] stage=write cycles=%u\n", t3 - t2);
    printf("[dma_l2_l1_bw] kernel takes %u clock cycles\n",
           (t1 - t0) + (t3 - t2));
  }
  mempool_barrier(num_cores);
  return 0;
}
