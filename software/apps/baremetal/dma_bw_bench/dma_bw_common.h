// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Shared body of the L1<->L2 DMA bandwidth comparison. The only thing that
// differs between the centralized (single mempool_dma frontend + split and
// distributed midends) and the distributed (one iDMA frontend per group)
// designs is DMA_XFER_IN/DMA_XFER_OUT, supplied by the including main.c.
//
// Both sides:
//   * move the same number of bytes,
//   * between the ordinary globally interleaved .l1 region and .l2,
//   * time with mcycle on core 0, which observes the WHOLE cluster transfer in
//     both designs (the centralized call polls the frontend's done flag; the
//     collective's wait fences every participating group's ready epoch),
//   * verify the data actually landed, so a no-op cannot look fast.

#ifndef DMA_BW_WORDS
#define DMA_BW_WORDS 16384 // 64 KiB
#endif

#define DMA_BW_BYTES ((size_t)DMA_BW_WORDS * sizeof(int32_t))

// The centralized engine hangs on an L1 side aligned below 8192. The default
// covers every flavour whose NUM_BANKS * 4 reaches it (dse_s1_winner: 16384);
// override only upward, and for both arms, never for one.
#ifndef DMA_BW_L1_ALIGN
#define DMA_BW_L1_ALIGN (NUM_BANKS * 4)
#endif
int32_t l1_buf[DMA_BW_WORDS]
    __attribute__((section(".l1"), aligned(DMA_BW_L1_ALIGN)));
// The L2 side must be aligned to the interleave granularity, not merely to
// int32_t: the centralized engine hangs outright on a 16-byte-aligned L2
// destination (L2->L1 completes, L1->L2 never retires). Match the stock
// memcpy test's 8192 so both arms see the same, supported case.
// 8192 is half the interleave period once NUM_BANKS * 4 = 16384
// (dse_s1_winner): pass DMA_BW_L2_ALIGN=16384 there for an aligned transfer.
#ifndef DMA_BW_L2_ALIGN
#define DMA_BW_L2_ALIGN 8192
#endif
int32_t l2_buf[DMA_BW_WORDS]
    __attribute__((section(".l2"), aligned(DMA_BW_L2_ALIGN)));

static uint32_t volatile cyc_in __attribute__((section(".l1")));
static uint32_t volatile cyc_out __attribute__((section(".l1")));
static uint32_t volatile errs __attribute__((section(".l1")));

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t ncores = mempool_get_core_count();

  mempool_barrier_init(core);
  if (core == 0u) {
    cyc_in = 0u;
    cyc_out = 0u;
    errs = 0u;
  }
  for (uint32_t i = core; i < DMA_BW_WORDS; i += ncores) {
    l2_buf[i] = (int32_t)(i + 1);
    l1_buf[i] = 0;
  }
  // Print BEFORE the barrier: a print between the barrier and t0 lets the
  // other coordinators start their transfer while core 0 is still printing.
  if (core == 0u)
    printf("[dma_bw] stage init done\n");
#ifdef DMA_BW_TRACE
  write_csr(trace, 1); // simulation trace of the L2->L1 leg (no cycle cost)
#endif
  mempool_barrier(ncores);

  // ---------------- L2 -> L1 ----------------
  mempool_timer_t t0 = mempool_get_timer();
  DMA_XFER_IN(l1_buf, l2_buf, DMA_BW_BYTES);
  mempool_timer_t t1 = mempool_get_timer();
#ifdef DMA_BW_TRACE
  write_csr(trace, 0);
#endif
  if (core == 0u) {
    cyc_in = (uint32_t)(t1 - t0);
    printf("[dma_bw] stage in done %u\n", (unsigned)cyc_in);
  }
  mempool_barrier(ncores);
#ifdef DMA_BW_BREAKDOWN
  DMA_BW_BREAKDOWN();
  mempool_barrier(ncores);
#endif

  // Prove the transfer happened; a silent no-op would otherwise time as free.
  for (uint32_t i = core; i < DMA_BW_WORDS; i += ncores)
    if (l1_buf[i] != (int32_t)(i + 1))
      __atomic_fetch_add(&errs, 1u, __ATOMIC_RELAXED);
  for (uint32_t i = core; i < DMA_BW_WORDS; i += ncores)
    l2_buf[i] = 0;
  mempool_barrier(ncores);

  // ---------------- L1 -> L2 ----------------
  t0 = mempool_get_timer();
  DMA_XFER_OUT(l2_buf, l1_buf, DMA_BW_BYTES);
  t1 = mempool_get_timer();
  if (core == 0u) {
    cyc_out = (uint32_t)(t1 - t0);
    printf("[dma_bw] stage out done %u\n", (unsigned)cyc_out);
  }
  mempool_barrier(ncores);

  for (uint32_t i = core; i < DMA_BW_WORDS; i += ncores)
    if (l2_buf[i] != (int32_t)(i + 1))
      __atomic_fetch_add(&errs, 1u, __ATOMIC_RELAXED);
  mempool_barrier(ncores);

  if (core == 0u)
    printf("[dma_bw] design=%s bytes=%u in=%u out=%u errors=%u\n", DMA_BW_NAME,
           (unsigned)DMA_BW_BYTES, (unsigned)cyc_in, (unsigned)cyc_out,
           (unsigned)errs);
  mempool_barrier(ncores);
  return (int)errs;
}
