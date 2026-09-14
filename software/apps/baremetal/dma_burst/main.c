// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Burst DMA: the shape a staged kernel actually issues, rather than one
// transfer measured in isolation.
//
//   in0: 512 KiB L2 -> L1      two operands staged in,
//   in1: 512 KiB L2 -> L1      one result written back,
//   out: 512 KiB L1 -> L2      and ONE synchronisation at the very end.
//
// That single fence is the point. A per-transfer wait serialises the engine
// behind the core three times; issuing all three and fencing once amortises
// the synchronisation over 1.5 MiB instead of 512 KiB.
//
// One source, built in both worktrees:
//   -DDMA_BURST_CENTRAL  monster:   the central mempool_dma
//   (default)            wfi-sync:  per-group iDMA, dma_collective_wait
//   -DDMA_BURST_READY    wfi-sync:  ditto, completed by wait_ready
//   -DDMA_BURST_SERIAL   wfi-sync:  a wait after every submit
//
// `out` reads back the buffer `in0` filled, so errors=0 is also a check that
// the three queued requests really did retire in issue order.

#include <stdint.h>

#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#ifdef DMA_BURST_CENTRAL
#ifdef DMA_BURST_CENTRAL_POLLDATA
#define DMA_BURST_NAME "centralized-polldata"
#else
#define DMA_BURST_NAME "centralized"
#endif
#elif defined(DMA_BURST_READY)
#define DMA_BURST_NAME "distributed-ready"
#elif defined(DMA_BURST_SERIAL)
#define DMA_BURST_NAME "distributed-serial"
#else
#define DMA_BURST_NAME "distributed"
#endif

#ifndef DMA_BURST_WORDS
#define DMA_BURST_WORDS 131072u // 512 KiB per transfer
#endif
#define BURST_BYTES (DMA_BURST_WORDS * 4u)

// L2 and L1 both need the interleave granularity, not just the bank stride:
// the central engine hangs outright on an under-aligned buffer.
#define BURST_ALIGN ((4 * NUM_BANKS) > 8192 ? (4 * NUM_BANKS) : 8192)

static uint32_t l2_a[DMA_BURST_WORDS]
    __attribute__((aligned(BURST_ALIGN), section(".l2")));
static uint32_t l2_b[DMA_BURST_WORDS]
    __attribute__((aligned(BURST_ALIGN), section(".l2")));
static uint32_t l2_z[DMA_BURST_WORDS]
    __attribute__((aligned(BURST_ALIGN), section(".l2")));
static uint32_t l1_a[DMA_BURST_WORDS]
    __attribute__((aligned(BURST_ALIGN), section(".l1")));
static uint32_t l1_b[DMA_BURST_WORDS]
    __attribute__((aligned(BURST_ALIGN), section(".l1")));
static volatile uint32_t burst_cyc __attribute__((section(".l1")));
static volatile uint32_t burst_errs __attribute__((section(".l1")));

// Stage timestamps, so the total can be attributed. `.l1` is NOBITS and powers
// up X, so these are written before they are read.
#define BURST_STAGES 8u
static volatile uint32_t burst_t[BURST_STAGES] __attribute__((section(".l1")));
#define BURST_T(i) (burst_t[i] = (uint32_t)mempool_get_timer())

#ifdef DMA_BURST_CENTRAL
// The midend fans one transfer out into per-group chunks of
// (4*NUM_BANKS)/NUM_GROUPS bytes, and they land OUT OF ORDER: the engine's own
// log shows 0x80183800, 0x80183c00, 0x80182000, 0x80183000 for consecutive
// chunks. So the last word of the buffer is NOT a completion signal -- it can
// match while other chunks are still in flight. Check the last word of EVERY
// chunk instead. (This is also why the single-transfer centralized figures,
// which test only the final word, are a slight under-report.)
#define BURST_CHUNK_WORDS (((4u * NUM_BANKS) / NUM_GROUPS) / 4u)

// Poll against the ANALYTIC expected value, not against the source buffer.
// Comparing dst[k] != src[k] costs two loads per chunk and one of them goes to
// L2, which dominated the measurement outright: the first 512 KiB stage timed
// 13133 cycles against an engine-reported transfer of 2104. One load per chunk
// is the cheapest sound test available, since the design offers no completion
// register to read instead.
#define BURST_EXP_A(k) ((k) + 1u)
#define BURST_EXP_B(k) (0xB0000000u ^ (k))

static void burst_central_settle(volatile const uint32_t *dst, uint32_t kind) {
  // k lands on the last word of every chunk, DMA_BURST_WORDS-1 included.
  for (uint32_t k = BURST_CHUNK_WORDS - 1u; k < DMA_BURST_WORDS;
       k += BURST_CHUNK_WORDS) {
    const uint32_t want = (kind == 1u) ? BURST_EXP_B(k) : BURST_EXP_A(k);
    while (dst[k] != want)
      ;
  }
  __sync_synchronize();
}
#endif

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t ncores = mempool_get_core_count();

  mempool_barrier_init(core);
  if (core == 0u) {
    burst_cyc = 0u;
    burst_errs = 0u;
    for (uint32_t i = 0u; i < BURST_STAGES; ++i)
      burst_t[i] = 0u;
  }
  for (uint32_t i = core; i < DMA_BURST_WORDS; i += ncores) {
    l2_a[i] = i + 1u;
    l2_b[i] = 0xB0000000u ^ i;
    l2_z[i] = 0u;
    l1_a[i] = 0u;
    l1_b[i] = 0u;
  }
  mempool_barrier(ncores);

  const mempool_timer_t t0 = mempool_get_timer();
#ifdef DMA_BURST_CENTRAL
  // Serialised: one transfer in flight at a time.
  //
  // Completion comes from the frontend's DONE register, which is a STICKY BIT
  // -- cleared when a launch is accepted (valid_o) and set by trans_complete_i
  // (mempool_dma.sv:61-78). The runtime's dma_wait() polls it correctly with
  // `while (!dma_done())`.
  //
  // Do NOT wait for DONE to *change*. mempool_dma_frontend.hjson:94-105
  // documents the register as "Get ID of finished transactions", i.e. a
  // counter, but the RTL drives a single bit. Sampling it before the launch and
  // waiting for a change exits on the launch's 1->0 CLEAR rather than on the
  // 0->1 completion, which times 512 KiB at 138 cycles (3799 B/cyc) and makes
  // every transfer after the first look instantaneous.
#ifdef DMA_BURST_CENTRAL_POLLDATA
  // Reference path: establish completion from the data instead, by checking the
  // last word of every midend chunk. Kept to show the cost of not having a
  // completion register, not because one is missing.
  if (core == 0u) {
    BURST_T(0);
    dma_memcpy_nonblocking(l1_a, l2_a, BURST_BYTES);
    burst_central_settle(l1_a, 0u);
    BURST_T(1);
    mempool_wait(512u);
    BURST_T(2);
    dma_memcpy_nonblocking(l1_b, l2_b, BURST_BYTES);
    burst_central_settle(l1_b, 1u);
    BURST_T(3);
    mempool_wait(512u);
    BURST_T(4);
    dma_memcpy_nonblocking(l2_z, l1_a, BURST_BYTES);
    burst_central_settle(l2_z, 0u);
    BURST_T(5);
  }
#else
  if (core == 0u) {
    BURST_T(0);
    dma_memcpy_blocking(l1_a, l2_a, BURST_BYTES);
    BURST_T(1);
    burst_t[2] = burst_t[1]; // no re-arm guard needed
    dma_memcpy_blocking(l1_b, l2_b, BURST_BYTES);
    BURST_T(3);
    burst_t[4] = burst_t[3];
    // Reads back what the first transfer wrote, so a premature DONE on
    // transfer 1 would corrupt l2_z and the final verify would catch it.
    dma_memcpy_blocking(l2_z, l1_a, BURST_BYTES);
    BURST_T(5);
  }
#endif
  mempool_barrier(ncores);
  if (core == 0u)
    BURST_T(6);
#else
  // Every core submits all three, then waits once on the last. Only the group
  // coordinators issue; waiting a group's last id retires the two before it.
  if (core == 0u)
    BURST_T(0);
  dma_collective_op_t op =
      dma_collective_memcpy_nonblocking(l1_a, l2_a, BURST_BYTES);
  uint32_t status = op.status;
#ifdef DMA_BURST_SERIAL
  status |= dma_collective_wait(&op);
#endif
  if (core == 0u)
    BURST_T(1);
  op = dma_collective_memcpy_nonblocking(l1_b, l2_b, BURST_BYTES);
  status |= op.status;
#ifdef DMA_BURST_SERIAL
  status |= dma_collective_wait(&op);
#endif
  if (core == 0u)
    BURST_T(2);
  op = dma_collective_memcpy_nonblocking(l2_z, l1_a, BURST_BYTES);
  status |= op.status;
  if (core == 0u)
    BURST_T(3);
#ifdef DMA_BURST_READY
  status |= dma_collective_wait_ready(&op);
#else
  status |= dma_collective_wait(&op);
#endif
  if (core == 0u)
    BURST_T(4);
  if (status != 0u)
    __atomic_fetch_add(&burst_errs, 1u, __ATOMIC_RELAXED);
#endif
  const mempool_timer_t t1 = mempool_get_timer();
  if (core == 0u)
    burst_cyc = (uint32_t)(t1 - t0);
  mempool_barrier(ncores);

  // Verify all three landed; a dropped transfer would otherwise time as fast.
  for (uint32_t i = core; i < DMA_BURST_WORDS; i += ncores) {
    if (l1_a[i] != i + 1u)
      __atomic_fetch_add(&burst_errs, 1u, __ATOMIC_RELAXED);
    if (l1_b[i] != (0xB0000000u ^ i))
      __atomic_fetch_add(&burst_errs, 1u, __ATOMIC_RELAXED);
    if (l2_z[i] != i + 1u)
      __atomic_fetch_add(&burst_errs, 1u, __ATOMIC_RELAXED);
  }
  mempool_barrier(ncores);

  if (core == 0u) {
    printf(
        "\n[dma_burst] design=%s per_xfer=%uB total=%uB cycles=%u errors=%u\n",
        DMA_BURST_NAME, (unsigned)BURST_BYTES, (unsigned)(3u * BURST_BYTES),
        (unsigned)burst_cyc, (unsigned)burst_errs);
#ifdef DMA_BURST_CENTRAL
    printf("[dma_burst] brk xfer1=%u guard1=%u xfer2=%u guard2=%u xfer3=%u "
           "endbar=%u\n",
           (unsigned)(burst_t[1] - burst_t[0]),
           (unsigned)(burst_t[2] - burst_t[1]),
           (unsigned)(burst_t[3] - burst_t[2]),
           (unsigned)(burst_t[4] - burst_t[3]),
           (unsigned)(burst_t[5] - burst_t[4]),
           (unsigned)(burst_t[6] - burst_t[5]));
#else
    printf("[dma_burst] brk issue1=%u issue2=%u issue3=%u wait=%u\n",
           (unsigned)(burst_t[1] - burst_t[0]),
           (unsigned)(burst_t[2] - burst_t[1]),
           (unsigned)(burst_t[3] - burst_t[2]),
           (unsigned)(burst_t[4] - burst_t[3]));
#endif
  }
  mempool_barrier(ncores);
  return (int)burst_errs;
}
