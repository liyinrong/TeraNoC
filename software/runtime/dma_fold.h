// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// The domain fold of iDMA addresses (DOMAINS > 1), split out of dma.h so a
// host harness can compile it: it needs NUM_BANKS, L1_BANK_SIZE, DOMAINS,
// DOMAIN_L1_DATA_BASE, L2_BASE, L2_SIZE and mempool_get_domain_id().

#ifndef _DMA_FOLD_H_
#define _DMA_FOLD_H_

// A domain build (addrmap.h) hands the VC-blind iDMA domain addresses: its
// global L1 [0, L1) and the same map seen through the domain's window at
// DOMAIN_L1_DATA_BASE. Both have rows of NUM_GROUPS stripes; the chip's rows
// are DOMAINS times wider and domain d owns stripes d*NUM_GROUPS.. of each, so
// row r of the domain is row r of the chip at offset d * row. Stacks (the
// physical sequential region) are never DMA'd and are not handled.
#define DMA_DOMAIN_L1_BYTES ((uint32_t)NUM_BANKS * L1_BANK_SIZE)
#define DMA_DOMAIN_ROW_BYTES ((uint32_t)NUM_BANKS * 4u)
extern uint32_t volatile dma_domain_errors;

static inline uint32_t dma_domain_l1(uint32_t a) {
  return a < DMA_DOMAIN_L1_BYTES ||
         (a >= DOMAIN_L1_DATA_BASE &&
          a - DOMAIN_L1_DATA_BASE < DMA_DOMAIN_L1_BYTES);
}
// Timing-only home channels (DOMAIN_L2_HOME): L2 past the program image is
// folded like L1, so domain d only ever reaches the banks of its own quarter
// of each chip row -- what a per-domain (or per-layer) weight placement on its
// home channels gives. Domains alias each other's data: values are garbage.
#ifndef DOMAIN_L2_HOME
#define DOMAIN_L2_HOME 0
#endif
#if DOMAIN_L2_HOME
// Folded L2 starts DMA_DOMAIN_HOME_LO bytes into L2. The default skips only
// the first 64 KiB, which today's images outgrow, so wrapped addresses can land
// on .text/.data. Raise it above the image's L2 end (arch.ld.c then asserts the
// image fits below it) to keep wrapped accesses on data. A plain integer: the
// linker script uses it too.
#ifndef DMA_DOMAIN_HOME_LO
#define DMA_DOMAIN_HOME_LO 0x10000
#endif
_Static_assert((DMA_DOMAIN_HOME_LO) % 0x10000 == 0,
               "DMA_DOMAIN_HOME_LO must be 64 KiB aligned");
#define DMA_DOMAIN_HOME_BASE ((uint32_t)L2_BASE + (uint32_t)DMA_DOMAIN_HOME_LO)
#define DMA_DOMAIN_HOME_WRAP                                                   \
  (((uint32_t)L2_SIZE - (uint32_t)DMA_DOMAIN_HOME_LO - 0x4000000u) /           \
   (DMA_DOMAIN_ROW_BYTES * DOMAINS) * (DMA_DOMAIN_ROW_BYTES * DOMAINS))
#endif
static inline uint32_t dma_domain_home(uint32_t a) {
#if DOMAIN_L2_HOME
  return a >= DMA_DOMAIN_HOME_BASE && a - (uint32_t)L2_BASE < (uint32_t)L2_SIZE;
#else
  (void)a;
  return 0;
#endif
}
// Offset of a translated address inside its domain map.
static inline uint32_t dma_domain_off(uint32_t a) {
#if DOMAIN_L2_HOME
  if (dma_domain_home(a))
    return a - DMA_DOMAIN_HOME_BASE;
#endif
  return a >= DMA_DOMAIN_L1_BYTES ? a - DOMAIN_L1_DATA_BASE : a;
}
static inline uint32_t dma_domain_addr(uint32_t a) {
  const uint32_t home = dma_domain_home(a);
  if (!home && !dma_domain_l1(a))
    return a;
  const uint32_t o = dma_domain_off(a);
  const uint32_t c =
      o / DMA_DOMAIN_ROW_BYTES * (DMA_DOMAIN_ROW_BYTES * DOMAINS) +
      mempool_get_domain_id() * DMA_DOMAIN_ROW_BYTES + o % DMA_DOMAIN_ROW_BYTES;
#if DOMAIN_L2_HOME
  if (home)
    return DMA_DOMAIN_HOME_BASE + c % DMA_DOMAIN_HOME_WRAP;
#endif
  return c;
}
// Rows crossed by a step of `stride` from in-row offset `at` (floor).
static inline int32_t dma_domain_carry(uint32_t at, uint32_t stride) {
  const int32_t to = (int32_t)(at % DMA_DOMAIN_ROW_BYTES) + (int32_t)stride;
  return (to >= 0 ? to : to - (int32_t)DMA_DOMAIN_ROW_BYTES + 1) /
         (int32_t)DMA_DOMAIN_ROW_BYTES;
}
// The chip stride of a domain stride first taken at domain offset `at` and
// then every `period` bytes: each row it crosses is DOMAINS rows wide. The
// carry must match at the last step, or no constant stride expresses it.
static inline uint32_t dma_domain_stride(uint32_t a, uint32_t at,
                                         uint32_t stride, uint32_t period,
                                         uint32_t reps) {
  if ((!dma_domain_l1(a) && !dma_domain_home(a)) || reps <= 1u)
    return stride;
  at = dma_domain_off(a) + (at - a);
  const int32_t rows = dma_domain_carry(at, stride);
  if (dma_domain_carry(at + (reps - 2u) * period, stride) != rows)
    ++dma_domain_errors;
  return stride +
         (uint32_t)(rows * (int32_t)DMA_DOMAIN_ROW_BYTES * (DOMAINS - 1));
}

// Whether a chip extent -- bursts of len bytes at c, c + a2, c + a3 and
// c + a2 + a3 (strides modulo 2^32) -- reaches [L2_BASE, end): the group
// RO-cache window for reads, the image below the home base for writes
// (DMA_DOMAIN_CHECK, dma.h).
static inline uint32_t dma_l2_low_overlap(uint32_t c, uint32_t a2, uint32_t a3,
                                          uint32_t len, uint32_t end) {
  uint32_t lo = c, hi = c;
  const uint32_t x[3] = {c + a2, c + a3, c + a2 + a3};
  for (uint32_t i = 0u; i < 3u; ++i) {
    lo = x[i] < lo ? x[i] : lo;
    hi = x[i] > hi ? x[i] : hi;
  }
  return lo < end && hi + len > (uint32_t)L2_BASE;
}

#endif // _DMA_FOLD_H_
