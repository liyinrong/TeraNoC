// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Directed test for the ONE-CONTEXT-PER-GROUP iDMA register file.
//
// Every other DMA test in-tree has ONE core per group touching the context at
// all, so none of them ever puts two tiles on the group's control arbiter at
// once -- which is the only thing the shared context adds. Here the group
// coordinator runs a transfer (the documented one-issuer contract) while every
// other core of the group polls the SAME context's status registers as fast as
// it can, so all NUM_TILES_PER_GROUP tiles drive the arbiter together.
//
// Missing responses time out; reserved status bits and bad payloads fail the
// check. Reads of the same status register cannot distinguish every possible
// response misroute, so this is also a contention/progress test.
//
// This test uses one contiguous chunk within both address-map stripes; see
// dma.h for the burst and multidimensional descriptor rules. The destination
// sits in the ISSUING group's own slice of the interleaved L1: a descriptor's
// L1 side is addressed with the group field of the L1 map replaced by the
// issuer's id, so a buffer whose stripe belongs to another group is written
// into this group at the same bank/word. A 128 B payload (the obvious first
// choice on minpool) crosses both the L2 and L1 stripe boundaries.

#include <stdint.h>

#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define CTX_ROUNDS 4u
#define CTX_POLLS 256u
// One group's slice of a row, matching DMA_COLLECTIVE_GROUP_BYTES. Reject
// configurations where that chunk would cross an L2 bank stripe.
#define CTX_WORDS (NUM_BANKS / NUM_GROUPS)
#define CTX_STRIPE_BYTES (CTX_WORDS * 4u) // 4 == sizeof(uint32_t)
#if CTX_STRIPE_BYTES > (AXI_DATA_WIDTH / 8u) * AXI_WIDTH_INTERLEAVED
#error "idma_ctx_share payload exceeds one L2 bank stripe"
#endif

// Aligned to a whole row: entry g then starts at the stripe that belongs to
// group g, which is what makes it addressable by that group's own DMA.
static uint32_t l2_src[NUM_GROUPS][CTX_WORDS]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l2")));
static uint32_t l1_dst[NUM_GROUPS][CTX_WORDS]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")));
static uint32_t errors __attribute__((section(".l1")));
static uint32_t polls __attribute__((section(".l1")));

static uint32_t pattern(uint32_t group, uint32_t round, uint32_t word) {
  return 0xC7000000u ^ (group << 16) ^ (round << 8) ^ word;
}

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t group = mempool_get_group_id();
  const uint32_t local_core = core % (uint32_t)NUM_CORES_PER_GROUP;
  // ONE issuer per group -- the contract dma.h states. The contention this
  // test is after comes from the readers: every other core of the group polls
  // the same context's status registers while that transfer runs, so the
  // group's control arbiter sees all NUM_TILES_PER_GROUP tiles at once.
  const uint32_t is_issuer = mempool_is_group_coordinator();

  mempool_barrier_init(core);
  if (core == 0u) {
    errors = 0u;
    polls = 0u;
  }
  mempool_barrier(NUM_CORES);

  for (uint32_t round = 1u; round <= CTX_ROUNDS; ++round) {
    // .l1 is NOLOAD. Give every destination word a known wrong value before
    // each transfer so a missing write fails deterministically.
    for (uint32_t w = local_core; w < CTX_WORDS;
         w += (uint32_t)NUM_CORES_PER_GROUP)
      l1_dst[group][w] = ~pattern(group, round, w);
    if (is_issuer)
      for (uint32_t w = 0; w < CTX_WORDS; ++w)
        l2_src[group][w] = pattern(group, round, w);
    mempool_barrier(NUM_CORES);

    if (is_issuer) {
      // ONE issuer per group, which the coordinator already is: the register
      // context is shared by the whole group and nothing brackets the
      // programming sequence any more (dma.h).
#ifndef CTX_NO_LAUNCH
      const dma_id_t id = dma_local_memcpy_nonblocking(
          0u, l1_dst[group], l2_src[group], CTX_STRIPE_BYTES);
      if (id == 0u)
        __atomic_fetch_add(&errors, 1u, __ATOMIC_RELAXED);
      else if (dma_local_wait_id(0u, id) != 0u)
        __atomic_fetch_add(&errors, 1u, __ATOMIC_RELAXED);
#endif
    } else {
      // Unlocked reads, concurrent with the issuer's programming and with each
      // other. STATUS exposes busy bits, not a Boolean. Count every completed
      // read and reject bits outside the documented register field.
      uint32_t local = 0u;
      uint32_t local_errors = 0u;
      for (uint32_t i = 0; i < CTX_POLLS; ++i) {
        uint32_t st = dma_local_status(0u);
#ifdef CTX_BAD_STATUS
        st |= 0x80000000u;
#endif
        if ((st & ~DMA_STATUS_BUSY_MASK) != 0u)
          ++local_errors;
        ++local;
      }
      __atomic_fetch_add(&polls, local, __ATOMIC_RELAXED);
      __atomic_fetch_add(&errors, local_errors, __ATOMIC_RELAXED);
    }
    mempool_barrier(NUM_CORES);

    if (is_issuer)
      for (uint32_t w = 0; w < CTX_WORDS; ++w)
        if (l1_dst[group][w] != pattern(group, round, w))
          __atomic_fetch_add(&errors, 1u, __ATOMIC_RELAXED);
    mempool_barrier(NUM_CORES);
  }

  if (core == 0u) {
    if (polls != CTX_ROUNDS * CTX_POLLS * (NUM_CORES - NUM_GROUPS))
      ++errors;
    printf("[idma_ctx_share] groups=%u tiles/group=%u stripe=%uB rounds=%u "
           "polls=%u errors=%u\n",
           (unsigned)NUM_GROUPS, (unsigned)NUM_TILES_PER_GROUP,
           (unsigned)CTX_STRIPE_BYTES, (unsigned)CTX_ROUNDS, (unsigned)polls,
           (unsigned)errors);
  }
  mempool_barrier(NUM_CORES);
  return (int)errors;
}
