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
// It fails if the arbiter drops a response (a reader wedges and the run hangs),
// misroutes one (a reader sees an impossible status), or corrupts the transfer
// the issuer programmed underneath the read traffic (the destination check).
//
// THE PAYLOAD IS ONE STRIPE, and that is the engine's contract, not a detail
// of this test -- see the contract in dma.h. The L1 destination must also sit
// in the ISSUING group's own slice of the interleaved L1: a descriptor's L1
// side is addressed with the group field of the L1 map replaced by the
// issuer's id, so a buffer whose stripe belongs to another group is written
// into this group at the same bank/word. A 128 B payload (the obvious first
// choice) breaks both: it crosses the stripe in the L2 and in the L1 map.

#include <stdint.h>

#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define CTX_ROUNDS 4u
// The payload must fit ONE stripe of both maps the transfer touches, and the
// two are not the same size on every config (mempool: 1024 B of L2 bank
// against 64 B of L1 group), so take the smaller -- a group's slice of one row,
// which is DMA_COLLECTIVE_GROUP_BYTES in dma_collective.c. dma_collective
// splits every linear transfer at exactly these boundaries; a raw descriptor
// has to respect them on its own.
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
  // ONE issuer per group -- the contract dma.h states. The contention this
  // test is after comes from the readers: every other core of the group polls
  // the same context's status registers while that transfer runs, so the
  // group's control arbiter sees all NUM_TILES_PER_GROUP tiles at once.
  const uint32_t is_tile_issuer = mempool_is_group_coordinator();

  mempool_barrier_init(core);
  if (core == 0u) {
    errors = 0u;
    polls = 0u;
  }
  for (uint32_t w = core; w < CTX_WORDS; w += (uint32_t)NUM_CORES) {
    l1_dst[group][w] = 0u;
    l2_src[group][w] = pattern(group, 0u, w);
  }
  mempool_barrier(NUM_CORES);

  for (uint32_t round = 1u; round <= CTX_ROUNDS; ++round) {
    if (is_tile_issuer)
      for (uint32_t w = 0; w < CTX_WORDS; ++w)
        l2_src[group][w] = pattern(group, round, w);
    mempool_barrier(NUM_CORES);

    if (is_tile_issuer) {
      // ONE issuer per group, which the coordinator already is: the register
      // context is shared by the whole group and nothing brackets the
      // programming sequence any more (dma.h).
      const dma_id_t id = dma_local_memcpy_nonblocking(
          0u, l1_dst[group], l2_src[group], CTX_STRIPE_BYTES);
      if (id == 0u)
        __atomic_fetch_add(&errors, 1u, __ATOMIC_RELAXED);
      else if (dma_local_wait_id(0u, id) != 0u)
        __atomic_fetch_add(&errors, 1u, __ATOMIC_RELAXED);
    } else {
      // Unlocked reads, concurrent with the issuer's programming and with each
      // other. A dropped or misrouted response wedges the reader; a read that
      // returns another requester's data shows up as an impossible status.
      // STATUS is the only per-stream register left to read this way: the
      // error window was removed with the rest of the error path.
      uint32_t local = 0u;
      for (uint32_t i = 0; i < 256u; ++i) {
        const uint32_t st = dma_local_status(0u);
        if (st <= 1u)
          ++local;
      }
      __atomic_fetch_add(&polls, local, __ATOMIC_RELAXED);
    }
    mempool_barrier(NUM_CORES);

    if (is_tile_issuer)
      for (uint32_t w = 0; w < CTX_WORDS; ++w)
        if (l1_dst[group][w] != pattern(group, round, w))
          __atomic_fetch_add(&errors, 1u, __ATOMIC_RELAXED);
    mempool_barrier(NUM_CORES);
  }

  if (core == 0u)
    printf("[idma_ctx_share] groups=%u tiles/group=%u stripe=%uB rounds=%u "
           "polls=%u errors=%u\n",
           (unsigned)NUM_GROUPS, (unsigned)NUM_TILES_PER_GROUP,
           (unsigned)CTX_STRIPE_BYTES, (unsigned)CTX_ROUNDS, (unsigned)polls,
           (unsigned)errors);
  mempool_barrier(NUM_CORES);
  return (int)errors;
}
