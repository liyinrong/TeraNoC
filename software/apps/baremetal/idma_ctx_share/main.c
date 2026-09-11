// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Directed test for the ONE-CONTEXT-PER-GROUP iDMA register file.
//
// Every other DMA test in-tree has ONE core per group touching the context at
// all, so none of them ever puts two tiles on the group's control arbiter at
// once -- which is the only thing the shared context adds. Here the group
// coordinator runs a transfer (the documented one-issuer contract) while every
// other core of the group polls the SAME context's status and error registers
// as fast as it can, so all NUM_TILES_PER_GROUP tiles drive the arbiter
// together.
//
// It fails if the arbiter drops a response (a reader wedges and the run hangs),
// misroutes one (a reader sees an impossible status, or a sticky error that was
// never raised), or corrupts the transfer the issuer programmed underneath the
// read traffic (the destination check).

#include <stdint.h>

#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define CTX_ROUNDS 4u
#define CTX_WORDS 32u

static uint32_t l2_src[NUM_CORES / NUM_CORES_PER_TILE][CTX_WORDS]
    __attribute__((aligned(64), section(".l2")));
static uint32_t l1_dst[NUM_CORES / NUM_CORES_PER_TILE][CTX_WORDS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1")));
static uint32_t errors __attribute__((section(".l1")));
static uint32_t polls __attribute__((section(".l1")));

static uint32_t pattern(uint32_t tile, uint32_t round, uint32_t word) {
  return 0xC7000000u ^ (tile << 16) ^ (round << 8) ^ word;
}

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t tile = mempool_get_tile_id();
  const uint32_t tiles = (uint32_t)NUM_CORES / (uint32_t)NUM_CORES_PER_TILE;
  // ONE issuer per group -- the contract dma.h states. The contention this
  // test is after comes from the readers: every other core of the group polls
  // the same context's status registers while that transfer runs, so the
  // group's control arbiter sees all NUM_TILES_PER_GROUP tiles at once.
  const uint32_t is_tile_issuer = mempool_is_group_coordinator();

  mempool_barrier_init(core);
  // The lock is `.l1` and powers up X; without this every issuer spins forever.
  dma_group_ctx_init();
  if (core == 0u) {
    errors = 0u;
    polls = 0u;
  }
  for (uint32_t t = core; t < tiles; t += (uint32_t)NUM_CORES)
    for (uint32_t w = 0; w < CTX_WORDS; ++w)
      l1_dst[t][w] = 0u;
  mempool_barrier(NUM_CORES);

  for (uint32_t round = 1u; round <= CTX_ROUNDS; ++round) {
    if (is_tile_issuer)
      for (uint32_t w = 0; w < CTX_WORDS; ++w)
        l2_src[tile][w] = pattern(tile, round, w);
    mempool_barrier(NUM_CORES);

    if (is_tile_issuer) {
      dma_group_ctx_lock();
      const dma_id_t id = dma_local_memcpy_nonblocking(
          0u, l1_dst[tile], l2_src[tile], CTX_WORDS * sizeof(uint32_t));
      dma_group_ctx_unlock();
      if (id == 0u)
        __atomic_fetch_add(&errors, 1u, __ATOMIC_RELAXED);
      else if (dma_local_wait_id(0u, id) != 0u)
        __atomic_fetch_add(&errors, 1u, __ATOMIC_RELAXED);
    } else {
      // Unlocked reads, concurrent with the issuer's programming and with each
      // other. A dropped or misrouted response wedges the reader; a read that
      // returns another requester's data shows up as an impossible status.
      uint32_t local = 0u;
      for (uint32_t i = 0; i < 256u; ++i) {
        const uint32_t st = dma_local_status(0u);
        const uint32_t er = dma_local_error(0u);
        if (st <= 1u)
          ++local;
        if (er != 0u)
          __atomic_fetch_add(&errors, 1u, __ATOMIC_RELAXED);
      }
      __atomic_fetch_add(&polls, local, __ATOMIC_RELAXED);
    }
    mempool_barrier(NUM_CORES);

    if (is_tile_issuer)
      for (uint32_t w = 0; w < CTX_WORDS; ++w)
        if (l1_dst[tile][w] != pattern(tile, round, w))
          __atomic_fetch_add(&errors, 1u, __ATOMIC_RELAXED);
    mempool_barrier(NUM_CORES);
  }

  if (core == 0u)
    printf("[idma_ctx_share] tiles=%u tiles/group=%u rounds=%u polls=%u "
           "errors=%u\n",
           (unsigned)tiles, (unsigned)NUM_TILES_PER_GROUP, (unsigned)CTX_ROUNDS,
           (unsigned)polls, (unsigned)errors);
  mempool_barrier(NUM_CORES);
  return (int)errors;
}
