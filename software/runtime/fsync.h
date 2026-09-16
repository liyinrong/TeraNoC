// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Hardware barrier over the fractal_sync H-tree.
//
// The register contract matches MAGIA's obi_slave_fsync / fsync_mm_utils.h so
// their software reads directly against ours:
//
//   +0x00 AGGR    W   aggregate bitmask for the next barrier
//   +0x04 ID      W   scope id; bit 0 picks h (even) or v (odd)
//   +0x08 CONTROL W   bit0 = trigger, bit1 = clear sticky error
//   +0x0C STATUS  R   bit0 done, bit1 error, bit2 BUSY, [7:4] lvl, [8+] rsp id
//
// The per-core event unit above that (SUBSCRIBE) is described
// in hardware/src/fsync_shim.sv.
//
// SCOPE MODEL. A barrier scope is a pow2-aligned SQUARE block of TILES, and
// the leaf of the tree is a TILE, not a core. That is what keeps every scope a
// COMPLETE subtree: a complete SxS subtree aggregates at every level it climbs,
// so aggr is all-ones and every member computes the SAME value independently,
// with no lock, no leader election and no communication. An incomplete scope
// (some cores excluded, an asymmetric set) makes the members' aggr DIVERGE and
// this whole O(1) scheme collapses (scripts/fsync/ checks the arithmetic).
//
// USAGE is two-level, because the hardware's finest scope is a PAIR of tiles
// and a single leaf is refused outright:
//   1. the participating cores of a tile meet at an atomic counter in L1;
//   2. the last of them triggers the tree -- ONE arrival per tile;
//   3. every participant sleeps until the tree's response reaches it.
// A dedicated DMA/control core stays out of step 1 and unsubscribes, which is
// why it does not perturb the scope.

#ifndef _FSYNC_H_
#define _FSYNC_H_

#include <stdint.h>

#include "addrmap.h"
#include "runtime.h"

#define FSYNC_AGGR_REG_OFFSET (0x00)
#define FSYNC_ID_REG_OFFSET (0x04)
#define FSYNC_CONTROL_REG_OFFSET (0x08)
#define FSYNC_STATUS_REG_OFFSET (0x0C)
#define FSYNC_SUBSCRIBE_REG_OFFSET 0x20u
#define FSYNC_EVT_FSYNC 0x1u
#define FSYNC_EVT_DMA 0x2u    // one group DMA transfer retired
#define FSYNC_EVT_ENGINE 0x4u // tile engine end-of-computation (RedMulE)

#define FSYNC_CONTROL_TRIGGER (1u << 0)

#define FSYNC_STATUS_ERROR_MASK (1u << 1)
// Bit 2, matching MAGIA's FSYNC_MM_STATUS_BUSY_MASK.
#define FSYNC_STATUS_BUSY_MASK (1u << 2)
#define FSYNC_STATUS_LEVEL_SHIFT 4u
#define FSYNC_STATUS_LEVEL_MASK (0xfu << FSYNC_STATUS_LEVEL_SHIFT)
#define FSYNC_STATUS_ID_SHIFT 8u

static inline uint32_t fsync_status_level(uint32_t status) {
  return (status & FSYNC_STATUS_LEVEL_MASK) >> FSYNC_STATUS_LEVEL_SHIFT;
}

static inline uint32_t fsync_status_id(uint32_t status) {
  return status >> FSYNC_STATUS_ID_SHIFT;
}

// id bit 0 selects the network. 0/1 are the 1D tree nodes, 2/3 the neighbour
// links used when two adjacent tiles sit under DIFFERENT sibling nodes.
#define FSYNC_ID_H (0u)
#define FSYNC_ID_V (1u)

// ---------------------------------------------------------------------------
// Scope arithmetic. Pure integer maths, no hardware needed, so it is unit
// testable on the host against the reference generator (scripts/fsync/).
// ---------------------------------------------------------------------------

// Tiles per group along one axis. Every flavour we build has a square tile
// grid AND a square tile block per group; a non-square one has no tree.
#define FSYNC_TPG_SIDE                                                         \
  (NUM_TILES_PER_GROUP == 1                                                    \
       ? 1                                                                     \
       : (NUM_TILES_PER_GROUP == 4                                             \
              ? 2                                                              \
              : (NUM_TILES_PER_GROUP == 16                                     \
                     ? 4                                                       \
                     : (NUM_TILES_PER_GROUP == 64 ? 8 : 0))))

// Side of the whole tile grid, in tiles.
#define FSYNC_TILE_SIDE (NUM_X * FSYNC_TPG_SIDE)

// A complete side x side subtree climbs 2*log2(side) levels and aggregates at
// every one of them, so the mask is all-ones of that width.
static inline uint32_t fsync_aggr_of_side(uint32_t side) {
  uint32_t levels = 0;
  for (uint32_t s = side; s > 1; s >>= 1)
    levels += 2;
  return (1u << levels) - 1u;
}

static inline uint32_t fsync_level_of_aggr(uint32_t aggr) {
  return aggr == 0u ? 0u : 31u - (uint32_t)__builtin_clz(aggr);
}

// General pow2-aligned RECTANGLE of tiles, w wide by h tall. This is what
// makes broadcast/reduce patterns expressible: "every VC in this row of VCs"
// is a rectangle, not a square.
//
// The tree alternates h and v levels STARTING WITH THE NETWORK ENTERED, and a
// w x h region aggregates at log2(w) horizontal levels and log2(h) vertical
// ones. Entering on the axis that needs MORE levels keeps the set bits
// contiguous from the LSB. Verified against the reference generator for every
// pow2-aligned rectangle at every position (scripts/fsync/).
//
// Returns the aggregate and writes the scope id, which selects the entry
// network -- so the caller must pass BOTH to the hardware.
static inline uint32_t fsync_aggr_of_rect(uint32_t w, uint32_t h,
                                          uint32_t *id) {
  uint32_t lw = 0, lh = 0;
  for (uint32_t t = w; t > 1; t >>= 1)
    lw++;
  for (uint32_t t = h; t > 1; t >>= 1)
    lh++;
  const uint32_t vertical = (lh > lw);
  const uint32_t n_entry = vertical ? lh : lw;
  const uint32_t n_other = vertical ? lw : lh;
  uint32_t aggr = 0;
  for (uint32_t i = 0; i < n_entry; i++)
    aggr |= 1u << (2 * i);
  for (uint32_t j = 0; j < n_other; j++)
    aggr |= 1u << (2 * j + 1);
  *id = vertical ? FSYNC_ID_V : FSYNC_ID_H;
  return aggr;
}

// A virtual-cluster level k spans 2^k GROUPS per axis, hence 2^k * tiles per
// group along each axis: a memory-side group level maps to a barrier-side TILE
// level one log2(tiles-per-group-side) higher.
static inline uint32_t fsync_aggr_of_vc_level(uint32_t k) {
  return fsync_aggr_of_side((1u << k) * (uint32_t)FSYNC_TPG_SIDE);
}

// Every tile of the machine.
static inline uint32_t fsync_aggr_global(void) {
  return fsync_aggr_of_side((uint32_t)FSYNC_TILE_SIDE);
}

// ---------------------------------------------------------------------------
// Hardware access. Only available once the shim is mapped into the tile.
// ---------------------------------------------------------------------------
#ifdef FSYNC_BASE

static inline uint32_t fsync_status(void) {
  return *(volatile uint32_t *)(FSYNC_BASE + FSYNC_STATUS_REG_OFFSET);
}

// Arm a barrier. Does not block.
static inline void fsync_trigger(uint32_t id, uint32_t aggr) {
  *(volatile uint32_t *)(FSYNC_BASE + FSYNC_AGGR_REG_OFFSET) = aggr;
  *(volatile uint32_t *)(FSYNC_BASE + FSYNC_ID_REG_OFFSET) = id;
  *(volatile uint32_t *)(FSYNC_BASE + FSYNC_CONTROL_REG_OFFSET) =
      FSYNC_CONTROL_TRIGGER;
}

// Which event sources may wake THIS core. Per-core address, so a core writes
// its own word: no read-modify-write and no race with its tile-mates.
static inline void fsync_subscribe_self(uint32_t sources) {
  const uint32_t lane = mempool_get_core_id() % (uint32_t)NUM_CORES_PER_TILE;
  *(volatile uint32_t *)(FSYNC_BASE + FSYNC_SUBSCRIBE_REG_OFFSET + 4u * lane) =
      sources;
}

// Leave the tile's barrier events entirely. A core that takes no part in its
// tile's barriers must unsubscribe: while subscribed, every response leaves a
// pending bit that wakes its next sleep, and mempool_barrier has no re-check.
static inline void fsync_unsubscribe_self(void) { fsync_subscribe_self(0u); }

// ---------------------------------------------------------------------------
// Two-level barrier. EVERY core of the tile calls these.
//
// The core that arrives LAST triggers the tree, and every participant, the
// trigger core included, sleeps once. A participant is subscribed to the tree
// event only (the reset value, which every event wait restores), and pending
// holds the response if it lands before the sleep, so the wake is always the
// response. That needs every legacy wake_up to be spent before the barrier: a
// software barrier before or after a phase of these is fine, one interleaved
// with them is not. A mismatched response traps.
//
// Each tile counter occupies one full tile-bank stripe, so entry t lands in
// tile t's own banks, and the array base is aligned to a whole all-tile stripe.
// The runtime owns one array, fsync_tile_counters, zeroed by
// mempool_barrier_init(); a test that wants its own must zero it first (`.l1`
// is NOBITS and powers up X).
//
// The epoch fields are bookkeeping for tests: they are only written when the
// includer defines FSYNC_EPOCH_TRACK.
// ---------------------------------------------------------------------------

#if NUM_BANKS_PER_TILE < 6
#error "fsync_tile_counter_t needs at least six banks per tile"
#endif

typedef struct {
  volatile uint32_t count;
  volatile uint32_t issue_epoch;
  volatile uint32_t done_epoch;
  volatile uint32_t expected_level;
  volatile uint32_t expected_id;
  uint32_t padding[NUM_BANKS_PER_TILE - 5];
} fsync_tile_counter_t;

#define FSYNC_TILE_COUNTER_ALIGNMENT (NUM_BANKS * sizeof(uint32_t))

#if FSYNC_TPG_SIDE != 0
extern fsync_tile_counter_t fsync_tile_counters[];
#endif

static inline void
fsync_tile_counter_init(volatile fsync_tile_counter_t *tile_cnt) {
  tile_cnt->count = 0u;
  tile_cnt->issue_epoch = 0u;
  tile_cnt->done_epoch = 0u;
  tile_cnt->expected_level = 0u;
  tile_cnt->expected_id = 0u;
}

static inline void fsync_barrier_n(volatile fsync_tile_counter_t *tile_cnt,
                                   uint32_t tile, uint32_t id, uint32_t aggr,
                                   uint32_t participants) {
  if (participants == 0u || participants > (uint32_t)NUM_CORES_PER_TILE)
    __builtin_trap();

  // Intra-tile rendezvous. tile_cnt[tile] sits in THIS tile's own bank stripe,
  // so the atomic is a local access; an arrival written to the shim instead
  // would serialise at the tile's fsync arbiter.
  if (__atomic_fetch_add(&tile_cnt[tile].count, 1, __ATOMIC_RELAXED) ==
      participants - 1u) {
    tile_cnt[tile].count = 0;
    fsync_trigger(id, aggr);
    // Everything below is off every other core's critical path: they are
    // released by the tree, not by this core.
    const uint32_t exp_level = fsync_level_of_aggr(aggr);
    mempool_wfi();
    // BUSY still set means something other than the response woke this core.
    const uint32_t status = fsync_status();
    if ((status & (FSYNC_STATUS_ERROR_MASK | FSYNC_STATUS_BUSY_MASK)) != 0u ||
        fsync_status_level(status) != exp_level ||
        fsync_status_id(status) != id)
      __builtin_trap();
#ifdef FSYNC_EPOCH_TRACK
    tile_cnt[tile].expected_level = exp_level;
    tile_cnt[tile].expected_id = id;
    ++tile_cnt[tile].issue_epoch;
    ++tile_cnt[tile].done_epoch;
#endif
  } else {
    // Released by the tree response.
    mempool_wfi();
  }
}

static inline void fsync_barrier(volatile fsync_tile_counter_t *tile_cnt,
                                 uint32_t tile, uint32_t id, uint32_t aggr) {
  fsync_barrier_n(tile_cnt, tile, id, aggr, NUM_CORES_PER_TILE);
}

static inline void fsync_barrier_all(volatile fsync_tile_counter_t *tile_cnt,
                                     uint32_t tile) {
  fsync_barrier(tile_cnt, tile, FSYNC_ID_H, fsync_aggr_global());
}

#endif // FSYNC_BASE
#endif // _FSYNC_H_
