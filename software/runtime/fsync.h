// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Hardware barrier over the fractal_sync H-tree (roadmap Phase 3).
//
// The register contract matches MAGIA's obi_slave_fsync / fsync_mm_utils.h so
// their software reads directly against ours:
//
//   +0x00 AGGR    W   aggregate bitmask for the next barrier
//   +0x04 ID      W   scope id; bit 0 picks h (even) or v (odd)
//   +0x08 CONTROL W   bit0 = trigger, bit1 = clear sticky error
//   +0x0C STATUS  R   bit0 done, bit1 error, bit2 BUSY, [7:4] lvl, [8+] rsp id
//
// SCOPE MODEL. A barrier scope is a pow2-aligned SQUARE block of TILES, and
// the leaf of the tree is a TILE, not a core. That is what keeps every scope a
// COMPLETE subtree: a complete SxS subtree aggregates at every level it climbs,
// so aggr is all-ones and every member computes the SAME value independently,
// with no lock, no leader election and no communication. An incomplete scope
// (some cores excluded, an asymmetric set) makes the members' aggr DIVERGE and
// this whole O(1) scheme collapses -- measured, see scripts/fsync/.
//
// USAGE is two-level, because the hardware's finest scope is a PAIR of tiles
// and a single leaf is refused outright:
//   1. the selected compute cores of a tile meet at a polling barrier in L1;
//   2. the tile leader calls fsync_barrier() -- ONE arrival per tile;
//   3. the last arrival polls the response and release-publishes a done epoch.
// A dedicated DMA/control core simply stays out of step 1, which is why it
// does not perturb the scope.

#ifndef _FSYNC_H_
#define _FSYNC_H_

#include <stdint.h>

#include "addrmap.h"

#define FSYNC_AGGR_REG_OFFSET (0x00)
#define FSYNC_ID_REG_OFFSET (0x04)
#define FSYNC_CONTROL_REG_OFFSET (0x08)
#define FSYNC_STATUS_REG_OFFSET (0x0C)

#define FSYNC_CONTROL_TRIGGER (1u << 0)
#define FSYNC_CONTROL_CLEAR (1u << 1)

#define FSYNC_STATUS_DONE_MASK (1u << 0)
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
#define FSYNC_ID_H_NBR (2u)
#define FSYNC_ID_V_NBR (3u)

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
// group along each axis. This is invariant #9: a memory-side group level maps
// to a barrier-side TILE level one log2(tiles-per-group-side) higher.
static inline uint32_t fsync_aggr_of_vc_level(uint32_t k) {
  return fsync_aggr_of_side((1u << k) * (uint32_t)FSYNC_TPG_SIDE);
}

// Distinct id for one of several CONCURRENT rectangular scopes.
//
// Required, not optional. A rectangle thinner than square has ZERO bits in its
// aggregate: it PROPAGATES through the levels it skips rather than aggregating
// there, so two such scopes meet at a shared node. With equal ids that node
// pairs them and completes a barrier that does not exist -- demonstrated in
// hardware/tb/tb_fractal_sync.sv test F, where row 0's left half and row 1's
// right half woke together despite neither row being complete.
//
// SQUARE scopes do not need this: their aggregate is contiguous all-ones, so
// they aggregate at every level up to their own root and never pass through a
// node another scope could match at. Disjoint square VCs may all use id 0.
//
// `index` is the scope's position along the axis it is stacked on (which row
// of VCs this is). Bit 0 must keep encoding the direction, so ids step by two.
// The id field is finite, so this wraps like MAGIA's row_id_lookup_mm: only
// scopes that can actually meet need to differ, and ones far enough apart
// resolve below any shared node and may reuse.
static inline uint32_t fsync_rect_id(uint32_t index, uint32_t vertical,
                                     uint32_t n_concurrent) {
  const uint32_t half = n_concurrent > 1 ? n_concurrent / 2 : 1;
  const uint32_t i = (index < half) ? index : (index - half);
  return (i << 1) | (vertical ? 1u : 0u);
}

// One VC-row: every virtual cluster sitting in the same row of VCs at level k.
// Full mesh width by one VC height -- the second half of a
// "reduce inside each VC, then combine across the row" pattern.
static inline uint32_t fsync_aggr_of_vc_row(uint32_t k, uint32_t *id) {
  return fsync_aggr_of_rect((uint32_t)FSYNC_TILE_SIDE,
                            (1u << k) * (uint32_t)FSYNC_TPG_SIDE, id);
}

// The column equivalent. Note it comes back with id = FSYNC_ID_V: a taller
// than wide region enters the VERTICAL network, so the id must be passed to
// the hardware along with the aggregate, not assumed to be 0.
static inline uint32_t fsync_aggr_of_vc_col(uint32_t k, uint32_t *id) {
  return fsync_aggr_of_rect((1u << k) * (uint32_t)FSYNC_TPG_SIDE,
                            (uint32_t)FSYNC_TILE_SIDE, id);
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

// Arm a barrier. Does NOT block; pair with fsync_wait().
static inline void fsync_trigger(uint32_t id, uint32_t aggr) {
  *(volatile uint32_t *)(FSYNC_BASE + FSYNC_AGGR_REG_OFFSET) = aggr;
  *(volatile uint32_t *)(FSYNC_BASE + FSYNC_ID_REG_OFFSET) = id;
  *(volatile uint32_t *)(FSYNC_BASE + FSYNC_CONTROL_REG_OFFSET) =
      FSYNC_CONTROL_TRIGGER;
}

// Observe BUSY assertion before accepting its deassertion. The MMIO trigger
// can take several cycles to reach the shim, so polling only for an idle level
// can mistake the pre-issue state for completion.
static inline uint32_t fsync_wait_response(void) {
  uint32_t status;
  do {
    status = fsync_status();
    if ((status & FSYNC_STATUS_BUSY_MASK) == 0u)
      mempool_wait(4u);
  } while ((status & FSYNC_STATUS_BUSY_MASK) == 0u);
  do {
    status = fsync_status();
    if ((status & FSYNC_STATUS_BUSY_MASK) != 0u)
      mempool_wait(4u);
  } while ((status & FSYNC_STATUS_BUSY_MASK) != 0u);
  return status;
}

// Returns non-zero if the tree reported an error (RF/CAM overflow), which
// must NOT be ignored: the scope did not synchronise.
static inline uint32_t fsync_wait(void) {
  return fsync_wait_response() & FSYNC_STATUS_ERROR_MASK;
}

static inline void fsync_clear_error(void) {
  *(volatile uint32_t *)(FSYNC_BASE + FSYNC_CONTROL_REG_OFFSET) =
      FSYNC_CONTROL_CLEAR;
}

// ---------------------------------------------------------------------------
// Two-level barrier. EVERY core of the tile calls these.
//
// The core that arrives LAST publishes an issue epoch, triggers the tree, polls
// BUSY, validates the returned level/id, and publishes the done epoch. Earlier
// cores poll those epochs, so an idle shim before issue cannot release them.
// Each tile counter occupies one full tile-bank stripe: address bits
// immediately above the bank select encode the tile, so adjacent words would
// all live in the first tile instead of beside their users. The array base must
// also be aligned to a complete all-tile bank stripe so entry zero maps to tile
// zero. Counters MUST be zeroed before first use (`.l1` is NOBITS and powers up
// X).
//
// A tree error or mismatched response traps before any participant proceeds.
// ---------------------------------------------------------------------------

typedef struct {
  volatile uint32_t count;
  volatile uint32_t issue_epoch;
  volatile uint32_t done_epoch;
  volatile uint32_t status_error;
  volatile uint32_t expected_level;
  volatile uint32_t expected_id;
  uint32_t padding[NUM_BANKS_PER_TILE > 6 ? NUM_BANKS_PER_TILE - 6 : 1];
} fsync_tile_counter_t;

#define FSYNC_TILE_COUNTER_ALIGNMENT (NUM_BANKS * sizeof(uint32_t))

static inline void
fsync_tile_counter_init(volatile fsync_tile_counter_t *tile_cnt) {
  tile_cnt->count = 0u;
  tile_cnt->issue_epoch = 0u;
  tile_cnt->done_epoch = 0u;
  tile_cnt->status_error = 0u;
  tile_cnt->expected_level = 0u;
  tile_cnt->expected_id = 0u;
  for (uint32_t i = 0u;
       i < (NUM_BANKS_PER_TILE > 6 ? NUM_BANKS_PER_TILE - 6 : 1); ++i)
    tile_cnt->padding[i] = 0u;
}

static inline void fsync_barrier_n(volatile fsync_tile_counter_t *tile_cnt,
                                   uint32_t tile, uint32_t id, uint32_t aggr,
                                   uint32_t participants) {
  if (participants == 0u || participants > (uint32_t)NUM_CORES_PER_TILE)
    __builtin_trap();
  const uint32_t expected =
      __atomic_load_n(&tile_cnt[tile].done_epoch, __ATOMIC_ACQUIRE);
  if (__atomic_fetch_add(&tile_cnt[tile].count, 1, __ATOMIC_RELAXED) ==
      participants - 1u) {
    // Ordered by the issue_epoch release below, like the three stores after it.
    tile_cnt[tile].count = 0;
    tile_cnt[tile].expected_level = fsync_level_of_aggr(aggr);
    tile_cnt[tile].expected_id = id;
    tile_cnt[tile].status_error = 0;
    __atomic_store_n(&tile_cnt[tile].issue_epoch, expected + 1u,
                     __ATOMIC_RELEASE);
    fsync_trigger(id, aggr);
    uint32_t status;
    do {
      status = fsync_status();
      if ((status & FSYNC_STATUS_BUSY_MASK) == 0u)
        mempool_wait(4u);
    } while ((status & FSYNC_STATUS_BUSY_MASK) == 0u);
    do {
      status = fsync_status();
      if ((status & FSYNC_STATUS_BUSY_MASK) != 0u)
        mempool_wait(4u);
    } while ((status & FSYNC_STATUS_BUSY_MASK) != 0u);
    if ((status & FSYNC_STATUS_ERROR_MASK) != 0u ||
        fsync_status_level(status) != tile_cnt[tile].expected_level ||
        fsync_status_id(status) != tile_cnt[tile].expected_id)
      tile_cnt[tile].status_error = status | 0x80000000u;
    __atomic_store_n(&tile_cnt[tile].done_epoch, expected + 1u,
                     __ATOMIC_RELEASE);
  } else {
#ifdef FSYNC_NEGATIVE_SKIP_ISSUE_EPOCH
    // Deliberate gate falsification: an idle pre-issue shim releases early.
    // Production builds never define this macro.
    while ((fsync_status() & FSYNC_STATUS_BUSY_MASK) != 0u) {
      mempool_wait(4u);
    }
#else
    while (__atomic_load_n(&tile_cnt[tile].issue_epoch, __ATOMIC_ACQUIRE) <=
           expected) {
      mempool_wait(4u);
    }
    while (__atomic_load_n(&tile_cnt[tile].done_epoch, __ATOMIC_ACQUIRE) <=
           expected) {
      mempool_wait(4u);
    }
#endif
  }
  // Spend the credit the tree just deposited.
  //
  // fs_wake is broadcast to EVERY core of the tile, unmasked
  // (mempool_tile.sv:1746), and nothing above ever sleeps -- mempool_wait is a
  // delay loop -- so every core is awake when the response lands and banks a
  // wake it never asked for (snitch.sv:447). wake_up_q is three bits and wraps
  // without saturating, so the eighth barrier destroys all eight credits and
  // the next mempool_wfi() sleeps forever. Measured exactly that way:
  // dma_dedicated_bench runs 18 of these and the RTL detector fired at barrier
  // 8 and 16, 256 cores each.
  //
  // This wfi cannot block. The credit is banked at the response edge, which is
  // strictly before the done_epoch handshake above can complete, so wake_up_q
  // is already nonzero here and wfi returns immediately while decrementing it
  // (snitch.sv:747-757). Net credit change per barrier: zero.
  //
  // NOT covered: a core of this tile that is not a participant still receives
  // the broadcast and has no drain. Partial-tile barriers (fsync_barrier_n with
  // participants < NUM_CORES_PER_TILE) therefore still leak into the
  // non-participants, and mixing those with sleeping barriers is unsafe.
  mempool_wfi();

  if (__atomic_load_n(&tile_cnt[tile].status_error, __ATOMIC_ACQUIRE) != 0u)
    __builtin_trap();
}

static inline void fsync_barrier(volatile fsync_tile_counter_t *tile_cnt,
                                 uint32_t tile, uint32_t id, uint32_t aggr) {
  fsync_barrier_n(tile_cnt, tile, id, aggr, NUM_CORES_PER_TILE);
}

static inline void fsync_barrier_vc(volatile fsync_tile_counter_t *tile_cnt,
                                    uint32_t tile, uint32_t k) {
  fsync_barrier(tile_cnt, tile, FSYNC_ID_H, fsync_aggr_of_vc_level(k));
}

// Combine across a row of VCs. Pair with fsync_barrier_vc() for the two-phase
// pattern: every VC reduces internally, then the row combines. Note the id is
// NOT always FSYNC_ID_H -- a taller-than-wide scope enters the vertical
// network -- so it is taken from the helper.
static inline void fsync_barrier_vc_row(volatile fsync_tile_counter_t *tile_cnt,
                                        uint32_t tile, uint32_t k) {
  uint32_t id;
  const uint32_t aggr = fsync_aggr_of_vc_row(k, &id);
  fsync_barrier(tile_cnt, tile, id, aggr);
}

static inline void fsync_barrier_vc_col(volatile fsync_tile_counter_t *tile_cnt,
                                        uint32_t tile, uint32_t k) {
  uint32_t id;
  const uint32_t aggr = fsync_aggr_of_vc_col(k, &id);
  fsync_barrier(tile_cnt, tile, id, aggr);
}

static inline void fsync_barrier_all(volatile fsync_tile_counter_t *tile_cnt,
                                     uint32_t tile) {
  fsync_barrier(tile_cnt, tile, FSYNC_ID_H, fsync_aggr_global());
}

static inline void fsync_barrier_all_n(volatile fsync_tile_counter_t *tile_cnt,
                                       uint32_t tile, uint32_t participants) {
  fsync_barrier_n(tile_cnt, tile, FSYNC_ID_H, fsync_aggr_global(),
                  participants);
}

// Single-arrival form, for a caller that has ALREADY established that exactly
// one core per tile reaches it. Other tile cores must not enter this helper.
static inline uint32_t fsync_barrier_global(void) {
  fsync_trigger(FSYNC_ID_H, fsync_aggr_global());
  const uint32_t status = fsync_wait_response();
  if ((status & FSYNC_STATUS_ERROR_MASK) != 0u ||
      fsync_status_level(status) != fsync_level_of_aggr(fsync_aggr_global()) ||
      fsync_status_id(status) != FSYNC_ID_H)
    return status | 0x80000000u;
  return 0u;
}

#endif // FSYNC_BASE
#endif // _FSYNC_H_
