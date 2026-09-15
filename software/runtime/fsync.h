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

// Self-contained on purpose: this header uses NUM_BANKS_PER_TILE,
// mempool_get_core_id() and mempool_wfi(), so an includer that reached it
// before runtime.h used to fail with an implicit-declaration cascade rather
// than a clear error.
#include <stdint.h>

#include "runtime.h"

#include <stdint.h>

#include "addrmap.h"

#define FSYNC_AGGR_REG_OFFSET (0x00)
#define FSYNC_ID_REG_OFFSET (0x04)
#define FSYNC_CONTROL_REG_OFFSET (0x08)
#define FSYNC_STATUS_REG_OFFSET (0x0C)
#define FSYNC_SUBSCRIBE_REG_OFFSET 0x20u
#define FSYNC_CAUSE_REG_OFFSET 0x30u
#define FSYNC_EVT_SEQ_REG_OFFSET 0x40u
#define FSYNC_EVT_FSYNC 0x1u
#define FSYNC_EVT_DMA 0x2u     // group engine went idle (busy falling)
#define FSYNC_EVT_DMA_ERR 0x4u // group engine latched a fault
#define FSYNC_EVT_ENGINE 0x8u  // tile engine end-of-computation (RedMulE)

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

// Which event sources may wake THIS core. Per-core address, so a core writes
// its own word: no read-modify-write and no race with its tile-mates.
static inline void fsync_subscribe_self(uint32_t sources) {
  const uint32_t lane = mempool_get_core_id() % (uint32_t)NUM_CORES_PER_TILE;
  *(volatile uint32_t *)(FSYNC_BASE + FSYNC_SUBSCRIBE_REG_OFFSET + 4u * lane) =
      sources;
}

// What woke this core, latched at the wake.
// Monotonic count of THIS core's event wakes. Sample it before sleeping and
// loop until it changes: that is the only way to tell "the tree answered me"
// from "wfi returned for some other reason". wfi is source-agnostic, and the
// legacy global wake banks a credit on any core that happens to be awake, so a
// bare wfi can return with nothing about the barrier having changed.
static inline uint32_t fsync_evt_seq_self(void) {
  const uint32_t lane = mempool_get_core_id() % (uint32_t)NUM_CORES_PER_TILE;
  return *(volatile uint32_t *)(FSYNC_BASE + FSYNC_EVT_SEQ_REG_OFFSET +
                                4u * lane) &
         0xffu;
}

static inline uint32_t fsync_cause_self(void) {
  const uint32_t lane = mempool_get_core_id() % (uint32_t)NUM_CORES_PER_TILE;
  return *(volatile uint32_t *)(FSYNC_BASE + FSYNC_CAUSE_REG_OFFSET +
                                4u * lane);
}

// Leave the tile's barrier events entirely. A core that takes no part in its
// tile's barriers must unsubscribe: while subscribed, every response leaves a
// pending bit that wakes its next sleep, and mempool_barrier has no re-check.
static inline void fsync_unsubscribe_self(void) { fsync_subscribe_self(0u); }

static inline void fsync_clear_error(void) {
  *(volatile uint32_t *)(FSYNC_BASE + FSYNC_CONTROL_REG_OFFSET) =
      FSYNC_CONTROL_CLEAR;
}

// ---------------------------------------------------------------------------
// Two-level barrier. EVERY core of the tile calls these.
//
// The core that arrives LAST triggers the tree. Every participant, the trigger
// core included, then sleeps until its own event count moves, which only the
// tree's response can do: pending holds subscribed sources only, and a legacy
// wake does not count.
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

  // Sample BEFORE arriving: a response cannot be missed, only observed as a
  // change that already happened.
  const uint32_t seq0 = fsync_evt_seq_self();

  // Intra-tile rendezvous. tile_cnt[tile] sits in THIS tile's own bank stripe,
  // so the atomic is a local access; moving it into hardware would cost more
  // than it saves (an arrival written to the shim serialises at the tile's
  // fsync arbiter, measured at +16 cycles for four cores).
  if (__atomic_fetch_add(&tile_cnt[tile].count, 1, __ATOMIC_RELAXED) ==
      participants - 1u) {
    tile_cnt[tile].count = 0;
    fsync_trigger(id, aggr);
    // Everything below is off every other core's critical path: they are
    // released by the tree, not by this core.
    const uint32_t exp_level = fsync_level_of_aggr(aggr);
    // Wait like every other participant. A bare wfi can return on a banked
    // legacy wake; the response would then land while this core is awake and
    // stay pending, releasing it early from its next barrier. The error bit
    // ends the wait too, since a tree error releases BUSY without a response.
    uint32_t status;
    do {
      mempool_wfi();
      status = fsync_status();
    } while ((status & FSYNC_STATUS_BUSY_MASK) != 0u ||
             ((status & FSYNC_STATUS_ERROR_MASK) == 0u &&
              fsync_evt_seq_self() == seq0));
    if ((status & FSYNC_STATUS_ERROR_MASK) != 0u ||
        fsync_status_level(status) != exp_level ||
        fsync_status_id(status) != id)
      __builtin_trap();
    // Observability only -- no core waits on these. They are what makes
    // fsync_epoch_smoke able to assert that every tile really did trigger on
    // every generation rather than merely that everyone got out.
    tile_cnt[tile].expected_level = exp_level;
    tile_cnt[tile].expected_id = id;
    ++tile_cnt[tile].issue_epoch;
    ++tile_cnt[tile].done_epoch;
  } else {
    // Released by the tree response, re-checked against the event sequence.
    //
    // A bare wfi is NOT sufficient: it is source-agnostic, and wake_up_all from
    // mempool_barrier banks a credit on any core that is awake when it fires --
    // measured, 15 such credits in a four-barrier run -- so a later wfi returns
    // immediately with the barrier still outstanding. Dropping the re-check
    // entirely deadlocked fsync_epoch_smoke: cores left barriers whose tiles
    // had not triggered.
    //
    // The sequence counter is what the two-epoch handshake used to provide,
    // without the cost: the old code waited for the TRIGGER core to read
    // STATUS, validate, and publish done_epoch, which put its MMIO read on
    // every other core's critical path. This is two local MMIO reads instead.
#ifdef FSYNC_NEGATIVE_SKIP_SEQ_RECHECK
    // Deliberate gate falsification: accept ANY wake, including a legacy credit
    // banked while this core was awake. Production builds never define this.
    mempool_wfi();
#else
    while (fsync_evt_seq_self() == seq0)
      mempool_wfi();
#endif
  }
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
