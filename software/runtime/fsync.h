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
//   1. all compute cores of a tile meet at a software barrier in local L1;
//   2. the tile leader calls fsync_barrier() -- ONE arrival per tile;
//   3. every core of the tile is released by the wake broadcast.
// A dedicated DMA/control core simply stays out of step 1, which is why it
// does not perturb the scope.

#ifndef _FSYNC_H_
#define _FSYNC_H_

#include <stdint.h>

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

// Poll until the barrier resolves. Returns non-zero if the tree reported an
// error (RF/CAM overflow), which must NOT be ignored: the scope did not
// synchronise and the caller has to trap rather than proceed.
static inline uint32_t fsync_wait(void) {
  uint32_t status;
  do {
    status = fsync_status();
  } while (status & FSYNC_STATUS_BUSY_MASK);
  return status & FSYNC_STATUS_ERROR_MASK;
}

static inline void fsync_clear_error(void) {
  *(volatile uint32_t *)(FSYNC_BASE + FSYNC_CONTROL_REG_OFFSET) =
      FSYNC_CONTROL_CLEAR;
}

// One arrival for this tile at a virtual-cluster level-k scope. Call from the
// tile leader only, after the intra-tile software barrier.
static inline uint32_t fsync_barrier_vc(uint32_t k) {
  fsync_trigger(FSYNC_ID_H, fsync_aggr_of_vc_level(k));
  return fsync_wait();
}

// Combine across a row of VCs. Pair with fsync_barrier_vc() for the two-phase
// pattern: every VC reduces internally, then the row combines.
static inline uint32_t fsync_barrier_vc_row(uint32_t k) {
  uint32_t id;
  const uint32_t aggr = fsync_aggr_of_vc_row(k, &id);
  fsync_trigger(id, aggr);
  return fsync_wait();
}

static inline uint32_t fsync_barrier_vc_col(uint32_t k) {
  uint32_t id;
  const uint32_t aggr = fsync_aggr_of_vc_col(k, &id);
  fsync_trigger(id, aggr);
  return fsync_wait();
}

static inline uint32_t fsync_barrier_global(void) {
  fsync_trigger(FSYNC_ID_H, fsync_aggr_global());
  return fsync_wait();
}

#endif // FSYNC_BASE
#endif // _FSYNC_H_
