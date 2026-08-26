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

// A virtual-cluster level k spans 2^k GROUPS per axis, hence 2^k * tiles per
// group along each axis. This is invariant #9: a memory-side group level maps
// to a barrier-side TILE level one log2(tiles-per-group-side) higher.
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

static inline uint32_t fsync_barrier_global(void) {
  fsync_trigger(FSYNC_ID_H, fsync_aggr_global());
  return fsync_wait();
}

#endif // FSYNC_BASE
#endif // _FSYNC_H_
