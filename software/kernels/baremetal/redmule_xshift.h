// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// RedMulE X contraction-start offset policy (REDMULE_XSHIFT, register index 18
// = REDMULE_REG_OFFS + 0x48, absolute 0x88; see archi_redmule.h for why not
// 0x18).
//
// Every engine walks the contraction from element 0, so after a barrier they all
// address the same L1 tile and drain through it one at a time. Starting each
// engine at a different element and wrapping fixes that: the set of accumulated
// products is unchanged, only the summation order.
//
// The offset is expressed in elements along N, but the useful granularity comes
// from the L1 address map. One RedMulE access covers PORT_WIDTH elements; one L1
// tile covers several accesses ("superbank slots"), and successive tiles then
// successive groups follow above that. So the shift space is two levels deep:
//
//     shift = (tile_slot * slots_per_tile + bank_slot) * PORT_WIDTH   [elements]
//
// Policy, in priority order:
//   1. Spend TILE offsets first -- two engines on different tiles never contend
//      for a tile port, whereas two engines on different superbanks of the same
//      tile only avoid a bank conflict.
//   2. Once every reachable tile is taken, engines that must share a tile are
//      separated in the BANK dimension instead.
//   3. Within each level, physically adjacent engines get maximally distant
//      offsets. The engine id's low bits are the tile within a group and the
//      next bits the neighbouring group, so a bit-reversal sends the closest
//      engines to opposite ends of the range.
//
// Reachable tiles depend on the workload: the contraction only spans
// n_size/PORT_WIDTH accesses, so a short contraction reaches fewer tiles than
// the machine has. Everything below is derived, nothing is hard-coded, so the
// policy follows the geometry and the workload without being re-tuned.

#ifndef __REDMULE_XSHIFT_H__
#define __REDMULE_XSHIFT_H__

#include <stdint.h>

#ifndef RMX_BANK_BYTES
#define RMX_BANK_BYTES 4 // L1 interleave granule (one bank word)
#endif

static inline uint32_t rmx_gcd(uint32_t a, uint32_t b) {
  while (b) {
    uint32_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

// Permutation of [0, m) that maps adjacent inputs to distant outputs. Exact
// bit-reversal when m is a power of two (maximal separation); otherwise a
// multiplier near m/2 that is coprime with m, which is still a permutation.
static inline uint32_t rmx_spread(uint32_t v, uint32_t m) {
  if (m < 2)
    return 0;
  v %= m;
  if ((m & (m - 1)) == 0) {
    uint32_t bits = 0;
    while ((1u << bits) < m)
      bits++;
    uint32_t r = 0;
    for (uint32_t b = 0; b < bits; b++) {
      r = (r << 1) | (v & 1);
      v >>= 1;
    }
    return r;
  }
  uint32_t c = m / 2 + 1;
  while (rmx_gcd(c, m) != 1)
    c++;
  return (v * c) % m;
}

// Contraction-start offset in ELEMENTS for one engine.
//
//   engine_id       dense RedMulE ordinal (0 .. n_engines-1)
//   n_size          contraction length in elements (the GEMM's N)
//   port_width      elements per RedMulE access = REDMULE_H * (REDMULE_P + 1)
//   elem_bytes      bytes per element (2 for fp16)
//   banks_per_tile  L1 banks in one tile   (NUM_BANKS_PER_TILE)
//   total_banks     L1 banks in the system (NUM_BANKS)
//
// Returns 0 when the workload leaves no room to shift, so callers can program
// the result unconditionally.
static inline uint16_t redmule_x_shift(uint32_t engine_id, uint32_t n_size,
                                       uint32_t port_width, uint32_t elem_bytes,
                                       uint32_t banks_per_tile,
                                       uint32_t total_banks) {
  const uint32_t access_bytes = port_width * elem_bytes;
  const uint32_t tile_bytes = banks_per_tile * RMX_BANK_BYTES;
  if (!access_bytes || tile_bytes < access_bytes)
    return 0;

  const uint32_t slots_per_tile = tile_bytes / access_bytes;
  const uint32_t chunks = n_size / port_width; // accesses in the contraction
  const uint32_t total_tiles = total_banks / banks_per_tile;
  if (chunks < 2)
    return 0;

  // Tiles this contraction can actually reach, capped by the interleave period.
  uint32_t tiles = chunks / slots_per_tile;
  if (tiles > total_tiles)
    tiles = total_tiles;

  uint32_t accesses;
  if (tiles >= 2) {
    // Level 1: a distinct tile per engine, bit-reversed for separation.
    // Level 2: engines beyond the tile count share a tile and separate by
    // superbank. Sharers are `tiles` apart in id, i.e. physically far apart.
    const uint32_t tile_slot = rmx_spread(engine_id % tiles, tiles);
    const uint32_t bank_slot =
        rmx_spread((engine_id / tiles) % slots_per_tile, slots_per_tile);
    accesses = tile_slot * slots_per_tile + bank_slot;
  } else {
    // The contraction is shorter than one tile: only bank offsets exist.
    accesses = rmx_spread(engine_id % chunks, chunks);
  }
  return (uint16_t)(accesses * port_width);
}

// W column-start offset in ELEMENTS, for REDMULE_MCFG1[31:16] (redmule_cfg's
// w_col_off). Identical policy to redmule_x_shift, applied to the OUTPUT COLUMN
// axis instead of the contraction: the engine runs [w_col_off, k_size) then
// wraps to [0, w_col_off). Because output columns are independent, this one is
// numerically EXACT -- unlike the X shift, it reorders nothing that is summed.
//
//   k_size   this engine's output-column count (cols_per_engine)
//
// The two axes decorrelate on their own: the X grouping is id % tiles_X and the
// W grouping id % tiles_W, and those capacities differ, so a pair sharing an X
// slot generally does not share a W slot. No extra seeding is needed.
//
// Note the reachable range is usually small: an engine's whole column span is
// k_size elements, which in a typical config is one L1 tile, so this resolves to
// a superbank offset rather than a tile offset. That is still useful -- a tile
// serves its superbanks concurrently -- but it is weaker than the X shift.
static inline uint16_t redmule_w_shift(uint32_t engine_id, uint32_t k_size,
                                       uint32_t port_width, uint32_t elem_bytes,
                                       uint32_t banks_per_tile,
                                       uint32_t total_banks) {
  return redmule_x_shift(engine_id, k_size, port_width, elem_bytes,
                         banks_per_tile, total_banks);
}

#endif // __REDMULE_XSHIFT_H__
