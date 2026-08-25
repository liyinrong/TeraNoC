// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#define VC_IMPL 1 // the library itself builds on any config; see vc.h
#include "vc.h"

#include "dma.h"
#include "runtime.h"

// ---------------------------------------------------------------------------
// Window table, from config/config.mk via runtime.mk. A level that this
// geometry does not have keeps base 0 / size 0, and vc_reserve rejects it.
// ---------------------------------------------------------------------------

const uint32_t vc_window_base_tab[VC_MAX_LEVELS] = {
#ifdef VC_WINDOW_BASE_0
    VC_WINDOW_BASE_0,
#else
    0,
#endif
#ifdef VC_WINDOW_BASE_1
    VC_WINDOW_BASE_1,
#else
    0,
#endif
#ifdef VC_WINDOW_BASE_2
    VC_WINDOW_BASE_2,
#else
    0,
#endif
#ifdef VC_WINDOW_BASE_3
    VC_WINDOW_BASE_3,
#else
    0,
#endif
};

const uint32_t vc_window_size_tab[VC_MAX_LEVELS] = {
#ifdef VC_WINDOW_SIZE_0
    VC_WINDOW_SIZE_0,
#else
    0,
#endif
#ifdef VC_WINDOW_SIZE_1
    VC_WINDOW_SIZE_1,
#else
    0,
#endif
#ifdef VC_WINDOW_SIZE_2
    VC_WINDOW_SIZE_2,
#else
    0,
#endif
#ifdef VC_WINDOW_SIZE_3
    VC_WINDOW_SIZE_3,
#else
    0,
#endif
};

// ---------------------------------------------------------------------------
// Allocator state, INDEXED BY CORE.
//
// This runtime has ONE address space: link.ld:20-25 puts `.bss` into `.l1`,
// so a plain file-scope `static` is a single object shared by every core, not
// a per-core variable. The allocator is collective and deterministic -- each
// core independently computes the same offsets -- which only works if each
// core has its OWN bump pointer. A shared one is incremented once per core
// and hands every core a different offset, which is precisely the bug this
// indexing fixes.
//
// Cost is 2 * VC_MAX_LEVELS * 4 bytes per core: 32 B/core, 32 KiB at
// terapool, under 1% of L1. `vc_rect_t` is NOT stored per core -- it is a
// pure function of the core's group id, so it is recomputed on demand and
// costs nothing.
// ---------------------------------------------------------------------------

static uint32_t vc_used_rows[NUM_CORES][VC_MAX_LEVELS]
    __attribute__((section(".l1")));
static uint32_t vc_reserved_rows[NUM_CORES][VC_MAX_LEVELS]
    __attribute__((section(".l1")));

// First bank row of a level's arena.
//
// The levels must occupy DISJOINT row ranges. The bank-row index is one axis
// shared by every level -- a level-k allocation occupies the same row indices
// in all 4^k groups of its rectangle -- so if two levels started at the same
// row their buffers would alias on whatever group the rectangles share. That
// is what the contract's capacity rule means by "the SUM of per-level usage":
// sum, not max. Reading "an independent bump pointer" as "each rewinds to the
// floor" gives overlapping arenas, which the coexistence check catches as a
// level-1 write showing up under a level-0 pointer.
// Lay the row axis out LARGEST SCOPE FIRST: level k starts above the levels
// above it, not below it.
//
// The direction is load-bearing. Agreement is required among the members of one
// rectangle at one level, and a level-k rectangle sits entirely INSIDE every
// level-j rectangle for j > k, so its members necessarily agree on the higher
// levels' reservations. They may legitimately differ on the LOWER ones -- that
// is exactly the heterogeneity the design allows, some rectangles using a level
// while others give the space to smaller private scopes. Summing the lower
// levels therefore let two members of one rectangle compute different bases for
// the same logical allocation, and put one member's buffer on top of another
// member's live data at a smaller scope. Summing the higher levels is uniform
// by construction.
static uint32_t vc_base_row(uint32_t c, uint32_t k) {
  uint32_t row = vc_floor_rows();
  for (uint32_t j = k + 1; j < VC_MAX_LEVELS; ++j) {
    row += vc_reserved_rows[c][j];
  }
  return row;
}

// Recomputed rather than stored; see above.
// De-interleave a Morton group id. Must mirror mempool_pkg::group_xy_of; both
// directions of the encoding live in exactly two places, one per language.
static void vc_unmorton(uint32_t gid, uint32_t *gx, uint32_t *gy) {
  uint32_t x = 0, y = 0;
  for (uint32_t i = 0; (1u << i) < (uint32_t)NUM_X; ++i) {
    y |= ((gid >> (2 * i)) & 1u) << i;
    x |= ((gid >> (2 * i + 1)) & 1u) << i;
  }
  *gx = x;
  *gy = y;
}

static vc_rect_t vc_rect_of(uint32_t k) {
  const uint32_t gid = mempool_get_group_id();
  uint32_t gx, gy;
  vc_unmorton(gid, &gx, &gy);
  vc_rect_t r;
  r.level = k;
  r.side = 1u << k;
  r.ngroups = r.side * r.side;
  // The whole point of Morton: the rectangle is a contiguous id range, so the
  // base is ONE flat mask rather than a per-axis one (the old INVARIANT #4).
  r.base_gid = gid & ~(r.ngroups - 1u);
  r.rx = gx >> k;
  r.ry = gy >> k;
  r.base_gx = r.rx << k;
  r.base_gy = r.ry << k;
  r.id = r.rx * ((uint32_t)NUM_Y >> k) + r.ry;
  return r;
}

// Global byte strides of the canonical layout. gy is the LOW half of the
// group field, so one gy step is one group's L1 and one gx step is NUM_Y of
// them (hardware/src/mempool_cluster_floonoc_wrapper.sv:249).
#define VC_GY_STRIDE ((uint32_t)VC_ROW_BYTES)
#define VC_GX_STRIDE ((uint32_t)VC_ROW_BYTES * NUM_Y)
#define VC_WORD_STRIDE ((uint32_t)VC_ROW_BYTES * NUM_X * NUM_Y)

void vc_init(void) {
  const uint32_t c = mempool_get_core_id();
  for (uint32_t k = 0; k < VC_MAX_LEVELS; ++k) {
    vc_used_rows[c][k] = 0;
    vc_reserved_rows[c][k] = 0;
  }
}

// Returned by value: there is no per-core table to point into.
vc_rect_t vc_rect(uint32_t level) { return vc_rect_of(level); }

// ---------------------------------------------------------------------------
// Translation
// ---------------------------------------------------------------------------

// The arithmetic twin of the hardware permuter. Decomposes the VC-local
// offset by division rather than by masking and reassembles the global
// address by addition, which is exactly the property invariant #4 exists to
// guarantee: with a per-axis base mask, OR equals +.
uint32_t vc_global_addr(const vc_rect_t *r, uint32_t offset) {
  // Morton group ids: the rectangle is a contiguous id range, so the offset's
  // group field is ONE 4^k-wide index and the base is one flat mask. The
  // per-axis split this used to do only existed because the raster id made a
  // rectangle discontiguous.
  const uint32_t ngroups = r->ngroups;
  const uint32_t byte_in_row = offset % VC_ROW_BYTES;
  const uint32_t rest = offset / VC_ROW_BYTES;
  const uint32_t vc_g = rest % ngroups;
  const uint32_t row = rest / ngroups;

  return byte_in_row + (r->base_gid + vc_g) * VC_GY_STRIDE +
         row * VC_WORD_STRIDE;
}

vc_desc_t vc_descriptor(const vc_rect_t *r, uint32_t offset, uint32_t bytes) {
  vc_desc_t d;
  const uint32_t side = r->side;
  // dim0 spans one bank row of one group TIMES the whole vc_gy field, since
  // the y-adjacent groups are contiguous in the global alias.
  // Under Morton a whole rectangle bank row is CONTIGUOUS, so dim0 is the
  // entire row and the gx dimension disappears: the descriptor is 2-D.
  const uint32_t inner = (uint32_t)VC_ROW_BYTES * side * side;
  const uint32_t stride = inner; // == vc_row_stride(r->level)
  // Round UP, as vc_malloc does. Truncating here made the natural idiom
  // p = vc_malloc(k,n); d = vc_descriptor(&r, off, n) under-copy, and a
  // sub-row `bytes` produced count[1] == 0, a descriptor that silently moves
  // nothing.
  const uint32_t rows = (bytes + stride - 1) / stride;

  // The shape is only valid at a whole level row. A misaligned offset does not
  // merely shift the copy, it walks into a NEIGHBOURING rectangle at the same
  // row indices its own members use. There is no error channel here, so refuse
  // by returning a descriptor that moves nothing -- vc_desc_valid() reports it.
  if (offset % stride != 0) {
    d.base = vc_global_addr(r, offset);
    d.len = 0;
    d.count[0] = 0;
    d.count[1] = 0;
    d.stride[0] = 0;
    d.stride[1] = 0;
    return d;
  }

  d.base = vc_global_addr(r, offset);
  d.len = inner;  // one whole rectangle row, contiguous
  d.count[0] = 1; // no gx dimension under Morton
  d.stride[0] = 0;
  d.count[1] = rows; // one step per bank row
  d.stride[1] = VC_WORD_STRIDE;
  return d;
}

void vc_dma_memcpy_blocking(const vc_desc_t *dst, const vc_desc_t *src) {
  // PRE-WP4: the frontend is 1-D, so walk the block-strided pattern as
  // count[0]*count[1] separate transfers. See the header for why this is a
  // bring-up path and not a benchmark path.
  // Walk the side that is actually block-strided. Looping on the SOURCE shape
  // alone silently discards the destination's: an L2 source is flat
  // (count {1,1}), so a strided VC destination collapsed into ONE contiguous
  // copy that landed in the wrong physical words and spilled across the rows
  // of neighbouring rectangles. Observed as a single 327680-byte transfer
  // where 160 runs of 2048 were required.
  const uint32_t d_runs = dst->count[0] * dst->count[1];
  const uint32_t s_runs = src->count[0] * src->count[1];

  if (s_runs == 1) { // flat source, strided destination
    for (uint32_t j = 0; j < dst->count[1]; ++j)
      for (uint32_t i = 0; i < dst->count[0]; ++i) {
        const uint32_t run = j * dst->count[0] + i;
        dma_memcpy_blocking(
            (void *)(dst->base + i * dst->stride[0] + j * dst->stride[1]),
            (void *)(src->base + run * dst->len), dst->len);
      }
  } else if (d_runs == 1) { // strided source, flat destination
    for (uint32_t j = 0; j < src->count[1]; ++j)
      for (uint32_t i = 0; i < src->count[0]; ++i) {
        const uint32_t run = j * src->count[0] + i;
        dma_memcpy_blocking(
            (void *)(dst->base + run * src->len),
            (void *)(src->base + i * src->stride[0] + j * src->stride[1]),
            src->len);
      }
  } else { // both strided: the shapes must agree run for run
    for (uint32_t j = 0; j < dst->count[1]; ++j)
      for (uint32_t i = 0; i < dst->count[0]; ++i)
        dma_memcpy_blocking(
            (void *)(dst->base + i * dst->stride[0] + j * dst->stride[1]),
            (void *)(src->base + i * src->stride[0] + j * src->stride[1]),
            dst->len);
  }
}

// ---------------------------------------------------------------------------
// Allocator
// ---------------------------------------------------------------------------

uint32_t vc_rows_committed(void) {
  const uint32_t c = mempool_get_core_id();
  uint32_t rows = vc_floor_rows();
  for (uint32_t k = 0; k < VC_MAX_LEVELS; ++k) {
    rows += vc_reserved_rows[c][k];
  }
  return rows;
}

int32_t vc_reserve(uint32_t level, uint32_t bytes) {
  if (level >= (uint32_t)VC_LEVELS) {
    return -1; // no window at this level for this geometry
  }
  const uint32_t stride = vc_row_stride(level);
  const uint32_t rows = (bytes + stride - 1) / stride;

  // LOCAL capacity check. The budget is per group and in bank rows, because
  // rows are the axis every level shares: a level-k allocation of `rows` rows
  // consumes exactly `rows` rows of EVERY group in its rectangle. Agreement
  // is required only inside one rectangle at one level, so this check is
  // deliberately local -- it never consults another group.
  const uint32_t c = mempool_get_core_id();

  // Reservation is per PHASE, before any allocation: a level's arena base is
  // the sum of the lower levels' reservations, so re-reserving after handing
  // out pointers would silently move arenas out from under them. Refuse it.
  for (uint32_t k = 0; k < VC_MAX_LEVELS; ++k) {
    if (vc_used_rows[c][k] != 0) {
      return -1;
    }
  }

  uint32_t others = vc_floor_rows();
  for (uint32_t k = 0; k < VC_MAX_LEVELS; ++k) {
    if (k != level) {
      others += vc_reserved_rows[c][k];
    }
  }
  if (others + rows > (uint32_t)VC_WORDS_PER_BANK) {
    return -1;
  }

  vc_reserved_rows[c][level] = rows;
  vc_used_rows[c][level] = 0;
  return 0;
}

void vc_arena_reset(uint32_t level) {
  vc_used_rows[mempool_get_core_id()][level] = 0;
}

void *vc_malloc(uint32_t level, uint32_t bytes) {
  if (level >= (uint32_t)VC_LEVELS) {
    return (void *)0;
  }
  const uint32_t stride = vc_row_stride(level);
  const uint32_t rows = (bytes + stride - 1) / stride;

  const uint32_t c = mempool_get_core_id();
  if (vc_used_rows[c][level] + rows > vc_reserved_rows[c][level]) {
    return (void *)0; // reservation exhausted
  }
  const uint32_t off =
      (vc_base_row(c, level) + vc_used_rows[c][level]) * stride;
  vc_used_rows[c][level] += rows;
  return vc_ptr(level, off);
}
