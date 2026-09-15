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
// This runtime has ONE address space: link.ld puts `.bss` into `.l1`,
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

// First bank row of a level's arena. The levels occupy disjoint row ranges
// (see the allocator notes in vc.h), laid out LARGEST SCOPE FIRST: level k
// starts above the levels above it, not below it.
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
  // Under Morton the rectangle is a contiguous id range, so the base is one
  // flat mask.
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
// them (hardware/src/mempool_cluster_floonoc_wrapper.sv).
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

// The arithmetic twin of the hardware permuter: decompose the VC-local offset
// by division and reassemble the global address by addition. Under Morton the
// offset's group field is one 4^k-wide index.
uint32_t vc_global_addr(const vc_rect_t *r, uint32_t offset) {
  const uint32_t ngroups = r->ngroups;
  const uint32_t byte_in_row = offset % VC_ROW_BYTES;
  const uint32_t rest = offset / VC_ROW_BYTES;
  const uint32_t vc_g = rest % ngroups;
  const uint32_t row = rest / ngroups;

  return byte_in_row + (r->base_gid + vc_g) * VC_GY_STRIDE +
         row * VC_WORD_STRIDE;
}

vc_desc_t vc_descriptor(const vc_rect_t *r, uint32_t offset, uint32_t bytes) {
  // One whole rectangle bank row is contiguous globally.
  const uint32_t row = vc_row_stride(r->level);
  vc_desc_t d = {.base = vc_global_addr(r, offset)};
  // Only a whole level row gives a valid shape: a misaligned offset walks into
  // a neighbouring rectangle at the rows its own members use. There is no error
  // channel here, so refuse with a descriptor that moves nothing.
  if (offset % row != 0)
    return d;
  d.len = row;
  // Round up, as vc_malloc does, so a sub-row `bytes` still moves its row.
  d.reps = (bytes + row - 1) / row;
  d.stride = VC_WORD_STRIDE;
  return d;
}

// A refused request is still submitted, under an invalid scope: every core of
// the rectangle then counts it, as the collective requires, and every wait on
// it returns the argument error without entering a barrier.
static dma_collective_op_t vc_dma_invalid_op(void) {
  const dma_collective_scope_t none = {
      .level = UINT32_MAX, .base_group = 0u, .num_groups = 0u};
  return dma_collective_submit_2d_nonblocking(&none, 0u, NULL, NULL, 0u, 0u, 0u,
                                              0u);
}

// Morton makes a level-k rectangle a contiguous range of 4^k group ids.
static dma_collective_scope_t vc_dma_scope(const vc_desc_t *vc) {
  const uint32_t num_groups = vc->len / (uint32_t)VC_ROW_BYTES;
  const dma_collective_scope_t scope = {
      .level = (31u - (uint32_t)__builtin_clz(num_groups)) / 2u,
      .base_group = (vc->base % VC_WORD_STRIDE) / (uint32_t)VC_ROW_BYTES,
      .num_groups = num_groups,
  };
  return scope;
}

dma_collective_op_t vc_dma_memcpy_nonblocking(const vc_desc_t *dst,
                                              const vc_desc_t *src) {
  // A VC descriptor always has a row stride, even for one row; a flat one
  // never.
  const uint32_t dst_vc = dst->stride != 0u;
  const vc_desc_t *const vc = dst_vc ? dst : src;
  const vc_desc_t *const flat = dst_vc ? src : dst;
  // The collective walks vc->reps whole rectangle rows on BOTH sides, so a
  // shorter flat buffer would be overrun.
  if (!vc_desc_valid(dst) || !vc_desc_valid(src) || vc->stride == 0u ||
      flat->stride != 0u || vc->len % VC_ROW_BYTES != 0u ||
      flat->len / vc->len < vc->reps)
    return vc_dma_invalid_op();

  // A group outside the scope is refused, and counted, by the collective.
  const dma_collective_scope_t scope = vc_dma_scope(vc);
  // Each group moves one bank-row stripe; the flat side advances by the full
  // rectangle row and the VC side by the global word-plane stride.
  const uint32_t group_offset =
      (mempool_get_group_id() - scope.base_group) * (uint32_t)VC_ROW_BYTES;
  const uint32_t dst_stride = dst_vc ? dst->stride : vc->len;
  const uint32_t src_stride = dst_vc ? vc->len : src->stride;

  return dma_collective_submit_2d_nonblocking(
      &scope, 0u, (void *)(dst->base + group_offset),
      (const void *)(src->base + group_offset), VC_ROW_BYTES, dst_stride,
      src_stride, vc->reps);
}

uint32_t vc_dma_memcpy_blocking(const vc_desc_t *dst, const vc_desc_t *src) {
  const dma_collective_op_t op = vc_dma_memcpy_nonblocking(dst, src);
  const uint32_t error = dma_collective_wait(&op);
  return op.status != DMA_COLLECTIVE_OK ? op.status : error;
}

// ---------------------------------------------------------------------------
// Address-derived entry point
//
// Stride is a property of the ADDRESS SPACE, not of the call: a pointer into
// a level-k window already implies that level's row stride, its rectangle and
// therefore the set of groups involved. Under virtual addressing the caller
// cannot know the stride anyway -- it is exactly what the window hides. So
// these take only (dest, src, len) and derive the rest.
//
// Exactly one side must be a VC pointer. VC<->VC has no flat side to give the
// descriptor its counterpart stride, and flat<->flat is the ordinary global
// collective, which dma_collective_memcpy_* already does more cheaply.
// ---------------------------------------------------------------------------

// The level whose window contains `addr`, or VC_LEVELS if none does. The
// windows are disjoint, so the first match is the only one.
static uint32_t vc_level_of_addr(uint32_t addr) {
  for (uint32_t k = 0; k < (uint32_t)VC_LEVELS; ++k) {
    const uint32_t base = vc_window_base(k);
    const uint32_t size = vc_window_size(k);
    if (size != 0u && addr >= base && (addr - base) < size)
      return k;
  }
  return (uint32_t)VC_LEVELS;
}

// Build both descriptors from the two addresses. Returns 0 on success.
static uint32_t vc_dma_auto_descs(void *dest, const void *src, size_t len,
                                  vc_desc_t *dst_desc, vc_desc_t *src_desc) {
  const uint32_t d_addr = (uint32_t)dest;
  const uint32_t s_addr = (uint32_t)src;
  const uint32_t d_lvl = vc_level_of_addr(d_addr);
  const uint32_t s_lvl = vc_level_of_addr(s_addr);
  const uint32_t d_vc = d_lvl < (uint32_t)VC_LEVELS;
  const uint32_t s_vc = s_lvl < (uint32_t)VC_LEVELS;

  if (d_vc == s_vc || len == 0u)
    return DMA_COLLECTIVE_ERROR_ARGUMENT;

  const uint32_t level = d_vc ? d_lvl : s_lvl;
  const uint32_t vc_addr = d_vc ? d_addr : s_addr;
  const vc_rect_t rect = vc_rect(level);
  const vc_desc_t vd =
      vc_descriptor(&rect, vc_addr - vc_window_base(level), (uint32_t)len);
  if (!vc_desc_valid(&vd))
    return DMA_COLLECTIVE_ERROR_ARGUMENT;

  // The flat side must be OUTSIDE L1. Cross-group L1<->L1 needs the wide TCDM
  // plane (see vc.h), and a global-L1 flat pointer would otherwise be accepted
  // here and produce a transfer the engines cannot make.
  const uint32_t flat_addr = d_vc ? s_addr : d_addr;
  if (flat_addr < (uint32_t)NUM_BANKS * (uint32_t)L1_BANK_SIZE)
    return DMA_COLLECTIVE_ERROR_ARGUMENT;

  // The flat side is one contiguous run; the collective gives it the VC side's
  // rectangle-row stride, which is what makes VC-local offset o hold flat
  // byte o.
  const vc_desc_t fd = {
      .base = flat_addr, .len = (uint32_t)len, .reps = 1u, .stride = 0u};
  *dst_desc = d_vc ? vd : fd;
  *src_desc = d_vc ? fd : vd;
  return DMA_COLLECTIVE_OK;
}

dma_collective_op_t vc_dma_memcpy_auto_nonblocking(void *dest, const void *src,
                                                   size_t len) {
  vc_desc_t dst_desc, src_desc;
  const uint32_t error =
      vc_dma_auto_descs(dest, src, len, &dst_desc, &src_desc);
  if (error != DMA_COLLECTIVE_OK)
    return vc_dma_invalid_op();
  return vc_dma_memcpy_nonblocking(&dst_desc, &src_desc);
}

uint32_t vc_dma_memcpy_auto_blocking(void *dest, const void *src, size_t len) {
  const dma_collective_op_t op = vc_dma_memcpy_auto_nonblocking(dest, src, len);
  const uint32_t error = dma_collective_wait(&op);
  return op.status != DMA_COLLECTIVE_OK ? op.status : error;
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
