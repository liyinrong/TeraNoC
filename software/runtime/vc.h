// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// WP3 virtual clusters: level and rectangle descriptors, the (level, id,
// offset) -> global ND block-strided translation, and the per-level arena
// allocator.
//
// The address map is computed ONCE in config/config.mk and reaches this file
// through software/runtime/runtime.mk; the RTL gets the identical numbers
// through hardware/Makefile's vlog_defs and re-derives them, failing
// elaboration on a mismatch. Never write a window constant here.
//
// Spec: docs/plan/wp2-wp4-implementation-roadmap.md §T2.1 (permuter) and
// §T2.2 (map, translation table, vc_malloc), and the pinned contract note
// "WP3 Address Map and Scrambler Contract".

#ifndef _VC_H_
#define _VC_H_

#include <stddef.h>
#include <stdint.h>

#include "dma_collective.h"
#include "runtime.h"

// VIRTUAL_CLUSTER gates the HARDWARE permuter. Without this guard the software
// API stayed fully functional on a build whose permuter was compiled out, so a
// VC pointer was just an address the cluster does not translate -- it leaves
// with no decode error and no fault. vc.c defines VC_IMPL because it is
// compiled into the runtime unconditionally (runtime.mk:211) and must still
// build; every other includer is an application, and an application that wants
// VC memory needs the hardware to be there.
#if !defined(VC_IMPL) && (!defined(VIRTUAL_CLUSTER) || !VIRTUAL_CLUSTER)
#error "vc.h needs virtual_cluster=1: the VC permuter is not in this hardware"
#endif

#ifndef VC_LEVELS
#error "VC_LEVELS undefined: the runtime must be built through runtime.mk"
#endif

// Levels run k = 0 .. VC_LEVELS-1. The TOP level k = log2(min(NUM_X,NUM_Y))
// deliberately has NO window: on a square mesh its permuter is the identity,
// so the global region already is that window. A "top level" allocation is
// therefore just a normal global pointer.
#define VC_MAX_LEVELS 4

// ---------------------------------------------------------------------------
// Derived geometry. Everything below is a compile-time constant folded by the
// preprocessor, so none of it costs anything at run time.
// ---------------------------------------------------------------------------

// Words per bank == bank rows. This is the axis every level partitions.
#define VC_WORDS_PER_BANK (L1_BANK_SIZE / 4)
// Banks in one group, and the bytes one bank row of one group spans.
#define VC_BANKS_PER_GROUP (VC_GROUP_BYTES / L1_BANK_SIZE)
#define VC_ROW_BYTES (VC_BANKS_PER_GROUP * 4)

// Bank rows the LINKER already owns at the bottom of L1. A level's window
// maps ALL bank rows of its rectangle, including these, so every allocator
// must start above them -- that separation is the allocator's job now that
// S12 removed chunk_base and with it the hardware-enforced split between
// levels.
//
// This is deliberately taken from `__heap_start` and not from
// NUM_CORES*SEQ_MEM_SIZE. The contract's [allocator] observation names only
// "the stack floor ... stacks and the sequential region", but link.ld:20-25
// places `.l1_prio`, `.l1` and `.bss` ABOVE __seq_end and below __heap_start,
// and those rows are just as clobberable. Deriving the floor from
// __heap_start covers all of it and keeps covering it when a kernel adds a
// static.
//
// One global row-plane spans every bank of every group, so a global address
// `a` lives in row a / (VC_ROW_BYTES * NUM_GROUPS).
#define VC_GLOBAL_ROW_BYTES ((uint32_t)VC_ROW_BYTES * NUM_GROUPS)

extern uint32_t __heap_start;

// Rows reserved for linker-placed data. Runtime value, not a constant,
// because __heap_start is a link-time symbol.
static inline uint32_t vc_floor_rows(void) {
  const uint32_t heap = (uint32_t)&__heap_start;
  return (heap + VC_GLOBAL_ROW_BYTES - 1) / VC_GLOBAL_ROW_BYTES;
}

// One bank row of a level-k rectangle, i.e. the byte distance between two
// consecutive rows in the VC-local offset space.
static inline uint32_t vc_row_stride(uint32_t k) {
  return ((uint32_t)VC_ROW_BYTES) << (2 * k);
}

// ---------------------------------------------------------------------------
// Rectangle descriptor
// ---------------------------------------------------------------------------

typedef struct {
  uint32_t level;    // k
  uint32_t side;     // 2^k groups per axis
  uint32_t base_gid; // rectangle origin as a flat Morton id, = gid & ~(4^k-1)
  uint32_t base_gx;  // the same origin in coordinates, for humans and the map
  uint32_t base_gy;
  uint32_t rx; // rectangle-grid coordinate, = gx >> k
  uint32_t ry;
  uint32_t id;      // flat rectangle id, rx * (NUM_Y >> k) + ry
  uint32_t ngroups; // side * side
} vc_rect_t;

// ---------------------------------------------------------------------------
// Global ND block-strided descriptor
//
// A VC-contiguous buffer is NOT contiguous globally; its global shape is a
// 3-D block-strided pattern, and that pattern is what a DMA must walk. The
// dimensions fall straight out of the canonical field order
// [ byte | bank | tile | gy | gx | word ]:
//   dim0  contiguous, and it spans the whole vc_gy field as well, because gy
//         is the LOW half of the group field, so the y-adjacent groups of a
//         rectangle are already adjacent in the global alias
//   dim1  one step per vc_gx
//   dim2  one step per bank row
// Track A checked this descriptor against the permuter on every probe (S8).
// ---------------------------------------------------------------------------

typedef struct {
  uint32_t base;      // global byte address of the first element
  uint32_t len;       // dim0, contiguous bytes
  uint32_t count[2];  // dim1, dim2 repeat counts
  uint32_t stride[2]; // dim1, dim2 byte strides
} vc_desc_t;

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

// Per-core init: rewinds this core's bump pointers to the linker floor. Call
// once after mempool_barrier_init and before any other vc_* call.
void vc_init(void);

// This core's rectangle at `level`. Returned by value -- it is a pure
// function of the core's group id, so nothing is stored per core.
vc_rect_t vc_rect(uint32_t level);

// Base address of a level's window. Offsets into it are VC-local: the
// hardware permuter turns them into the requester's own rectangle.
static inline uint32_t vc_window_base(uint32_t level);
static inline uint32_t vc_window_size(uint32_t level);

// A VC-local pointer. Every member of the rectangle passing the same
// (level, offset) addresses the SAME physical words.
static inline void *vc_ptr(uint32_t level, uint32_t offset) {
  return (void *)(vc_window_base(level) + offset);
}

// The global alias of a VC-local offset, computed by arithmetic rather than
// by the permuter's bit tricks. This is the DMA / cross-VC view, and gate
// item 2 asserts the two resolve to the same physical bank.
uint32_t vc_global_addr(const vc_rect_t *r, uint32_t offset);

// The global ND descriptor for `bytes` starting at a VC-local `offset`.
// `offset` must be a multiple of the FULL vc_row_stride(level), not of
// vc_row_stride(level)/side: a dim0-aligned but not row-aligned offset builds a
// shape that walks into a neighbouring rectangle. `bytes` rounds up to whole
// rows, matching vc_malloc. A misaligned offset yields a descriptor that moves
// nothing; check it with vc_desc_valid().
vc_desc_t vc_descriptor(const vc_rect_t *r, uint32_t offset, uint32_t bytes);

// False when vc_descriptor refused the arguments.
static inline uint32_t vc_desc_valid(const vc_desc_t *d) {
  return d->len != 0 && d->count[0] != 0 && d->count[1] != 0;
}

// One L2<->VC transfer over the caller's rectangle. Every core of the
// rectangle calls, in the same order; each group's coordinator issues its own
// stripe (dma_collective.h). Complete the returned op with dma_collective_wait
// or dma_collective_wait_ready.
//
// Both descriptors use global aliases. One side must be a VC descriptor and
// the other flat L2 memory. Cross-group L1<->L1 still needs the WP2 wide plane.
dma_collective_op_t vc_dma_memcpy_nonblocking(const vc_desc_t *dst,
                                              const vc_desc_t *src);
uint32_t vc_dma_memcpy_blocking(const vc_desc_t *dst, const vc_desc_t *src);

// Address-derived forms. Stride, rectangle and group scope all follow from
// the addresses, so the caller passes only (dest, src, len); exactly one side
// must point into a VC window.
dma_collective_op_t vc_dma_memcpy_auto_nonblocking(void *dest, const void *src,
                                                   size_t len);
uint32_t vc_dma_memcpy_auto_blocking(void *dest, const void *src, size_t len);

// --- allocator -------------------------------------------------------------
//
// Per group ONE arena; per level an INDEPENDENT bump pointer, so a level-k
// allocation moves only level k's pointer and every member of that rectangle
// lands on the same rows regardless of what it does at other levels.
//
// The levels' arenas are DISJOINT ranges of the bank-row axis, stacked above
// the linker floor in level order. They have to be: a level-k allocation
// occupies the same row indices in all 4^k groups of its rectangle, so two
// levels sharing a row alias on whatever group their rectangles share. That
// is what the capacity rule means by the SUM of per-level usage. (Reading
// "independent bump pointer" as "each rewinds to the floor" instead produces
// exactly that aliasing, and the coexistence check catches it as a level-1
// write appearing under a level-0 pointer.)
//
// Consequence: reserve every level BEFORE allocating from any of them. A
// level's base depends on the lower levels' reservations, so a reservation
// made after pointers are live would move arenas underneath them;
// vc_reserve refuses that rather than aliasing silently.
//
// Agreement is required among the members of ONE rectangle at ONE level, not
// across the mesh: two different rectangles at the same level need not agree,
// and a group with no allocation at a level constrains nobody. That is what
// makes heterogeneous configurations legal.

// Collective, once per phase, by every member of the rectangle. Sets the
// level's row budget and rewinds its bump pointer. Returns 0 on success, -1
// if the local capacity check fails OR if any level already has live
// allocations (see the ordering note above).
int32_t vc_reserve(uint32_t level, uint32_t bytes);

// Bump-allocate inside the level's reservation. Deterministic and lock-free:
// every member of the rectangle issuing the same call sequence gets the same
// offset. Returns NULL when the reservation is exhausted.
void *vc_malloc(uint32_t level, uint32_t bytes);

// Rewind one level's bump pointer, keeping its reservation.
void vc_arena_reset(uint32_t level);

// Bank rows currently reserved across all levels, plus the stack floor. The
// local capacity invariant is that this never exceeds VC_WORDS_PER_BANK.
uint32_t vc_rows_committed(void);

// ---------------------------------------------------------------------------
// Window table. Built from the config defines so there is exactly one source.
// ---------------------------------------------------------------------------

extern const uint32_t vc_window_base_tab[VC_MAX_LEVELS];
extern const uint32_t vc_window_size_tab[VC_MAX_LEVELS];

static inline uint32_t vc_window_base(uint32_t level) {
  return vc_window_base_tab[level];
}
static inline uint32_t vc_window_size(uint32_t level) {
  return vc_window_size_tab[level];
}

#endif // _VC_H_
