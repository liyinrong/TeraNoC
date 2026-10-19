// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Virtual clusters: level and rectangle descriptors, the (level, offset) ->
// global block-strided translation, and the per-level arena allocator.
//
// The address map is computed ONCE in config/config.mk and reaches this file
// through software/runtime/runtime.mk; the RTL gets the identical numbers
// through hardware/Makefile's vlog_defs and re-derives them, failing
// elaboration on a mismatch. Never write a window constant here.

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
// compiled into the runtime unconditionally (runtime.mk) and must still build;
// every other includer is an application, and an application that wants VC
// memory needs the hardware to be there.
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
// must start above them; the hardware does not separate the levels.
//
// The floor is taken from `__heap_start`, not from NUM_CORES*SEQ_MEM_SIZE:
// link.ld places `.l1_prio`, `.l1` and `.bss` above the sequential region and
// below __heap_start, and those rows are just as clobberable. It keeps covering
// them when a kernel adds a static.
//
// One global row-plane spans every bank of every group, so a global address
// `a` lives in row a / (VC_ROW_BYTES * NUM_GROUPS).
#define VC_GLOBAL_ROW_BYTES ((uint32_t)VC_ROW_BYTES * NUM_GROUPS)
// Where cores see the global map: 0, or a domain's top-level window
// (addrmap.h). Row r of the map is at VC_GLOBAL_BASE + r * VC_GLOBAL_ROW_BYTES
// for cores; a global alias (VC_GLOBAL_BASE excluded) is what the DMA takes.
#define VC_GLOBAL_BASE DOMAIN_L1_DATA_BASE
#define VC_GLOBAL_ROW(a) (((uint32_t)(a)-VC_GLOBAL_BASE) / VC_GLOBAL_ROW_BYTES)

extern uint32_t __heap_start;

// Rows reserved for linker-placed data. Runtime value, not a constant,
// because __heap_start is a link-time symbol.
static inline uint32_t vc_floor_rows(void) {
  // In a domain build the heap is in the domain's top-level window.
  const uint32_t heap = (uint32_t)&__heap_start - VC_GLOBAL_BASE;
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
// Global block-strided descriptor
//
// A VC-contiguous buffer is NOT contiguous globally. Under Morton group ids a
// whole bank row of a rectangle is one contiguous global run, so the global
// shape is 2-D: `len` bytes per bank row, repeated `reps` times at the global
// word-plane `stride`. A flat L2 buffer is the same struct with reps = 1 and
// stride = 0.
// ---------------------------------------------------------------------------

typedef struct {
  uint32_t base;   // global byte address of the first element
  uint32_t len;    // contiguous bytes per step
  uint32_t reps;   // steps: bank rows for a VC descriptor, 1 for a flat one
  uint32_t stride; // bytes between steps; 0 marks a flat descriptor
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
// by the permuter's bit tricks. This is the DMA / cross-VC view; vc_smoke
// checks that both resolve to the same word.
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
  return d->len != 0 && d->reps != 0;
}

// One L2<->VC transfer over the caller's rectangle. Every core of the
// rectangle calls, in the same order; each group's coordinator issues its own
// stripe (dma_collective.h). Complete the returned op with dma_collective_wait
// or dma_collective_wait_ready.
//
// Both descriptors use global aliases. One side must be a VC descriptor and
// the other flat L2 memory at least as long as the VC side's whole rows.
// Cross-group L1<->L1 needs the wide TCDM plane. A refused request still counts
// as an operation (dma_collective.h), and waiting it returns the error.
dma_collective_op_t vc_dma_memcpy_nonblocking(const vc_desc_t *dst,
                                              const vc_desc_t *src);
uint32_t vc_dma_memcpy_blocking(const vc_desc_t *dst, const vc_desc_t *src);

// Address-derived forms. Stride, rectangle and group scope all follow from
// the addresses, so the caller passes only (dest, src, len); exactly one side
// must point into a VC window, and `len` must be whole rows of its level.
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

// --- tile window -----------------------------------------------------------
//
// One tile's L1 at VC_TILE_WINDOW_BASE, right after the level-0 window. The
// scramblers re-place an offset into the requester's OWN tile, so every tile
// sees a private window over the same offsets. An offset is [bank | row]: the
// tile's slice of a global row plane, so row r is global row r and the rows
// below vc_floor_rows() belong to the linker here too. Like every VC window it
// is compute-only; the DMA takes the global alias.

#define VC_TILE_ROW_BYTES (VC_ROW_BYTES / NUM_TILES_PER_GROUP)

static inline void *vc_tile_ptr(uint32_t offset) {
  return (void *)(VC_TILE_WINDOW_BASE + offset);
}

// The global alias of a tile-window offset as seen by `tile` (a flat tile id).
static inline uint32_t vc_tile_global_addr(uint32_t tile, uint32_t offset) {
  return (offset / VC_TILE_ROW_BYTES) * VC_GLOBAL_ROW_BYTES +
         tile * VC_TILE_ROW_BYTES + offset % VC_TILE_ROW_BYTES;
}

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
