// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// WP3 virtual-cluster gate, software half.
//
// Covers gate items 2 and the "all levels coexisting" tail of
// docs/plan/wp2-wp4-implementation-roadmap.md §T2.3:
//   A  alias equivalence  - a word written through the VC-local pointer reads
//      back through the global alias, for every level and every offset shape.
//   B  descriptor agreement - the DMA global ND descriptor and the compute
//      pointer resolve to the same physical address (Track A finding S8).
//   C  all levels coexisting - one buffer per level, live at the same time,
//      content-verified after every level has been written.
//   D  allocator - the local capacity check, the stack floor, and per-level
//      independence of the bump pointers.
//
// Gate item 1 (bijection) is the standalone replay in
// hardware/tb/tb_vc_scrambler.sv; items 4 and 5 are elaboration asserts in
// hardware/src/mempool_addr_scrambler.sv.

#include <stdint.h>
#include <string.h>

#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"
#include "vc.h"

// Per core: `.bss` lives in the shared `.l1` (link.ld:20-25), so a plain
// static would be one counter raced by every core.
static uint32_t errors[NUM_CORES] __attribute__((section(".l1")));

static void check(uint32_t cond, const char *what, uint32_t a, uint32_t b) {
  if (!cond) {
    const uint32_t c = mempool_get_core_id();
    errors[c]++;
    // Only core 0 prints, so a failure storm cannot interleave 1024 printfs.
    if (c == 0 && errors[c] <= 8) {
      printf("[VC] FAIL %s: %08x vs %08x\n", what, a, b);
    }
  }
}

// A word this core owns at (level, offset), distinct per group so a collision
// between rectangles shows up as wrong content rather than as a silent pass.
// Sections A-C write and read each group's OWN slot, so folding the group id
// in is what proves no other group clobbered it. The cross-group sweep is the
// opposite case -- one core writes, every member reads -- so it needs a value
// that does not depend on who is looking.
static uint32_t shared_pattern(uint32_t level, uint32_t offset) {
  return 0x5C000000u ^ (level << 24) ^ (offset >> 2);
}

static uint32_t pattern(uint32_t level, uint32_t offset) {
  return 0xA5000000u ^ (level << 24) ^ (mempool_get_group_id() << 16) ^
         (offset >> 2);
}

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();

  mempool_barrier_init(core_id);
  errors[core_id] = 0;
  vc_init();
  mempool_barrier(num_cores);

  if (core_id == 0) {
    printf("[VC] levels=%d group_bytes=%d row_bytes=%d floor_rows=%d "
           "words_per_bank=%d\n",
           VC_LEVELS, VC_GROUP_BYTES, VC_ROW_BYTES, vc_floor_rows(),
           VC_WORDS_PER_BANK);
  }

  // ---- D: allocator ------------------------------------------------------
  // Reserve one row per level. Every member of a rectangle runs the identical
  // sequence, so the offsets agree without a lock.
  for (uint32_t k = 0; k < (uint32_t)VC_LEVELS; k++) {
    check(vc_reserve(k, vc_row_stride(k)) == 0, "reserve", k, 0);
  }
  // The stack floor must be respected: the first allocation of any level
  // starts at or above row VC_STACK_ROWS.
  for (uint32_t k = 0; k < (uint32_t)VC_LEVELS; k++) {
    uint32_t off = (uint32_t)vc_malloc(k, 4) - vc_window_base(k);
    check(off >= vc_floor_rows() * vc_row_stride(k), "linker floor", off,
          vc_floor_rows() * vc_row_stride(k));
    vc_arena_reset(k);
  }
  // An over-large reservation must be refused, not silently truncated.
  check(vc_reserve(0, (uint32_t)VC_WORDS_PER_BANK * vc_row_stride(0)) == -1,
        "oversize reserve refused", 0, 0);
  check(vc_reserve(0, vc_row_stride(0)) == 0, "reserve after refusal", 0, 0);
  check(vc_rows_committed() <= (uint32_t)VC_WORDS_PER_BANK, "capacity",
        vc_rows_committed(), (uint32_t)VC_WORDS_PER_BANK);

  mempool_barrier(num_cores);

  // ---- A + C: write every level's buffer through the VC-local pointer ----
  // One core per group does the writing, so a collision between two groups of
  // the same rectangle is visible as wrong content.
  volatile uint32_t *buf[VC_MAX_LEVELS];
  const uint32_t group_leader = (core_id % (NUM_CORES / NUM_GROUPS)) == 0;

  for (uint32_t k = 0; k < (uint32_t)VC_LEVELS; k++) {
    buf[k] = (volatile uint32_t *)vc_malloc(k, vc_row_stride(k));
    check(buf[k] != 0, "vc_malloc", k, 0);
  }
  mempool_barrier(num_cores);

  if (group_leader) {
    for (uint32_t k = 0; k < (uint32_t)VC_LEVELS; k++) {
      const vc_rect_t rr = vc_rect(k);
      const vc_rect_t *r = &rr;
      const uint32_t off = (uint32_t)buf[k] - vc_window_base(k);
      // Each group writes only the slice of the row that lives in its own
      // L1, so leaders of the same rectangle do not race.
      // Morton: a group's position inside its rectangle is simply the low
      // 4^k bits of its id. The raster form needed a per-axis recomposition.
      const uint32_t slot = mempool_get_group_id() % r->ngroups;
      const uint32_t word = slot * (VC_ROW_BYTES / 4);
      buf[k][word] = pattern(k, off + word * 4);
    }
  }
  mempool_barrier(num_cores);

  // ---- A: read back through the GLOBAL alias -----------------------------
  if (group_leader) {
    for (uint32_t k = 0; k < (uint32_t)VC_LEVELS; k++) {
      const vc_rect_t rr = vc_rect(k);
      const vc_rect_t *r = &rr;
      const uint32_t off = (uint32_t)buf[k] - vc_window_base(k);
      // Morton: a group's position inside its rectangle is simply the low
      // 4^k bits of its id. The raster form needed a per-axis recomposition.
      const uint32_t slot = mempool_get_group_id() % r->ngroups;
      const uint32_t word = slot * (VC_ROW_BYTES / 4);
      const uint32_t voff = off + word * 4;

      // The alias: same physical word, reached by global arithmetic.
      volatile uint32_t *g = (volatile uint32_t *)vc_global_addr(r, voff);
      check(*g == pattern(k, voff), "alias equivalence", *g, pattern(k, voff));

      // ---- B: the DMA descriptor -----------------------------------------
      // Checking d.base alone was a tautology: vc_descriptor computes it with
      // vc_global_addr from the same arguments used for `g`. Check the SHAPE,
      // and check it at a row-aligned offset, which is the real precondition.
      const uint32_t aoff = off - (off % vc_row_stride(k));
      vc_desc_t d = vc_descriptor(r, aoff, vc_row_stride(k));
      check(vc_desc_valid(&d), "descriptor valid", d.len, d.count[1]);
      check(d.len * d.count[0] * d.count[1] == vc_row_stride(k),
            "descriptor covers one row", d.len * d.count[0] * d.count[1],
            vc_row_stride(k));
      // Under Morton dim0 is the WHOLE rectangle row -- the gx dimension is
      // gone, which is the point of the encoding. It was VC_ROW_BYTES * side.
      check(d.len == vc_row_stride(k) && d.count[0] == 1, "descriptor dim0",
            d.len, vc_row_stride(k));
      // A misaligned offset must be refused, not silently aimed at a neighbour.
      if (vc_row_stride(k) > VC_ROW_BYTES) {
        vc_desc_t bad = vc_descriptor(r, aoff + VC_ROW_BYTES, vc_row_stride(k));
        check(!vc_desc_valid(&bad), "misaligned descriptor refused", bad.len,
              bad.count[1]);
      }
    }
  }
  mempool_barrier(num_cores);

  // ---- C: every level still holds its own content ------------------------
  if (group_leader) {
    for (uint32_t k = 0; k < (uint32_t)VC_LEVELS; k++) {
      const vc_rect_t rr = vc_rect(k);
      const vc_rect_t *r = &rr;
      const uint32_t off = (uint32_t)buf[k] - vc_window_base(k);
      // Morton: a group's position inside its rectangle is simply the low
      // 4^k bits of its id. The raster form needed a per-axis recomposition.
      const uint32_t slot = mempool_get_group_id() % r->ngroups;
      const uint32_t word = slot * (VC_ROW_BYTES / 4);
      check(buf[k][word] == pattern(k, off + word * 4), "coexistence",
            buf[k][word], pattern(k, off + word * 4));
    }
  }
  mempool_barrier(num_cores);

  // ---- D: cross-group sweep of the whole rectangle -----------------------
  //
  // Sections A-C only ever touch the accessing core's OWN slot, and at this
  // geometry voff % VC_ROW_BYTES == 0 with VC_ROW_BYTES equal to the
  // byte|bank|tile field width, so every one of those accesses resolves to the
  // requester's own group and stays inside its tile. Nothing in the gate had
  // put a virtual-cluster address on an inter-tile wire, which is why the
  // confinement assertion had never been armed.
  //
  // Here ONE core -- the rectangle's leader -- writes every slot of its
  // rectangle, so ngroups-1 of the writes leave the group, and then EVERY
  // group leader in the rectangle reads all of them back through both address
  // paths. That checks the sharing contract of vc.h: any member passing the
  // same (level, offset) reaches the same physical words.
  for (uint32_t k = 0; k < (uint32_t)VC_LEVELS; k++) {
    const vc_rect_t rr = vc_rect(k);
    const vc_rect_t *r = &rr;
    const uint32_t gid = mempool_get_group_id();
    const uint32_t off = (uint32_t)buf[k] - vc_window_base(k);
    // Morton: the rectangle's base IS an id, so leadership is one compare.
    const uint32_t rect_leader = group_leader && (gid == r->base_gid);

    if (rect_leader)
      for (uint32_t slot = 0; slot < r->ngroups; slot++) {
        const uint32_t voff = off + slot * VC_ROW_BYTES;
        *(volatile uint32_t *)vc_ptr(k, voff) = shared_pattern(k, voff);
      }
    mempool_barrier(num_cores);

    if (group_leader)
      for (uint32_t slot = 0; slot < r->ngroups; slot++) {
        const uint32_t voff = off + slot * VC_ROW_BYTES;
        const uint32_t want = shared_pattern(k, voff);
        check(*(volatile uint32_t *)vc_ptr(k, voff) == want, "rect sweep vc",
              *(volatile uint32_t *)vc_ptr(k, voff), want);
        check(*(volatile uint32_t *)vc_global_addr(r, voff) == want,
              "rect sweep alias", *(volatile uint32_t *)vc_global_addr(r, voff),
              want);
      }
    mempool_barrier(num_cores);
  }

  // ---- E: heterogeneous lower-level budgets ------------------------------
  //
  // The regression for the arena-layout bug. Members of one level-1 rectangle
  // are explicitly allowed to disagree about level 0 -- some rectangles use a
  // level while others give the space to smaller private scopes -- so a level-1
  // base that depended on level-0 reservations made them stop sharing the
  // buffer they believe they share, and put one member's rows on another
  // member's live level-0 data. Here the odd groups reserve twice the level-0
  // rows of the even ones, and the level-1 sweep must still agree.
  if ((uint32_t)VC_LEVELS > 1) {
    for (uint32_t k = 0; k < (uint32_t)VC_LEVELS; k++)
      vc_arena_reset(k);
    const uint32_t gid = mempool_get_group_id();
    check(vc_reserve(0, vc_row_stride(0) * (1 + (gid & 1))) == 0,
          "heterogeneous level-0 reserve", gid, 0);
    check(vc_reserve(1, vc_row_stride(1)) == 0, "level-1 reserve", gid, 0);

    volatile uint32_t *hbuf =
        (volatile uint32_t *)vc_malloc(1, vc_row_stride(1));
    check(hbuf != 0, "heterogeneous vc_malloc", 0, 0);
    mempool_barrier(num_cores);

    const vc_rect_t hr = vc_rect(1);
    const uint32_t hoff = (uint32_t)hbuf - vc_window_base(1);
    const uint32_t hlead = group_leader && (gid == hr.base_gid);

    if (hlead)
      for (uint32_t slot = 0; slot < hr.ngroups; slot++)
        *(volatile uint32_t *)vc_ptr(1, hoff + slot * VC_ROW_BYTES) =
            shared_pattern(1, hoff + slot * VC_ROW_BYTES);
    mempool_barrier(num_cores);

    if (group_leader)
      for (uint32_t slot = 0; slot < hr.ngroups; slot++) {
        const uint32_t voff = hoff + slot * VC_ROW_BYTES;
        check(*(volatile uint32_t *)vc_ptr(1, voff) == shared_pattern(1, voff),
              "heterogeneous sharing", *(volatile uint32_t *)vc_ptr(1, voff),
              shared_pattern(1, voff));
      }
    mempool_barrier(num_cores);
  }

  if (core_id == 0) {
    uint32_t total = 0;
    for (uint32_t c = 0; c < num_cores; c++) {
      total += errors[c];
    }
    printf("[VC] %d errors\n", total);
    errors[0] = total;
  }
  mempool_barrier(num_cores);
  return (int)errors[0];
}
