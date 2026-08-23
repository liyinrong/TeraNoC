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
      const uint32_t slot =
          (mempool_get_group_id() % r->side) * 1 +
          ((mempool_get_group_id() / NUM_Y) % r->side) * r->side;
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
      const uint32_t slot =
          (mempool_get_group_id() % r->side) * 1 +
          ((mempool_get_group_id() / NUM_Y) % r->side) * r->side;
      const uint32_t word = slot * (VC_ROW_BYTES / 4);
      const uint32_t voff = off + word * 4;

      // The alias: same physical word, reached by global arithmetic.
      volatile uint32_t *g = (volatile uint32_t *)vc_global_addr(r, voff);
      check(*g == pattern(k, voff), "alias equivalence", *g, pattern(k, voff));

      // ---- B: the DMA descriptor's base must be that same address --------
      vc_desc_t d = vc_descriptor(r, voff, vc_row_stride(k));
      check(d.base == (uint32_t)g, "descriptor base", d.base, (uint32_t)g);
    }
  }
  mempool_barrier(num_cores);

  // ---- C: every level still holds its own content ------------------------
  if (group_leader) {
    for (uint32_t k = 0; k < (uint32_t)VC_LEVELS; k++) {
      const vc_rect_t rr = vc_rect(k);
      const vc_rect_t *r = &rr;
      const uint32_t off = (uint32_t)buf[k] - vc_window_base(k);
      const uint32_t slot =
          (mempool_get_group_id() % r->side) * 1 +
          ((mempool_get_group_id() / NUM_Y) % r->side) * r->side;
      const uint32_t word = slot * (VC_ROW_BYTES / 4);
      check(buf[k][word] == pattern(k, off + word * 4), "coexistence",
            buf[k][word], pattern(k, off + word * 4));
    }
  }
  mempool_barrier(num_cores);

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
