// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Hardware barrier vs software barrier for a GLOBAL synchronisation.
//
// fsync_bar shows the hardware barrier is CORRECT; it says nothing about
// whether it is worth having. This measures that, on the rendezvous kernels
// actually use: every core in the machine.
//
// BASELINE CHOICE. mempool_log_barrier is the primary software baseline, not
// mempool_barrier. The log barrier is the binary-tree one that real kernels
// call (axpy_f32:50, qwen_spatz_op_main.h:187, ...); mempool_barrier is a
// single L1 atomic counter that every core contends on, so beating it would
// flatter the hardware against something nobody uses at scale. Both are
// reported, because the gap between them is itself the interesting number: it
// says how much of the win is "a tree" versus "a tree in hardware".
//
//   GLOBAL   sw_log : mempool_log_barrier(2, core_id)   -- every core
//            sw_lin : mempool_barrier(num_cores)        -- every core
//            hw     : fsync_barrier_all() -- last arrival triggers, all wfi
//   MECHONLY hw     : one core per tile driving the tree directly
//     (the software half of this pair is omitted: mempool_strided_barrier has
//      no prototype in synchronization.h, and adding one to shared runtime was
//      not worth doing mid-handoff. Re-add it for a mechanism-only comparison.)
//
// The MECHONLY pair has the same participant set on both sides and no
// intra-tile term on either, so it isolates the barrier MECHANISM. The GLOBAL
// pair is what a kernel would actually pay, including the intra-tile arrival
// rendezvous that one-arrival-per-tile forces.
//
// PHASE DISCIPLINE. Both mechanisms drive wake_up_sync_i, the only wake a
// Snitch has, and the software barriers end in an unconditional wfi. So every
// software barrier here runs BEFORE any fsync is triggered, and nothing after
// the hardware phase uses one.
//
// No core polls in the GLOBAL path. The core that arrives LAST in a tile
// triggers the tree and then waits with the others, so every core is already
// in WFI when the wake arrives. A FIXED leader would have to stay awake to
// poll, which would force its tile-mates to spin so they could not steal its
// wake -- a spin barrier plus a poll loop plus a release barrier, all to avoid
// a problem that disappears when the last arrival does the trigger.

#include <stdint.h>

#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

// Separate block on purpose: clang-format sorts within a block, and
// fsync.h must come AFTER runtime.h (it calls mempool_wfi, which
// runtime.h defines static while synchronization.h declares non-static).
#include "fsync.h"

#define NUM_TILES (NUM_CORES / NUM_CORES_PER_TILE)
#define WARMUP 4
#define ITERS 32

// `.l1` is NOBITS with an empty bss range, so nothing zeroes it.
static uint32_t tile_cnt[NUM_TILES] __attribute__((section(".l1")));
static uint32_t sw_bar __attribute__((section(".l1")));
static uint32_t mech_done __attribute__((section(".l1")));
static uint32_t c_log __attribute__((section(".l1")));
static uint32_t c_lin __attribute__((section(".l1")));
static uint32_t c_hw __attribute__((section(".l1")));
static uint32_t c_sw_ldr __attribute__((section(".l1")));
static uint32_t c_hw_ldr __attribute__((section(".l1")));

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  const uint32_t tile = core_id / NUM_CORES_PER_TILE;
  const uint32_t is_leader = (core_id % NUM_CORES_PER_TILE) == 0;
  mempool_timer_t t0, t1;

  mempool_barrier_init(core_id);
  for (uint32_t t = core_id; t < NUM_TILES; t += num_cores) {
    tile_cnt[t] = 0;
  }
  if (core_id == 0) {
    sw_bar = 0;
    mech_done = 0;
    c_log = c_lin = c_hw = c_sw_ldr = c_hw_ldr = 0;
  }
  mempool_barrier(num_cores);

  // ============ SOFTWARE PHASE (no fsync may fire here) =================
  for (uint32_t i = 0; i < WARMUP; i++)
    mempool_log_barrier(2, core_id);
  t0 = mempool_get_timer();
  for (uint32_t i = 0; i < ITERS; i++)
    mempool_log_barrier(2, core_id);
  t1 = mempool_get_timer();
  if (core_id == 0)
    c_log = (uint32_t)(t1 - t0);
  mempool_barrier(num_cores);

  for (uint32_t i = 0; i < WARMUP; i++)
    mempool_barrier(num_cores);
  t0 = mempool_get_timer();
  for (uint32_t i = 0; i < ITERS; i++)
    mempool_barrier(num_cores);
  t1 = mempool_get_timer();
  if (core_id == 0)
    c_lin = (uint32_t)(t1 - t0);
  mempool_barrier(num_cores);

  // ============ HARDWARE PHASE (no software barrier past here) ==========
  // One call, every core. The last arrival in each tile triggers the tree and
  // then waits with the rest; the wake releases the whole tile. No spin, no
  // poll, no designated leader.
  for (uint32_t i = 0; i < WARMUP; i++)
    fsync_barrier_all(tile_cnt, tile);
  t0 = mempool_get_timer();
  for (uint32_t i = 0; i < ITERS; i++)
    fsync_barrier_all(tile_cnt, tile);
  t1 = mempool_get_timer();
  if (core_id == 0)
    c_hw = (uint32_t)(t1 - t0);

  // Mechanism only: ONE core per tile drives the tree directly, so there is no
  // intra-tile term on either side of the MECHONLY comparison. Tile-mates spin
  // on a flag rather than WFI so they cannot swallow the leader's wake.
  if (is_leader) {
    t0 = mempool_get_timer();
    for (uint32_t i = 0; i < ITERS; i++)
      fsync_barrier_global();
    t1 = mempool_get_timer();
    if (core_id == 0) {
      c_hw_ldr = (uint32_t)(t1 - t0);
      __sync_synchronize();
      __atomic_store_n(&mech_done, 1, __ATOMIC_RELAXED);
    }
  } else {
    while (__atomic_load_n(&mech_done, __ATOMIC_RELAXED) == 0)
      ;
  }

  // Final rendezvous on the HARDWARE barrier -- everything after the hw phase
  // must stay on the same mechanism. Without it the non-printing cores fall
  // straight to `return 0` and _eoc (crt0.S:141, which runs on EVERY core)
  // ends the simulation before core 0 has printed anything.
  fsync_barrier_all(tile_cnt, tile);

  if (core_id != 0)
    while (1)
      mempool_wfi();

  if (core_id == 0) {
    printf("[fsync_perf] cores=%u tiles=%u iters=%u\n", (unsigned)num_cores,
           (unsigned)NUM_TILES, (unsigned)ITERS);
    printf("[fsync_perf] GLOBAL sw_log=%u sw_lin=%u hw=%u cyc/barrier\n",
           (unsigned)(c_log / ITERS), (unsigned)(c_lin / ITERS),
           (unsigned)(c_hw / ITERS));
    printf("[fsync_perf] MECHONLY hw=%u cyc/barrier\n",
           (unsigned)(c_hw_ldr / ITERS));
    if (c_hw)
      printf("[fsync_perf] hw vs sw_log x100 = %u\n",
             (unsigned)((c_log * 100u) / c_hw));
  }
  return 0;
}
