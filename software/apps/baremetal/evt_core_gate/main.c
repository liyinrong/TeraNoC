// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// EVT_CORE and the epoch-checked fsync barrier.
//
// Tile 0 runs four directed cases, driven by core 0 (on a 1+3 build the
// Snitch-only scheduler, so its path is integer-only):
//   A1 core 0 raises to cores 1-3 while they sleep; all wake. Reports the
//      latency from the raise store to the target resuming.
//   A2 a raise landing between a core's check and its wfi still wakes it.
//   A3 three raises before a wake give ONE wake: the second wfi sleeps until a
//      later raise.
//   A4 a core not subscribed to EVT_CORE is not woken by a raise; once
//      subscribed, it is.
// Then every core of the machine runs B: tile 0's cores 1-3 carry a pending
// EVT_CORE into fsync_barrier_all. A barrier that slept only once would let
// them out before the other tiles arrive; the epoch check must hold them.
//
// A lost wake hangs rather than fails, so A2 and A4 re-raise after a deadline
// and count an error instead.

#include <stdint.h>

#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "fsync.h"

#define NUM_TILES (NUM_CORES / NUM_CORES_PER_TILE)
#define SETTLE 2000u   // cycles for a target to reach its wfi
#define DEADLINE 5000u // cycles before a missing wake counts as lost

static fsync_tile_counter_t tile_cnt[NUM_TILES]
    __attribute__((aligned(FSYNC_TILE_COUNTER_ALIGNMENT), section(".l1")));

static volatile uint32_t __attribute__((section(".l1"))) ready, go, arm2, go2,
    done2, arm3, raised3, arm4;
static volatile uint32_t __attribute__((section(".l1")))
woke[NUM_CORES_PER_TILE],
    first3, second3, woke4;
static volatile uint32_t __attribute__((section(".l1"))) arm_b, raised_b,
    arrived, errs;

static void err(uint32_t code) {
  __atomic_fetch_add(&errs, 1u, __ATOMIC_RELAXED);
  printf("[evt_core] FAIL %u\n", (unsigned)code);
}

static void subscribe_lane(uint32_t lane, uint32_t sources) {
  *(volatile uint32_t *)(FSYNC_BASE + FSYNC_SUBSCRIBE_REG_OFFSET + 4u * lane) =
      sources;
}

static void scheduler(void) {
  // A1: sleep, raise, wake.
  while (ready != NUM_CORES_PER_TILE - 1u)
    ;
  mempool_wait(SETTLE);
  go = 1u;
  __sync_synchronize();
  const uint32_t t_raise = (uint32_t)mempool_get_timer();
  fsync_raise(0xeu);
  for (uint32_t c = 1; c < NUM_CORES_PER_TILE; ++c)
    while (woke[c] == 0u)
      ;
  uint32_t lo = 0xffffffffu, hi = 0u;
  for (uint32_t c = 1; c < NUM_CORES_PER_TILE; ++c) {
    const uint32_t lat = woke[c] - t_raise;
    lo = lat < lo ? lat : lo;
    hi = lat > hi ? lat : hi;
  }
  printf("[evt_core] A1 raise->resume latency min=%u max=%u\n", (unsigned)lo,
         (unsigned)hi);

  // A2: core 1 has checked go2 and is about to wfi when the raise lands.
  while (arm2 == 0u)
    ;
  go2 = 1u;
  __sync_synchronize();
  fsync_raise(0x2u);
  uint32_t t = (uint32_t)mempool_get_timer();
  while (done2 == 0u)
    if ((uint32_t)mempool_get_timer() - t > DEADLINE) {
      err(2u);
      fsync_raise(0x2u);
      t = (uint32_t)mempool_get_timer();
    }

  // A3: three raises while core 2 is awake, then one more later.
  while (arm3 == 0u)
    ;
  fsync_raise(0x4u);
  fsync_raise(0x4u);
  fsync_raise(0x4u);
  __sync_synchronize();
  raised3 = 1u;
  while (first3 == 0u)
    ;
  mempool_wait(SETTLE);
  const uint32_t t_final = (uint32_t)mempool_get_timer();
  fsync_raise(0x4u);
  while (second3 == 0u)
    ;
  if ((int32_t)(second3 - t_final) < 0)
    err(3u); // the second wfi returned on a second, uncoalesced event

  // A4: core 3 subscribes to nothing; a raise must not wake it.
  while (arm4 == 0u)
    ;
  mempool_wait(SETTLE);
  fsync_raise(0x8u);
  mempool_wait(SETTLE);
  const uint32_t t_resub = (uint32_t)mempool_get_timer();
  subscribe_lane(3u, FSYNC_EVT_CORE);
  fsync_raise(0x8u);
  t = (uint32_t)mempool_get_timer();
  while (woke4 == 0u)
    if ((uint32_t)mempool_get_timer() - t > DEADLINE) {
      err(5u);
      break;
    }
  if (woke4 != 0u && (int32_t)(woke4 - t_resub) < 0)
    err(4u); // woken by the raise it was not subscribed to

  // B: leave an EVT_CORE pending on cores 1-3 while they are awake.
  while (arm_b != NUM_CORES_PER_TILE - 1u)
    ;
  fsync_raise(0xeu);
  __sync_synchronize();
  raised_b = 1u;
}

static void worker(uint32_t lane) {
  fsync_subscribe_self(FSYNC_EVT_CORE);
  __atomic_fetch_add(&ready, 1u, __ATOMIC_RELAXED);
  while (go == 0u)
    mempool_wfi();
  woke[lane] = (uint32_t)mempool_get_timer();

  if (lane == 1u) {
    if (go2 == 0u) {
      arm2 = 1u;
      mempool_wait(500u); // the check-to-wfi window the raise lands in
      mempool_wfi();
    }
    done2 = 1u;
  } else if (lane == 2u) {
    arm3 = 1u;
    while (raised3 == 0u)
      ;
    mempool_wfi();
    first3 = (uint32_t)mempool_get_timer();
    mempool_wfi();
    second3 = (uint32_t)mempool_get_timer();
  } else {
    fsync_subscribe_self(0u);
    arm4 = 1u;
    mempool_wfi();
    woke4 = (uint32_t)mempool_get_timer();
  }

  fsync_subscribe_self(FSYNC_EVT_FSYNC | FSYNC_EVT_CORE);
  __atomic_fetch_add(&arm_b, 1u, __ATOMIC_RELAXED);
  while (raised_b == 0u)
    ;
}

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  const uint32_t tile = core_id / NUM_CORES_PER_TILE;
  const uint32_t lane = core_id % NUM_CORES_PER_TILE;

  mempool_barrier_init(core_id);
  for (uint32_t t = core_id; t < NUM_TILES; t += num_cores)
    fsync_tile_counter_init(&tile_cnt[t]);
  if (core_id == 0u) {
    ready = go = arm2 = go2 = done2 = arm3 = raised3 = arm4 = 0u;
    first3 = second3 = woke4 = arm_b = raised_b = arrived = errs = 0u;
    for (uint32_t c = 0; c < NUM_CORES_PER_TILE; ++c)
      woke[c] = 0u;
  }
  mempool_barrier(num_cores);

  if (tile == 0u) {
    if (lane == 0u)
      scheduler();
    else
      worker(lane);
  }

  // B: count in, then the barrier. Fence so the count is visible before this
  // core's arrival can release anyone.
  __atomic_fetch_add(&arrived, 1u, __ATOMIC_RELAXED);
  __sync_synchronize();
  fsync_barrier_all(tile_cnt, tile);
  const uint32_t seen = arrived;
  if (seen != num_cores) {
    err(6u); // released before every core arrived
    if (core_id == 0u)
      printf("[evt_core] arrived=%u at release\n", (unsigned)seen);
  }

  fsync_subscribe_self(FSYNC_EVT_FSYNC);
  fsync_barrier_all(tile_cnt, tile);
  if (core_id == 0u)
    printf("[evt_core] %s errors=%u\n", errs ? "FAIL" : "PASS", (unsigned)errs);
  return (int)errs;
}
