// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Vector store, fence, flag: the release pattern of a Spatz producer.
//
// Core 1 of tile 0 vector-stores a 512 B row into the banks of the last group
// (remote), executes fence, then stores a flag into tile 0's own bank. Core 2
// of tile 0 polls the flag and reads the row back; it must see the new data.
// The flag is local and the row remote, so a flag that overtook the row would
// show up as stale words.
//
// It also measures the fence itself, as the cycles between two mcycle reads
// around it: once with nothing outstanding, once right after the row's
// vse32.v was handed to Spatz.

#include <stdint.h>

#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "fsync.h"

#define NUM_TILES (NUM_CORES / NUM_CORES_PER_TILE)
#define ROW_WORDS 128u // 512 B: one e32, LMUL=8 register group at VLEN 512
#define ITERS 8u

static fsync_tile_counter_t tile_cnt[NUM_TILES]
    __attribute__((aligned(FSYNC_TILE_COUNTER_ALIGNMENT), section(".l1")));
// One word per bank of the machine: index i lands in bank i, so tile t owns
// words [t * NUM_BANKS_PER_TILE, (t + 1) * NUM_BANKS_PER_TILE).
static uint32_t row[NUM_BANKS]
    __attribute__((aligned(NUM_BANKS * 4), section(".l1")));
static volatile uint32_t __attribute__((section(".l1"))) ack, errs;
static volatile uint32_t __attribute__((section(".l1"))) cost_idle[ITERS],
    cost_busy[ITERS];

static void writer(void) {
  volatile uint32_t *flag = &row[0];
  uint32_t *dst = &row[(NUM_TILES - 4u) * (NUM_BANKS_PER_TILE)];
  uint32_t vl;
  asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(ROW_WORDS));
  for (uint32_t it = 1; it <= ITERS; ++it) {
    uint32_t t0, t1, t2, t3;
    asm volatile("vmv.v.x v8, %0" ::"r"(it));
    asm volatile("csrr %0, mcycle" : "=r"(t0));
    asm volatile("fence" ::: "memory");
    asm volatile("csrr %0, mcycle" : "=r"(t1));
    asm volatile("vse32.v v8, (%0)" ::"r"(dst) : "memory");
    asm volatile("csrr %0, mcycle" : "=r"(t2));
    asm volatile("fence" ::: "memory");
    asm volatile("csrr %0, mcycle" : "=r"(t3));
    *flag = it;
    cost_idle[it - 1u] = t1 - t0;
    cost_busy[it - 1u] = t3 - t2;
    while (ack != it)
      ;
  }
}

static void reader(void) {
  volatile uint32_t *flag = &row[0];
  volatile uint32_t *src = &row[(NUM_TILES - 4u) * (NUM_BANKS_PER_TILE)];
  for (uint32_t it = 1; it <= ITERS; ++it) {
    while (*flag != it)
      ;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < ROW_WORDS; ++i)
      bad += src[i] != it;
    if (bad) {
      errs += bad;
      printf("[vfence] iter %u: %u stale words\n", (unsigned)it, (unsigned)bad);
    }
    ack = it;
  }
}

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  const uint32_t tile = core_id / NUM_CORES_PER_TILE;

  mempool_barrier_init(core_id);
  for (uint32_t t = core_id; t < NUM_TILES; t += num_cores)
    fsync_tile_counter_init(&tile_cnt[t]);
  for (uint32_t i = core_id; i < NUM_BANKS; i += num_cores)
    row[i] = 0u;
  if (core_id == 0u)
    ack = errs = 0u;
  mempool_barrier(num_cores);

  if (core_id == 1u)
    writer();
  else if (core_id == 2u)
    reader();

  fsync_barrier_all(tile_cnt, tile);
  if (core_id == 0u) {
    for (uint32_t i = 0; i < ITERS; ++i)
      printf("[vfence] iter %u fence cycles: idle=%u after_512B_vse=%u\n",
             (unsigned)(i + 1u), (unsigned)cost_idle[i],
             (unsigned)cost_busy[i]);
    printf("[vfence] %s errors=%u\n", errs ? "FAIL" : "PASS", (unsigned)errs);
  }
  return (int)errs;
}
