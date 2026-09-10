// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// One-line hardware barrier for an application whose cores all participate.
//
// fsync_barrier_all() needs a per-tile counter array, correctly aligned so that
// entry t lands in tile t's own bank stripe, and zeroed before the first
// arrival. Every app that wants the tree would otherwise repeat that setup.
//
// The storage is a file-scope static, so include this from exactly ONE
// translation unit. That is what a baremetal app is -- the Makefile compiles
// <app>/main.c and nothing else -- but it is the reason this is not in
// fsync.h.
//
// It falls back to the software barrier wherever the tree cannot serve, so an
// app can call it unconditionally and still build everywhere. Check
// FSYNC_APP_BARRIER if the app wants to report which one it used.

#ifndef _FSYNC_APP_H_
#define _FSYNC_APP_H_

#include <stdint.h>

#include "fsync.h"
#include "runtime.h"
#include "synchronization.h"

// FSYNC_TPG_SIDE truncates: the H-tree only covers a tile count that is a power
// of FOUR, and is 0 for the grids it cannot express.
#if defined(FSYNC_BASE) && (FSYNC_TPG_SIDE != 0)
#define FSYNC_APP_BARRIER 1
#else
#define FSYNC_APP_BARRIER 0
#endif

#if FSYNC_APP_BARRIER

#define FSYNC_APP_TILES ((uint32_t)NUM_CORES / (uint32_t)NUM_CORES_PER_TILE)

static fsync_tile_counter_t fsync_app_counter[FSYNC_APP_TILES]
    __attribute__((aligned(FSYNC_TILE_COUNTER_ALIGNMENT), section(".l1")));

// Call INSTEAD of mempool_barrier_init. The software barrier is still set up:
// the counters live in `.l1`, which is NOBITS and powers up X, so they must be
// zeroed and that zeroing must be complete before any core can arrive -- which
// needs a barrier that does not itself depend on them.
static inline void fsync_app_barrier_init(uint32_t core_id) {
  mempool_barrier_init(core_id);
  for (uint32_t t = core_id; t < FSYNC_APP_TILES; t += (uint32_t)NUM_CORES)
    fsync_tile_counter_init(&fsync_app_counter[t]);
  mempool_barrier((uint32_t)NUM_CORES);
}

static inline void fsync_app_barrier(void) {
  fsync_barrier_all(fsync_app_counter, mempool_get_tile_id());
}

#else

static inline void fsync_app_barrier_init(uint32_t core_id) {
  mempool_barrier_init(core_id);
}

static inline void fsync_app_barrier(void) {
  mempool_barrier((uint32_t)NUM_CORES);
}

#endif // FSYNC_APP_BARRIER

#endif // _FSYNC_APP_H_
