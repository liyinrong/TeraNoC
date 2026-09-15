// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// One-line barrier for an application whose cores all participate: the
// fractal_sync tree where it covers the tile grid, the software barrier
// otherwise. FSYNC_APP_BARRIER says which one the build uses.

#ifndef _FSYNC_APP_H_
#define _FSYNC_APP_H_

#include <stdint.h>

#include "fsync.h"
#include "runtime.h"
#include "synchronization.h"

#if defined(FSYNC_BASE) && (FSYNC_TPG_SIDE != 0)
#define FSYNC_APP_BARRIER 1
static inline void fsync_app_barrier(void) {
  fsync_barrier_all(fsync_tile_counters, mempool_get_tile_id());
}
#else
#define FSYNC_APP_BARRIER 0
static inline void fsync_app_barrier(void) {
  mempool_barrier((uint32_t)NUM_CORES);
}
#endif

#endif // _FSYNC_APP_H_
