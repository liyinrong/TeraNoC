// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Coordinator timestamps of the most recent collective operation, for
// benchmarks. Build the runtime and the app with -DDMA_COLLECTIVE_PROFILE.
// One full bank stripe per group, so each coordinator writes only its own
// group's banks.
//  0 submit entered     2 wait entered         4 barrier released
//  1 descriptor issued  3 own engine retired   5 wait returned

#ifndef _DMA_COLLECTIVE_PROF_H_
#define _DMA_COLLECTIVE_PROF_H_

#include <stdint.h>

#include "runtime.h"

#ifdef DMA_COLLECTIVE_PROFILE
#define DMA_PROF_SLOTS 6u
typedef struct {
  volatile uint32_t t[DMA_PROF_SLOTS];
  uint32_t pad[NUM_BANKS / NUM_GROUPS - DMA_PROF_SLOTS];
} dma_prof_group_t;
extern dma_prof_group_t dma_prof[];
#define DMA_PROF(i)                                                            \
  dma_prof[mempool_get_group_id()].t[i] = (uint32_t)mempool_get_timer()
#else
#define DMA_PROF(i)                                                            \
  do {                                                                         \
  } while (0)
#endif

#endif // _DMA_COLLECTIVE_PROF_H_
