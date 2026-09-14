// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Distributed arm: one iDMA v0.6.5 frontend per group. Every group's
// coordinator programs only its own engine; the wait fences all of them.

#include <stddef.h>
#include <stdint.h>

#include "dma.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define DMA_BW_NAME "distributed"

// The fixed-coordinator collective: every core calls, each group's coordinator
// issues its own stripe, and the wait's fractal_sync barrier is what makes the
// data visible cluster-wide.
#define DMA_XFER_IN(d, s, n) (void)dma_collective_memcpy_blocking((d), (s), (n))
#define DMA_XFER_OUT(d, s, n) DMA_XFER_IN(d, s, n)

// Stage breakdown from the runtime's own probes. Each coordinator writes only
// its own group's bank stripe, so reading group 0 is a local read.
#ifdef DMA_COLLECTIVE_PROFILE
#define DMA_BW_BREAKDOWN()                                                     \
  do {                                                                         \
    if (core == 0u) {                                                          \
      const volatile uint32_t *const t = dma_prof[0].t;                        \
      printf("[dma_bw] brk submit=%d to_wait=%d engine=%d barrier=%d ret=%d "  \
             "total=%d\n",                                                     \
             (int)(t[1] - t[0]), (int)(t[2] - t[1]), (int)(t[3] - t[2]),       \
             (int)(t[4] - t[3]), (int)(t[5] - t[4]), (int)(t[5] - t[0]));      \
      for (uint32_t g = 0u; g < (uint32_t)NUM_GROUPS; ++g) {                   \
        const volatile uint32_t *const u = dma_prof[g].t;                      \
        printf("[dma_bw] grp %u t=%u %u %u %u %u %u\n", (unsigned)g,           \
               (unsigned)u[0], (unsigned)u[1], (unsigned)u[2], (unsigned)u[3], \
               (unsigned)u[4], (unsigned)u[5]);                                \
      }                                                                        \
    }                                                                          \
  } while (0)
#endif

#include "dma_bw_common.h"
