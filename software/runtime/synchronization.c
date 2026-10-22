// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Samuel Riedel, ETH Zurich

#include <stdbool.h>
#include <stdint.h>

#include "async_task.h"
#include "dma.h"
#include "fsync.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#if defined(__GNUC__) || defined(__clang__)
#define LOG2_NUM_CORES (31 - __builtin_clz(NUM_CORES))
#else
#error                                                                         \
    "Compiler not supported for compile-time LOG2. Please add your own or use the #elif chain."
#endif

uint32_t volatile barrier __attribute__((section(".l1")));
uint32_t volatile log_barrier[NUM_CORES * 4]
    __attribute__((aligned(NUM_CORES * 4), section(".l1")));
uint32_t volatile partial_barrier[NUM_CORES * 4]
    __attribute__((aligned(NUM_CORES * 4), section(".l1")));

#if FSYNC_TPG_SIDE != 0
fsync_tile_counter_t fsync_tile_counters[NUM_CORES / NUM_CORES_PER_TILE]
    __attribute__((aligned(FSYNC_TILE_COUNTER_ALIGNMENT), section(".l1_rows")));
#endif

#if DOMAINS > 1
// Descriptors dma_domain_stride could not translate (dma.h), per domain.
uint32_t volatile dma_domain_errors __attribute__((section(".l1")));
#if DMA_DOMAIN_CHECK
// Descriptors DMA_DOMAIN_CHECK dropped (dma.h), per domain.
uint32_t volatile dma_check_hits __attribute__((section(".l1")));
// DMA reads of the RO-cache window (dma.h), per domain, and every domain's as
// reported to domain 0.
uint32_t volatile dma_check_ro_hits __attribute__((section(".l1")));
static uint32_t volatile domain_ro_hits[DOMAINS]
    __attribute__((section(".l1")));
// DMA writes below the home base or into the image (dma.h), likewise.
uint32_t volatile dma_check_wr_hits __attribute__((section(".l1")));
static uint32_t volatile domain_wr_hits[DOMAINS]
    __attribute__((section(".l1")));
#endif
#if DOMAIN_L2_EXACT
// Descriptors the exact fold added by splitting (dma_fold_exact.h), per
// domain, and every domain's two counters as reported to domain 0.
uint32_t volatile dma_domain_splits __attribute__((section(".l1")));
static uint32_t volatile domain_counts[DOMAINS][2]
    __attribute__((section(".l1")));
// Per-core stacks of the exact fold's call path (dma.h), whole global rows:
// each core uses them through the VC tile window (dma_fold_slot).
dma_fold_slot_t dma_fold_slots[NUM_CORES]
    __attribute__((section(".l1_rows"), aligned(NUM_BANKS * 4)));
#if DMA_FOLD_PROFILE
// The fold's per-site profile (dma.h), per domain, and every domain's copy as
// reported to domain 0.
dma_fold_prof_t dma_fold_prof[DMA_FOLD_PROF_SITES]
    __attribute__((section(".l1")));
uint32_t volatile dma_fold_prof_lost __attribute__((section(".l1")));
static dma_fold_prof_t domain_prof[DOMAINS][DMA_FOLD_PROF_SITES]
    __attribute__((section(".l1")));
static uint32_t volatile domain_prof_lost[DOMAINS]
    __attribute__((section(".l1")));
#endif
#endif
// End-of-run join. Each domain has its own copy (it is in the domain window);
// domains d > 0 report into domain 0's through its physical global alias, so
// hart 0 ends the run only after every domain has finished main.
static uint32_t volatile domain_done[DOMAINS] __attribute__((section(".l1")));
static uint32_t volatile domain_ret[DOMAINS] __attribute__((section(".l1")));

static uint32_t volatile *domain0_alias(uint32_t volatile *p) {
  const uint32_t a = (uint32_t)p - DOMAIN_L1_DATA_BASE;
  return (uint32_t volatile *)(a / DMA_DOMAIN_ROW_BYTES *
                                   (DMA_DOMAIN_ROW_BYTES * DOMAINS) +
                               a % DMA_DOMAIN_ROW_BYTES);
}

// Physical hart 0, from crt0 before main: no other domain can be past main.
void domain_init(void) {
  for (uint32_t d = 0; d < DOMAINS; ++d)
    *domain0_alias(&domain_done[d]) = 0u;
}

// Every core, from crt0 after main; returns the run's retval for hart 0.
int domain_exit(int ret) {
  if (mempool_get_core_id() != 0u)
    return ret;
  const uint32_t me = mempool_get_domain_id();
  if (dma_domain_errors) {
    printf("[domain] %u untranslatable DMA descriptors in domain %u\n",
           dma_domain_errors, me);
    ret |= 1;
  }
#if DOMAIN_L2_EXACT
  // Other domains cannot print: domain 0 prints every domain's counters.
  *domain0_alias(&domain_counts[me][0]) = dma_domain_errors;
  *domain0_alias(&domain_counts[me][1]) = dma_domain_splits;
#if DMA_DOMAIN_CHECK
  *domain0_alias(&domain_ro_hits[me]) = dma_check_ro_hits;
  *domain0_alias(&domain_wr_hits[me]) = dma_check_wr_hits;
#endif
#if DMA_FOLD_PROFILE
  for (uint32_t i = 0; i < DMA_FOLD_PROF_SITES; ++i) {
    const uint32_t *const src = (const uint32_t *)&dma_fold_prof[i];
    uint32_t *const dst = (uint32_t *)&domain_prof[me][i];
    for (uint32_t k = 0; k < sizeof(dma_fold_prof_t) / 4u; ++k)
      *domain0_alias(&dst[k]) = src[k];
  }
  *domain0_alias(&domain_prof_lost[me]) = dma_fold_prof_lost;
#endif
#endif
  if (me != 0u) {
    *domain0_alias(&domain_ret[me]) = (uint32_t)ret;
    __sync_synchronize();
    *domain0_alias(&domain_done[me]) = 1u;
    return ret;
  }
  for (uint32_t d = 1; d < DOMAINS; ++d) {
    while (!*domain0_alias(&domain_done[d]))
      mempool_wait(64u);
    ret |= (int)*domain0_alias(&domain_ret[d]);
  }
#if DOMAIN_L2_EXACT
  for (uint32_t d = 0; d < DOMAINS; ++d)
    printf("[domain] exact fold: domain %u errors %u splits %u\n", d,
           domain_counts[d][0], domain_counts[d][1]);
#if DMA_DOMAIN_CHECK
  for (uint32_t d = 0; d < DOMAINS; ++d)
    printf("[domain] ro-window reads: domain %u hits %u\n", d,
           domain_ro_hits[d]);
  for (uint32_t d = 0; d < DOMAINS; ++d)
    printf("[domain] below-base writes: domain %u hits %u\n", d,
           domain_wr_hits[d]);
#endif
#if DMA_FOLD_PROFILE
  for (uint32_t d = 0; d < DOMAINS; ++d) {
    printf("[fold-prof] domain %u lost %u\n", d, domain_prof_lost[d]);
    for (uint32_t i = 0; i < DMA_FOLD_PROF_SITES; ++i) {
      const dma_fold_prof_t *const e = &domain_prof[d][i];
      if (e->calls)
        printf("[fold-prof] domain %u pc 0x%08x calls %u pieces %u min %u "
               "max %u cycles %u\n",
               d, e->pc, e->calls, e->pieces, e->minp, e->maxp, e->cycles);
    }
  }
#endif
#endif
  return ret;
}
#endif

void mempool_barrier_init(uint32_t core_id) {
#if DOMAINS > 1
  if (core_id == 0) {
    dma_domain_errors = 0u;
#if DMA_DOMAIN_CHECK
    dma_check_hits = 0u;
    dma_check_ro_hits = 0u;
    dma_check_wr_hits = 0u;
#if DOMAIN_L2_EXACT
    {
      // The fold slot stacks must be 16-B aligned: Snitch traps on a
      // misaligned word access (the model splits it and can wedge).
      const uint32_t slot = (uint32_t)dma_fold_slot();
      const uint32_t top =
          (uint32_t)&dma_fold_slot()->stack[DMA_FOLD_STACK_BYTES / 4u];
      if ((slot | top) % 16u) {
        extern char fake_uart;
        for (const char *p = "[domain] FOLD SLOT MISALIGNED\n"; *p; ++p)
          *(volatile char *)&fake_uart = *p;
        dma_domain_errors = 1000u;
      }
    }
#endif
#endif
#if DMA_RO_PROBE
    {
      // Planted control: one DMA read of the RO-cache window (dma.h).
      // Row-aligned: the L1 side of a raw DMA must be the issuing group's
      // stripe, and core 0's group owns offset 0 of every domain row.
      static uint32_t ro_probe[16]
          __attribute__((section(".l1"), aligned(DMA_DOMAIN_ROW_BYTES)));
      dma_local_wait_id(
          0u,
          dma_local_memcpy_nonblocking(
              0u, ro_probe, (const void *)((uint32_t)L2_BASE + 0x100u), 64u));
    }
#endif
#if DOMAIN_L2_EXACT
    dma_domain_splits = 0u;
#endif
#if DMA_FOLD_PROFILE
    for (uint32_t i = 0; i < DMA_FOLD_PROF_SITES; ++i)
      dma_fold_prof[i] = (dma_fold_prof_t){0u, 0u, 0u, ~0u, 0u, 0u};
    dma_fold_prof_lost = 0u;
#endif
  }
#endif
  if (core_id == 0) {
    // Initialize the barrier
    barrier = 0;
    wake_up_all();
    mempool_wfi();
  } else {
    mempool_wfi();
  }
  // Initialize log-barriers synch variables in parallel
  for (uint32_t i = core_id; i < NUM_CORES * 4; i += NUM_CORES) {
    log_barrier[i] = 0;
    partial_barrier[i] = 0;
  }
#if FSYNC_TPG_SIDE != 0
  for (uint32_t t = core_id; t < NUM_CORES / NUM_CORES_PER_TILE; t += NUM_CORES)
    fsync_tile_counter_init(&fsync_tile_counters[t]);
#endif
  dma_collective_runtime_init();
  tnc_engine_runtime_init();
  mempool_barrier(NUM_CORES);
}

/* POLLING BARRIER */

void mempool_barrier_poll_slot(mempool_poll_slot_t *slot, uint32_t num_cores) {
  if (num_cores <= 1u) {
    __atomic_thread_fence(__ATOMIC_ACQ_REL);
    return;
  }
  const uint32_t expected =
      __atomic_load_n(&slot->generation, __ATOMIC_ACQUIRE);
  if (__atomic_fetch_add(&slot->count, 1u, __ATOMIC_ACQ_REL) ==
      num_cores - 1u) {
    // Ordered by the generation release below, which is what waiters acquire.
    slot->count = 0u;
    __atomic_store_n(&slot->generation, expected + 1u, __ATOMIC_RELEASE);
    return;
  }
  while (__atomic_load_n(&slot->generation, __ATOMIC_ACQUIRE) == expected) {
    mempool_wait(4u);
  }
}

/* PLAIN BARRIER */

// Why these sleep rather than poll: Snitch banks wakes that arrive while a core
// is AWAKE in a 3-bit counter that WRAPS (snitch.sv), so the eighth
// unconsumed wake discards the other seven. The barriers below are safe because
// they are credit-NEUTRAL: every participant executes exactly one wfi per wake
// pulse it can receive. The last arrival wakes the scope and then falls into
// the same wfi -- which returns immediately, spending the credit it just
// created. Two rules keep that true, and both are already honoured here:
//   * wake the SCOPE, not the machine: wake_up_all() only when the barrier
//     covers every core, else the group/tile-scoped variants, so a core outside
//     the scope never banks a pulse it will not spend;
//   * never arm a second wake source in the same phase (see fsync.h).

void mempool_barrier(uint32_t num_cores) {
  // Increment the barrier counter
  if ((num_cores - 1) == __atomic_fetch_add(&barrier, 1, __ATOMIC_RELAXED)) {
    __atomic_store_n(&barrier, 0, __ATOMIC_RELAXED);
    __sync_synchronize(); // Full memory barrier
    wake_up_all();
  }
  // Some threads have not reached the barrier --> Let's wait
  // Clear the wake-up trigger for the last core reaching the barrier as well
  mempool_wfi();
}

void mempool_strided_barrier(uint32_t *barrier, uint32_t num_cores,
                             uint32_t stride, uint32_t offset) {

  // Increment the barrier counter
  if ((num_cores - 1) == __atomic_fetch_add(barrier, 1, __ATOMIC_RELAXED)) {
    __atomic_store_n(barrier, 0, __ATOMIC_RELAXED);
    __sync_synchronize(); // Full memory barrier
    set_wake_up_stride(stride);
    set_wake_up_offset(offset);
    wake_up_all();
    set_wake_up_stride(1U);
    set_wake_up_offset(0U);
  }
  mempool_wfi();
}

/* LOG BARRIER */

void mempool_log_barrier(uint32_t step, uint32_t core_id) {
  uint32_t idx = (step * (core_id / step)) * 4;
  uint32_t next_step, previous_step;
  uint32_t num_cores = mempool_get_core_count();
  previous_step = step >> 1;
  if ((step - previous_step) ==
      __atomic_fetch_add(&log_barrier[idx + previous_step - 1], previous_step,
                         __ATOMIC_RELAXED)) {
    next_step = step << 1;
    __atomic_store_n(&log_barrier[idx + previous_step - 1], 0,
                     __ATOMIC_RELAXED);
    if (num_cores == step) {
      __sync_synchronize(); // Full memory barrier
      wake_up_all();
      mempool_wfi();
    } else {
      mempool_log_barrier(next_step, core_id);
    }
  } else
    mempool_wfi();
}

void mempool_anyradixlog_barrier(uint32_t radix, uint32_t core_id) {
  uint32_t num_cores = mempool_get_core_count();
  uint32_t first_step = (LOG2_NUM_CORES % radix) == 0
                            ? (1U << radix)
                            : 1U << (LOG2_NUM_CORES % radix);
  uint32_t step = 0, previous_step = 0;
  // At first step you take care of the remainder
  uint32_t idx = (first_step * (core_id / first_step)) * 4;
  if ((first_step - 1) ==
      __atomic_fetch_add(&log_barrier[idx], 1, __ATOMIC_RELAXED)) {
    __atomic_store_n(&log_barrier[idx], 0, __ATOMIC_RELAXED);
    num_cores /= first_step;
    previous_step = first_step;
    step = (first_step << radix);
    // Following steps proceed with the radix chosen
    while (num_cores > 1U) {
      idx = (step * (core_id / step)) * 4;
      if ((step - previous_step) ==
          __atomic_fetch_add(&log_barrier[idx + previous_step - 1],
                             previous_step, __ATOMIC_RELAXED)) {
        __atomic_store_n(&log_barrier[idx + previous_step - 1], 0,
                         __ATOMIC_RELAXED);
        num_cores >>= radix;
        previous_step = step;
        step <<= radix;
      } else {
        break;
      }
    }
    // Last core wakes-up everyone
    if (num_cores == 1U) {
      __sync_synchronize(); // Full memory barrier
      wake_up_all();
    }
  }
  mempool_wfi();
}

void mempool_linlog_barrier(uint32_t step, uint32_t core_id) {

  uint32_t idx = (step * (core_id / step)) * 4;
  uint32_t next_step, previous_step;
  uint32_t num_cores = mempool_get_core_count();

  previous_step = step >> 1;
  if ((step - 1) == __atomic_fetch_add(&log_barrier[idx + previous_step - 1], 1,
                                       __ATOMIC_RELAXED)) {
    next_step = step << 1;
    __atomic_store_n(&log_barrier[idx + previous_step - 1], 0,
                     __ATOMIC_RELAXED);
    if (num_cores == step) {
      __sync_synchronize(); // Full memory barrier
      wake_up_all();
      mempool_wfi();
    } else {
      mempool_log_barrier(next_step, core_id);
    }
  } else
    mempool_wfi();
}

void mempool_strided_log_barrier(uint32_t step, uint32_t core_id,
                                 uint32_t stride, uint32_t offset) {

  uint32_t idx = (step * (core_id / step)) * 4 + offset;
  uint32_t next_step, previous_step;
  uint32_t num_cores = mempool_get_core_count();

  previous_step = step >> 1;
  if ((step - previous_step) ==
      __atomic_fetch_add(&log_barrier[idx + previous_step - 1], previous_step,
                         __ATOMIC_RELAXED)) {
    next_step = step << 1;
    __atomic_store_n(&log_barrier[idx + previous_step - 1], 0,
                     __ATOMIC_RELAXED);
    if (num_cores == step) {
      __sync_synchronize(); // Full memory barrier
      set_wake_up_stride(stride);
      set_wake_up_offset(offset);
      wake_up_all();
      set_wake_up_stride(1U);
      set_wake_up_offset(0U);
      mempool_wfi();
    } else {
      mempool_log_barrier(next_step, core_id);
    }
  } else
    mempool_wfi();
}

/* PARTIAL BARRIER */

void mempool_log_partial_barrier(uint32_t step, uint32_t core_id,
                                 uint32_t num_cores_barrier) {

  uint32_t core_init = num_cores_barrier * (core_id / num_cores_barrier);
  uint32_t core_end = core_init + num_cores_barrier;
  uint32_t num_cores = mempool_get_core_count();
  uint32_t num_cores_per_tile = mempool_get_core_count_per_tile();
  uint32_t num_cores_per_group = mempool_get_core_count_per_group();

  if (core_id >= core_init && core_id < core_end) {

    uint32_t idx = (step * (core_id / step)) * 4;
    uint32_t next_step, previous_step;
    previous_step = step >> 1;
    if ((step - previous_step) ==
        __atomic_fetch_add(&log_barrier[idx + previous_step - 1], previous_step,
                           __ATOMIC_RELAXED)) {
      next_step = step << 1;
      __atomic_store_n(&log_barrier[idx + previous_step - 1], 0,
                       __ATOMIC_RELAXED);
      if (num_cores_barrier == step) {

        __sync_synchronize(); // Full memory barrier
        if (num_cores_barrier >= num_cores) {
          wake_up_all();
        } else if (num_cores_barrier >= num_cores_per_group) {
          uint32_t volatile group_init = core_init / num_cores_per_group;
          uint32_t volatile group_end = core_end / num_cores_per_group;
          wake_up_group(((1U << (group_end - group_init)) - 1) << group_init);
        } else if (num_cores_barrier >= num_cores_per_tile) {
          uint32_t volatile tile_init = core_init / num_cores_per_tile;
          uint32_t volatile tile_end = core_end / num_cores_per_tile;
          wake_up_tile(tile_init / NUM_TILES_PER_GROUP,
                       ((1U << (tile_end - tile_init)) - 1)
                           << tile_init % NUM_TILES_PER_GROUP);
        } else {
          while (core_init < core_end) {
            wake_up(core_init);
            core_init++;
          }
        }
        mempool_wfi();
      } else {
        mempool_log_partial_barrier(next_step, core_id, num_cores_barrier);
      }
    } else
      mempool_wfi();
  }
}

void mempool_partial_barrier(uint32_t volatile core_id,
                             uint32_t volatile core_init,
                             uint32_t volatile num_sleeping_cores,
                             uint32_t volatile memloc) {

  uint32_t volatile core_end = core_init + num_sleeping_cores;
  uint32_t num_cores_per_tile = mempool_get_core_count_per_tile();
  uint32_t num_cores_per_group = mempool_get_core_count_per_group();

  if (core_id >= core_init && core_id < core_end) {

    if (num_sleeping_cores - 1 ==
        __atomic_fetch_add(&partial_barrier[(core_init * 4) + memloc], 1,
                           __ATOMIC_RELAXED)) {

      __atomic_store_n(&partial_barrier[(core_init * 4) + memloc], 0,
                       __ATOMIC_RELAXED);
      __sync_synchronize(); // Full memory barrier
      /* Wake-up the core remainder */
      if ((core_end - core_init) > num_cores_per_tile) {
        while (core_init % num_cores_per_tile != 0) {
          wake_up(core_init);
          core_init++;
        }
        while (core_end % num_cores_per_tile != 0) {
          core_end--;
          wake_up(core_end);
        }
      } else if ((core_end - core_init) < num_cores_per_tile) {
        while (core_init < core_end) {
          wake_up(core_init);
          core_init++;
        }
      }

      /* Wake-up the tile remainder */
      uint32_t volatile tile_init = core_init / num_cores_per_tile;
      uint32_t volatile tile_end = core_end / num_cores_per_tile;
      if ((tile_end - tile_init) > NUM_TILES_PER_GROUP) {
        wake_up_tile(tile_init / NUM_TILES_PER_GROUP,
                     ((1U << (16 - tile_init % NUM_TILES_PER_GROUP)) - 1)
                         << tile_init % NUM_TILES_PER_GROUP);
        wake_up_tile(tile_end / NUM_TILES_PER_GROUP,
                     ((1U << tile_end % NUM_TILES_PER_GROUP) - 1));
        core_init += num_cores_per_tile * (16 - tile_init);
        core_end -= num_cores_per_tile * tile_end % NUM_TILES_PER_GROUP;
      } else if ((tile_end - tile_init) < NUM_TILES_PER_GROUP) {
        wake_up_tile(tile_init / NUM_TILES_PER_GROUP,
                     ((1U << (tile_end - tile_init)) - 1)
                         << tile_init % NUM_TILES_PER_GROUP);
        core_init += num_cores_per_tile * (tile_end - tile_init);
      }

      /* Wake-up the group remainder */
      uint32_t volatile group_init = core_init / num_cores_per_group;
      uint32_t volatile group_end = core_end / num_cores_per_group;
      if (group_end - group_init > 0) {
        wake_up_group(((1U << (group_end - group_init)) - 1) << group_init);
        core_init += num_cores_per_group * (group_end - group_init);
      }
    }
    mempool_wfi();
  }
}
