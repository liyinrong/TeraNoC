// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Shared machinery of the phases scoped to a 2x2 virtual cluster (VC): the
// VC geometry, the per-buffer-switch fractal_sync barrier and the event
// discipline. Included by layer_resident.c; not a translation unit.

#if Q38_FFN_VC || Q38_ATTN_VC
#define Q38_VC_GROUPS 4u
#define Q38_VC_COUNT (NUM_GROUPS / Q38_VC_GROUPS)
#define Q38_VC_ENGINES (NUM_REDMULE_TILES / Q38_VC_COUNT)
// One VC-local row: the four groups' stripes of one bank row.
#define Q38_VC_ROW (4u * VC_ROW_BYTES)
#define Q38_VC_SMALL __attribute__((section(".l1"), aligned(8)))
// One record per tile, aligned so its bank stripe belongs to that tile.
#define Q38_VC_TILE_LOCAL                                                      \
  __attribute__((section(".l1"), aligned(NUM_BANKS * sizeof(uint32_t))))
_Static_assert(NUM_BANKS_PER_TILE >= 6u &&
                   NUM_BANKS == NUM_REDMULE_TILES * NUM_BANKS_PER_TILE &&
                   VC_ROW_BYTES % (NT * sizeof(__fp16)) == 0u,
               "VC handoffs need owner-tile stripes and whole Spatz rows");

// A whole global bank row becomes one row in every group's level-0 window.
static __fp16 *vc_group_window(uint32_t global) {
  return (__fp16 *)(VC_WINDOW_BASE_0 + VC_GLOBAL_ROW(global) * VC_ROW_BYTES);
}

// Spatz uses the global alias of the same group-local words. Translate each
// row access separately: successive window rows are discontiguous globally.
static void *vc_group_global(const void *window) {
  const uint32_t offset = (uint32_t)window - VC_WINDOW_BASE_0;
  return (void *)(VC_GLOBAL_BASE + offset % VC_ROW_BYTES +
                  mempool_get_group_id() * VC_ROW_BYTES +
                  offset / VC_ROW_BYTES * VC_GLOBAL_ROW_BYTES);
}
#define Q38_VC_GLOBAL(ptr) ((__typeof__(ptr))vc_group_global(ptr))

// Debug: poll instead of sleeping on events.
#ifndef Q38_VC_POLL
#define Q38_VC_POLL 0
#endif
// Debug: overlap an L1 -> L2 DMA with every job (see vc_agent).
#ifndef Q38_VC_DMA_STRESS
#define Q38_VC_DMA_STRESS 0
#endif
#if Q38_VC_POLL
// With polled waits every waiting core keeps running, so one that has waited
// too long dumps each core's wait site (tag << 24 | step) and ends the run.
static uint32_t vc_where[NUM_CORES] Q38_VC_SMALL;
static uint32_t vc_since[NUM_CORES] Q38_VC_SMALL;
static uint32_t vc_dumped Q38_VC_SMALL;
static NOSTACK void vc_watch(uint32_t tag, uint32_t step) {
  const uint32_t core = mempool_get_core_id(), where = tag << 24 | step;
  const uint32_t now = mempool_get_timer();
  if (vc_where[core] != where) {
    vc_where[core] = where;
    vc_since[core] = now;
    return;
  }
  if (now - vc_since[core] < 400000u ||
      __atomic_fetch_add(&vc_dumped, 1u, __ATOMIC_ACQ_REL) != 0u)
    return;
  for (uint32_t c = 0u; c < NUM_CORES; ++c) {
    report_fatal("[vc_dog] core=", c);
    report_fatal("[vc_dog]   where=", vc_where[c]);
  }
  report_fatal("[vc_dog] fsync_status=", fsync_status());
  q38_fatal("[vc_dog] stuck core=", core);
}
#endif
static void vc_sleep(uint32_t tag, uint32_t step) {
#if Q38_VC_POLL
  vc_watch(tag, step);
  mempool_wait(256u);
#else
  (void)tag;
  (void)step;
  mempool_wfi();
#endif
}

// The VC's 16 tiles meet (one arrival per tile, from its agent). Split
// phase: arrive, work, then wait for the response.
static void vc_barrier_arrive(void) {
  fsync_trigger(FSYNC_ID_H, fsync_aggr_of_vc_level(1u));
}

static void vc_barrier_wait(void) {
  while (fsync_status() & FSYNC_STATUS_BUSY_MASK)
    vc_sleep(2u, 0u);
}

static void vc_barrier(void) {
  vc_barrier_arrive();
  vc_barrier_wait();
}

// Event discipline. A VC barrier's response reaches every core of the tile
// that subscribes to barrier events, and a stale one would release that core
// early from the next cluster-wide fractal_sync barrier (the collective DMA's),
// which cannot tell. So on entering the section every core sets its own
// sources -- setting them drops the pending bits of the removed ones, including
// a response that landed before it got here -- and on leaving it subscribes to
// nothing until every agent is past its last VC barrier (the block-end
// sync_all), and only then restores the barrier events the rest expects.
static void vc_subscribe(uint32_t sources) {
  fsync_subscribe_self(0u);
  fsync_subscribe_self(sources);
}

// A worker waits for its agent's monotone flag: it sleeps on its RedMulE's
// end events and looks for the flag shortly after each.
static uint32_t vc_worker_wait(const uint32_t *flag, uint32_t value) {
  while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) < value) {
    vc_sleep(5u, value);
    for (uint32_t i = 0u;
         i < 64u && __atomic_load_n(flag, __ATOMIC_ACQUIRE) < value; ++i)
      mempool_wait(16u);
  }
  return value;
}
#endif
