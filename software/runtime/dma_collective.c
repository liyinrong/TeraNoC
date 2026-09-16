// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "dma_collective.h"

#include <stdint.h>

#include "dma.h"
#include "dma_collective_prof.h"
#include "fsync.h"
#include "runtime.h"
#include "synchronization.h"

#define DMA_COLLECTIVE_GROUP_WORDS (NUM_BANKS / NUM_GROUPS)
#define DMA_COLLECTIVE_CORE_WORDS (NUM_BANKS / NUM_CORES)
#define DMA_COLLECTIVE_TCDM_SIZE (NUM_BANKS * L1_BANK_SIZE)
#define DMA_COLLECTIVE_ROW_BYTES (NUM_BANKS * sizeof(uint32_t))
#define DMA_COLLECTIVE_GROUP_BYTES (DMA_COLLECTIVE_ROW_BYTES / NUM_GROUPS)
#define DMA_COLLECTIVE_L2_STRIPE_BYTES                                         \
  ((AXI_DATA_WIDTH / 8u) * AXI_WIDTH_INTERLEAVED)
#define DMA_COLLECTIVE_L2_PERIOD_BYTES                                         \
  (DMA_COLLECTIVE_L2_STRIPE_BYTES * L2_BANKS)
#define DMA_COLLECTIVE_LEVELS ((uint32_t)VC_LEVELS + 1u)
#define DMA_COLLECTIVE_STORAGE                                                 \
  __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")))

#define DMA_COLLECTIVE_POLL_CYCLES 4u // delay between lock and READY polls

#ifndef RO_LINE_WIDTH
#error "RO_LINE_WIDTH undefined: the runtime must be built through runtime.mk"
#endif
#if RO_LINE_WIDTH > 512
#error "link.ld aligns .text.dma_hot to 64 bytes, less than one cache line"
#endif
// One line of the group read-only cache, in bytes.
#define DMA_COLLECTIVE_CACHE_LINE ((uint32_t)RO_LINE_WIDTH / 8u)

// The whole-cluster issue and wait paths live in their own line-aligned
// section, kept contiguous by link.ld and prefetched into each group's
// read-only cache at init, so their first run does not fetch every line from
// L2 in every group at once.
#define DMA_HOT                                                                \
  __attribute__((section(".text.dma_hot"), aligned(RO_LINE_WIDTH / 8)))
#define DMA_COLD __attribute__((section(".text.dma_cold"), noinline, cold))
// L2 folds whole rows and whole group stripes, so a row-aligned range needs no
// splitting at L2 stripe boundaries.
#define DMA_COLLECTIVE_ROWS_FOLD                                               \
  (((DMA_COLLECTIVE_ROW_BYTES % DMA_COLLECTIVE_L2_PERIOD_BYTES) == 0u) &&      \
   ((DMA_COLLECTIVE_L2_STRIPE_BYTES % DMA_COLLECTIVE_GROUP_BYTES) == 0u))
// The fast paths also use masks for these, so they need powers of two.
#define DMA_COLLECTIVE_FAST_GEOMETRY                                           \
  (((DMA_COLLECTIVE_ROW_BYTES & (DMA_COLLECTIVE_ROW_BYTES - 1u)) == 0u) &&     \
   ((NUM_CORES_PER_GROUP & (NUM_CORES_PER_GROUP - 1u)) == 0u) &&               \
   DMA_COLLECTIVE_ROWS_FOLD)

// Per-group control block, written by the group coordinator. Padded to one
// group bank stripe, so entry g lands in group g's banks.
#define DMA_COLLECTIVE_GROUP_FIELDS (5 + NUM_DMAS_PER_GROUP)
#if DMA_COLLECTIVE_GROUP_WORDS <= DMA_COLLECTIVE_GROUP_FIELDS
#error "The DMA group control block does not fit in one group bank stripe"
#endif

typedef struct {
  volatile dma_id_t last_id[NUM_DMAS_PER_GROUP]; // newest issued id, 0 = none
  volatile uint32_t error;       // first launch error since the last wait
  volatile uint32_t ready_seq;   // every operation up to this one has retired
  volatile uint32_t ready_error; // error published with ready_seq
  volatile uint32_t fence_seq;   // operation whose barrier fence_error names
  volatile uint32_t fence_error; // stored into EVERY group of a failing scope
  uint32_t padding[DMA_COLLECTIVE_GROUP_WORDS - DMA_COLLECTIVE_GROUP_FIELDS];
} dma_collective_group_t;

// Per-core operation count, padded to one core's bank share so entry c lands
// in core c's tile.
#if DMA_COLLECTIVE_CORE_WORDS < 3
#error "The DMA per-core state does not fit in one core bank share"
#endif

typedef struct {
  volatile uint32_t seq;
  uint32_t aggr_all; // fractal_sync aggregate of the whole cluster, 0 = none
  uint32_t padding[DMA_COLLECTIVE_CORE_WORDS - 2];
} dma_collective_core_t;

static dma_collective_group_t
    dma_collective_group[NUM_GROUPS] DMA_COLLECTIVE_STORAGE;
static dma_collective_core_t
    dma_collective_core[NUM_CORES] DMA_COLLECTIVE_STORAGE;
// Rendezvous for scopes the fractal_sync tree cannot express.
static mempool_poll_slot_t dma_collective_poll[DMA_COLLECTIVE_LEVELS]
                                              [NUM_GROUPS]
    __attribute__((section(".l1")));

extern const char __dma_hot_start[], __dma_hot_end[];
// Load the hot code into the group read-only cache by reading one word per
// cache line as data, so its first run fills lines from the group cache.
DMA_COLD static void dma_collective_prefetch_hot(void) {
  volatile uint32_t sink = 0u;
  for (const char *p = __dma_hot_start; p < __dma_hot_end;
       p += DMA_COLLECTIVE_CACHE_LINE)
    sink = *(const uint32_t *)(const void *)p;
  (void)sink;
}

#ifdef DMA_COLLECTIVE_PROFILE
dma_prof_group_t dma_prof[NUM_GROUPS] DMA_COLLECTIVE_STORAGE;
#define DMA_PROF_COORD(i)                                                      \
  do {                                                                         \
    if (mempool_is_group_coordinator())                                        \
      DMA_PROF(i);                                                             \
  } while (0)
#else
#define DMA_PROF_COORD(i)                                                      \
  do {                                                                         \
  } while (0)
#endif

// The engine's descriptor registers are one context per group, so the lock is
// per group. The collective does not need it -- it has one issuer per group --
// but raw dma_local_* callers issuing from several tiles do (dma.h). `.l1`
// powers up X, so the lock must be zeroed before the first acquire.
void dma_collective_runtime_init(void) {
  const uint32_t core = mempool_get_core_id();
  dma_collective_core[core].seq = 0u;
#if FSYNC_TPG_SIDE != 0
  dma_collective_core[core].aggr_all = fsync_aggr_global();
#else
  dma_collective_core[core].aggr_all = 0u;
#endif
  if (mempool_is_group_coordinator()) {
    dma_collective_group_t *const control =
        &dma_collective_group[mempool_get_group_id()];
    for (uint32_t s = 0u; s < (uint32_t)NUM_DMAS_PER_GROUP; ++s)
      control->last_id[s] = 0u;
    control->error = DMA_COLLECTIVE_OK;
    control->ready_seq = 0u;
    control->ready_error = DMA_COLLECTIVE_OK;
    control->fence_seq = 0u;
    control->fence_error = DMA_COLLECTIVE_OK;
    dma_collective_prefetch_hot();
  }
  if (core == 0u) {
    for (uint32_t k = 0u; k < DMA_COLLECTIVE_LEVELS; ++k)
      for (uint32_t g = 0u; g < (uint32_t)NUM_GROUPS; ++g)
        mempool_barrier_poll_init(&dma_collective_poll[k][g]);
  }
}

// The engine accepted no descriptor: NEXT_ID read back as 0. Errors are a
// software matter only -- the engine has no error report to fold in.
static inline uint32_t dma_collective_launch_error(void) {
  return DMA_COLLECTIVE_ERROR_LAUNCH;
}

static uint32_t
dma_collective_scope_valid(const dma_collective_scope_t *scope) {
  if (scope == NULL || scope->level > (uint32_t)VC_LEVELS)
    return 0;
  const uint32_t n = scope->level == (uint32_t)VC_LEVELS
                         ? (uint32_t)NUM_GROUPS
                         : 1u << (2u * scope->level);
  return scope->num_groups == n && (scope->base_group & (n - 1u)) == 0u &&
         scope->base_group < (uint32_t)NUM_GROUPS &&
         n <= (uint32_t)NUM_GROUPS - scope->base_group;
}

static uint32_t
dma_collective_group_in_scope(const dma_collective_scope_t *scope,
                              uint32_t group) {
  return group >= scope->base_group &&
         group < scope->base_group + scope->num_groups;
}

// The scope's core rendezvous: the fractal_sync tree where it can express the
// scope, software otherwise. Every core of the scope calls.
static void dma_collective_barrier(const dma_collective_scope_t *scope) {
  const uint32_t all = scope->num_groups == (uint32_t)NUM_GROUPS;
#if FSYNC_TPG_SIDE != 0
  const uint32_t aggr =
      all ? fsync_aggr_global() : fsync_aggr_of_vc_level(scope->level);
  // A zero aggregate is a single-tile scope, which the tree refuses.
  if (aggr != 0u) {
    fsync_barrier(fsync_tile_counters, mempool_get_tile_id(), FSYNC_ID_H, aggr);
    return;
  }
#endif
  if (all)
    mempool_barrier((uint32_t)NUM_CORES);
  else
    mempool_barrier_poll_slot(
        &dma_collective_poll[scope->level][scope->base_group],
        scope->num_groups * (uint32_t)NUM_CORES_PER_GROUP);
}

static uint32_t dma_collective_group_row_range(uint32_t start, uint32_t end,
                                               uint32_t group,
                                               uint32_t *first_row,
                                               uint32_t *last_row) {
  const uint32_t group_start = group * DMA_COLLECTIVE_GROUP_BYTES;
  const uint32_t group_end = group_start + DMA_COLLECTIVE_GROUP_BYTES;

  uint32_t first = start / DMA_COLLECTIVE_ROW_BYTES;
  if (first * DMA_COLLECTIVE_ROW_BYTES + group_end <= start)
    ++first;

  uint32_t last = (end - 1u) / DMA_COLLECTIVE_ROW_BYTES;
  if (last * DMA_COLLECTIVE_ROW_BYTES + group_start >= end) {
    if (last == 0)
      return 0;
    --last;
  }
  if (first > last)
    return 0;

  *first_row = first;
  *last_row = last;
  return 1;
}

// Does every group that owns part of [start, end) lie inside the scope? The
// whole cluster and a single-row range are settled directly; a partial scope
// spanning rows tests each group.
static uint32_t
dma_collective_scope_covers_range(const dma_collective_scope_t *scope,
                                  uint32_t start, uint32_t end) {
  if (scope->base_group == 0u && scope->num_groups == (uint32_t)NUM_GROUPS)
    return 1;

  if (start / DMA_COLLECTIVE_ROW_BYTES ==
      (end - 1u) / DMA_COLLECTIVE_ROW_BYTES) {
    const uint32_t g_first =
        (start % DMA_COLLECTIVE_ROW_BYTES) / DMA_COLLECTIVE_GROUP_BYTES;
    const uint32_t g_last =
        ((end - 1u) % DMA_COLLECTIVE_ROW_BYTES) / DMA_COLLECTIVE_GROUP_BYTES;
    return g_first >= scope->base_group &&
           g_last < scope->base_group + scope->num_groups;
  }

  for (uint32_t group = 0; group < (uint32_t)NUM_GROUPS; ++group) {
    uint32_t first_row;
    uint32_t last_row;
    if (dma_collective_group_row_range(start, end, group, &first_row,
                                       &last_row) &&
        !dma_collective_group_in_scope(scope, group))
      return 0;
  }
  return 1;
}

static dma_id_t dma_collective_issue_piece(uint32_t stream, uint32_t dest,
                                           uint32_t src, uint32_t l1_start,
                                           uint32_t l1_piece, uint32_t len,
                                           uint32_t reps, uint32_t dest_in_l1,
                                           uint32_t *error,
                                           uint32_t *descriptors) {
  dma_id_t last_id = 0;
  if (reps > 1 &&
      DMA_COLLECTIVE_ROW_BYTES % DMA_COLLECTIVE_L2_PERIOD_BYTES != 0) {
    *error = DMA_COLLECTIVE_ERROR_ARGUMENT;
    return 0;
  }

  while (len != 0) {
    const uint32_t delta = l1_piece - l1_start;
    const uint32_t external_addr = dest_in_l1 ? src + delta : dest + delta;
    const uint32_t bytes_to_l2_boundary =
        DMA_COLLECTIVE_L2_STRIPE_BYTES -
        external_addr % DMA_COLLECTIVE_L2_STRIPE_BYTES;
    const uint32_t chunk =
        len < bytes_to_l2_boundary ? len : bytes_to_l2_boundary;
    void *const piece_dest = (void *)(dest_in_l1 ? l1_piece : dest + delta);
    const void *const piece_src =
        (const void *)(dest_in_l1 ? src + delta : l1_piece);

    if (reps == 1)
      last_id =
          dma_local_memcpy_nonblocking(stream, piece_dest, piece_src, chunk);
    else
      last_id = dma_local_memcpy_2d_nonblocking(stream, piece_dest, piece_src,
                                                chunk, DMA_COLLECTIVE_ROW_BYTES,
                                                DMA_COLLECTIVE_ROW_BYTES, reps);
    ++*descriptors;
    if (last_id == 0) {
      *error = dma_collective_launch_error();
      return 0;
    }
    l1_piece += chunk;
    len -= chunk;
  }
  return last_id;
}

static dma_id_t dma_collective_issue_linear(uint32_t stream, void *dest,
                                            const void *src, size_t len,
                                            uint32_t group, uint32_t *error,
                                            uint32_t *descriptors) {
  const uint32_t dest_addr = (uint32_t)dest;
  const uint32_t src_addr = (uint32_t)src;
  const uint32_t dest_in_l1 = dest_addr < (uint32_t)DMA_COLLECTIVE_TCDM_SIZE;
  const uint32_t src_in_l1 = src_addr < (uint32_t)DMA_COLLECTIVE_TCDM_SIZE;
  const uint32_t l1_start = dest_in_l1 ? dest_addr : src_addr;
  const uint32_t l1_end = l1_start + (uint32_t)len;
  uint32_t first_row;
  uint32_t last_row;

  // The caller validated the range. Row-aligned, every group has the same
  // shape -- GROUP_BYTES per row at stride ROW_BYTES -- so one 2-D descriptor
  // replaces the head/middle/tail split below.
  *error = 0;
  *descriptors = 0;
  if ((l1_start % DMA_COLLECTIVE_ROW_BYTES) == 0u &&
      ((uint32_t)len % DMA_COLLECTIVE_ROW_BYTES) == 0u &&
      DMA_COLLECTIVE_ROWS_FOLD) {
    const uint32_t rows = (uint32_t)len / DMA_COLLECTIVE_ROW_BYTES;
    const uint32_t off = group * DMA_COLLECTIVE_GROUP_BYTES;
    void *const piece_dest = (void *)(dest_addr + off);
    const void *const piece_src = (const void *)(src_addr + off);
    *descriptors = 1;
    const dma_id_t id = dma_local_memcpy_2d_nonblocking(
        stream, piece_dest, piece_src, DMA_COLLECTIVE_GROUP_BYTES,
        DMA_COLLECTIVE_ROW_BYTES, DMA_COLLECTIVE_ROW_BYTES, rows);
    if (id == 0)
      *error = dma_collective_launch_error();
    return id;
  }

  if (!dma_collective_group_row_range(l1_start, l1_end, group, &first_row,
                                      &last_row))
    return 0;

  const uint32_t group_start = group * DMA_COLLECTIVE_GROUP_BYTES;
  dma_id_t last_id = 0;

  const uint32_t first_region =
      first_row * DMA_COLLECTIVE_ROW_BYTES + group_start;
  const uint32_t first_start =
      l1_start > first_region ? l1_start : first_region;
  const uint32_t first_limit = first_region + DMA_COLLECTIVE_GROUP_BYTES;
  const uint32_t first_end = l1_end < first_limit ? l1_end : first_limit;
  last_id = dma_collective_issue_piece(stream, dest_addr, src_addr, l1_start,
                                       first_start, first_end - first_start, 1,
                                       dest_in_l1, error, descriptors);
  if (*error != 0)
    return last_id;

  if (last_row > first_row + 1u) {
    const uint32_t middle_start =
        (first_row + 1u) * DMA_COLLECTIVE_ROW_BYTES + group_start;
    const uint32_t middle_reps = last_row - first_row - 1u;
    last_id =
        dma_collective_issue_piece(stream, dest_addr, src_addr, l1_start,
                                   middle_start, DMA_COLLECTIVE_GROUP_BYTES,
                                   middle_reps, dest_in_l1, error, descriptors);
    if (*error != 0)
      return last_id;
  }

  if (last_row != first_row) {
    const uint32_t last_region =
        last_row * DMA_COLLECTIVE_ROW_BYTES + group_start;
    const uint32_t last_start = l1_start > last_region ? l1_start : last_region;
    const uint32_t last_limit = last_region + DMA_COLLECTIVE_GROUP_BYTES;
    const uint32_t last_end = l1_end < last_limit ? l1_end : last_limit;
    last_id = dma_collective_issue_piece(stream, dest_addr, src_addr, l1_start,
                                         last_start, last_end - last_start, 1,
                                         dest_in_l1, error, descriptors);
    if (*error != 0)
      return last_id;
  }
  return last_id;
}

// Every core counts every submit; the count names the operation. An invalid
// scope still counts, so every core stays in step.
static dma_collective_op_t
dma_collective_begin(const dma_collective_scope_t *scope, uint32_t stream) {
  const uint32_t core = mempool_get_core_id();
  dma_collective_op_t op;
  op.seq = dma_collective_core[core].seq + 1u;
  dma_collective_core[core].seq = op.seq;
  op.status = DMA_COLLECTIVE_OK;
  if (!dma_collective_scope_valid(scope)) {
    op.scope.level = UINT32_MAX;
    op.scope.base_group = 0u;
    op.scope.num_groups = 0u;
    op.status = DMA_COLLECTIVE_ERROR_ARGUMENT;
    return op;
  }
  op.scope = *scope;
  if (stream >= (uint32_t)NUM_DMAS_PER_GROUP)
    op.status = DMA_COLLECTIVE_ERROR_ARGUMENT;
  else if (!dma_collective_group_in_scope(scope, mempool_get_group_id()))
    op.status = DMA_COLLECTIVE_ERROR_SCOPE;
  return op;
}

// Coordinator only: remember the newest id and the first launch error.
static void dma_collective_record(dma_collective_op_t *op, uint32_t stream,
                                  dma_id_t id, uint32_t error) {
  dma_collective_group_t *const control =
      &dma_collective_group[mempool_get_group_id()];
  if (id != 0u)
    control->last_id[stream] = id;
  if (error != DMA_COLLECTIVE_OK) {
    op->status = error;
    if (control->error == DMA_COLLECTIVE_OK)
      control->error = error;
  }
}

// Any scope and range; dma_collective_submit_nonblocking() takes it for
// everything its whole-cluster fast path does not cover.
DMA_COLD static dma_collective_op_t
dma_collective_submit_general(const dma_collective_scope_t *scope,
                              uint32_t stream, void *dest, const void *src,
                              size_t len) {
  DMA_PROF_COORD(0);
  dma_collective_op_t op = dma_collective_begin(scope, stream);
  if (op.status != DMA_COLLECTIVE_OK)
    return op;

  const uint32_t dest_addr = (uint32_t)dest;
  const uint32_t src_addr = (uint32_t)src;
  const uint32_t dest_in_l1 = dest_addr < (uint32_t)DMA_COLLECTIVE_TCDM_SIZE;
  const uint32_t src_in_l1 = src_addr < (uint32_t)DMA_COLLECTIVE_TCDM_SIZE;
  const uint32_t l1_start = dest_in_l1 ? dest_addr : src_addr;
  if (dest_in_l1 == src_in_l1 || len == 0 ||
      len > (size_t)((uint32_t)DMA_COLLECTIVE_TCDM_SIZE - l1_start)) {
    op.status = DMA_COLLECTIVE_ERROR_ARGUMENT;
    return op;
  }
  if (!dma_collective_scope_covers_range(scope, l1_start,
                                         l1_start + (uint32_t)len)) {
    op.status = DMA_COLLECTIVE_ERROR_SCOPE;
    return op;
  }
  if (!mempool_is_group_coordinator())
    return op;

  uint32_t error = DMA_COLLECTIVE_OK;
  uint32_t descriptors = 0u;
  const dma_id_t id = dma_collective_issue_linear(
      stream, dest, src, len, mempool_get_group_id(), &error, &descriptors);
  dma_collective_record(&op, stream, id, error);
  DMA_PROF_COORD(1);
  return op;
}

DMA_COLD static void dma_collective_launch_failed(dma_collective_op_t *op,
                                                  uint32_t group,
                                                  uint32_t stream) {
  const uint32_t error = dma_collective_launch_error();
  op->status = error;
  if (dma_collective_group[group].error == DMA_COLLECTIVE_OK)
    dma_collective_group[group].error = error;
}

// Only the fallback of the scope-less entry keeps a scope in memory.
DMA_COLD static dma_collective_op_t
dma_collective_submit_all_general(uint32_t stream, void *dest, const void *src,
                                  size_t len) {
  const dma_collective_scope_t scope = dma_collective_scope_all();
  return dma_collective_submit_general(&scope, stream, dest, src, len);
}

// Hot path: the whole cluster and a row-aligned global L1 <-> L2 range, which
// is what dma_collective_memcpy_*() issues. Short and linear, so it spans few
// 64-byte instruction-cache lines; everything else takes the general path.
DMA_HOT dma_collective_op_t dma_collective_submit_all_nonblocking(
    uint32_t stream, void *dest, const void *src, size_t len) {
  const uint32_t dest_addr = (uint32_t)dest;
  const uint32_t src_addr = (uint32_t)src;
  const uint32_t dest_in_l1 = dest_addr < (uint32_t)DMA_COLLECTIVE_TCDM_SIZE;
  const uint32_t src_in_l1 = src_addr < (uint32_t)DMA_COLLECTIVE_TCDM_SIZE;
  const uint32_t l1_start = dest_in_l1 ? dest_addr : src_addr;
  const uint32_t n = (uint32_t)len;
  if (__builtin_expect(
          DMA_COLLECTIVE_FAST_GEOMETRY &&
              stream < (uint32_t)NUM_DMAS_PER_GROUP &&
              dest_in_l1 != src_in_l1 && n != 0u && (size_t)n == len &&
              ((l1_start | n) & (DMA_COLLECTIVE_ROW_BYTES - 1u)) == 0u &&
              n <= (uint32_t)DMA_COLLECTIVE_TCDM_SIZE - l1_start,
          1)) {
    const uint32_t hart = mempool_get_core_id();
    dma_collective_op_t op;
    op.scope.level = (uint32_t)VC_LEVELS;
    op.scope.base_group = 0u;
    op.scope.num_groups = (uint32_t)NUM_GROUPS;
    op.seq = dma_collective_core[hart].seq + 1u;
    dma_collective_core[hart].seq = op.seq;
    op.status = DMA_COLLECTIVE_OK;
    if ((hart & ((uint32_t)NUM_CORES_PER_GROUP - 1u)) == 0u) {
      DMA_PROF(0);
      const uint32_t group = hart / (uint32_t)NUM_CORES_PER_GROUP;
      const uint32_t off = group * (uint32_t)DMA_COLLECTIVE_GROUP_BYTES;
      const dma_id_t id = dma_local_memcpy_2d_nonblocking(
          stream, (void *)(dest_addr + off), (const void *)(src_addr + off),
          DMA_COLLECTIVE_GROUP_BYTES, DMA_COLLECTIVE_ROW_BYTES,
          DMA_COLLECTIVE_ROW_BYTES, n / (uint32_t)DMA_COLLECTIVE_ROW_BYTES);
      if (__builtin_expect(id != 0u, 1))
        dma_collective_group[group].last_id[stream] = id;
      else
        dma_collective_launch_failed(&op, group, stream);
      DMA_PROF(1);
    }
    return op;
  }
  return dma_collective_submit_all_general(stream, dest, src, len);
}

dma_collective_op_t
dma_collective_submit_nonblocking(const dma_collective_scope_t *scope,
                                  uint32_t stream, void *dest, const void *src,
                                  size_t len) {
  if (scope != NULL && scope->num_groups == (uint32_t)NUM_GROUPS &&
      scope->base_group == 0u && scope->level == (uint32_t)VC_LEVELS)
    return dma_collective_submit_all_nonblocking(stream, dest, src, len);
  return dma_collective_submit_general(scope, stream, dest, src, len);
}

dma_collective_op_t dma_collective_submit_2d_nonblocking(
    const dma_collective_scope_t *scope, uint32_t stream, void *dest,
    const void *src, size_t len, uint32_t dst_stride, uint32_t src_stride,
    uint32_t reps) {
  DMA_PROF_COORD(0);
  dma_collective_op_t op = dma_collective_begin(scope, stream);
  if (op.status != DMA_COLLECTIVE_OK)
    return op;
  if (len == 0 || reps == 0) {
    op.status = DMA_COLLECTIVE_ERROR_ARGUMENT;
    return op;
  }
  if (!mempool_is_group_coordinator())
    return op;

  const dma_id_t id = dma_local_memcpy_2d_nonblocking(
      stream, dest, src, len, dst_stride, src_stride, reps);
  const uint32_t error =
      id != 0u ? DMA_COLLECTIVE_OK : dma_collective_launch_error();
  dma_collective_record(&op, stream, id, error);
  DMA_PROF_COORD(1);
  return op;
}

// Coordinator only: sleep until everything its group has outstanding retires,
// then publish READY through `seq`. Waiting the last id waits every earlier
// one.
static inline __attribute__((always_inline)) uint32_t
dma_collective_retire(dma_collective_group_t *control, uint32_t seq) {
  uint32_t error = control->error;
  for (uint32_t s = 0u; s < (uint32_t)NUM_DMAS_PER_GROUP; ++s) {
    const dma_id_t id = control->last_id[s];
    if (id == 0u)
      continue;
    const uint32_t wait_error = dma_local_wait_id(s, id);
    if (error == DMA_COLLECTIVE_OK)
      error = wait_error;
    control->last_id[s] = 0u;
  }
  control->error = DMA_COLLECTIVE_OK;
  // Plain store: readers reach it only after acquiring ready_seq below.
  control->ready_error = error;
  __atomic_store_n(&control->ready_seq, seq, __ATOMIC_RELEASE);
  return error;
}

// Tell every group of a failing scope before the barrier releases anyone, so
// each core learns it from its own group's banks.
DMA_COLD static void
dma_collective_fence_scope(const dma_collective_scope_t *scope, uint32_t seq,
                           uint32_t error) {
  const uint32_t end = scope->base_group + scope->num_groups;
  for (uint32_t g = scope->base_group; g < end; ++g) {
    dma_collective_group[g].fence_error = error;
    __atomic_store_n(&dma_collective_group[g].fence_seq, seq, __ATOMIC_RELEASE);
  }
  __sync_synchronize();
}

// Any scope; dma_collective_wait() takes it beyond its whole-cluster fast path.
DMA_COLD static uint32_t
dma_collective_wait_general(const dma_collective_op_t *last) {
  if (last == NULL || !dma_collective_scope_valid(&last->scope))
    return DMA_COLLECTIVE_ERROR_ARGUMENT;
  const uint32_t group = mempool_get_group_id();
  if (!dma_collective_group_in_scope(&last->scope, group))
    return DMA_COLLECTIVE_ERROR_SCOPE;

  const uint32_t coordinator = mempool_is_group_coordinator();
  const uint32_t seq = dma_collective_core[mempool_get_core_id()].seq;
  uint32_t error = DMA_COLLECTIVE_OK;
  if (coordinator) {
    DMA_PROF(2);
    error = dma_collective_retire(&dma_collective_group[group], seq);
    DMA_PROF(3);
    if (error != DMA_COLLECTIVE_OK)
      dma_collective_fence_scope(&last->scope, seq, error);
  }

  dma_collective_barrier(&last->scope);
  DMA_PROF_COORD(4);

  if (error == DMA_COLLECTIVE_OK &&
      __atomic_load_n(&dma_collective_group[group].fence_seq,
                      __ATOMIC_ACQUIRE) == seq)
    error = dma_collective_group[group].fence_error;
  __sync_synchronize();
  DMA_PROF_COORD(5);
  return error;
}

// Hot path: a whole-cluster operation. The identity is computed once and the
// barrier aggregate was precomputed by dma_collective_runtime_init().
DMA_HOT uint32_t dma_collective_wait(const dma_collective_op_t *last) {
  if (__builtin_expect(DMA_COLLECTIVE_FAST_GEOMETRY && last != NULL &&
                           last->scope.num_groups == (uint32_t)NUM_GROUPS &&
                           last->scope.base_group == 0u &&
                           last->scope.level == (uint32_t)VC_LEVELS,
                       1)) {
    const uint32_t hart = mempool_get_core_id();
    const uint32_t group = hart / (uint32_t)NUM_CORES_PER_GROUP;
    const uint32_t seq = dma_collective_core[hart].seq;
    dma_collective_group_t *const control = &dma_collective_group[group];
    uint32_t error = DMA_COLLECTIVE_OK;
    if ((hart & ((uint32_t)NUM_CORES_PER_GROUP - 1u)) == 0u) {
      DMA_PROF(2);
      error = dma_collective_retire(control, seq);
      DMA_PROF(3);
      if (__builtin_expect(error != DMA_COLLECTIVE_OK, 0))
        dma_collective_fence_scope(&last->scope, seq, error);
    }
#if FSYNC_TPG_SIDE != 0
    const uint32_t aggr = dma_collective_core[hart].aggr_all;
    if (aggr != 0u)
      fsync_barrier(fsync_tile_counters, hart / (uint32_t)NUM_CORES_PER_TILE,
                    FSYNC_ID_H, aggr);
    else
#endif
      mempool_barrier((uint32_t)NUM_CORES);
    DMA_PROF_COORD(4);
    if (error == DMA_COLLECTIVE_OK &&
        __atomic_load_n(&control->fence_seq, __ATOMIC_ACQUIRE) == seq)
      error = control->fence_error;
    __sync_synchronize();
    DMA_PROF_COORD(5);
    return error;
  }
  return dma_collective_wait_general(last);
}

uint32_t dma_collective_wait_ready(const dma_collective_op_t *op) {
  if (op == NULL || !dma_collective_scope_valid(&op->scope))
    return DMA_COLLECTIVE_ERROR_ARGUMENT;
  if (!dma_collective_group_in_scope(&op->scope, mempool_get_group_id()))
    return DMA_COLLECTIVE_ERROR_SCOPE;

  uint32_t error = DMA_COLLECTIVE_OK;
  if (mempool_is_group_coordinator())
    error =
        dma_collective_retire(&dma_collective_group[mempool_get_group_id()],
                              dma_collective_core[mempool_get_core_id()].seq);

  const uint32_t end = op->scope.base_group + op->scope.num_groups;
  for (uint32_t g = op->scope.base_group; g < end; ++g) {
    dma_collective_group_t *const control = &dma_collective_group[g];
    while (!dma_local_id_done(
        __atomic_load_n(&control->ready_seq, __ATOMIC_ACQUIRE), op->seq))
      mempool_wait(DMA_COLLECTIVE_POLL_CYCLES);
    if (error == DMA_COLLECTIVE_OK)
      error = control->ready_error;
  }
  __sync_synchronize();
  return error;
}
