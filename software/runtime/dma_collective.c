// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "dma_collective.h"

#include <stdbool.h>
#include <stdint.h>

#include "fsync.h"
#include "runtime.h"
#include "synchronization.h"

#define DMA_COLLECTIVE_TILE_WORDS NUM_BANKS_PER_TILE
#define DMA_COLLECTIVE_NUM_TILES (NUM_CORES / NUM_CORES_PER_TILE)
#define DMA_COLLECTIVE_TCDM_SIZE (NUM_BANKS * L1_BANK_SIZE)
#define DMA_COLLECTIVE_ROW_BYTES (NUM_BANKS * sizeof(uint32_t))
#define DMA_COLLECTIVE_GROUP_BYTES (DMA_COLLECTIVE_ROW_BYTES / NUM_GROUPS)
#define DMA_COLLECTIVE_L2_STRIPE_BYTES                                         \
  ((AXI_DATA_WIDTH / 8u) * AXI_WIDTH_INTERLEAVED)
#define DMA_COLLECTIVE_L2_PERIOD_BYTES                                         \
  (DMA_COLLECTIVE_L2_STRIPE_BYTES * L2_BANKS)

typedef struct {
  volatile uint32_t value;
  uint32_t padding[DMA_COLLECTIVE_TILE_WORDS - 1];
} dma_collective_tile_lock_t;

#define DMA_COLLECTIVE_ALL_FIELDS 14u

#if DMA_COLLECTIVE_GROUP_WORDS < DMA_COLLECTIVE_ALL_FIELDS
#error "The default collective control does not fit in one group bank stripe"
#endif

typedef struct {
  volatile uint32_t generation;
  dma_collective_handle_t handle;
  // Failure report for the hardware fence. A group that fails writes its code
  // into EVERY group's copy before the barrier, so the common path costs each
  // reader one load from its own group's banks instead of a remote poll.
  volatile uint32_t fence_error;
  volatile uint32_t fence_generation;
  // Solo path: this group's in-flight transfer. Written and read by the same
  // core, in its own group's banks, so it replaces the whole slot machinery.
  volatile uint32_t solo_id;
  volatile uint32_t solo_error;
  uint32_t padding[DMA_COLLECTIVE_GROUP_WORDS - DMA_COLLECTIVE_ALL_FIELDS];
} dma_collective_all_group_t;

#ifdef DMA_COLLECTIVE_PROFILE
dma_prof_group_t dma_prof[NUM_GROUPS] DMA_COLLECTIVE_STORAGE;
#endif

static dma_collective_tile_lock_t
    dma_collective_tile_lock[DMA_COLLECTIVE_NUM_TILES] DMA_COLLECTIVE_STORAGE;
static dma_collective_state_t dma_collective_all_state DMA_COLLECTIVE_STORAGE;
static dma_collective_all_group_t
    dma_collective_all_group[NUM_GROUPS] DMA_COLLECTIVE_STORAGE;
#if FSYNC_TPG_SIDE != 0
// One full bank stripe per tile, so a tile's counter lands in its own banks.
static fsync_tile_counter_t dma_collective_fsync_cnt[DMA_COLLECTIVE_NUM_TILES]
    __attribute__((aligned(FSYNC_TILE_COUNTER_ALIGNMENT), section(".l1")));
#endif

static uint32_t
dma_collective_scope_valid(const dma_collective_scope_t *scope) {
  return scope != NULL && scope->num_groups != 0 &&
         scope->base_group < (uint32_t)NUM_GROUPS &&
         scope->num_groups <= (uint32_t)NUM_GROUPS - scope->base_group &&
         scope->cores_per_group == (uint32_t)NUM_CORES_PER_GROUP;
}

static uint32_t
dma_collective_group_in_scope(const dma_collective_scope_t *scope,
                              uint32_t group) {
  return group >= scope->base_group &&
         group < scope->base_group + scope->num_groups;
}

static uint32_t dma_collective_slot_index(uint32_t generation) {
  return (generation - 1u) % DMA_COLLECTIVE_SLOTS;
}

static uint32_t dma_collective_epoch_done(uint32_t done, uint32_t target) {
  return (int32_t)(done - target) >= 0;
}

static dma_collective_handle_t
dma_collective_invalid_handle(dma_collective_state_t *state,
                              uint32_t generation, uint32_t stream,
                              uint32_t status) {
  const dma_collective_handle_t handle = {
      .state = state,
      .generation = generation,
      .slot = generation == 0 ? 0 : dma_collective_slot_index(generation),
      .group_id = mempool_get_group_id(),
      .base_group = 0,
      .num_groups = 0,
      .stream = stream,
      .local_id = 0,
      .status = status,
  };
  return handle;
}

// The engine's descriptor registers are ONE context per group, so this lock is
// per group. It used to be per tile, which was correct only while every tile
// owned its own context: with a shared one, two tiles writing their descriptors
// at the same time interleave SRC/DST/LENGTH into a single register set and
// each launches the other's transfer.
//
// The array entries are padded to a tile stripe, so entry t lands in TILE t's
// banks. Index by the group's FIRST tile rather than by the group id, or group
// g's lock would sit in whichever group owns tile g and every acquire would
// leave the group.
#define DMA_COLLECTIVE_LOCK_INDEX                                              \
  (mempool_get_group_id() * (uint32_t)NUM_TILES_PER_GROUP)

// The lock lives in `.l1`, which is NOBITS and powers up X, so it MUST be
// zeroed before the first acquire or __atomic_exchange_n never returns 0 and
// every issuer spins forever. Exported next to the lock itself: a caller
// outside this file has no other way to reach the storage.
void dma_group_ctx_init(void) {
  if (mempool_is_tile_agent()) {
    dma_collective_tile_lock_t *const lock =
        &dma_collective_tile_lock[DMA_COLLECTIVE_LOCK_INDEX];
    lock->value = 0u;
    for (uint32_t i = 0u; i < DMA_COLLECTIVE_TILE_WORDS - 1u; ++i)
      lock->padding[i] = 0u;
  }
  __sync_synchronize();
}

void dma_group_ctx_lock(void) {
  const uint32_t idx = DMA_COLLECTIVE_LOCK_INDEX;
  while (__atomic_exchange_n(&dma_collective_tile_lock[idx].value, 1,
                             __ATOMIC_ACQUIRE) != 0) {
    mempool_wait(4u);
  }
}

void dma_group_ctx_unlock(void) {
  const uint32_t idx = DMA_COLLECTIVE_LOCK_INDEX;
  __atomic_store_n(&dma_collective_tile_lock[idx].value, 0, __ATOMIC_RELEASE);
}

// Claim for a caller that is the SOLE issuer of this group -- the agent path.
//
// The SPMD claim below must elect one issuer among every core of the group,
// which costs a CAS plus three fenced acquire loads. dma_collective_all_* is
// coordinator-only (see dma_collective_all_memcpy_nonblocking), so there is
// exactly one caller and that election is provably uncontended: the CAS can
// never fail and the non-owner path can never run.
//
// What does NOT go away is the slot-reuse guard. With DMA_COLLECTIVE_SLOTS
// generations in flight, generation N reuses the slot of N-SLOTS, so that
// transfer must have landed. A sole issuer is also the sole publisher of
// submit_epoch/issued_epoch, so `issued == submit` holds by construction and
// only ready_epoch has to be read -- one acquire load instead of four fenced
// operations.
static dma_collective_handle_t
dma_collective_claim_agent(dma_collective_state_t *state,
                           const dma_collective_scope_t *scope,
                           uint32_t generation, uint32_t stream) {
  if (state == NULL || !dma_collective_scope_valid(scope) || generation == 0 ||
      stream >= (uint32_t)NUM_DMAS_PER_GROUP)
    return dma_collective_invalid_handle(state, generation, stream,
                                         DMA_COLLECTIVE_ERROR_ARGUMENT);

  const uint32_t group = mempool_get_group_id();
  if (!dma_collective_group_in_scope(scope, group))
    return dma_collective_invalid_handle(state, generation, stream,
                                         DMA_COLLECTIVE_ERROR_SCOPE);

  const uint32_t slot = dma_collective_slot_index(generation);
  dma_collective_slot_state_t *const slot_state =
      &state->group[group].slot[slot];
  dma_collective_handle_t handle = {
      .state = state,
      .generation = generation,
      .slot = slot,
      .group_id = group,
      .base_group = scope->base_group,
      .num_groups = scope->num_groups,
      .stream = stream,
      .local_id = 0,
      .status = DMA_COLLECTIVE_OK,
  };

  const uint32_t submit =
      __atomic_load_n(&slot_state->submit_epoch, __ATOMIC_RELAXED);
  if (submit == generation)
    return handle; // already claimed by this same agent; idempotent
  if (dma_collective_epoch_done(submit, generation)) {
    handle.status = DMA_COLLECTIVE_ERROR_STALE;
    return handle;
  }
  if (submit != 0u &&
      !dma_collective_epoch_done(
          __atomic_load_n(&slot_state->ready_epoch, __ATOMIC_ACQUIRE),
          submit)) {
    handle.status = DMA_COLLECTIVE_ERROR_REUSE;
    return handle;
  }
  // No CAS: a sole issuer has no competitor. Release so the drain and any
  // observer see a consistent slot before issued_epoch is published.
  __atomic_store_n(&slot_state->submit_epoch, generation, __ATOMIC_RELEASE);
  return handle;
}

static dma_collective_handle_t
dma_collective_claim(dma_collective_state_t *state,
                     const dma_collective_scope_t *scope, uint32_t generation,
                     uint32_t stream, uint32_t *owner) {
  *owner = 0;
  if (state == NULL || !dma_collective_scope_valid(scope) || generation == 0 ||
      stream >= (uint32_t)NUM_DMAS_PER_GROUP)
    return dma_collective_invalid_handle(state, generation, stream,
                                         DMA_COLLECTIVE_ERROR_ARGUMENT);

  const uint32_t group = mempool_get_group_id();
  if (!dma_collective_group_in_scope(scope, group))
    return dma_collective_invalid_handle(state, generation, stream,
                                         DMA_COLLECTIVE_ERROR_SCOPE);

  const uint32_t slot = dma_collective_slot_index(generation);
  dma_collective_slot_state_t *const slot_state =
      &state->group[group].slot[slot];
  dma_collective_handle_t handle = {
      .state = state,
      .generation = generation,
      .slot = slot,
      .group_id = group,
      .base_group = scope->base_group,
      .num_groups = scope->num_groups,
      .stream = stream,
      .local_id = 0,
      .status = DMA_COLLECTIVE_OK,
  };

  while (1) {
    uint32_t submit =
        __atomic_load_n(&slot_state->submit_epoch, __ATOMIC_ACQUIRE);
    if (submit != generation && dma_collective_epoch_done(submit, generation)) {
      handle.status = DMA_COLLECTIVE_ERROR_STALE;
      return handle;
    }
    if (submit == generation) {
      uint32_t issued;
      do {
        issued = __atomic_load_n(&slot_state->issued_epoch, __ATOMIC_ACQUIRE);
        if (issued != generation)
          mempool_wait(4u);
      } while (issued != generation);
      if (issued != generation) {
        handle.status = DMA_COLLECTIVE_ERROR_STALE;
        return handle;
      }
      handle.local_id =
          __atomic_load_n(&slot_state->local_id, __ATOMIC_RELAXED);
      return handle;
    }

    const uint32_t issued =
        __atomic_load_n(&slot_state->issued_epoch, __ATOMIC_ACQUIRE);
    const uint32_t ready =
        __atomic_load_n(&slot_state->ready_epoch, __ATOMIC_ACQUIRE);
    if (issued != submit || !dma_collective_epoch_done(ready, issued)) {
      handle.status = DMA_COLLECTIVE_ERROR_REUSE;
      return handle;
    }

    uint32_t expected = submit;
    if (__atomic_compare_exchange_n(&slot_state->submit_epoch, &expected,
                                    generation, false, __ATOMIC_ACQ_REL,
                                    __ATOMIC_ACQUIRE)) {
      *owner = 1;
      return handle;
    }
  }
}

static void dma_collective_publish(dma_collective_handle_t *handle,
                                   dma_id_t local_id, uint32_t error,
                                   uint32_t descriptors) {
  dma_collective_slot_state_t *const slot_state =
      &handle->state->group[handle->group_id].slot[handle->slot];
  // Plain volatile stores: the release store on issued_epoch below orders all
  // of them, and every reader reaches them only after acquiring that word. A
  // relaxed __atomic_store_n would add no ordering here but does compile to
  // amoswap.w, which blocks the bank for an extra cycle (tcdm_adapter.sv:262).
  slot_state->local_id = local_id;
  slot_state->error = error;
  // No runtime reader for these two: dma_collective_smoke asserts on them, and
  // a build-time gate would let that check pass vacuously. Two plain stores.
  slot_state->descriptors = descriptors;
  slot_state->issuer_core = mempool_get_core_id();
  slot_state->stream = handle->stream;
  __atomic_store_n(&slot_state->issued_epoch, handle->generation,
                   __ATOMIC_RELEASE);
  handle->local_id = local_id;
}

void dma_collective_init(dma_collective_state_t *state,
                         const dma_collective_scope_t *scope) {
  if (state == NULL || !dma_collective_scope_valid(scope))
    return;

  const uint32_t group = mempool_get_group_id();
  if (!dma_collective_group_in_scope(scope, group))
    return;

  if (mempool_is_group_coordinator()) {
    dma_collective_group_state_t *const group_state = &state->group[group];
    for (uint32_t slot = 0; slot < DMA_COLLECTIVE_SLOTS; ++slot) {
      group_state->slot[slot].submit_epoch = 0;
      group_state->slot[slot].issued_epoch = 0;
      group_state->slot[slot].wait_epoch = 0;
      group_state->slot[slot].ready_epoch = 0;
      group_state->slot[slot].local_id = 0;
      group_state->slot[slot].error = 0;
      group_state->slot[slot].descriptors = 0;
      group_state->slot[slot].issuer_core = 0;
      group_state->slot[slot].poller_core = 0;
      group_state->slot[slot].stream = 0;
    }
    for (uint32_t i = 0u;
         i < DMA_COLLECTIVE_GROUP_WORDS - DMA_COLLECTIVE_STATE_FIELDS; ++i)
      group_state->padding[i] = 0u;
  }
  dma_group_ctx_init();
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

// Does every group that owns part of [start, end) lie inside the scope?
//
// The general answer needs one row-range test per group, which is O(NUM_GROUPS)
// and grows with the mesh. Two cases settle it in O(1) first:
//
//   * a scope covering every group is a tautology -- whatever the range
//     touches is in scope. This is the whole-cluster collective, i.e. the
//     common case;
//   * a range confined to ONE row touches exactly the contiguous group span
//     [start/GROUP_BYTES, (end-1)/GROUP_BYTES] within that row, so the two
//     endpoints answer it directly.
//
// Anything else (a partial scope spanning rows) falls through to the loop.
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
      *error = dma_local_error(stream);
      if (*error == 0)
        *error = DMA_COLLECTIVE_ERROR_LAUNCH;
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

  *error = 0;
  *descriptors = 0;
  if (dest_in_l1 == src_in_l1 || len == 0 || len > UINT32_MAX ||
      l1_start >= (uint32_t)DMA_COLLECTIVE_TCDM_SIZE ||
      len > (size_t)((uint32_t)DMA_COLLECTIVE_TCDM_SIZE - l1_start)) {
    *error = DMA_COLLECTIVE_ERROR_ARGUMENT;
    return 0;
  }
  // Row-aligned fast path: O(1), one descriptor.
  //
  // A buffer that starts on a row boundary and spans a whole number of rows
  // gives every group the SAME shape -- GROUP_BYTES per row, at stride
  // ROW_BYTES, for `rows` rows -- so the head/middle/tail split below
  // degenerates into three descriptors that describe one rectangle. Emit that
  // rectangle directly and skip the decomposition entirely.
  //
  // The L2 side needs no stripe splitting either: L2_STRIPE_BYTES is
  // groups_per_l2_bank * GROUP_BYTES by construction (config.mk derives
  // axi_width_interleaved from the group L1 width), so a GROUP_BYTES chunk
  // starting at a multiple of GROUP_BYTES always lies inside one stripe. The
  // reps>1 precondition that issue_piece checks is asserted here too.
  if ((l1_start % DMA_COLLECTIVE_ROW_BYTES) == 0u &&
      ((uint32_t)len % DMA_COLLECTIVE_ROW_BYTES) == 0u &&
      (DMA_COLLECTIVE_ROW_BYTES % DMA_COLLECTIVE_L2_PERIOD_BYTES) == 0u &&
      (DMA_COLLECTIVE_L2_STRIPE_BYTES % DMA_COLLECTIVE_GROUP_BYTES) == 0u) {
    const uint32_t rows = (uint32_t)len / DMA_COLLECTIVE_ROW_BYTES;
    const uint32_t off = group * DMA_COLLECTIVE_GROUP_BYTES;
    void *const piece_dest = (void *)(dest_addr + off);
    const void *const piece_src = (const void *)(src_addr + off);
    *descriptors = 1;
    const dma_id_t id = dma_local_memcpy_2d_nonblocking(
        stream, piece_dest, piece_src, DMA_COLLECTIVE_GROUP_BYTES,
        DMA_COLLECTIVE_ROW_BYTES, DMA_COLLECTIVE_ROW_BYTES, rows);
    if (id == 0) {
      *error = dma_local_error(stream);
      if (*error == 0)
        *error = DMA_COLLECTIVE_ERROR_LAUNCH;
    }
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

static dma_collective_handle_t
dma_collective_memcpy_common(dma_collective_state_t *state,
                             const dma_collective_scope_t *scope,
                             uint32_t generation, uint32_t stream, void *dest,
                             const void *src, size_t len, uint32_t agent) {
  uint32_t owner = 1u;
  DMA_PROF(1);
  dma_collective_handle_t handle =
      agent ? dma_collective_claim_agent(state, scope, generation, stream)
            : dma_collective_claim(state, scope, generation, stream, &owner);
  DMA_PROF(2);
  if (handle.status != DMA_COLLECTIVE_OK || !owner)
    return handle;

  const uint32_t dest_addr = (uint32_t)dest;
  const uint32_t src_addr = (uint32_t)src;
  const uint32_t dest_in_l1 = dest_addr < (uint32_t)DMA_COLLECTIVE_TCDM_SIZE;
  const uint32_t l1_start = dest_in_l1 ? dest_addr : src_addr;
  uint32_t error = 0;
  uint32_t descriptors = 0;
  dma_id_t local_id = 0;

  if (len == 0 || len > UINT32_MAX ||
      l1_start >= (uint32_t)DMA_COLLECTIVE_TCDM_SIZE ||
      len > (size_t)((uint32_t)DMA_COLLECTIVE_TCDM_SIZE - l1_start) ||
      !dma_collective_scope_covers_range(scope, l1_start,
                                         l1_start + (uint32_t)len)) {
    error = DMA_COLLECTIVE_ERROR_SCOPE;
    DMA_PROF(3);
    DMA_PROF(4);
    DMA_PROF(5);
  } else {
    DMA_PROF(3);
    dma_group_ctx_lock();
    DMA_PROF(4);
    local_id = dma_collective_issue_linear(
        stream, dest, src, len, handle.group_id, &error, &descriptors);
    dma_group_ctx_unlock();
    DMA_PROF(5);
  }
  DMA_PROF(6);
  dma_collective_publish(&handle, local_id, error, descriptors);
  DMA_PROF(7);
  return handle;
}

// SPMD form: every core of every group in scope calls this. One wins the
// election and issues; the others come away with the same handle.
dma_collective_handle_t
dma_collective_memcpy_nonblocking(dma_collective_state_t *state,
                                  const dma_collective_scope_t *scope,
                                  uint32_t generation, uint32_t stream,
                                  void *dest, const void *src, size_t len) {
  return dma_collective_memcpy_common(state, scope, generation, stream, dest,
                                      src, len, 0u);
}

// Agent form: the caller asserts it is the SOLE issuer for its group. Skips the
// election, which is uncontended by construction in that case. Behaviour is
// otherwise identical; calling it from more than one core per group is a
// programming error and will corrupt the slot.
dma_collective_handle_t dma_collective_memcpy_agent_nonblocking(
    dma_collective_state_t *state, const dma_collective_scope_t *scope,
    uint32_t generation, uint32_t stream, void *dest, const void *src,
    size_t len) {
  return dma_collective_memcpy_common(state, scope, generation, stream, dest,
                                      src, len, 1u);
}

dma_collective_handle_t dma_collective_submit_2d_nonblocking(
    dma_collective_state_t *state, const dma_collective_scope_t *scope,
    uint32_t generation, uint32_t stream, void *dest, const void *src,
    size_t len, uint32_t dst_stride, uint32_t src_stride, uint32_t reps) {
  uint32_t owner;
  dma_collective_handle_t handle =
      dma_collective_claim(state, scope, generation, stream, &owner);
  if (handle.status != DMA_COLLECTIVE_OK || !owner)
    return handle;

  uint32_t error = 0;
  dma_id_t local_id = 0;
  if (len == 0 || reps == 0) {
    error = DMA_COLLECTIVE_ERROR_ARGUMENT;
  } else {
    dma_group_ctx_lock();
    local_id = dma_local_memcpy_2d_nonblocking(stream, dest, src, len,
                                               dst_stride, src_stride, reps);
    if (local_id == 0) {
      error = dma_local_error(stream);
      if (error == 0)
        error = DMA_COLLECTIVE_ERROR_LAUNCH;
    }
    dma_group_ctx_unlock();
  }
  dma_collective_publish(&handle, local_id, error, local_id != 0);
  return handle;
}

uint32_t dma_collective_wait_local(const dma_collective_handle_t *handle) {
  if (handle == NULL || handle->state == NULL || handle->generation == 0 ||
      handle->slot >= DMA_COLLECTIVE_SLOTS ||
      handle->group_id >= (uint32_t)NUM_GROUPS ||
      handle->stream >= (uint32_t)NUM_DMAS_PER_GROUP)
    return DMA_COLLECTIVE_ERROR_ARGUMENT;
  if (handle->status != DMA_COLLECTIVE_OK)
    return handle->status;

  dma_collective_group_state_t *const group_state =
      &handle->state->group[handle->group_id];
  dma_collective_slot_state_t *const slot_state =
      &group_state->slot[handle->slot];
  while (1) {
    const uint32_t ready =
        __atomic_load_n(&slot_state->ready_epoch, __ATOMIC_ACQUIRE);
    if (dma_collective_epoch_done(ready, handle->generation))
      return __atomic_load_n(&slot_state->error, __ATOMIC_RELAXED);

    uint32_t wait = __atomic_load_n(&slot_state->wait_epoch, __ATOMIC_ACQUIRE);
    if (wait != handle->generation &&
        dma_collective_epoch_done(wait, handle->generation))
      return DMA_COLLECTIVE_ERROR_STALE;
    if (wait == handle->generation)
      mempool_wait(4u);
    if (wait == handle->generation)
      continue;

    uint32_t expected = wait;
    if (__atomic_compare_exchange_n(&slot_state->wait_epoch, &expected,
                                    handle->generation, false, __ATOMIC_ACQ_REL,
                                    __ATOMIC_ACQUIRE)) {
      // Ordered by the ready_epoch release at the end of the drain loop.
      slot_state->poller_core = mempool_get_core_id();
      uint32_t error = __atomic_load_n(&slot_state->error, __ATOMIC_RELAXED);
      dma_id_t completed_id = 0;
      if (handle->local_id != 0) {
        const uint32_t wait_error =
            dma_local_wait_id(handle->stream, handle->local_id);
        DMA_PROF(10);
        if (error == 0)
          error = wait_error;
        completed_id = dma_local_completed_id(handle->stream);
      }
      slot_state->error = error;

      // Completion ids are cumulative. Retiring the named request therefore
      // also retires every older slot on the same stream, which makes an
      // explicit last handle a wait-through fence for queued submissions.
      for (uint32_t slot = 0; slot < DMA_COLLECTIVE_SLOTS; ++slot) {
        dma_collective_slot_state_t *const completed = &group_state->slot[slot];
        const uint32_t issued =
            __atomic_load_n(&completed->issued_epoch, __ATOMIC_ACQUIRE);
        const dma_id_t local_id =
            __atomic_load_n(&completed->local_id, __ATOMIC_RELAXED);
        if (issued == 0 ||
            dma_collective_epoch_done(
                __atomic_load_n(&completed->ready_epoch, __ATOMIC_ACQUIRE),
                issued) ||
            __atomic_load_n(&completed->stream, __ATOMIC_RELAXED) !=
                handle->stream ||
            (slot != handle->slot &&
             (completed_id == 0 || local_id == 0 ||
              !dma_local_id_done(completed_id, local_id))))
          continue;
        if (__atomic_load_n(&completed->error, __ATOMIC_RELAXED) == 0)
          completed->error = error;
        __atomic_store_n(&completed->ready_epoch, issued, __ATOMIC_RELEASE);
      }
      return error;
    }
  }
}

uint32_t dma_collective_wait(const dma_collective_scope_t *scope,
                             const dma_collective_handle_t *handle) {
  if (handle == NULL || !dma_collective_scope_valid(scope))
    return DMA_COLLECTIVE_ERROR_ARGUMENT;
  if (handle->status != DMA_COLLECTIVE_OK)
    return handle->status;
  if (handle->base_group != scope->base_group ||
      handle->num_groups != scope->num_groups)
    return DMA_COLLECTIVE_ERROR_SCOPE;
  uint32_t error = dma_collective_wait_local(handle);

  for (uint32_t group = scope->base_group;
       group < scope->base_group + scope->num_groups; ++group) {
    dma_collective_slot_state_t *const slot_state =
        &handle->state->group[group].slot[handle->slot];
    while (!dma_collective_epoch_done(
        __atomic_load_n(&slot_state->ready_epoch, __ATOMIC_ACQUIRE),
        handle->generation)) {
      mempool_wait(4u);
    }
    error |= __atomic_load_n(&slot_state->error, __ATOMIC_RELAXED);
  }
  __sync_synchronize();
  return error;
}

uint32_t
dma_collective_wait_through(const dma_collective_scope_t *scope,
                            const dma_collective_handle_t *last_handle) {
  return dma_collective_wait(scope, last_handle);
}

uint32_t dma_collective_done_local(const dma_collective_handle_t *handle) {
  if (handle == NULL || handle->state == NULL || handle->generation == 0 ||
      handle->slot >= DMA_COLLECTIVE_SLOTS ||
      handle->group_id >= (uint32_t)NUM_GROUPS ||
      handle->stream >= (uint32_t)NUM_DMAS_PER_GROUP ||
      handle->status != DMA_COLLECTIVE_OK)
    return 0;
  dma_collective_slot_state_t *const slot_state =
      &handle->state->group[handle->group_id].slot[handle->slot];
  if (__atomic_load_n(&slot_state->submit_epoch, __ATOMIC_ACQUIRE) !=
      handle->generation)
    return 0;
  if (dma_collective_epoch_done(
          __atomic_load_n(&slot_state->ready_epoch, __ATOMIC_ACQUIRE),
          handle->generation))
    return 1;
  return handle->local_id == 0 ||
         dma_local_done_id(handle->stream, handle->local_id);
}

uint32_t dma_collective_memcpy_blocking(dma_collective_state_t *state,
                                        const dma_collective_scope_t *scope,
                                        uint32_t generation, uint32_t stream,
                                        void *dest, const void *src,
                                        size_t len) {
  const dma_collective_handle_t handle = dma_collective_memcpy_nonblocking(
      state, scope, generation, stream, dest, src, len);
  return dma_collective_wait(scope, &handle);
}

void dma_collective_all_runtime_init(void) {
  const dma_collective_scope_t scope = dma_collective_scope_all();
  const uint32_t group = mempool_get_group_id();

  dma_collective_init(&dma_collective_all_state, &scope);
#if FSYNC_TPG_SIDE != 0
  for (uint32_t t = mempool_get_core_id(); t < DMA_COLLECTIVE_NUM_TILES;
       t += (uint32_t)NUM_CORES)
    fsync_tile_counter_init(&dma_collective_fsync_cnt[t]);
#endif
  if (mempool_is_group_coordinator()) {
    dma_collective_all_group_t *const control =
        &dma_collective_all_group[group];
    control->generation = 0;
    control->handle = dma_collective_invalid_handle(
        &dma_collective_all_state, 0, 0, DMA_COLLECTIVE_ERROR_STALE);
    control->fence_error = 0;
    control->fence_generation = 0;
    control->solo_id = 0;
    control->solo_error = 0;
    for (uint32_t i = 0u;
         i < DMA_COLLECTIVE_GROUP_WORDS - DMA_COLLECTIVE_ALL_FIELDS; ++i)
      control->padding[i] = 0u;
  }
  __sync_synchronize();
}

uint32_t dma_collective_all_memcpy_nonblocking(void *dest, const void *src,
                                               size_t len) {
  DMA_PROF(0);
  const uint32_t group = mempool_get_group_id();
  const dma_collective_scope_t scope = dma_collective_scope_all();
  if (!mempool_is_group_coordinator())
    return DMA_COLLECTIVE_ERROR_ARGUMENT;

  dma_collective_all_group_t *const control = &dma_collective_all_group[group];
  uint32_t generation = control->generation + 1u;
  if (generation == 0)
    generation = 1;

  dma_collective_handle_t handle = dma_collective_memcpy_agent_nonblocking(
      &dma_collective_all_state, &scope, generation, 0, dest, src, len);
  if (handle.status == DMA_COLLECTIVE_ERROR_REUSE &&
      control->handle.status == DMA_COLLECTIVE_OK) {
    const uint32_t wait_error =
        dma_collective_wait_through(&scope, &control->handle);
    if (wait_error != DMA_COLLECTIVE_OK)
      return wait_error;
    handle = dma_collective_memcpy_agent_nonblocking(
        &dma_collective_all_state, &scope, generation, 0, dest, src, len);
  }

  control->handle = handle;
  if (handle.status == DMA_COLLECTIVE_OK)
    control->generation = generation;
  DMA_PROF(8);
  return handle.status;
}

uint32_t dma_collective_all_wait(void) {
  const uint32_t group = mempool_get_group_id();
  const dma_collective_scope_t scope = dma_collective_scope_all();
  if (!mempool_is_group_coordinator())
    return DMA_COLLECTIVE_ERROR_ARGUMENT;
  return dma_collective_wait_through(&scope,
                                     &dma_collective_all_group[group].handle);
}

// ---------------------------------------------------------------------------
// Solo path
//
// The slot machinery exists for the SPMD contract: any core may call, any core
// may wait, several generations may be in flight. Under the _all_ contract --
// one designated agent per group, the SAME core waits, one transfer at a time,
// completion established by a rendezvous rather than by polling peers -- every
// piece of it has no work to do:
//
//   issuer election   one caller, no competitor
//   poller election   the issuer is the waiter
//   publish           local_id/error never leave the core that produced them
//   ready_epoch       the rendezvous below establishes what it was polled for
//   slot drain        one live transfer, nothing to retire
//
// So skip it. The transfer's identity lives in this group's own control block,
// two words written and read by one core in its own banks.
//
// The cost of the contract is that only ONE transfer per group may be in
// flight. That is what double buffering actually needs -- issue the next tile,
// compute on the current one, wait -- so it is not a restriction in practice;
// deeper queueing must use the slot API.
uint32_t dma_collective_all_memcpy_solo(void *dest, const void *src,
                                        size_t len) {
  DMA_PROF(0);
  if (!mempool_is_group_coordinator())
    return DMA_COLLECTIVE_ERROR_ARGUMENT;

  const uint32_t group = mempool_get_group_id();
  dma_collective_all_group_t *const control = &dma_collective_all_group[group];
  uint32_t error = 0;
  uint32_t descriptors = 0;
  DMA_PROF(1);

  // The _all_ scope is every group, so scope coverage is a tautology and only
  // the L1 range needs checking. issue_linear repeats these, but doing them
  // here keeps the failure out of the engine.
  const uint32_t dest_addr = (uint32_t)dest;
  const uint32_t src_addr = (uint32_t)src;
  const uint32_t dest_in_l1 = dest_addr < (uint32_t)DMA_COLLECTIVE_TCDM_SIZE;
  const uint32_t src_in_l1 = src_addr < (uint32_t)DMA_COLLECTIVE_TCDM_SIZE;
  if (dest_in_l1 == src_in_l1 || len == 0 || len > UINT32_MAX) {
    control->solo_id = 0;
    control->solo_error = DMA_COLLECTIVE_ERROR_ARGUMENT;
    return DMA_COLLECTIVE_ERROR_ARGUMENT;
  }

  DMA_PROF(2);
  const dma_id_t id = dma_collective_issue_linear(0u, dest, src, len, group,
                                                  &error, &descriptors);
  DMA_PROF(3);
  control->solo_id = id;
  control->solo_error = error;
  DMA_PROF(4);
  return error;
}

// Completion for the solo path. EVERY core calls it: the rendezvous is what
// establishes that every group's engine has retired, and the fsync leaf is a
// tile, so every tile must take part.
uint32_t dma_collective_all_wait_solo(void) {
  DMA_PROF(5);
  uint32_t error = DMA_COLLECTIVE_OK;
  const uint32_t coordinator = mempool_is_group_coordinator();
  dma_collective_all_group_t *const control =
      &dma_collective_all_group[mempool_get_group_id()];

  if (coordinator) {
    error = control->solo_error;
    const uint32_t id = control->solo_id;
    if (id != 0) {
      const uint32_t wait_error = dma_local_wait_id(0u, (dma_id_t)id);
      if (error == 0)
        error = wait_error;
    }
    DMA_PROF(6);
    control->solo_id = 0;
    if (error != DMA_COLLECTIVE_OK) {
      // Rare path: tell every group before the rendezvous releases readers.
      for (uint32_t g = 0; g < (uint32_t)NUM_GROUPS; ++g)
        __atomic_store_n(&dma_collective_all_group[g].fence_error, error,
                         __ATOMIC_RELEASE);
      __sync_synchronize();
    }
  }

#if FSYNC_TPG_SIDE == 0
  // No tree on this geometry; a software all-core barrier establishes the same
  // thing, just slower. No ready_epoch is needed either way.
  mempool_barrier(NUM_CORES);
#else
  fsync_barrier_all(dma_collective_fsync_cnt, mempool_get_tile_id());
#endif
  DMA_PROF(7);

  if (coordinator && error == DMA_COLLECTIVE_OK) {
    const uint32_t reported =
        __atomic_load_n(&control->fence_error, __ATOMIC_ACQUIRE);
    if (reported != DMA_COLLECTIVE_OK) {
      error = reported;
      control->fence_error = DMA_COLLECTIVE_OK;
    }
  }
  __sync_synchronize();
  DMA_PROF(8);
  return error;
}

uint32_t dma_collective_all_memcpy_solo_blocking(void *dest, const void *src,
                                                 size_t len) {
  const uint32_t issue_error = dma_collective_all_memcpy_solo(dest, src, len);
  const uint32_t wait_error = dma_collective_all_wait_solo();
  return issue_error != DMA_COLLECTIVE_OK ? issue_error : wait_error;
}

// Hardware cross-group fence. EVERY core calls this, unlike the rest of the
// _all_ API: the fractal_sync leaf is a TILE, so a barrier over the cluster
// only completes if every tile triggers its shim.
//
// Each group's coordinator confirms its OWN engine locally, then the whole
// machine rendezvous on the tree. That is the invariant the software fence
// pays 16 remote spin-loops to establish; the tree establishes it in one
// round trip. Errors ride the per-group report written before the barrier.
uint32_t dma_collective_all_wait_sync(void) {
  DMA_PROF(9);
  const uint32_t group = mempool_get_group_id();
  dma_collective_all_group_t *const control = &dma_collective_all_group[group];
  const uint32_t coordinator = mempool_is_group_coordinator();
  const uint32_t generation = control->generation;
  uint32_t error = DMA_COLLECTIVE_OK;

#if FSYNC_TPG_SIDE == 0
  // No square tile grid, so no tree (see FSyncTpgSide in mempool_pkg.sv).
  // Keep the all-cores contract with the software fence underneath.
  const dma_collective_scope_t scope = dma_collective_scope_all();
  if (coordinator)
    error = dma_collective_wait_through(&scope, &control->handle);
  mempool_barrier(NUM_CORES);
  return error;
#else
  if (coordinator) {
    error = dma_collective_wait_local(&control->handle);
    DMA_PROF(11);
    if (error != DMA_COLLECTIVE_OK) {
      // Rare path: tell every group, then make the stores visible before the
      // barrier that releases the readers.
      for (uint32_t g = 0; g < (uint32_t)NUM_GROUPS; ++g) {
        dma_collective_all_group[g].fence_error = error;
        __atomic_store_n(&dma_collective_all_group[g].fence_generation,
                         generation, __ATOMIC_RELEASE);
      }
      __sync_synchronize();
    }
  }

  fsync_barrier_all(dma_collective_fsync_cnt, mempool_get_tile_id());
  DMA_PROF(12);

  // One load from this group's own banks, not a remote poll of all sixteen.
  if (coordinator && error == DMA_COLLECTIVE_OK &&
      __atomic_load_n(&control->fence_generation, __ATOMIC_ACQUIRE) ==
          generation)
    error = control->fence_error;
  __sync_synchronize();
  DMA_PROF(13);
  return error;
#endif
}

uint32_t dma_collective_all_memcpy_blocking(void *dest, const void *src,
                                            size_t len) {
  const uint32_t issue_error =
      dma_collective_all_memcpy_nonblocking(dest, src, len);
  if (issue_error != DMA_COLLECTIVE_OK)
    return issue_error;
  return dma_collective_all_wait();
}
