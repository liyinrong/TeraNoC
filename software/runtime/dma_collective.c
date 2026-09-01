// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "dma_collective.h"

#include <stdbool.h>
#include <stdint.h>

#include "runtime.h"

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

static dma_collective_tile_lock_t
    dma_collective_tile_lock[DMA_COLLECTIVE_NUM_TILES] DMA_COLLECTIVE_STORAGE;

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

static void dma_collective_lock(void) {
  const uint32_t tile = mempool_get_tile_id();
  while (__atomic_exchange_n(&dma_collective_tile_lock[tile].value, 1,
                             __ATOMIC_ACQUIRE) != 0) {
  }
}

static void dma_collective_unlock(void) {
  const uint32_t tile = mempool_get_tile_id();
  __atomic_store_n(&dma_collective_tile_lock[tile].value, 0, __ATOMIC_RELEASE);
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
    if (submit > generation) {
      handle.status = DMA_COLLECTIVE_ERROR_STALE;
      return handle;
    }
    if (submit == generation) {
      uint32_t issued;
      do {
        issued = __atomic_load_n(&slot_state->issued_epoch, __ATOMIC_ACQUIRE);
      } while (issued < generation);
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
    if (issued != submit || ready < issued) {
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
  __atomic_store_n(&slot_state->local_id, local_id, __ATOMIC_RELAXED);
  __atomic_store_n(&slot_state->error, error, __ATOMIC_RELAXED);
  __atomic_store_n(&slot_state->descriptors, descriptors, __ATOMIC_RELAXED);
  __atomic_store_n(&slot_state->issuer_core, mempool_get_core_id(),
                   __ATOMIC_RELAXED);
  __atomic_store_n(&slot_state->stream, handle->stream, __ATOMIC_RELAXED);
  __atomic_store_n(&slot_state->issued_epoch, handle->generation,
                   __ATOMIC_RELEASE);
  handle->local_id = local_id;
}

void dma_collective_init(dma_collective_state_t *state,
                         const dma_collective_scope_t *scope) {
  if (state == NULL || !dma_collective_scope_valid(scope))
    return;

  const uint32_t core = mempool_get_core_id();
  const uint32_t group = mempool_get_group_id();
  if (!dma_collective_group_in_scope(scope, group))
    return;

  if (core % (uint32_t)NUM_CORES_PER_GROUP == 0) {
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
  }
  if (core % (uint32_t)NUM_CORES_PER_TILE == 0)
    dma_collective_tile_lock[mempool_get_tile_id()].value = 0;
  __sync_synchronize();
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

static uint32_t
dma_collective_scope_covers_range(const dma_collective_scope_t *scope,
                                  uint32_t start, uint32_t end) {
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

dma_collective_handle_t
dma_collective_memcpy_nonblocking(dma_collective_state_t *state,
                                  const dma_collective_scope_t *scope,
                                  uint32_t generation, uint32_t stream,
                                  void *dest, const void *src, size_t len) {
  uint32_t owner;
  dma_collective_handle_t handle =
      dma_collective_claim(state, scope, generation, stream, &owner);
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
  } else {
    dma_collective_lock();
    local_id = dma_collective_issue_linear(
        stream, dest, src, len, handle.group_id, &error, &descriptors);
    dma_collective_unlock();
  }
  dma_collective_publish(&handle, local_id, error, descriptors);
  return handle;
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
    dma_collective_lock();
    local_id = dma_local_memcpy_2d_nonblocking(stream, dest, src, len,
                                               dst_stride, src_stride, reps);
    if (local_id == 0) {
      error = dma_local_error(stream);
      if (error == 0)
        error = DMA_COLLECTIVE_ERROR_LAUNCH;
    }
    dma_collective_unlock();
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
    if (ready >= handle->generation)
      return __atomic_load_n(&slot_state->error, __ATOMIC_RELAXED);

    uint32_t wait = __atomic_load_n(&slot_state->wait_epoch, __ATOMIC_ACQUIRE);
    if (wait > handle->generation)
      return DMA_COLLECTIVE_ERROR_STALE;
    if (wait == handle->generation)
      continue;

    uint32_t expected = wait;
    if (__atomic_compare_exchange_n(&slot_state->wait_epoch, &expected,
                                    handle->generation, false, __ATOMIC_ACQ_REL,
                                    __ATOMIC_ACQUIRE)) {
      __atomic_store_n(&slot_state->poller_core, mempool_get_core_id(),
                       __ATOMIC_RELAXED);
      uint32_t error = __atomic_load_n(&slot_state->error, __ATOMIC_RELAXED);
      dma_id_t completed_id = 0;
      if (handle->local_id != 0) {
        const uint32_t wait_error =
            dma_local_wait_id(handle->stream, handle->local_id);
        if (error == 0)
          error = wait_error;
        completed_id = dma_local_completed_id(handle->stream);
      }
      __atomic_store_n(&slot_state->error, error, __ATOMIC_RELAXED);

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
            __atomic_load_n(&completed->ready_epoch, __ATOMIC_ACQUIRE) >=
                issued ||
            __atomic_load_n(&completed->stream, __ATOMIC_RELAXED) !=
                handle->stream ||
            (slot != handle->slot &&
             (completed_id == 0 || local_id == 0 ||
              !dma_local_id_done(completed_id, local_id))))
          continue;
        if (__atomic_load_n(&completed->error, __ATOMIC_RELAXED) == 0)
          __atomic_store_n(&completed->error, error, __ATOMIC_RELAXED);
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
    while (__atomic_load_n(&slot_state->ready_epoch, __ATOMIC_ACQUIRE) <
           handle->generation) {
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
  if (__atomic_load_n(&slot_state->ready_epoch, __ATOMIC_ACQUIRE) >=
      handle->generation)
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
