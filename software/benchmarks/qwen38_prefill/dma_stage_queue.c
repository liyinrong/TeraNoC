// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "dma_stage_queue.h"

#include "runtime.h"

static q38_dma_stage_slot_t *q38_dma_slot(q38_dma_stage_queue_t *queue,
                                          uint32_t epoch) {
  return &queue->slot[epoch % Q38_DMA_QUEUE_SLOTS];
}

void q38_dma_stage_init(q38_dma_stage_queue_t *queue) {
  if (queue == 0 || mempool_get_core_id() != 0u)
    return;
  for (uint32_t i = 0u; i < Q38_DMA_QUEUE_SLOTS; ++i) {
    q38_dma_stage_slot_t *slot = &queue->slot[i];
    slot->published_epoch = 0u;
    slot->free_epoch = 0u;
    slot->dest = 0;
    slot->src = 0;
    slot->bytes = 0u;
    slot->pieces = 1u;
    slot->src_stride = 0u;
    slot->stream = 0u;
    for (uint32_t g = 0u; g < NUM_GROUPS; ++g) {
      slot->group[g].issued_epoch = 0u;
      slot->group[g].completed_epoch = 0u;
      slot->group[g].id = 0u;
      slot->group[g].status = Q38_DMA_QUEUE_OK;
    }
  }
  for (uint32_t g = 0u; g < NUM_GROUPS; ++g) {
    queue->next_issue[g] = 1u;
    queue->next_complete[g] = 1u;
  }
  __atomic_thread_fence(__ATOMIC_RELEASE);
}

uint32_t q38_dma_stage_publish(q38_dma_stage_queue_t *queue, uint32_t epoch,
                               void *dest, const void *src, size_t bytes,
                               uint32_t stream) {
  if (mempool_get_core_id() != 0u)
    return Q38_DMA_QUEUE_INVALID;
  return q38_dma_stage_publish_any(queue, epoch, dest, src, bytes, stream);
}

uint32_t q38_dma_stage_publish_any(q38_dma_stage_queue_t *queue, uint32_t epoch,
                                   void *dest, const void *src, size_t bytes,
                                   uint32_t stream) {
  if (queue == 0 || epoch == 0u || dest == 0 || src == 0 || bytes == 0u ||
      stream >= (uint32_t)NUM_DMAS_PER_GROUP)
    return Q38_DMA_QUEUE_INVALID;
  q38_dma_stage_slot_t *slot = q38_dma_slot(queue, epoch);
  const uint32_t prior =
      epoch > Q38_DMA_QUEUE_SLOTS ? epoch - Q38_DMA_QUEUE_SLOTS : 0u;
  if (__atomic_load_n(&slot->free_epoch, __ATOMIC_ACQUIRE) != prior)
    return Q38_DMA_QUEUE_BUSY;
  slot->dest = dest;
  slot->src = src;
  slot->bytes = bytes;
  slot->pieces = 1u;
  slot->src_stride = bytes;
  slot->stream = stream;
  __atomic_store_n(&slot->published_epoch, epoch, __ATOMIC_RELEASE);
  return Q38_DMA_QUEUE_OK;
}

uint32_t q38_dma_stage_publish_pieces(q38_dma_stage_queue_t *queue,
                                      uint32_t epoch, void *dest,
                                      const void *src, size_t piece_bytes,
                                      size_t src_stride, uint32_t pieces,
                                      uint32_t stream) {
  if (queue == 0 || epoch == 0u || dest == 0 || src == 0 || piece_bytes == 0u ||
      pieces == 0u || stream >= (uint32_t)NUM_DMAS_PER_GROUP)
    return Q38_DMA_QUEUE_INVALID;
  q38_dma_stage_slot_t *slot = q38_dma_slot(queue, epoch);
  const uint32_t prior =
      epoch > Q38_DMA_QUEUE_SLOTS ? epoch - Q38_DMA_QUEUE_SLOTS : 0u;
  if (__atomic_load_n(&slot->free_epoch, __ATOMIC_ACQUIRE) != prior)
    return Q38_DMA_QUEUE_BUSY;
  slot->dest = dest;
  slot->src = src;
  slot->bytes = piece_bytes;
  slot->pieces = pieces;
  slot->src_stride = src_stride;
  slot->stream = stream;
  __atomic_store_n(&slot->published_epoch, epoch, __ATOMIC_RELEASE);
  return Q38_DMA_QUEUE_OK;
}

uint32_t q38_dma_stage_progress(q38_dma_stage_queue_t *queue) {
  if (queue == 0 || !mempool_is_group_coordinator())
    return Q38_DMA_QUEUE_INVALID;
  const uint32_t group = mempool_get_group_id();
  uint32_t progressed = 0u;
  for (uint32_t i = 0u; i < Q38_DMA_QUEUE_SLOTS; ++i) {
    const uint32_t epoch = queue->next_issue[group];
    if (epoch - queue->next_complete[group] >= Q38_DMA_QUEUE_MAX_INFLIGHT)
      break;
    q38_dma_stage_slot_t *slot = q38_dma_slot(queue, epoch);
    if (__atomic_load_n(&slot->published_epoch, __ATOMIC_ACQUIRE) != epoch)
      break;
    const dma_coordinator_op_t op =
        slot->pieces > 1u
            ? dma_coordinator_submit_pieces(slot->stream, slot->dest, slot->src,
                                            slot->bytes, slot->src_stride,
                                            slot->pieces)
            : dma_coordinator_submit_linear(slot->stream, slot->dest, slot->src,
                                            slot->bytes);
    q38_dma_group_progress_t *state = &slot->group[group];
    state->id = op.id;
    state->status = op.status;
    __atomic_store_n(&state->issued_epoch, epoch, __ATOMIC_RELEASE);
    queue->next_issue[group] = epoch + 1u;
    ++progressed;
  }
  for (uint32_t i = 0u; i < Q38_DMA_QUEUE_SLOTS; ++i) {
    const uint32_t epoch = queue->next_complete[group];
    if (epoch == queue->next_issue[group])
      break;
    q38_dma_stage_slot_t *slot = q38_dma_slot(queue, epoch);
    q38_dma_group_progress_t *state = &slot->group[group];
    if (state->id != 0u && !dma_local_done_id(slot->stream, state->id))
      break;
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    __atomic_store_n(&state->completed_epoch, epoch, __ATOMIC_RELEASE);
    queue->next_complete[group] = epoch + 1u;
    ++progressed;
  }
  return progressed;
}

uint32_t q38_dma_stage_ready(const q38_dma_stage_queue_t *queue,
                             uint32_t epoch) {
  if (queue == 0 || epoch == 0u)
    return Q38_DMA_QUEUE_INVALID;
  const q38_dma_stage_slot_t *slot = &queue->slot[epoch % Q38_DMA_QUEUE_SLOTS];
  if (__atomic_load_n(&slot->published_epoch, __ATOMIC_ACQUIRE) != epoch)
    return Q38_DMA_QUEUE_PENDING;
  for (uint32_t g = 0u; g < NUM_GROUPS; ++g) {
    const q38_dma_group_progress_t *state = &slot->group[g];
    if (__atomic_load_n(&state->completed_epoch, __ATOMIC_ACQUIRE) != epoch)
      return Q38_DMA_QUEUE_PENDING;
    if (state->status != DMA_COLLECTIVE_OK)
      return state->status;
  }
  return Q38_DMA_QUEUE_OK;
}

uint32_t q38_dma_stage_release(q38_dma_stage_queue_t *queue, uint32_t epoch) {
  if (mempool_get_core_id() != 0u)
    return Q38_DMA_QUEUE_INVALID;
  return q38_dma_stage_release_any(queue, epoch);
}

uint32_t q38_dma_stage_release_any(q38_dma_stage_queue_t *queue,
                                   uint32_t epoch) {
  if (queue == 0 || epoch == 0u)
    return Q38_DMA_QUEUE_INVALID;
  const uint32_t status = q38_dma_stage_ready(queue, epoch);
  if (status != Q38_DMA_QUEUE_OK)
    return status;
  q38_dma_stage_slot_t *slot = q38_dma_slot(queue, epoch);
  __atomic_store_n(&slot->free_epoch, epoch, __ATOMIC_RELEASE);
  return Q38_DMA_QUEUE_OK;
}
