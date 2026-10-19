// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef QWEN38_DMA_STAGE_QUEUE_H_
#define QWEN38_DMA_STAGE_QUEUE_H_

#include <stddef.h>
#include <stdint.h>

#include "dma.h"

// Eight slots let a dataflow stage keep two weight groups and the next
// stage's first pieces in flight.
#ifndef Q38_DMA_QUEUE_SLOTS
#define Q38_DMA_QUEUE_SLOTS 8u
#endif
// Epochs a group keeps issued but not completed. One epoch can take up to
// three iDMA descriptors (dma_collective_issue_linear), so a deep queue must
// cap this below the group iDMA's request FIFO (16): a launch into a full FIFO
// stalls the coordinator on its register read.
#ifndef Q38_DMA_QUEUE_MAX_INFLIGHT
#define Q38_DMA_QUEUE_MAX_INFLIGHT Q38_DMA_QUEUE_SLOTS
#endif

enum {
  Q38_DMA_QUEUE_OK = 0u,
  Q38_DMA_QUEUE_PENDING = 1u,
  Q38_DMA_QUEUE_INVALID = 0x82000001u,
  Q38_DMA_QUEUE_BUSY = 0x82000002u,
};

typedef struct {
  volatile uint32_t issued_epoch;
  volatile uint32_t completed_epoch;
  uint32_t id;
  uint32_t status;
} q38_dma_group_progress_t;

typedef struct {
  volatile uint32_t published_epoch;
  volatile uint32_t free_epoch;
  void *dest;
  const void *src;
  size_t bytes;
  // Above one: `pieces` pieces of `bytes`, `src_stride` apart in L2 and back
  // to back in L1 (dma_coordinator_submit_pieces).
  uint32_t pieces;
  size_t src_stride;
  uint32_t stream;
  q38_dma_group_progress_t group[NUM_GROUPS];
} q38_dma_stage_slot_t;

typedef struct {
  q38_dma_stage_slot_t slot[Q38_DMA_QUEUE_SLOTS];
  uint32_t next_issue[NUM_GROUPS];
  uint32_t next_complete[NUM_GROUPS];
} q38_dma_stage_queue_t;

void q38_dma_stage_init(q38_dma_stage_queue_t *queue);
uint32_t q38_dma_stage_publish(q38_dma_stage_queue_t *queue, uint32_t epoch,
                               void *dest, const void *src, size_t bytes,
                               uint32_t stream);
// The _any forms drop the core-0 guard for divergent workers. The caller
// must still be the only publisher or releaser of that epoch.
uint32_t q38_dma_stage_publish_any(q38_dma_stage_queue_t *queue, uint32_t epoch,
                                   void *dest, const void *src, size_t bytes,
                                   uint32_t stream);
uint32_t q38_dma_stage_publish_pieces(q38_dma_stage_queue_t *queue,
                                      uint32_t epoch, void *dest,
                                      const void *src, size_t piece_bytes,
                                      size_t src_stride, uint32_t pieces,
                                      uint32_t stream);
uint32_t q38_dma_stage_release_any(q38_dma_stage_queue_t *queue,
                                   uint32_t epoch);
uint32_t q38_dma_stage_progress(q38_dma_stage_queue_t *queue);
uint32_t q38_dma_stage_ready(const q38_dma_stage_queue_t *queue,
                             uint32_t epoch);
// The caller must also wait for all consumers of the copied buffer before
// releasing this request's generation for reuse.
uint32_t q38_dma_stage_release(q38_dma_stage_queue_t *queue, uint32_t epoch);

#endif
