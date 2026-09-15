// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Samuel Riedel, ETH Zurich

#ifndef _DMA_H_
#define _DMA_H_

#include <stddef.h>
#include <stdint.h>

#include "addrmap.h"
#include "fsync.h"
#include "runtime.h"

// iDMA v0.6.5 reg32_3d register map. The descriptor registers are one context
// per group; status, transfer ids and errors are per stream.
// Second 64 KiB of the peripheral window (mempool_pkg::DMABaseAddr).
#define DMA_BASE (CONTROL_REGISTER_OFFSET + 0x10000u)
#define DMA_CONF_OFFSET 0x000u
#define DMA_STATUS_OFFSET(stream) (0x004u + 4u * (stream))
#define DMA_NEXT_ID_OFFSET(stream) (0x044u + 4u * (stream))
#define DMA_DONE_ID_OFFSET(stream) (0x084u + 4u * (stream))
#define DMA_DST_ADDR_OFFSET 0x0d0u
#define DMA_SRC_ADDR_OFFSET 0x0d8u
#define DMA_LENGTH_OFFSET 0x0e0u
#define DMA_DST_STRIDE_2_OFFSET 0x0e8u
#define DMA_SRC_STRIDE_2_OFFSET 0x0f0u
#define DMA_REPS_2_OFFSET 0x0f8u
#define DMA_DST_STRIDE_3_OFFSET 0x100u
#define DMA_SRC_STRIDE_3_OFFSET 0x108u
#define DMA_REPS_3_OFFSET 0x110u
#define DMA_ERROR_STATUS_OFFSET(stream) (0x120u + 4u * (stream))

#define DMA_CONF_ENABLE_ND_BIT 10u
#define DMA_CONF_ENABLE_ND_MASK (0x3u << DMA_CONF_ENABLE_ND_BIT)
#define DMA_STATUS_BUSY_MASK 0x3ffu

typedef uint32_t dma_id_t;

enum {
  DMA_LOCAL_ERROR_ARGUMENT = 0x80000001u,
};

// THE ENGINE'S DESCRIPTOR REGISTERS ARE ONE CONTEXT PER GROUP.
//
// DMA_BASE carries no tile index: the context is selected by which tile's port
// the access arrives on, and every tile of a group now reaches the SAME one
// (mempool_group.sv arbitrates them). So two tiles of one group programming a
// transfer concurrently interleave SRC/DST/LENGTH into a single register set
// and each launches the other's descriptor.
//
// One issuer per group is therefore the contract for everything below --
// mempool_is_group_coordinator() is how every caller in-tree elects one. A
// caller that genuinely needs several tiles of a group to issue must bracket
// the programming sequence with dma_group_ctx_lock()/dma_group_ctx_unlock().
// Reads (status, done id, error) need no lock; they are single-register and
// the arbiter serialises them. A violation is silent: the transfer completes,
// with the wrong address or length.
//
// dma_group_ctx_init() zeroes the lock; mempool_barrier_init() calls it on
// every core.
void dma_group_ctx_init(void);
void dma_group_ctx_lock(void);
void dma_group_ctx_unlock(void);

static inline volatile uint32_t *dma_local_reg(uint32_t offset) {
  return (volatile uint32_t *)(DMA_BASE + offset);
}

static inline uint32_t dma_local_status(uint32_t stream) {
  return *dma_local_reg(DMA_STATUS_OFFSET(stream));
}

static inline uint32_t dma_local_idle(uint32_t stream) {
  return (dma_local_status(stream) & DMA_STATUS_BUSY_MASK) == 0;
}

static inline dma_id_t dma_local_completed_id(uint32_t stream) {
  return *dma_local_reg(DMA_DONE_ID_OFFSET(stream));
}

static inline uint32_t dma_local_error(uint32_t stream) {
  return *dma_local_reg(DMA_ERROR_STATUS_OFFSET(stream));
}

static inline uint32_t dma_local_id_done(dma_id_t done, dma_id_t target) {
  return (int32_t)(done - target) >= 0;
}

static inline uint32_t dma_local_done_id(uint32_t stream, dma_id_t id) {
  return dma_local_id_done(dma_local_completed_id(stream), id);
}

// The waits below sleep. The event unit has no per-transfer event: it wakes a
// subscribed core when the group engine goes idle (busy falling) or latches a
// fault. Busy covers a request from its launch until the cycle DONE_ID passes
// it (mempool_idma_group.sv busy_o, through the backend's outstanding count),
// so an outstanding request always has an idle edge still to come, no earlier
// than its completion. With later requests queued behind it that edge comes
// when the last one retires: late, never lost.
//
// Subscribe, and fence so the subscription is in place, BEFORE the first test:
// an edge after that is held pending and the wfi returns at once. The loops
// re-test because a wfi also returns for an unrelated wake. The exit restores
// the barrier subscription, which also drops an engine event taken while awake.
static inline void dma_local_sleep_enter(void) {
  fsync_subscribe_self(FSYNC_EVT_DMA | FSYNC_EVT_DMA_ERR);
  __sync_synchronize();
}

static inline uint32_t dma_local_sleep_exit(uint32_t stream) {
  fsync_subscribe_self(FSYNC_EVT_FSYNC);
  __sync_synchronize();
  // Error and completion can become visible together: re-read the sticky error.
  return dma_local_error(stream);
}

// Wait until one request has landed; waiting an id waits every earlier one.
static inline uint32_t dma_local_wait_id(uint32_t stream, dma_id_t id) {
  if (stream >= (uint32_t)NUM_DMAS_PER_GROUP || id == 0)
    return DMA_LOCAL_ERROR_ARGUMENT;
  dma_local_sleep_enter();
  while (!dma_local_done_id(stream, id) && dma_local_error(stream) == 0u)
    mempool_wfi();
  return dma_local_sleep_exit(stream);
}

// Wait until every request accepted by one stream has landed. The status
// includes the frontend FIFO, ND midend and backend, so a wait right after an
// asynchronous launch cannot observe a false idle gap.
static inline uint32_t dma_local_wait(uint32_t stream) {
  if (stream >= (uint32_t)NUM_DMAS_PER_GROUP)
    return DMA_LOCAL_ERROR_ARGUMENT;
  dma_local_sleep_enter();
  while (!dma_local_idle(stream) && dma_local_error(stream) == 0u)
    mempool_wfi();
  return dma_local_sleep_exit(stream);
}

static inline dma_id_t
dma_local_memcpy_nd_nonblocking(uint32_t stream, void *dest, const void *src,
                                size_t len, uint32_t dst_stride_2,
                                uint32_t src_stride_2, uint32_t reps_2,
                                uint32_t dst_stride_3, uint32_t src_stride_3,
                                uint32_t reps_3, uint32_t dimensions) {
  if (stream >= (uint32_t)NUM_DMAS_PER_GROUP || len == 0 || dimensions < 1 ||
      dimensions > 3 || dma_local_error(stream) != 0)
    return 0;

  // Program only the dimensions this descriptor uses: CONF's ENABLE_ND field
  // tells the ND midend how many to walk, so the registers above it are dead.
  // The sticky-error read above fails a faulted engine early; the wait would
  // report it too.
  uint32_t conf = *dma_local_reg(DMA_CONF_OFFSET);
  conf &= ~DMA_CONF_ENABLE_ND_MASK;
  if (dimensions >= 2)
    conf |= (dimensions - 1u) << DMA_CONF_ENABLE_ND_BIT;
  *dma_local_reg(DMA_CONF_OFFSET) = conf;
  *dma_local_reg(DMA_SRC_ADDR_OFFSET) = (uint32_t)src;
  *dma_local_reg(DMA_DST_ADDR_OFFSET) = (uint32_t)dest;
  *dma_local_reg(DMA_LENGTH_OFFSET) = (uint32_t)len;
  if (dimensions >= 2) {
    *dma_local_reg(DMA_SRC_STRIDE_2_OFFSET) = src_stride_2;
    *dma_local_reg(DMA_DST_STRIDE_2_OFFSET) = dst_stride_2;
    *dma_local_reg(DMA_REPS_2_OFFSET) = reps_2;
  }
  if (dimensions >= 3) {
    *dma_local_reg(DMA_SRC_STRIDE_3_OFFSET) = src_stride_3;
    *dma_local_reg(DMA_DST_STRIDE_3_OFFSET) = dst_stride_3;
    *dma_local_reg(DMA_REPS_3_OFFSET) = reps_3;
  }
  __sync_synchronize();
  const dma_id_t id = *dma_local_reg(DMA_NEXT_ID_OFFSET(stream));
  __sync_synchronize();
  return id;
}

static inline dma_id_t dma_local_memcpy_nonblocking(uint32_t stream, void *dest,
                                                    const void *src,
                                                    size_t len) {
  return dma_local_memcpy_nd_nonblocking(stream, dest, src, len, 0, 0, 0, 0, 0,
                                         1, 1);
}

static inline dma_id_t
dma_local_memcpy_2d_nonblocking(uint32_t stream, void *dest, const void *src,
                                size_t len, uint32_t dst_stride,
                                uint32_t src_stride, uint32_t reps) {
  if (reps == 0)
    return 0;
  return dma_local_memcpy_nd_nonblocking(stream, dest, src, len, dst_stride,
                                         src_stride, reps, 0, 0, 1, 2);
}

static inline dma_id_t dma_local_memcpy_3d_nonblocking(
    uint32_t stream, void *dest, const void *src, size_t len,
    uint32_t dst_stride_2, uint32_t src_stride_2, uint32_t reps_2,
    uint32_t dst_stride_3, uint32_t src_stride_3, uint32_t reps_3) {
  if (reps_2 == 0 || reps_3 == 0)
    return 0;
  return dma_local_memcpy_nd_nonblocking(stream, dest, src, len, dst_stride_2,
                                         src_stride_2, reps_2, dst_stride_3,
                                         src_stride_3, reps_3, 3);
}

// The collective API, for every includer of dma.h. Last, because it builds on
// everything above.
#include "dma_collective.h"

#endif // _DMA_H_
