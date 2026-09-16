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
// per group; status and transfer ids are per stream.
//
// The engine has no interrupt and no error register: software polls STATUS
// (busy) and DONE_ID (completion), and both are exact. The one event the
// hardware emits is the completion pulse behind FSYNC_EVT_DMA, which
// dma_event_gate uses; the runtime polls.
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

#define DMA_CONF_ENABLE_ND_BIT 10u
#define DMA_CONF_ENABLE_ND_MASK (0x3u << DMA_CONF_ENABLE_ND_BIT)
#define DMA_STATUS_BUSY_MASK 0x3ffu

typedef uint32_t dma_id_t;

enum {
  DMA_LOCAL_OK = 0,
  DMA_LOCAL_ERROR_ARGUMENT = 0x80000001u,
};

// THE ENGINE'S DESCRIPTOR REGISTERS ARE ONE CONTEXT PER GROUP.
//
// DMA_BASE carries no tile index: the context is selected by which tile's port
// the access arrives on, and every tile of a group now reaches the SAME one
// (mempool_group.sv arbitrates them). Programming a transfer takes SEVERAL
// writes -- CONF, SRC, DST, LENGTH, the strides, then the NEXT_ID read that
// launches it -- and the arbiters make each access atomic but not the
// sequence. Two issuers of one group therefore interleave their writes into
// the single register set and each launches the other's descriptor, silently:
// the transfer completes, with the wrong address or length.
//
// ONE ISSUER PER GROUP IS THEREFORE ABSOLUTE, and there is no lock: the
// centralized DMA this replaced had one context for the whole machine and no
// lock either. mempool_is_group_coordinator() is how every caller in-tree
// elects its issuer, and the collective has exactly one per group by
// construction. A caller that cannot keep to one issuer has no safe option
// here; it must serialise at a higher level.
//
// Reads (status, done id) are single-register, so the arbiter serialises them
// and EVERY core may poll them freely -- that is what idma_ctx_share
// exercises.

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

static inline uint32_t dma_local_id_done(dma_id_t done, dma_id_t target) {
  return (int32_t)(done - target) >= 0;
}

static inline uint32_t dma_local_done_id(uint32_t stream, dma_id_t id) {
  return dma_local_id_done(dma_local_completed_id(stream), id);
}

// Both waits POLL. The engine's completion pulse (FSYNC_EVT_DMA) is exact, but
// a poll is as cheap here -- the register file is ~4 cycles a read, L1 speed --
// and it needs no subscription discipline: no unsubscribe to restore, no stale
// pending bit to keep out of the next barrier. Measured on dse_s1_winner at
// 512 KiB, sleeping instead of polling moved nothing: bw in 1559/1559.
//
// Wait until one request has landed; waiting an id waits every earlier one.
static inline uint32_t dma_local_wait_id(uint32_t stream, dma_id_t id) {
  if (stream >= (uint32_t)NUM_DMAS_PER_GROUP || id == 0)
    return DMA_LOCAL_ERROR_ARGUMENT;
  while (!dma_local_done_id(stream, id))
    mempool_wait(4u);
  __sync_synchronize();
  return DMA_LOCAL_OK;
}

// Wait until every request accepted by one stream has landed. The status
// includes the frontend FIFO, ND midend and backend, so a wait right after an
// asynchronous launch cannot observe a false idle gap.
static inline uint32_t dma_local_wait(uint32_t stream) {
  if (stream >= (uint32_t)NUM_DMAS_PER_GROUP)
    return DMA_LOCAL_ERROR_ARGUMENT;
  while (!dma_local_idle(stream))
    mempool_wait(4u);
  __sync_synchronize();
  return DMA_LOCAL_OK;
}

static inline dma_id_t
dma_local_memcpy_nd_nonblocking(uint32_t stream, void *dest, const void *src,
                                size_t len, uint32_t dst_stride_2,
                                uint32_t src_stride_2, uint32_t reps_2,
                                uint32_t dst_stride_3, uint32_t src_stride_3,
                                uint32_t reps_3, uint32_t dimensions) {
  if (stream >= (uint32_t)NUM_DMAS_PER_GROUP || len == 0 || dimensions < 1 ||
      dimensions > 3)
    return 0;

  // Program only the dimensions this descriptor uses: CONF's ENABLE_ND field
  // tells the ND midend how many to walk, so the registers above it are dead.
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
