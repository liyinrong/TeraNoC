// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Samuel Riedel, ETH Zurich

#ifndef _DMA_H_
#define _DMA_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "addrmap.h"
#include "fsync.h"
#include "runtime.h"

// iDMA v0.6.5 reg32_3d register map. Every tile has an independent descriptor
// bank, while status and transfer IDs observe the selected group-local stream.
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
#define DMA_ERROR_ADDRESS_OFFSET(stream) (0x160u + 4u * (stream))

#define DMA_CONF_ENABLE_ND_BIT 10u
#define DMA_CONF_ENABLE_ND_MASK (0x3u << DMA_CONF_ENABLE_ND_BIT)
#define DMA_STATUS_BUSY_MASK 0x3ffu

typedef uint32_t dma_id_t;

enum {
  DMA_LOCAL_OK = 0,
  DMA_LOCAL_ERROR_ARGUMENT = 0x80000001u,
};

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

static inline uint32_t dma_local_error_address(uint32_t stream) {
  return *dma_local_reg(DMA_ERROR_ADDRESS_OFFSET(stream));
}

static inline void dma_local_error_clear(uint32_t stream) {
  *dma_local_reg(DMA_ERROR_STATUS_OFFSET(stream)) = 1;
}

static inline uint32_t dma_local_id_done(dma_id_t done, dma_id_t target) {
  return (int32_t)(done - target) >= 0;
}

static inline uint32_t dma_local_done_id(uint32_t stream, dma_id_t id) {
  return dma_local_id_done(dma_local_completed_id(stream), id);
}

static inline uint32_t dma_local_wait_id(uint32_t stream, dma_id_t id) {
  if (stream >= (uint32_t)NUM_DMAS_PER_GROUP || id == 0)
    return DMA_LOCAL_ERROR_ARGUMENT;
  while (!dma_local_done_id(stream, id)) {
    const uint32_t error = dma_local_error(stream);
    if (error != 0)
      return error;
    mempool_wait(4u);
  }
  __sync_synchronize();
  // Error and retirement can become visible together. Re-read the sticky
  // status after observing the target ID so that completion cannot bypass an
  // error which landed between the loop's last two MMIO reads.
  return dma_local_error(stream);
}

// Wait until every request accepted by one stream has landed. The hardware
// status includes the frontend FIFO, ND midend, and backend, so an immediate
// wait after an asynchronous launch cannot observe a false idle gap.
static inline uint32_t dma_local_wait(uint32_t stream) {
  if (stream >= (uint32_t)NUM_DMAS_PER_GROUP)
    return DMA_LOCAL_ERROR_ARGUMENT;
  while (!dma_local_idle(stream)) {
    const uint32_t error = dma_local_error(stream);
    if (error != 0)
      return error;
    mempool_wait(4u);
  }
  __sync_synchronize();
  return dma_local_error(stream);
}

// Sleep until this group's engine is idle, instead of polling done_id.
//
// The iDMA exposes no completion pulse, so the event unit derives one from busy
// FALLING. That is exact while a single transfer is outstanding -- the solo
// collective's contract -- and conservative otherwise: it fires when the last
// queued transfer retires, not a named one. Do not build a per-id sleep on it.
//
// Order matters. Subscribe BEFORE testing idle: an edge that lands after the
// subscribe is recorded in the level-held pending and the wfi returns at once,
// but one that landed before the subscribe was never recorded and the core
// would sleep forever. The loop re-tests because a wfi can also return for an
// unrelated wake.
//
// The subscription is restored to the barrier event on the way out. A core that
// stayed subscribed to a source it no longer sleeps for accumulates pending and
// wakes early from the next barrier.
static inline uint32_t dma_local_wait_sleep(uint32_t stream) {
  if (stream >= (uint32_t)NUM_DMAS_PER_GROUP)
    return DMA_LOCAL_ERROR_ARGUMENT;
  fsync_subscribe_self(FSYNC_EVT_DMA | FSYNC_EVT_DMA_ERR);
  while (!dma_local_idle(stream)) {
    const uint32_t error = dma_local_error(stream);
    if (error != 0) {
      fsync_subscribe_self(FSYNC_EVT_FSYNC);
      return error;
    }
    mempool_wfi();
  }
  fsync_subscribe_self(FSYNC_EVT_FSYNC);
  __sync_synchronize();
  return dma_local_error(stream);
}

// Whole-cluster compatibility path for applications whose L1 objects use the
// ordinary global interleave. Exactly one coordinator per group calls these
// functions in the same program order. The implementation remains distributed:
// every coordinator programs only its own group-local frontend.
void dma_collective_all_runtime_init(void);
uint32_t dma_collective_all_memcpy_nonblocking(void *dest, const void *src,
                                               size_t len);
uint32_t dma_collective_all_wait(void);
// Hardware cross-group fence on the fractal_sync tree. Unlike the rest of the
// _all_ API this is called by EVERY core, and only group coordinators get a
// meaningful return code.
uint32_t dma_collective_all_wait_sync(void);

// Solo path: one designated agent per group issues, the SAME core waits, one
// transfer per group in flight. Skips the slot machinery entirely -- issuer and
// poller election, publish, ready_epoch, drain -- because none of it has work
// to do under that contract. _memcpy_solo is coordinator-only; _wait_solo is
// called by EVERY core, because the rendezvous is what establishes completion.
uint32_t dma_collective_all_memcpy_solo(void *dest, const void *src,
                                        size_t len);
uint32_t dma_collective_all_wait_solo(void);
uint32_t dma_collective_all_memcpy_solo_blocking(void *dest, const void *src,
                                                 size_t len);
uint32_t dma_collective_all_memcpy_blocking(void *dest, const void *src,
                                            size_t len);

static inline dma_id_t
dma_local_memcpy_nd_nonblocking(uint32_t stream, void *dest, const void *src,
                                size_t len, uint32_t dst_stride_2,
                                uint32_t src_stride_2, uint32_t reps_2,
                                uint32_t dst_stride_3, uint32_t src_stride_3,
                                uint32_t reps_3, uint32_t dimensions) {
  if (stream >= (uint32_t)NUM_DMAS_PER_GROUP || len == 0 || dimensions < 1 ||
      dimensions > 3 || len > UINT32_MAX || dma_local_error(stream) != 0)
    return 0;

  // Only program the dimensions this descriptor actually uses. CONF's
  // ENABLE_ND field tells the ND midend how many dimensions to walk, so the
  // registers above that are dead -- writing them costs an MMIO round trip
  // each (~8 cycles) for a value the engine ignores. A 1-D transfer needs 3
  // registers, a 2-D one 6, where the unconditional form always wrote 10.
  //
  // The sticky-error read in the guard above is kept, but NOT for the reason
  // first given here. It is redundant for hang-avoidance: dma_local_wait_id
  // re-reads the sticky status on every poll iteration and returns it, so a
  // faulted engine surfaces at the wait rather than hanging it. All the guard
  // buys is failing earlier.
  //
  // It is kept because removing it is free in the wrong direction -- measured
  // 76 cycles for one 1-D descriptor with it and 76 without. An isolated probe
  // showed 105 -> 40 for the same removal, but that shape has nothing to
  // overlap the read; here its latency already hides behind the argument
  // checks and the CONF access, exactly as the CONF read's own latency hides
  // behind the address writes (shadowing CONF also gained nothing, 40 vs 44).
  //
  // The rule this exposes: on this register file an access is ~4 cycles of
  // throughput, as cheap as L1. Cost comes from a dependent read with no
  // independent work behind it, not from distance.
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

static inline uint32_t dma_local_memcpy_blocking(uint32_t stream, void *dest,
                                                 const void *src, size_t len) {
  const dma_id_t id = dma_local_memcpy_nonblocking(stream, dest, src, len);
  if (id != 0)
    return dma_local_wait_id(stream, id);
  const uint32_t error = dma_local_error(stream);
  return error != 0 ? error : DMA_LOCAL_ERROR_ARGUMENT;
}
#endif // _DMA_H_
