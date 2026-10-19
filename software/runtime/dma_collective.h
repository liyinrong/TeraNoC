// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef _DMA_COLLECTIVE_H_
#define _DMA_COLLECTIVE_H_

#include <stddef.h>
#include <stdint.h>

#include "runtime.h"

// Collective DMA over a virtual-cluster scope, from one group to the whole
// cluster.
//
// FIXED ISSUER. Only the group coordinator (mempool_is_group_coordinator())
// programs its group's engine, so the one-issuer contract of dma.h holds by
// construction: no issuer or poller election and no descriptor lock.
//
// SPMD. Every core of the scope makes every submit and wait call, in the same
// order. Non-coordinators issue nothing but count the call, so every core names
// an operation by the same sequence number.
//
// QUEUEING. Submits may stack up. An engine retires requests in issue order and
// counts retirements (idma_transfer_id_gen.sv), so waiting on a group's last id
// waits every earlier one. The request FIFO holds ReqFifoDepth per stream
// (mempool_idma_group.sv); a full FIFO stalls the submit, it does not fail it.
//
// COMPLETION, two forms:
//   dma_collective_wait        every core of the scope. Coordinators wait their
//                              own engine, then the scope's core barrier.
//   dma_collective_wait_ready  any core of the scope, no core barrier. Returns
//                              once every group has published READY through
//                              the operation. A group publishes only when its
//                              coordinator waits (either form), so a
//                              coordinator that is still computing delays its
//                              group.

enum {
  DMA_COLLECTIVE_OK = 0,
  DMA_COLLECTIVE_ERROR_ARGUMENT = 0x80000001u,
  DMA_COLLECTIVE_ERROR_SCOPE = 0x80000002u,
  DMA_COLLECTIVE_ERROR_LAUNCH = 0x80000005u,
};

// A VC rectangle: level k spans 2^k groups per axis, i.e. 4^k groups forming
// one contiguous Morton id range. Level VC_LEVELS is the whole cluster.
typedef struct {
  uint32_t level;
  uint32_t base_group;
  uint32_t num_groups;
} dma_collective_scope_t;

typedef struct {
  dma_collective_scope_t scope;
  uint32_t seq;    // identical on every core of the scope
  uint32_t status; // this core's submit error
} dma_collective_op_t;

// Divergent stage workers use one fixed issuer per group. The group
// coordinator calls this once for a global linear L1 <-> L2 range; it issues
// only the descriptors for its group's L1 stripes. No other core participates
// and no barrier is implied. id==0 with status OK means this group owns no
// bytes. The coordinator must publish completion for every participating
// group before consumers read the destination or reuse the source buffer.
typedef struct {
  uint32_t id;
  uint32_t status;
  uint32_t descriptors;
} dma_coordinator_op_t;

dma_coordinator_op_t dma_coordinator_submit_linear(uint32_t stream, void *dest,
                                                   const void *src, size_t len);

// As dma_coordinator_submit_linear for `pieces` source pieces of `piece_len`
// bytes each, `src_stride` bytes apart in L2, landing back to back at `dest`
// in L1 (a strided gather into one contiguous L1 range). Row-aligned pieces
// take one 3-D descriptor per group; anything else falls back to one linear
// submit per piece.
dma_coordinator_op_t dma_coordinator_submit_pieces(uint32_t stream, void *dest,
                                                   const void *src,
                                                   size_t piece_len,
                                                   size_t src_stride,
                                                   uint32_t pieces);

// The level-`level` rectangle containing this core's group.
static inline dma_collective_scope_t
dma_collective_scope_level(uint32_t level) {
  const uint32_t all = level >= (uint32_t)VC_LEVELS;
  const uint32_t n = all ? (uint32_t)NUM_GROUPS : 1u << (2u * level);
  const dma_collective_scope_t scope = {
      .level = all ? (uint32_t)VC_LEVELS : level,
      .base_group = mempool_get_group_id() & ~(n - 1u),
      .num_groups = n,
  };
  return scope;
}

// The whole cluster is a constant: no hart-id read on the caller's path.
static inline dma_collective_scope_t dma_collective_scope_all(void) {
  const dma_collective_scope_t scope = {.level = (uint32_t)VC_LEVELS,
                                        .base_group = 0u,
                                        .num_groups = (uint32_t)NUM_GROUPS};
  return scope;
}

// Every core, once; mempool_barrier_init() does it.
void dma_collective_runtime_init(void);

// One contiguous global L2 <-> L1 range under `scope`. Exactly one side must be
// in the interleaved L1 range, and every group owning part of it must be in the
// scope.
dma_collective_op_t
dma_collective_submit_nonblocking(const dma_collective_scope_t *scope,
                                  uint32_t stream, void *dest, const void *src,
                                  size_t len);

// The same for the whole cluster, without a scope argument: a scope struct in
// memory is a copy of a .rodata template, i.e. an L2 read on the hot path.
dma_collective_op_t dma_collective_submit_all_nonblocking(uint32_t stream,
                                                          void *dest,
                                                          const void *src,
                                                          size_t len);

// An already group-specialised 2-D descriptor: each core passes ITS group's
// shape. The lower layer for VC descriptors and manually partitioned transfers.
dma_collective_op_t dma_collective_submit_2d_nonblocking(
    const dma_collective_scope_t *scope, uint32_t stream, void *dest,
    const void *src, size_t len, uint32_t dst_stride, uint32_t src_stride,
    uint32_t reps);

// Both wait `op` and every earlier operation; see COMPLETION above. Only
// coordinators see their own launch errors in op.status; every core sees
// engine errors in the return value.
uint32_t dma_collective_wait(const dma_collective_op_t *last);
uint32_t dma_collective_wait_ready(const dma_collective_op_t *op);

// Whole-cluster forms on stream 0. Every core calls.
static inline dma_collective_op_t
dma_collective_memcpy_nonblocking(void *dest, const void *src, size_t len) {
  return dma_collective_submit_all_nonblocking(0u, dest, src, len);
}

static inline uint32_t
dma_collective_memcpy_blocking(void *dest, const void *src, size_t len) {
  const dma_collective_op_t op =
      dma_collective_memcpy_nonblocking(dest, src, len);
  const uint32_t error = dma_collective_wait(&op);
  return op.status != DMA_COLLECTIVE_OK ? op.status : error;
}

#endif // _DMA_COLLECTIVE_H_
