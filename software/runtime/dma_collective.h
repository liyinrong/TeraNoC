// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef _DMA_COLLECTIVE_H_
#define _DMA_COLLECTIVE_H_

#include <stddef.h>
#include <stdint.h>

#include "dma.h"

#define DMA_COLLECTIVE_GROUP_WORDS (NUM_BANKS / NUM_GROUPS)
#define DMA_COLLECTIVE_SLOT_FIELDS 10u

// One slot per collective that can be in flight at once. The budget is the
// group's bank stripe: the whole per-group state has to fit inside it so that
// two groups never share a bank, which is the property the padding exists to
// keep. A hardcoded 2 did not fit minpool at all (16 words against 20), and
// because dma_collective.c.o links into every binary that stopped the flavour
// from building ANY software.
//
// The cap stays at 2 until the _all_ API keeps a ring of handles: extra slots
// buy nothing while each issue overwrites the last handle, and raising it
// would silently weaken dma_collective_smoke, whose four generations currently
// wrap the slot index and so cover reuse. Raise it with that change, and
// lengthen the smoke to keep wrapping. The engine is not the constraint --
// its request FIFO is 16 deep per stream (mempool_idma_group.sv:12) and every
// accepted request gets its own id.
#define DMA_COLLECTIVE_MAX_SLOTS 2u
#define DMA_COLLECTIVE_SLOTS                                                   \
  ((DMA_COLLECTIVE_GROUP_WORDS / DMA_COLLECTIVE_SLOT_FIELDS) >                 \
           DMA_COLLECTIVE_MAX_SLOTS                                            \
       ? DMA_COLLECTIVE_MAX_SLOTS                                              \
       : (DMA_COLLECTIVE_GROUP_WORDS / DMA_COLLECTIVE_SLOT_FIELDS))
#define DMA_COLLECTIVE_STATE_FIELDS                                            \
  (DMA_COLLECTIVE_SLOTS * DMA_COLLECTIVE_SLOT_FIELDS)
#define DMA_COLLECTIVE_ALIGNMENT (NUM_BANKS * sizeof(uint32_t))

#if DMA_COLLECTIVE_SLOTS < 1u
#error "A DMA collective slot does not fit in one group bank stripe"
#endif
#if DMA_COLLECTIVE_GROUP_WORDS < DMA_COLLECTIVE_STATE_FIELDS
#error "A DMA collective group state does not fit in one group bank stripe"
#endif

#define DMA_COLLECTIVE_STORAGE                                                 \
  __attribute__((aligned(DMA_COLLECTIVE_ALIGNMENT), section(".l1")))

enum {
  DMA_COLLECTIVE_OK = 0,
  DMA_COLLECTIVE_ERROR_ARGUMENT = 0x80000001u,
  DMA_COLLECTIVE_ERROR_SCOPE = 0x80000002u,
  DMA_COLLECTIVE_ERROR_STALE = 0x80000003u,
  DMA_COLLECTIVE_ERROR_REUSE = 0x80000004u,
  DMA_COLLECTIVE_ERROR_LAUNCH = 0x80000005u,
};

typedef struct {
  uint32_t base_group;
  uint32_t num_groups;
  uint32_t cores_per_group;
} dma_collective_scope_t;

typedef struct {
  volatile uint32_t submit_epoch;
  volatile uint32_t issued_epoch;
  volatile uint32_t wait_epoch;
  volatile uint32_t ready_epoch;
  volatile dma_id_t local_id;
  volatile uint32_t error;
  volatile uint32_t descriptors;
  volatile uint32_t issuer_core;
  volatile uint32_t poller_core;
  volatile uint32_t stream;
} dma_collective_slot_state_t;

typedef struct {
  dma_collective_slot_state_t slot[DMA_COLLECTIVE_SLOTS];
  uint32_t padding[DMA_COLLECTIVE_GROUP_WORDS - DMA_COLLECTIVE_STATE_FIELDS];
} dma_collective_group_state_t;

typedef struct dma_collective_state {
  dma_collective_group_state_t group[NUM_GROUPS];
} dma_collective_state_t;

typedef struct {
  dma_collective_state_t *state;
  uint32_t generation;
  uint32_t slot;
  uint32_t group_id;
  uint32_t base_group;
  uint32_t num_groups;
  uint32_t stream;
  dma_id_t local_id;
  uint32_t status;
} dma_collective_handle_t;

static inline dma_collective_scope_t dma_collective_scope_all(void) {
  const dma_collective_scope_t scope = {
      .base_group = 0,
      .num_groups = NUM_GROUPS,
      .cores_per_group = NUM_CORES_PER_GROUP,
  };
  return scope;
}

// Every core in the scope calls init after mempool_barrier_init(), followed by
// a scope-appropriate barrier before the state is used. The deterministic first
// core of each group initializes that group's bank-local state; the first core
// of every tile initializes the tile-local descriptor lock.
void dma_collective_init(dma_collective_state_t *state,
                         const dma_collective_scope_t *scope);

// Split one contiguous L2<->L1 transfer over the participating group engines.
// Exactly one side must be in the ordinary interleaved L1 range. Every core in
// every group in scope calls this with the same nonzero generation. The first
// arrival in each group issues that group's stripe and publishes its local id.
// Up to DMA_COLLECTIVE_SLOTS generations may be outstanding concurrently.
dma_collective_handle_t
dma_collective_memcpy_nonblocking(dma_collective_state_t *state,
                                  const dma_collective_scope_t *scope,
                                  uint32_t generation, uint32_t stream,
                                  void *dest, const void *src, size_t len);

// Agent form of the above: the caller asserts it is the SOLE issuer for its
// group, so the issuer election is skipped. Calling it from more than one core
// per group is a programming error.
dma_collective_handle_t dma_collective_memcpy_agent_nonblocking(
    dma_collective_state_t *state, const dma_collective_scope_t *scope,
    uint32_t generation, uint32_t stream, void *dest, const void *src,
    size_t len);

// Submit an already group-specialized 2-D transfer. This is the common lower
// layer for virtual-cluster descriptors and manually partitioned workloads.
dma_collective_handle_t dma_collective_submit_2d_nonblocking(
    dma_collective_state_t *state, const dma_collective_scope_t *scope,
    uint32_t generation, uint32_t stream, void *dest, const void *src,
    size_t len, uint32_t dst_stride, uint32_t src_stride, uint32_t reps);

// Wait only for the caller's group-local id. Any core can become the completion
// poller; this does not wait for other groups or impose a core barrier.
uint32_t dma_collective_wait_local(const dma_collective_handle_t *handle);

// Fence every group engine in the scope through handle->generation. All groups
// must have at least one caller publishing local completion, but no
// core-arrival barrier is implied.
uint32_t dma_collective_wait(const dma_collective_scope_t *scope,
                             const dma_collective_handle_t *handle);

// Wait for the named operation and every earlier request accepted by the same
// group streams. iDMA transfer ids retire in issue order, so the explicit last
// handle is the non-racy stream fence. This remains a DMA-only wait: it does
// not impose a core barrier.
uint32_t
dma_collective_wait_through(const dma_collective_scope_t *scope,
                            const dma_collective_handle_t *last_handle);

uint32_t dma_collective_done_local(const dma_collective_handle_t *handle);

uint32_t dma_collective_memcpy_blocking(dma_collective_state_t *state,
                                        const dma_collective_scope_t *scope,
                                        uint32_t generation, uint32_t stream,
                                        void *dest, const void *src,
                                        size_t len);

#endif // _DMA_COLLECTIVE_H_
