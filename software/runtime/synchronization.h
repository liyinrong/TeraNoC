// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Samuel Riedel, ETH Zurich

#ifndef __SYNCHRONIZATION_H__
#define __SYNCHRONIZATION_H__

// Barrier functions
void mempool_barrier_init(uint32_t core_id);
void mempool_barrier(uint32_t num_cores);
void mempool_log_barrier(uint32_t step, uint32_t core_id);
void mempool_anyradixlog_barrier(uint32_t radix, uint32_t core_id);
void mempool_log_partial_barrier(uint32_t step, uint32_t core_id,
                                 uint32_t num_cores_barrier);
void mempool_linlog_barrier(uint32_t step, uint32_t core_id);
// ---------------------------------------------------------------------------
// Polling barriers.
//
// The barriers above SLEEP, which is correct and cheap while the scope is
// credit-neutral (see synchronization.c). Use these instead only where that
// does not hold -- a caller that cannot guarantee one wfi per wake it may
// receive, e.g. because a second wake source (RedMulE EOC, fsync) is armed in
// the same phase, or a non-participant may be woken by a wider scope.
//
// Cost: waiters spin on `generation` instead of sleeping, so they consume L1
// bandwidth for the whole wait. Prefer the sleeping barriers.
//
// ABA-safe by construction: waiters watch a monotonically increasing
// generation, never the reset-to-zero count.
typedef struct {
  volatile uint32_t count;
  volatile uint32_t generation;
} mempool_poll_slot_t;

// `.l1` is NOBITS: every slot must be zeroed before first use.
static inline void mempool_barrier_poll_init(mempool_poll_slot_t *slot) {
  slot->count = 0u;
  slot->generation = 0u;
}

void mempool_barrier_poll_slot(mempool_poll_slot_t *slot, uint32_t num_cores);

void mempool_partial_barrier(uint32_t volatile core_id,
                             uint32_t volatile core_init,
                             uint32_t volatile num_sleeping_cores,
                             uint32_t volatile memloc);

#endif // __SYNCHRONIZATION_H__
