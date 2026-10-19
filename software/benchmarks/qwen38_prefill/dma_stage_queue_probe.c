// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Six divergent DMA requests exercise publication and ring-slot reuse. Four
// more fill the queue before any wait. Only core 0 publishes and each group
// coordinator issues; no worker joins a DMA collective.
#include <stdint.h>

#include "dma_stage_queue.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#ifndef PROBE_ROWS
#define PROBE_ROWS 8u
#endif
#define PROBE_WORDS (NUM_BANKS * PROBE_ROWS)
#define PROBE_ALIGN (NUM_BANKS * sizeof(uint32_t))
#define PROBE_ROUNDS 6u
#define PROBE_QUEUED 4u

static uint32_t source[PROBE_WORDS]
    __attribute__((section(".l2_bss"), aligned(PROBE_ALIGN)));
static uint32_t destination[PROBE_WORDS]
    __attribute__((section(".l1"), aligned(PROBE_ALIGN)));
static uint32_t returned[PROBE_WORDS]
    __attribute__((section(".l2_bss"), aligned(PROBE_ALIGN)));
static uint32_t queued_source[PROBE_QUEUED][PROBE_WORDS]
    __attribute__((section(".l2_bss"), aligned(PROBE_ALIGN)));
static uint32_t queued_destination[PROBE_QUEUED][PROBE_WORDS]
    __attribute__((section(".l1"), aligned(PROBE_ALIGN)));
static q38_dma_stage_queue_t queue
    __attribute__((section(".l1"), aligned(PROBE_ALIGN)));
static uint32_t failures[NUM_CORES] __attribute__((section(".l1")));
static uint32_t worker_progress[NUM_CORES] __attribute__((section(".l1")));

static uint32_t pattern(uint32_t round, uint32_t word) {
  return 0x8c380000u ^ (round << 20) ^ (word * 0x1021u);
}

int main(void) {
  const uint32_t core = mempool_get_core_id();
  uint32_t error = 0u;
  mempool_barrier_init(core);
  q38_dma_stage_init(&queue);
  failures[core] = 0u;
  worker_progress[core] = 0u;
  mempool_barrier(NUM_CORES);

  for (uint32_t round = 1u; round <= PROBE_ROUNDS; round += 2u) {
    for (uint32_t i = core; i < PROBE_WORDS; i += NUM_CORES) {
      source[i] = pattern(round, i);
      destination[i] = ~pattern(round, i);
      returned[i] = 0u;
    }
    mempool_barrier(NUM_CORES);

    if (core == 0u)
      error += q38_dma_stage_publish(&queue, round, destination, source,
                                     sizeof(source), 0u) != Q38_DMA_QUEUE_OK;
    if (mempool_is_group_coordinator()) {
      while (q38_dma_stage_ready(&queue, round) == Q38_DMA_QUEUE_PENDING)
        error += q38_dma_stage_progress(&queue) == Q38_DMA_QUEUE_INVALID;
    } else {
      uint32_t progress = core ^ round;
      for (uint32_t i = 0u; i < 128u; ++i)
        progress = progress * 1664525u + 1013904223u;
      worker_progress[core] ^= progress;
      while (q38_dma_stage_ready(&queue, round) == Q38_DMA_QUEUE_PENDING)
        mempool_wait(4u);
    }
    error += q38_dma_stage_ready(&queue, round) != Q38_DMA_QUEUE_OK;
    mempool_barrier(NUM_CORES);
    for (uint32_t i = core; i < PROBE_WORDS; i += NUM_CORES)
      error += destination[i] != pattern(round, i);
    mempool_barrier(NUM_CORES);
    if (core == 0u)
      error += q38_dma_stage_release(&queue, round) != Q38_DMA_QUEUE_OK;
    mempool_barrier(NUM_CORES);

    if (core == 0u)
      error += q38_dma_stage_publish(&queue, round + 1u, returned, destination,
                                     sizeof(returned), 0u) != Q38_DMA_QUEUE_OK;
    if (mempool_is_group_coordinator()) {
      while (q38_dma_stage_ready(&queue, round + 1u) == Q38_DMA_QUEUE_PENDING)
        error += q38_dma_stage_progress(&queue) == Q38_DMA_QUEUE_INVALID;
    } else {
      while (q38_dma_stage_ready(&queue, round + 1u) == Q38_DMA_QUEUE_PENDING)
        mempool_wait(4u);
    }
    error += q38_dma_stage_ready(&queue, round + 1u) != Q38_DMA_QUEUE_OK;
    mempool_barrier(NUM_CORES);
    for (uint32_t i = core; i < PROBE_WORDS; i += NUM_CORES)
      error += returned[i] != pattern(round, i);
    mempool_barrier(NUM_CORES);
    if (core == 0u)
      error += q38_dma_stage_release(&queue, round + 1u) != Q38_DMA_QUEUE_OK;
    mempool_barrier(NUM_CORES);
    if (core == 0u)
      printf("[qwen38_dma_stage_queue] completed_round=%u\n",
             (unsigned)(round + 1u));
  }
  for (uint32_t i = core; i < PROBE_QUEUED * PROBE_WORDS; i += NUM_CORES) {
    const uint32_t operation = i / PROBE_WORDS;
    const uint32_t word = i % PROBE_WORDS;
    queued_source[operation][word] =
        pattern(PROBE_ROUNDS + operation + 1u, word);
    queued_destination[operation][word] = 0u;
  }
  mempool_barrier(NUM_CORES);
  if (core == 0u)
    for (uint32_t operation = 0u; operation < PROBE_QUEUED; ++operation)
      error += q38_dma_stage_publish(
                   &queue, PROBE_ROUNDS + operation + 1u,
                   queued_destination[operation], queued_source[operation],
                   sizeof(queued_source[operation]), 0u) != Q38_DMA_QUEUE_OK;
  const uint32_t last_epoch = PROBE_ROUNDS + PROBE_QUEUED;
  if (mempool_is_group_coordinator()) {
    while (q38_dma_stage_ready(&queue, last_epoch) == Q38_DMA_QUEUE_PENDING)
      error += q38_dma_stage_progress(&queue) == Q38_DMA_QUEUE_INVALID;
  } else {
    while (q38_dma_stage_ready(&queue, last_epoch) == Q38_DMA_QUEUE_PENDING)
      mempool_wait(4u);
  }
  for (uint32_t operation = 0u; operation < PROBE_QUEUED; ++operation)
    error += q38_dma_stage_ready(&queue, PROBE_ROUNDS + operation + 1u) !=
             Q38_DMA_QUEUE_OK;
  mempool_barrier(NUM_CORES);
  for (uint32_t i = core; i < PROBE_QUEUED * PROBE_WORDS; i += NUM_CORES) {
    const uint32_t operation = i / PROBE_WORDS;
    const uint32_t word = i % PROBE_WORDS;
    error += queued_destination[operation][word] !=
             pattern(PROBE_ROUNDS + operation + 1u, word);
  }
  mempool_barrier(NUM_CORES);
  if (core == 0u)
    for (uint32_t operation = 0u; operation < PROBE_QUEUED; ++operation)
      error += q38_dma_stage_release(&queue, PROBE_ROUNDS + operation + 1u) !=
               Q38_DMA_QUEUE_OK;
  mempool_barrier(NUM_CORES);
  if (!mempool_is_group_coordinator())
    error += worker_progress[core] == 0u;
  failures[core] = error;
  mempool_barrier(NUM_CORES);
  if (core == 0u) {
    uint32_t total = 0u;
    for (uint32_t i = 0u; i < NUM_CORES; ++i)
      total += failures[i];
    printf("[qwen38_dma_stage_queue] rounds=%u queued=%u bytes=%u "
           "groups=%u errors=%u\n",
           (unsigned)PROBE_ROUNDS, (unsigned)PROBE_QUEUED,
           (unsigned)sizeof(source), (unsigned)NUM_GROUPS, (unsigned)total);
    failures[0] = total;
  }
  mempool_barrier(NUM_CORES);
  return (int)failures[0];
}
