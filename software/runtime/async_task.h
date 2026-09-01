// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef _ASYNC_TASK_H_
#define _ASYNC_TASK_H_

#include <stdint.h>

#include "dma.h"

#define TNC_TASK_SLOTS 4u
#define TNC_BUFFER_COUNT 2u
#define TNC_GROUP_WORDS (NUM_BANKS / NUM_GROUPS)
#define TNC_TILE_WORDS NUM_BANKS_PER_TILE
#define TNC_TASK_STORAGE                                                       \
  __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")))

#if NUM_TILES_PER_GROUP > 16
#error "Task tile masks support at most 16 tiles per group"
#endif

enum {
  TNC_OK = 0,
  TNC_ERROR_ARGUMENT = 0x81000001u,
  TNC_ERROR_SCOPE = 0x81000002u,
  TNC_ERROR_STALE = 0x81000003u,
  TNC_ERROR_REUSE = 0x81000004u,
  TNC_ERROR_NOT_READY = 0x81000005u,
};

typedef enum {
  TNC_TOKEN_NONE = 0,
  TNC_TOKEN_DMA = 1,
  TNC_TOKEN_REDMULE = 2,
  TNC_TOKEN_WORKERS = 3,
} tnc_token_kind_t;

typedef enum {
  TNC_GRANULARITY_GROUP = 0,
  TNC_GRANULARITY_TILE = 1,
  TNC_GRANULARITY_WORKER = 2,
} tnc_granularity_t;

typedef struct {
  uint16_t task_id;
  uint16_t generation;
  uint16_t base_group;
  uint16_t num_groups;
  uint16_t tile_mask;
  uint8_t granularity;
  uint8_t reserved;
} tnc_task_key_t;

typedef struct {
  uint8_t kind;
  uint8_t reserved;
  uint16_t unit;
  uint32_t target;
} tnc_token_t;

#define TNC_TASK_GLOBAL_FIELDS 7u
#define TNC_TASK_GROUP_FIELDS 8u

#if NUM_BANKS < TNC_TASK_GLOBAL_FIELDS
#error "Task-global state does not fit in one interleaved L1 row"
#endif
#if TNC_GROUP_WORDS < TNC_TASK_GROUP_FIELDS
#error "Task group state does not fit in one group bank stripe"
#endif

typedef struct {
  volatile uint32_t active_epoch;
  volatile uint32_t ready_epoch;
  volatile uint32_t free_epoch;
  volatile uint32_t status;
  tnc_task_key_t key;
  uint32_t padding[NUM_BANKS - TNC_TASK_GLOBAL_FIELDS];
} tnc_task_global_state_t;

typedef struct {
  volatile uint32_t token_epoch;
  volatile uint32_t ready_epoch;
  volatile uint32_t release_epoch;
  volatile uint32_t consumer_epoch;
  volatile uint32_t consumer_count;
  volatile uint32_t consumer_target;
  tnc_token_t token;
  uint32_t padding[TNC_GROUP_WORDS - TNC_TASK_GROUP_FIELDS];
} tnc_task_group_state_t;

typedef struct {
  tnc_task_global_state_t global;
  tnc_task_group_state_t group[NUM_GROUPS];
} tnc_task_slot_t;

typedef struct {
  tnc_task_slot_t slot[TNC_TASK_SLOTS];
} tnc_task_context_t;

#define TNC_AGENT_FIELDS 10u
#if NUM_CORES_PER_TILE > 32
#error "Worker completion bitmap supports at most 32 cores per tile"
#endif
#if TNC_TILE_WORDS < TNC_AGENT_FIELDS
#error "Tile-agent state does not fit in one tile bank stripe"
#endif

typedef struct {
  volatile uint32_t worker_issue_epoch;
  volatile uint32_t worker_done_count;
  volatile uint32_t worker_done_mask;
  volatile uint32_t worker_target;
  volatile uint32_t mailbox_epoch;
  volatile uint32_t mailbox_arg;
  volatile uint32_t redmule_target;
  volatile uint32_t redmule_epoch;
  volatile uint32_t redmule_done_epoch;
  volatile uint32_t status;
  uint32_t padding[TNC_TILE_WORDS - TNC_AGENT_FIELDS];
} tnc_tile_agent_t;

typedef struct {
  tnc_tile_agent_t tile[NUM_CORES / NUM_CORES_PER_TILE];
} tnc_agent_context_t;

typedef struct {
  volatile uint32_t ready_epoch[TNC_BUFFER_COUNT];
  volatile uint32_t free_epoch[TNC_BUFFER_COUNT];
  uint32_t padding[NUM_BANKS - 2u * TNC_BUFFER_COUNT];
} tnc_buffer_global_state_t;

#define TNC_BUFFER_GROUP_FIELDS 10u
#if TNC_GROUP_WORDS < TNC_BUFFER_GROUP_FIELDS
#error "Buffer group state does not fit in one group bank stripe"
#endif

typedef struct {
  volatile uint32_t ready_epoch[TNC_BUFFER_COUNT];
  volatile uint32_t consumer_epoch[TNC_BUFFER_COUNT];
  volatile uint32_t consumer_count[TNC_BUFFER_COUNT];
  volatile uint32_t release_epoch[TNC_BUFFER_COUNT];
  volatile uint32_t consumer_target[TNC_BUFFER_COUNT];
  uint32_t padding[TNC_GROUP_WORDS - TNC_BUFFER_GROUP_FIELDS];
} tnc_buffer_group_state_t;

typedef struct {
  tnc_buffer_global_state_t global;
  tnc_buffer_group_state_t group[NUM_GROUPS];
} tnc_buffer_state_t;

static inline uint32_t tnc_counter8_done(uint8_t done, uint8_t target) {
  return (int8_t)(done - target) >= 0;
}

static inline uint32_t tnc_counter16_done(uint16_t done, uint16_t target) {
  return (int16_t)(done - target) >= 0;
}

static inline uint8_t tnc_redmule_target(int32_t acquire_id) {
  return (uint8_t)(acquire_id + 1);
}

int32_t tnc_redmule_acquire_target(void);
void tnc_redmule_trigger(void);
uint8_t tnc_redmule_completed(void);
uint32_t tnc_redmule_done(uint8_t target);
void tnc_redmule_wait(uint8_t target);
void tnc_engine_runtime_init(void);

uint32_t tnc_is_group_coordinator(void);
uint32_t tnc_is_tile_agent(void);
uint32_t tnc_task_key_equal(const tnc_task_key_t *a, const tnc_task_key_t *b);
void tnc_task_context_init(tnc_task_context_t *context);
uint32_t tnc_task_open(tnc_task_context_t *context, const tnc_task_key_t *key);
uint32_t tnc_task_publish_token(tnc_task_context_t *context,
                                const tnc_task_key_t *key,
                                const tnc_token_t *token);
uint32_t tnc_task_token_done(const tnc_token_t *token,
                             const tnc_agent_context_t *agents);
uint32_t tnc_task_complete_local(tnc_task_context_t *context,
                                 const tnc_task_key_t *key,
                                 const tnc_agent_context_t *agents);
uint32_t tnc_task_progress_ready(tnc_task_context_t *context,
                                 const tnc_task_key_t *key);
uint32_t tnc_task_wait_ready(tnc_task_context_t *context,
                             const tnc_task_key_t *key);
uint32_t tnc_task_ready(const tnc_task_context_t *context,
                        const tnc_task_key_t *key);
uint32_t tnc_task_all_ready(const tnc_task_context_t *context,
                            const tnc_task_key_t *keys, uint32_t count);
uint32_t tnc_task_wait_all(tnc_task_context_t *context,
                           const tnc_task_key_t *keys, uint32_t count);
uint32_t tnc_task_retire(tnc_task_context_t *context,
                         const tnc_task_key_t *key);

void tnc_agent_context_init(tnc_agent_context_t *agents);
uint32_t tnc_worker_dispatch(tnc_agent_context_t *agents, uint32_t tile,
                             uint32_t workers, uint32_t argument);
uint32_t tnc_worker_wait_dispatch(const tnc_agent_context_t *agents,
                                  uint32_t tile, uint32_t after_epoch,
                                  uint32_t *argument);
void tnc_worker_complete(tnc_agent_context_t *agents, uint32_t tile,
                         uint32_t epoch);
uint32_t tnc_worker_done(const tnc_agent_context_t *agents, uint32_t tile,
                         uint32_t epoch);
uint32_t tnc_redmule_agent_publish(tnc_agent_context_t *agents,
                                   const tnc_task_key_t *key, uint8_t target);
uint32_t tnc_redmule_agent_progress(tnc_agent_context_t *agents,
                                    const tnc_task_key_t *key);
uint32_t tnc_redmule_group_done(const tnc_agent_context_t *agents,
                                const tnc_task_key_t *key);

void tnc_buffer_init(tnc_buffer_state_t *buffer);
uint32_t tnc_buffer_publish_ready(tnc_buffer_state_t *buffer,
                                  const tnc_task_key_t *key);
uint32_t tnc_buffer_progress_ready(tnc_buffer_state_t *buffer,
                                   const tnc_task_key_t *key);
void tnc_buffer_wait_ready(tnc_buffer_state_t *buffer,
                           const tnc_task_key_t *key);
uint32_t tnc_buffer_begin_consumption(tnc_buffer_state_t *buffer,
                                      const tnc_task_key_t *key,
                                      uint32_t consumers_per_group);
uint32_t tnc_buffer_consumer_release(tnc_buffer_state_t *buffer,
                                     const tnc_task_key_t *key);
uint32_t tnc_buffer_progress_free(tnc_buffer_state_t *buffer,
                                  const tnc_task_key_t *key);
void tnc_buffer_wait_free(tnc_buffer_state_t *buffer,
                          const tnc_task_key_t *key);

#endif // _ASYNC_TASK_H_
