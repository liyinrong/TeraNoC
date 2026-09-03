// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "async_task.h"

#include <stddef.h>

#include "runtime.h"

#if NUM_REDMULE_TILES > 0
#include "archi_redmule.h"
#if defined(__has_include)
#if __has_include("hwpe_ctrl_target.h")
#include "hwpe_ctrl_target.h"
#define TNC_REDMULE_TARGET_REGS 1
#endif
#endif

#if defined(TNC_REDMULE_TARGET_REGS)
static inline volatile redmule_regif_t *tnc_redmule_regs(void) {
  return (volatile redmule_regif_t *)REDMULE_BASE_ADD;
}
#else
static inline volatile uint32_t *tnc_redmule_reg(uint32_t offset) {
  return (volatile uint32_t *)(REDMULE_BASE_ADD + offset);
}
#endif
#endif

static uint32_t tnc_scope_valid(const tnc_task_key_t *key) {
  return key != NULL && key->generation != 0u && key->num_groups != 0u &&
         key->base_group < (uint16_t)NUM_GROUPS &&
         key->num_groups <= (uint16_t)NUM_GROUPS - key->base_group &&
         key->granularity <= TNC_GRANULARITY_WORKER &&
         (key->tile_mask & (uint16_t) ~((1u << NUM_TILES_PER_GROUP) - 1u)) ==
             0u &&
         (key->granularity != TNC_GRANULARITY_TILE || key->tile_mask != 0u);
}

static uint32_t tnc_group_in_scope(const tnc_task_key_t *key, uint32_t group) {
  return group >= key->base_group &&
         group < (uint32_t)key->base_group + key->num_groups;
}

static uint32_t tnc_slot_index(const tnc_task_key_t *key) {
  return (uint32_t)key->task_id % TNC_TASK_SLOTS;
}

static uint32_t tnc_task_key_status(const tnc_task_global_state_t *state,
                                    const tnc_task_key_t *key) {
  const uint16_t active =
      (uint16_t)__atomic_load_n(&state->active_epoch, __ATOMIC_ACQUIRE);
  if (active == key->generation) {
    if (tnc_counter16_done(
            (uint16_t)__atomic_load_n(&state->free_epoch, __ATOMIC_ACQUIRE),
            active))
      return TNC_ERROR_STALE;
    return tnc_task_key_equal(&state->key, key) ? TNC_OK : TNC_ERROR_SCOPE;
  }
  if (active == 0u)
    return TNC_ERROR_NOT_READY;
  return tnc_counter16_done(active, key->generation) ? TNC_ERROR_STALE
                                                     : TNC_ERROR_REUSE;
}

static inline void tnc_poll_backoff(void) { mempool_wait(4u); }

static inline uint32_t tnc_epoch16_done(uint32_t done, uint16_t target) {
  return tnc_counter16_done((uint16_t)done, target);
}

int32_t tnc_redmule_acquire_target(void) {
#if NUM_REDMULE_TILES > 0
  int32_t acquire_id;
  do {
#if defined(TNC_REDMULE_TARGET_REGS)
    acquire_id = (int32_t)tnc_redmule_regs()->hwpe_ctrl.acquire;
#else
    acquire_id = (int32_t)*tnc_redmule_reg(REDMULE_ACQUIRE);
#endif
    if (acquire_id < 0)
      tnc_poll_backoff();
  } while (acquire_id < 0);
  return (int32_t)tnc_redmule_target(acquire_id);
#else
  return -1;
#endif
}

void tnc_redmule_trigger(void) {
#if NUM_REDMULE_TILES > 0
#if defined(TNC_REDMULE_TARGET_REGS)
  tnc_redmule_regs()->hwpe_ctrl.commit_trigger = 0u;
#else
  *tnc_redmule_reg(REDMULE_TRIGGER) = 0u;
#endif
#endif
}

uint8_t tnc_redmule_completed(void) {
#if NUM_REDMULE_TILES > 0
#if defined(TNC_REDMULE_TARGET_REGS)
  return (uint8_t)tnc_redmule_regs()->hwpe_ctrl.running_job;
#else
  return (uint8_t)*tnc_redmule_reg(REDMULE_RUNNING_JOB);
#endif
#else
  return 0u;
#endif
}

uint32_t tnc_redmule_done(uint8_t target) {
  return tnc_counter8_done(tnc_redmule_completed(), target);
}

// Sleep on RedMulE's EOC rather than polling RUNNING_JOB.
//
// This is not only cheaper, it is required for the sleeping barriers to stay
// safe. redmule_evt is OR-ed into the wake of core 0 of every RedMulE tile
// (mempool_tile.sv:1745). Polling here left every completion pulse banked and
// unspent in Snitch's 3-bit wake counter, so eight jobs between two sleeps
// wrapped it and destroyed seven pending wakes -- after which the next barrier
// wake could be lost outright.
//
// do/while, not while/do: entering the loop body first means the wfi always
// runs, so a pulse that already landed is SPENT (wfi returns immediately when
// the counter is non-zero) instead of being left as a credit. Spurious wakes
// are harmless -- each spends exactly the credit that caused it, and the
// predicate is re-checked every turn.
void tnc_redmule_wait(uint8_t target) {
  do {
    mempool_wfi();
  } while (!tnc_redmule_done(target));
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
}

void tnc_engine_runtime_init(void) {
#if NUM_REDMULE_TILES > 0
  if (!mempool_is_tile_agent() || !mempool_tile_has_redmule())
    return;
#if defined(TNC_REDMULE_TARGET_REGS)
  tnc_redmule_regs()->hwpe_ctrl.soft_clear = 0u;
#else
  *tnc_redmule_reg(REDMULE_SOFT_CLEAR) = 0u;
#endif
  mempool_wait(10u);
#endif
}

uint32_t tnc_is_group_coordinator(void) {
  return mempool_is_group_coordinator();
}

uint32_t tnc_is_tile_agent(void) { return mempool_is_tile_agent(); }

uint32_t tnc_task_key_equal(const tnc_task_key_t *a, const tnc_task_key_t *b) {
  return a != NULL && b != NULL && a->task_id == b->task_id &&
         a->generation == b->generation && a->base_group == b->base_group &&
         a->num_groups == b->num_groups && a->tile_mask == b->tile_mask &&
         a->granularity == b->granularity;
}

void tnc_task_context_init(tnc_task_context_t *context) {
  if (context == NULL)
    return;
  const uint32_t core = mempool_get_core_id();
  const uint32_t group = mempool_get_group_id();
  for (uint32_t slot = 0; slot < TNC_TASK_SLOTS; ++slot) {
    if (core == 0u) {
      context->slot[slot].global.active_epoch = 0u;
      context->slot[slot].global.ready_epoch = 0u;
      context->slot[slot].global.free_epoch = 0u;
      context->slot[slot].global.status = TNC_OK;
      context->slot[slot].global.key.task_id = 0u;
      context->slot[slot].global.key.generation = 0u;
      context->slot[slot].global.key.base_group = 0u;
      context->slot[slot].global.key.num_groups = 0u;
      context->slot[slot].global.key.tile_mask = 0u;
      context->slot[slot].global.key.granularity = 0u;
      context->slot[slot].global.key.reserved = 0u;
      for (uint32_t i = 0u; i < NUM_BANKS - TNC_TASK_GLOBAL_FIELDS; ++i)
        context->slot[slot].global.padding[i] = 0u;
    }
    if (tnc_is_group_coordinator()) {
      tnc_task_group_state_t *const state = &context->slot[slot].group[group];
      state->token_epoch = 0u;
      state->ready_epoch = 0u;
      state->release_epoch = 0u;
      state->consumer_epoch = 0u;
      state->consumer_count = 0u;
      state->consumer_target = 0u;
      state->token.kind = TNC_TOKEN_NONE;
      state->token.reserved = 0u;
      state->token.unit = 0u;
      state->token.target = 0u;
      for (uint32_t i = 0u; i < TNC_GROUP_WORDS - TNC_TASK_GROUP_FIELDS; ++i)
        state->padding[i] = 0u;
    }
  }
  __sync_synchronize();
}

uint32_t tnc_task_open(tnc_task_context_t *context, const tnc_task_key_t *key) {
  if (context == NULL || !tnc_scope_valid(key) || !tnc_is_group_coordinator())
    return TNC_ERROR_ARGUMENT;
  const uint32_t group = mempool_get_group_id();
  if (!tnc_group_in_scope(key, group))
    return TNC_ERROR_SCOPE;
  tnc_task_global_state_t *const state =
      &context->slot[tnc_slot_index(key)].global;

  if (group == key->base_group && tnc_is_group_coordinator()) {
    const uint32_t active =
        __atomic_load_n(&state->active_epoch, __ATOMIC_ACQUIRE);
    if (active == key->generation &&
        tnc_counter16_done(
            (uint16_t)__atomic_load_n(&state->free_epoch, __ATOMIC_ACQUIRE),
            (uint16_t)active))
      return TNC_ERROR_STALE;
    if (active != 0u && active != key->generation &&
        tnc_counter16_done((uint16_t)active, key->generation))
      return TNC_ERROR_STALE;
    if (active != key->generation && active != 0u &&
        !tnc_counter16_done(
            (uint16_t)__atomic_load_n(&state->free_epoch, __ATOMIC_ACQUIRE),
            (uint16_t)active))
      return TNC_ERROR_REUSE;
    state->key.task_id = key->task_id;
    state->key.generation = key->generation;
    state->key.base_group = key->base_group;
    state->key.num_groups = key->num_groups;
    state->key.tile_mask = key->tile_mask;
    state->key.granularity = key->granularity;
    state->key.reserved = key->reserved;
    state->status = TNC_OK;
    __atomic_store_n(&state->active_epoch, key->generation, __ATOMIC_RELEASE);
  }
  while (__atomic_load_n(&state->active_epoch, __ATOMIC_ACQUIRE) !=
         key->generation) {
    const uint32_t active =
        __atomic_load_n(&state->active_epoch, __ATOMIC_ACQUIRE);
    if (active != 0u && active != key->generation &&
        tnc_counter16_done((uint16_t)active, key->generation))
      return TNC_ERROR_STALE;
    if (active != 0u && active != key->generation &&
        !tnc_counter16_done(
            (uint16_t)__atomic_load_n(&state->free_epoch, __ATOMIC_ACQUIRE),
            (uint16_t)active))
      return TNC_ERROR_REUSE;
    tnc_poll_backoff();
  }
  if (!tnc_task_key_equal(&state->key, key))
    return TNC_ERROR_SCOPE;
  return state->status;
}

uint32_t tnc_task_publish_token(tnc_task_context_t *context,
                                const tnc_task_key_t *key,
                                const tnc_token_t *token) {
  if (context == NULL || token == NULL || !tnc_scope_valid(key) ||
      !tnc_is_group_coordinator())
    return TNC_ERROR_ARGUMENT;
  const uint32_t group = mempool_get_group_id();
  if (!tnc_group_in_scope(key, group))
    return TNC_ERROR_SCOPE;
  const uint32_t key_status =
      tnc_task_key_status(&context->slot[tnc_slot_index(key)].global, key);
  if (key_status != TNC_OK)
    return key_status;
  if (token->kind <= TNC_TOKEN_NONE || token->kind > TNC_TOKEN_WORKERS)
    return TNC_ERROR_ARGUMENT;
  tnc_task_group_state_t *const state =
      &context->slot[tnc_slot_index(key)].group[group];
  state->token = *token;
  __atomic_store_n(&state->token_epoch, key->generation, __ATOMIC_RELEASE);
  return TNC_OK;
}

uint32_t tnc_task_token_done(const tnc_token_t *token,
                             const tnc_agent_context_t *agents) {
  if (token == NULL)
    return 0u;
  switch ((tnc_token_kind_t)token->kind) {
  case TNC_TOKEN_DMA:
    return dma_local_done_id(token->unit, token->target);
  case TNC_TOKEN_REDMULE:
    return tnc_redmule_done((uint8_t)token->target);
  case TNC_TOKEN_WORKERS:
    return agents != NULL && token->unit < NUM_CORES / NUM_CORES_PER_TILE &&
           tnc_worker_done(agents, token->unit, token->target);
  default:
    return 0u;
  }
}

uint32_t tnc_task_complete_local(tnc_task_context_t *context,
                                 const tnc_task_key_t *key,
                                 const tnc_agent_context_t *agents) {
  if (context == NULL || !tnc_scope_valid(key) || !tnc_is_group_coordinator())
    return TNC_ERROR_ARGUMENT;
  const uint32_t group = mempool_get_group_id();
  if (!tnc_group_in_scope(key, group))
    return TNC_ERROR_SCOPE;
  const uint32_t key_status =
      tnc_task_key_status(&context->slot[tnc_slot_index(key)].global, key);
  if (key_status != TNC_OK)
    return key_status;
  tnc_task_group_state_t *const state =
      &context->slot[tnc_slot_index(key)].group[group];
  while (__atomic_load_n(&state->token_epoch, __ATOMIC_ACQUIRE) !=
         key->generation) {
    tnc_poll_backoff();
  }
  while ((state->token.kind == TNC_TOKEN_REDMULE &&
          !tnc_redmule_group_done(agents, key)) ||
         (state->token.kind != TNC_TOKEN_REDMULE &&
          !tnc_task_token_done(&state->token, agents))) {
    tnc_poll_backoff();
  }
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  __atomic_store_n(&state->ready_epoch, key->generation, __ATOMIC_RELEASE);
  return TNC_OK;
}

uint32_t tnc_task_progress_ready(tnc_task_context_t *context,
                                 const tnc_task_key_t *key) {
  if (context == NULL || !tnc_scope_valid(key))
    return TNC_ERROR_ARGUMENT;
  if (mempool_get_group_id() != key->base_group || !tnc_is_group_coordinator())
    return TNC_ERROR_SCOPE;
  tnc_task_slot_t *const slot = &context->slot[tnc_slot_index(key)];
  const uint32_t key_status = tnc_task_key_status(&slot->global, key);
  if (key_status != TNC_OK)
    return key_status;
  for (uint32_t group = key->base_group;
       group < (uint32_t)key->base_group + key->num_groups; ++group)
    if (__atomic_load_n(&slot->group[group].ready_epoch, __ATOMIC_ACQUIRE) !=
        key->generation)
      return TNC_ERROR_NOT_READY;
  __atomic_store_n(&slot->global.ready_epoch, key->generation,
                   __ATOMIC_RELEASE);
  return TNC_OK;
}

uint32_t tnc_task_wait_ready(tnc_task_context_t *context,
                             const tnc_task_key_t *key) {
  if (context == NULL || !tnc_scope_valid(key))
    return TNC_ERROR_ARGUMENT;
  tnc_task_global_state_t *const global =
      &context->slot[tnc_slot_index(key)].global;
  const uint32_t key_status = tnc_task_key_status(global, key);
  if (key_status != TNC_OK)
    return key_status;
  while (__atomic_load_n(&global->ready_epoch, __ATOMIC_ACQUIRE) !=
         key->generation) {
    const uint32_t status = tnc_task_key_status(global, key);
    if (status != TNC_OK)
      return status;
    if (mempool_get_group_id() == key->base_group && tnc_is_group_coordinator())
      (void)tnc_task_progress_ready(context, key);
    tnc_poll_backoff();
  }
  const uint32_t final_status = tnc_task_key_status(global, key);
  return final_status == TNC_OK ? global->status : final_status;
}

uint32_t tnc_task_ready(const tnc_task_context_t *context,
                        const tnc_task_key_t *key) {
  if (context == NULL || !tnc_scope_valid(key))
    return 0u;
  const tnc_task_global_state_t *const global =
      &context->slot[tnc_slot_index(key)].global;
  if (tnc_task_key_status(global, key) != TNC_OK ||
      __atomic_load_n(&global->ready_epoch, __ATOMIC_ACQUIRE) !=
          key->generation)
    return 0u;
  return tnc_task_key_status(global, key) == TNC_OK;
}

uint32_t tnc_task_all_ready(const tnc_task_context_t *context,
                            const tnc_task_key_t *keys, uint32_t count) {
  if (context == NULL || keys == NULL || count == 0u)
    return 0u;
  for (uint32_t i = 0u; i < count; ++i)
    if (!tnc_task_ready(context, &keys[i]))
      return 0u;
  return 1u;
}

uint32_t tnc_task_wait_all(tnc_task_context_t *context,
                           const tnc_task_key_t *keys, uint32_t count) {
  if (context == NULL || keys == NULL || count == 0u)
    return TNC_ERROR_ARGUMENT;
  for (uint32_t i = 0u; i < count; ++i) {
    const uint32_t status = tnc_task_wait_ready(context, &keys[i]);
    if (status != TNC_OK)
      return status;
  }
  return TNC_OK;
}

uint32_t tnc_task_retire(tnc_task_context_t *context,
                         const tnc_task_key_t *key) {
  if (context == NULL || !tnc_scope_valid(key) ||
      mempool_get_group_id() != key->base_group || !tnc_is_group_coordinator())
    return TNC_ERROR_ARGUMENT;
  tnc_task_global_state_t *const global =
      &context->slot[tnc_slot_index(key)].global;
  if (__atomic_load_n(&global->active_epoch, __ATOMIC_ACQUIRE) !=
          key->generation ||
      !tnc_task_key_equal(&global->key, key))
    return TNC_ERROR_STALE;
  if (__atomic_load_n(&global->ready_epoch, __ATOMIC_ACQUIRE) !=
      key->generation)
    return TNC_ERROR_NOT_READY;
  __atomic_store_n(&global->free_epoch, key->generation, __ATOMIC_RELEASE);
  return TNC_OK;
}

void tnc_agent_context_init(tnc_agent_context_t *agents) {
  if (agents == NULL || !tnc_is_tile_agent())
    return;
  tnc_tile_agent_t *const agent = &agents->tile[mempool_get_tile_id()];
  agent->worker_issue_epoch = 0u;
  agent->worker_done_count = 0u;
  agent->worker_done_mask = 0u;
  agent->worker_target = 0u;
  agent->mailbox_epoch = 0u;
  agent->mailbox_arg = 0u;
  agent->redmule_target = 0u;
  agent->redmule_epoch = 0u;
  agent->redmule_done_epoch = 0u;
  agent->status = TNC_OK;
  for (uint32_t i = 0u; i < TNC_TILE_WORDS - TNC_AGENT_FIELDS; ++i)
    agent->padding[i] = 0u;
  __sync_synchronize();
}

uint32_t tnc_worker_dispatch(tnc_agent_context_t *agents, uint32_t tile,
                             uint32_t workers, uint32_t argument) {
  if (agents == NULL || tile >= NUM_CORES / NUM_CORES_PER_TILE ||
      tile != mempool_get_tile_id() || !tnc_is_tile_agent() || workers == 0u ||
      workers > (uint32_t)NUM_CORES_PER_TILE - 1u)
    return TNC_ERROR_ARGUMENT;
  tnc_tile_agent_t *const agent = &agents->tile[tile];
  const uint32_t epoch = agent->worker_issue_epoch + 1u;
  agent->worker_done_count = 0u;
  agent->worker_done_mask = 0u;
  agent->worker_target = workers;
  agent->mailbox_arg = argument;
  __atomic_store_n(&agent->mailbox_epoch, epoch, __ATOMIC_RELEASE);
  __atomic_store_n(&agent->worker_issue_epoch, epoch, __ATOMIC_RELEASE);
  return epoch;
}

uint32_t tnc_worker_wait_dispatch(const tnc_agent_context_t *agents,
                                  uint32_t tile, uint32_t after_epoch,
                                  uint32_t *argument) {
  if (agents == NULL || argument == NULL ||
      tile >= NUM_CORES / NUM_CORES_PER_TILE || tile != mempool_get_tile_id() ||
      tnc_is_tile_agent())
    return 0u;
  const tnc_tile_agent_t *const agent = &agents->tile[tile];
  uint32_t epoch;
  do {
    epoch = __atomic_load_n(&agent->mailbox_epoch, __ATOMIC_ACQUIRE);
    if (epoch == after_epoch)
      tnc_poll_backoff();
  } while (epoch == after_epoch);
  *argument = __atomic_load_n(&agent->mailbox_arg, __ATOMIC_RELAXED);
  return epoch;
}

void tnc_worker_complete(tnc_agent_context_t *agents, uint32_t tile,
                         uint32_t epoch) {
  if (agents == NULL || tile >= NUM_CORES / NUM_CORES_PER_TILE ||
      tile != mempool_get_tile_id() || tnc_is_tile_agent())
    return;
  tnc_tile_agent_t *const agent = &agents->tile[tile];
  while (__atomic_load_n(&agent->worker_issue_epoch, __ATOMIC_ACQUIRE) <
         epoch) {
    tnc_poll_backoff();
  }
  const uint32_t tile_core =
      mempool_get_core_id() % (uint32_t)NUM_CORES_PER_TILE;
  const uint32_t bit = 1u << tile_core;
  const uint32_t prior =
      __atomic_fetch_or(&agent->worker_done_mask, bit, __ATOMIC_ACQ_REL);
  if ((prior & bit) == 0u)
    __atomic_fetch_add(&agent->worker_done_count, 1u, __ATOMIC_RELEASE);
}

uint32_t tnc_worker_done(const tnc_agent_context_t *agents, uint32_t tile,
                         uint32_t epoch) {
  if (agents == NULL || tile >= NUM_CORES / NUM_CORES_PER_TILE)
    return 0u;
  const tnc_tile_agent_t *const agent = &agents->tile[tile];
  return __atomic_load_n(&agent->worker_issue_epoch, __ATOMIC_ACQUIRE) ==
             epoch &&
         __atomic_load_n(&agent->worker_done_count, __ATOMIC_ACQUIRE) >=
             agent->worker_target;
}

uint32_t tnc_redmule_agent_publish(tnc_agent_context_t *agents,
                                   const tnc_task_key_t *key, uint8_t target) {
  if (agents == NULL || !tnc_scope_valid(key) || !tnc_is_tile_agent() ||
      !mempool_tile_has_redmule())
    return TNC_ERROR_ARGUMENT;
  const uint32_t group = mempool_get_group_id();
  const uint32_t tile = mempool_get_tile_id();
  const uint32_t local_tile = tile % (uint32_t)NUM_TILES_PER_GROUP;
  if (!tnc_group_in_scope(key, group) ||
      (key->tile_mask & (uint16_t)(1u << local_tile)) == 0u)
    return TNC_ERROR_SCOPE;
  tnc_tile_agent_t *const agent = &agents->tile[tile];
  agent->redmule_target = target;
  agent->redmule_done_epoch = 0u;
  __atomic_store_n(&agent->redmule_epoch, key->generation, __ATOMIC_RELEASE);
  return TNC_OK;
}

uint32_t tnc_redmule_agent_progress(tnc_agent_context_t *agents,
                                    const tnc_task_key_t *key) {
  if (agents == NULL || !tnc_scope_valid(key) || !tnc_is_tile_agent() ||
      !mempool_tile_has_redmule())
    return TNC_ERROR_ARGUMENT;
  const uint32_t tile = mempool_get_tile_id();
  const uint32_t local_tile = tile % (uint32_t)NUM_TILES_PER_GROUP;
  if (!tnc_group_in_scope(key, mempool_get_group_id()) ||
      (key->tile_mask & (uint16_t)(1u << local_tile)) == 0u)
    return TNC_ERROR_SCOPE;
  tnc_tile_agent_t *const agent = &agents->tile[tile];
  if (__atomic_load_n(&agent->redmule_epoch, __ATOMIC_ACQUIRE) !=
      key->generation)
    return TNC_ERROR_NOT_READY;
  if (!tnc_redmule_done((uint8_t)agent->redmule_target))
    return TNC_ERROR_NOT_READY;
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  __atomic_store_n(&agent->redmule_done_epoch, key->generation,
                   __ATOMIC_RELEASE);
  return TNC_OK;
}

uint32_t tnc_redmule_group_done(const tnc_agent_context_t *agents,
                                const tnc_task_key_t *key) {
  if (agents == NULL || !tnc_scope_valid(key) ||
      !tnc_group_in_scope(key, mempool_get_group_id()))
    return 0u;
  const uint32_t first_tile =
      mempool_get_group_id() * (uint32_t)NUM_TILES_PER_GROUP;
  for (uint32_t local_tile = 0u; local_tile < NUM_TILES_PER_GROUP;
       ++local_tile) {
    if ((key->tile_mask & (uint16_t)(1u << local_tile)) == 0u)
      continue;
    if (__atomic_load_n(
            &agents->tile[first_tile + local_tile].redmule_done_epoch,
            __ATOMIC_ACQUIRE) != key->generation)
      return 0u;
  }
  return 1u;
}

void tnc_buffer_init(tnc_buffer_state_t *buffer) {
  if (buffer == NULL)
    return;
  const uint32_t core = mempool_get_core_id();
  const uint32_t group = mempool_get_group_id();
  if (core == 0u) {
    buffer->global.ready_epoch[0] = 0u;
    buffer->global.ready_epoch[1] = 0u;
    buffer->global.free_epoch[0] = 2u;
    buffer->global.free_epoch[1] = 1u;
    for (uint32_t i = 0u; i < NUM_BANKS - 2u * TNC_BUFFER_COUNT; ++i)
      buffer->global.padding[i] = 0u;
  }
  if (tnc_is_group_coordinator()) {
    for (uint32_t physical = 0; physical < TNC_BUFFER_COUNT; ++physical) {
      buffer->group[group].ready_epoch[physical] = 0u;
      buffer->group[group].consumer_epoch[physical] = 0u;
      buffer->group[group].consumer_count[physical] = 0u;
      buffer->group[group].release_epoch[physical] = 0u;
      buffer->group[group].consumer_target[physical] = 0u;
    }
    for (uint32_t i = 0u; i < TNC_GROUP_WORDS - TNC_BUFFER_GROUP_FIELDS; ++i)
      buffer->group[group].padding[i] = 0u;
  }
  __sync_synchronize();
}

uint32_t tnc_buffer_publish_ready(tnc_buffer_state_t *buffer,
                                  const tnc_task_key_t *key) {
  if (buffer == NULL || !tnc_scope_valid(key) || !tnc_is_group_coordinator())
    return TNC_ERROR_ARGUMENT;
  const uint32_t group = mempool_get_group_id();
  if (!tnc_group_in_scope(key, group))
    return TNC_ERROR_SCOPE;
  const uint32_t physical = key->generation & 1u;
  __atomic_store_n(&buffer->group[group].ready_epoch[physical], key->generation,
                   __ATOMIC_RELEASE);
  return TNC_OK;
}

uint32_t tnc_buffer_progress_ready(tnc_buffer_state_t *buffer,
                                   const tnc_task_key_t *key) {
  if (buffer == NULL || !tnc_scope_valid(key) ||
      mempool_get_group_id() != key->base_group || !tnc_is_group_coordinator())
    return TNC_ERROR_ARGUMENT;
  const uint32_t physical = key->generation & 1u;
  for (uint32_t group = key->base_group;
       group < (uint32_t)key->base_group + key->num_groups; ++group)
    if (!tnc_epoch16_done(
            __atomic_load_n(&buffer->group[group].ready_epoch[physical],
                            __ATOMIC_ACQUIRE),
            key->generation))
      return TNC_ERROR_NOT_READY;
  __atomic_store_n(&buffer->global.ready_epoch[physical], key->generation,
                   __ATOMIC_RELEASE);
  return TNC_OK;
}

void tnc_buffer_wait_ready(tnc_buffer_state_t *buffer,
                           const tnc_task_key_t *key) {
  const uint32_t physical = key->generation & 1u;
  while (!tnc_epoch16_done(
      __atomic_load_n(&buffer->global.ready_epoch[physical], __ATOMIC_ACQUIRE),
      key->generation)) {
    if (mempool_get_group_id() == key->base_group && tnc_is_group_coordinator())
      (void)tnc_buffer_progress_ready(buffer, key);
    tnc_poll_backoff();
  }
}

uint32_t tnc_buffer_begin_consumption(tnc_buffer_state_t *buffer,
                                      const tnc_task_key_t *key,
                                      uint32_t consumers_per_group) {
  if (buffer == NULL || !tnc_scope_valid(key) || consumers_per_group == 0u ||
      !tnc_is_group_coordinator())
    return TNC_ERROR_ARGUMENT;
  const uint32_t group = mempool_get_group_id();
  if (!tnc_group_in_scope(key, group))
    return TNC_ERROR_SCOPE;
  const uint32_t physical = key->generation & 1u;
  buffer->group[group].consumer_count[physical] = 0u;
  buffer->group[group].consumer_target[physical] = consumers_per_group;
  buffer->group[group].release_epoch[physical] = 0u;
  __atomic_store_n(&buffer->group[group].consumer_epoch[physical],
                   key->generation, __ATOMIC_RELEASE);
  return TNC_OK;
}

uint32_t tnc_buffer_consumer_release(tnc_buffer_state_t *buffer,
                                     const tnc_task_key_t *key) {
  if (buffer == NULL || !tnc_scope_valid(key))
    return TNC_ERROR_ARGUMENT;
  const uint32_t group = mempool_get_group_id();
  if (!tnc_group_in_scope(key, group))
    return TNC_ERROR_SCOPE;
  const uint32_t physical = key->generation & 1u;
  while (!tnc_epoch16_done(
      __atomic_load_n(&buffer->group[group].consumer_epoch[physical],
                      __ATOMIC_ACQUIRE),
      key->generation)) {
    tnc_poll_backoff();
  }
  const uint32_t target = buffer->group[group].consumer_target[physical];
  const uint32_t prior = __atomic_fetch_add(
      &buffer->group[group].consumer_count[physical], 1u, __ATOMIC_ACQ_REL);
  if (prior + 1u == target)
    __atomic_store_n(&buffer->group[group].release_epoch[physical],
                     key->generation, __ATOMIC_RELEASE);
  return prior + 1u > target ? TNC_ERROR_REUSE : TNC_OK;
}

uint32_t tnc_buffer_progress_free(tnc_buffer_state_t *buffer,
                                  const tnc_task_key_t *key) {
  if (buffer == NULL || !tnc_scope_valid(key) ||
      mempool_get_group_id() != key->base_group || !tnc_is_group_coordinator())
    return TNC_ERROR_ARGUMENT;
  const uint32_t physical = key->generation & 1u;
  for (uint32_t group = key->base_group;
       group < (uint32_t)key->base_group + key->num_groups; ++group)
    if (!tnc_epoch16_done(
            __atomic_load_n(&buffer->group[group].release_epoch[physical],
                            __ATOMIC_ACQUIRE),
            key->generation))
      return TNC_ERROR_NOT_READY;
  __atomic_store_n(&buffer->global.free_epoch[physical],
                   (uint16_t)(key->generation + 2u), __ATOMIC_RELEASE);
  return TNC_OK;
}

void tnc_buffer_wait_free(tnc_buffer_state_t *buffer,
                          const tnc_task_key_t *key) {
  const uint32_t physical = key->generation & 1u;
  while (!tnc_epoch16_done(
      __atomic_load_n(&buffer->global.free_epoch[physical], __ATOMIC_ACQUIRE),
      key->generation)) {
    if (key->generation >= 2u && mempool_get_group_id() == key->base_group &&
        tnc_is_group_coordinator()) {
      const tnc_task_key_t prior = {
          .task_id = key->task_id,
          .generation = (uint16_t)(key->generation - 2u),
          .base_group = key->base_group,
          .num_groups = key->num_groups,
          .tile_mask = key->tile_mask,
          .granularity = key->granularity,
          .reserved = key->reserved,
      };
      (void)tnc_buffer_progress_free(buffer, &prior);
    }
    tnc_poll_backoff();
  }
}
