// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "async_task.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#if NUM_GROUPS < 8
#error "async_task_smoke needs at least eight groups"
#endif

static tnc_task_context_t task_context TNC_TASK_STORAGE;
static tnc_agent_context_t agent_context TNC_TASK_STORAGE;
static tnc_buffer_state_t buffer_state TNC_TASK_STORAGE;
static uint32_t double_buffer_payload[TNC_BUFFER_COUNT][NUM_GROUPS]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")));
static uint32_t single_buffer_payload[NUM_GROUPS]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")));
static uint32_t total_errors __attribute__((section(".l1")));
static uint32_t fast_advance __attribute__((section(".l1")));
static uint32_t conjunction_before __attribute__((section(".l1")));
static uint32_t conjunction_after __attribute__((section(".l1")));
static uint32_t distinct_before __attribute__((section(".l1")));
static uint32_t alias_false_completion __attribute__((section(".l1")));
static uint32_t exclusion_rejected __attribute__((section(".l1")));
static uint32_t single_buffer_hazard __attribute__((section(".l1")));
static uint32_t double_buffer_preserved __attribute__((section(".l1")));

static uint32_t buffer_value(uint32_t generation, uint32_t group) {
  return 0xb0000000u ^ (generation << 16) ^ group;
}

static tnc_task_key_t task_key(uint16_t task_id, uint16_t generation,
                               uint16_t base_group, uint16_t num_groups) {
  const tnc_task_key_t key = {
      .task_id = task_id,
      .generation = generation,
      .base_group = base_group,
      .num_groups = num_groups,
      .tile_mask = 1,
      .granularity = 1,
      .reserved = 0,
  };
  return key;
}

static uint32_t group_in_key(const tnc_task_key_t *key, uint32_t group) {
  return group >= key->base_group &&
         group < (uint32_t)key->base_group + key->num_groups;
}

static void complete_worker_task(const tnc_task_key_t *key) {
  const uint32_t group = mempool_get_group_id();
  const uint32_t tile = group * NUM_TILES_PER_GROUP;
  const uint32_t tile_core = mempool_get_core_id() % NUM_CORES_PER_TILE;

  if (group_in_key(key, group) && tnc_is_group_coordinator()) {
    if (tnc_task_open(&task_context, key) != TNC_OK) {
      __atomic_fetch_add(&total_errors, 1u, __ATOMIC_RELAXED);
    } else {
      const uint32_t epoch = tnc_worker_dispatch(
          &agent_context, tile, NUM_CORES_PER_TILE - 1u, key->task_id);
      const tnc_token_t token = {
          .kind = TNC_TOKEN_WORKERS,
          .reserved = 0,
          .unit = (uint16_t)tile,
          .target = epoch,
      };
      if (epoch == TNC_ERROR_ARGUMENT ||
          tnc_task_publish_token(&task_context, key, &token) != TNC_OK)
        __atomic_fetch_add(&total_errors, 1u, __ATOMIC_RELAXED);
    }
  }
  mempool_barrier(NUM_CORES);

  if (group_in_key(key, group) && mempool_get_tile_id() == tile &&
      tile_core != 0u) {
    uint32_t argument;
    const uint32_t epoch =
        tnc_worker_wait_dispatch(&agent_context, tile, 0u, &argument);
    if (argument != key->task_id)
      __atomic_fetch_add(&total_errors, 1u, __ATOMIC_RELAXED);
    tnc_worker_complete(&agent_context, tile, epoch);
    tnc_worker_complete(&agent_context, tile, epoch);
  }
  mempool_barrier(NUM_CORES);

  if (group_in_key(key, group) && tnc_is_group_coordinator() &&
      tnc_task_complete_local(&task_context, key, &agent_context) != TNC_OK)
    __atomic_fetch_add(&total_errors, 1u, __ATOMIC_RELAXED);
  mempool_barrier(NUM_CORES);

  if (group == key->base_group && tnc_is_group_coordinator() &&
      tnc_task_progress_ready(&task_context, key) != TNC_OK)
    __atomic_fetch_add(&total_errors, 1u, __ATOMIC_RELAXED);
  mempool_barrier(NUM_CORES);
}

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t group = mempool_get_group_id();
  const uint32_t local_core = core % NUM_CORES_PER_GROUP;
  const uint32_t num_cores = mempool_get_core_count();
  const tnc_task_key_t a = task_key(0, 1, 0, 4);
  const tnc_task_key_t b = task_key(1, 1, 0, NUM_GROUPS);
  const tnc_task_key_t c = task_key(2, 1, 0, 1);
  const tnc_task_key_t d = task_key(3, 1, 4, 4);
  const tnc_task_key_t next = task_key(0, 2, 0, 4);
  const tnc_task_key_t fence[3] = {a, b, c};

  mempool_barrier_init(core);
  tnc_task_context_init(&task_context);
  tnc_agent_context_init(&agent_context);
  tnc_buffer_init(&buffer_state);
  for (uint32_t i = core; i < TNC_BUFFER_COUNT * NUM_GROUPS; i += num_cores)
    ((uint32_t *)double_buffer_payload)[i] = 0u;
  for (uint32_t i = core; i < NUM_GROUPS; i += num_cores)
    single_buffer_payload[i] = 0u;
  if (core == 0u) {
    total_errors = 0u;
    fast_advance = 0u;
    conjunction_before = 0u;
    conjunction_after = 0u;
    distinct_before = 0u;
    alias_false_completion = 0u;
    exclusion_rejected = 0u;
    single_buffer_hazard = 0u;
    double_buffer_preserved = 1u;
  }
  mempool_barrier(num_cores);

  if (group == 0u && tnc_is_group_coordinator()) {
    const uint32_t wrong_target = tnc_worker_dispatch(
        &agent_context, group * NUM_TILES_PER_GROUP, NUM_CORES_PER_TILE, 0u);
    exclusion_rejected = wrong_target == TNC_ERROR_ARGUMENT;
    if (!exclusion_rejected)
      __atomic_fetch_add(&total_errors, 1u, __ATOMIC_RELAXED);
  }
  mempool_barrier(num_cores);

  // A is one 2x2 VC; B is global. C remains absent for the first conjunction
  // test, and D is a disjoint 2x2 VC at A's level.
  complete_worker_task(&a);
  complete_worker_task(&b);
  if (core == 0u) {
    conjunction_before = tnc_task_all_ready(&task_context, fence, 3u);
    distinct_before = tnc_task_ready(&task_context, &d);
    const tnc_task_key_t aliased_d = a;
    alias_false_completion = tnc_task_ready(&task_context, &aliased_d);
    if (conjunction_before || distinct_before || !alias_false_completion)
      ++total_errors;
  }
  mempool_barrier(num_cores);

  complete_worker_task(&c);
  complete_worker_task(&d);

  if (core == 0u) {
    conjunction_after = tnc_task_all_ready(&task_context, fence, 3u);
    if (!conjunction_after ||
        tnc_task_wait_all(&task_context, fence, 3u) != TNC_OK ||
        !tnc_task_ready(&task_context, &d))
      ++total_errors;

    const tnc_task_key_t exhausted = task_key(4, 2, 0, 4);
    const uint32_t exhausted_status = tnc_task_open(&task_context, &exhausted);
    const uint32_t retire_status = tnc_task_retire(&task_context, &a);
    const uint32_t reuse_status = tnc_task_open(&task_context, &next);
    printf("[async_task_smoke] conjunction=%u,%u distinct=%u alias=%u "
           "equal=%u "
           "exhausted=0x%x retire=%u reuse=%u wrap=%u,%u,%u,%u,%u,%u\n",
           (unsigned)conjunction_before, (unsigned)conjunction_after,
           (unsigned)distinct_before, (unsigned)alias_false_completion,
           (unsigned)tnc_task_key_equal(&a, &d), (unsigned)exhausted_status,
           (unsigned)retire_status, (unsigned)reuse_status,
           (unsigned)tnc_counter8_done(0u, 1u),
           (unsigned)tnc_counter8_done(1u, 1u),
           (unsigned)tnc_counter8_done(121u, 250u),
           (unsigned)tnc_counter8_done(122u, 250u),
           (unsigned)tnc_counter16_done(1u, 65535u),
           (unsigned)tnc_counter16_done(32767u, 65535u));
    if (tnc_task_key_equal(&a, &d) || exhausted_status != TNC_ERROR_REUSE ||
        retire_status != TNC_OK || reuse_status != TNC_OK)
      ++total_errors;
    if (tnc_counter8_done(0u, 1u) || !tnc_counter8_done(1u, 1u) ||
        !tnc_counter8_done(121u, 250u) || tnc_counter8_done(122u, 250u) ||
        !tnc_counter16_done(1u, 65535u) || tnc_counter16_done(32767u, 65535u))
      ++total_errors;
  }
  mempool_barrier(num_cores);

  // READY(2) is independent of FREE(3) for the other physical buffer.
  if (group_in_key(&a, group) && tnc_is_group_coordinator()) {
    double_buffer_payload[a.generation & 1u][group] =
        buffer_value(a.generation, group);
    single_buffer_payload[group] = buffer_value(a.generation, group);
    (void)tnc_buffer_publish_ready(&buffer_state, &a);
    double_buffer_payload[next.generation & 1u][group] =
        buffer_value(next.generation, group);
    // Deliberate single-buffer falsification: N+1 overwrites N before N is
    // released, while the real two-buffer payload remains intact.
    single_buffer_payload[group] = buffer_value(next.generation, group);
    (void)tnc_buffer_publish_ready(&buffer_state, &next);
  }
  mempool_barrier(num_cores);
  if (group == a.base_group && tnc_is_group_coordinator()) {
    (void)tnc_buffer_progress_ready(&buffer_state, &a);
    (void)tnc_buffer_progress_ready(&buffer_state, &next);
  }
  mempool_barrier(num_cores);
  if (group_in_key(&a, group)) {
    tnc_buffer_wait_ready(&buffer_state, &a);
    tnc_buffer_wait_ready(&buffer_state, &next);
  }
  if (group_in_key(&a, group)) {
    if (double_buffer_payload[a.generation & 1u][group] !=
        buffer_value(a.generation, group)) {
      __atomic_store_n(&double_buffer_preserved, 0u, __ATOMIC_RELAXED);
      __atomic_fetch_add(&total_errors, 1u, __ATOMIC_RELAXED);
    }
    if (single_buffer_payload[group] != buffer_value(a.generation, group))
      __atomic_store_n(&single_buffer_hazard, 1u, __ATOMIC_RELAXED);
  }
  if (group_in_key(&a, group) && tnc_is_group_coordinator())
    (void)tnc_buffer_begin_consumption(
        &buffer_state, &a, NUM_TILES_PER_GROUP * (NUM_CORES_PER_TILE - 1u));
  mempool_barrier(num_cores);
  if (group_in_key(&a, group) && core % (uint32_t)NUM_CORES_PER_TILE != 0u) {
    if (local_core == NUM_CORES_PER_GROUP - 1u)
      mempool_wait(500u);
    if (local_core == 1u &&
        buffer_state.global.free_epoch[a.generation & 1u] < a.generation + 2u)
      __atomic_store_n(&fast_advance, 1u, __ATOMIC_RELEASE);
    if (tnc_buffer_consumer_release(&buffer_state, &a) != TNC_OK)
      __atomic_fetch_add(&total_errors, 1u, __ATOMIC_RELAXED);
  }
  mempool_barrier(num_cores);
  if (group == a.base_group && tnc_is_group_coordinator())
    if (tnc_buffer_progress_free(&buffer_state, &a) != TNC_OK)
      __atomic_fetch_add(&total_errors, 1u, __ATOMIC_RELAXED);
  mempool_barrier(num_cores);

  if (core == 0u) {
    if (!fast_advance || !single_buffer_hazard || !double_buffer_preserved ||
        buffer_state.global.free_epoch[1] != 3u)
      ++total_errors;
    printf("[async_task_smoke] slots=%u exclusion=%u fast=%u single=%u "
           "double=%u free=%u errors=%u\n",
           (unsigned)TNC_TASK_SLOTS, (unsigned)exclusion_rejected,
           (unsigned)fast_advance, (unsigned)single_buffer_hazard,
           (unsigned)double_buffer_preserved,
           (unsigned)buffer_state.global.free_epoch[1], (unsigned)total_errors);
  }
  mempool_barrier(num_cores);
  return (int)total_errors;
}
