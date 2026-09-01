// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "async_task.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "archi_redmule.h"
#include "hal_redmule.h"

#if NUM_REDMULE_TILES < 1
#error "rm_async_smoke requires RedMulE"
#endif

#define RM_SMOKE_JOBS 2u
#define RM_SMOKE_M 64u
#define RM_SMOKE_N 64u
#define RM_SMOKE_K 64u
#define RM_SMOKE_X_WORDS (RM_SMOKE_M * RM_SMOKE_N)
#define RM_SMOKE_W_WORDS (RM_SMOKE_N * RM_SMOKE_K)
#define RM_SMOKE_Z_WORDS (RM_SMOKE_M * RM_SMOKE_K)
#define FP16_ONE 0x3c00u
#define FP16_SIXTY_FOUR 0x5400u

static uint16_t x[RM_SMOKE_JOBS][RM_SMOKE_X_WORDS]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")));
static uint16_t w[RM_SMOKE_JOBS][RM_SMOKE_W_WORDS]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")));
static uint16_t z[RM_SMOKE_JOBS][RM_SMOKE_Z_WORDS]
    __attribute__((aligned(NUM_BANKS * sizeof(uint32_t)), section(".l1")));
static tnc_task_context_t task_context TNC_TASK_STORAGE;
static tnc_agent_context_t agent_context TNC_TASK_STORAGE;
static uint32_t targets[RM_SMOKE_JOBS] __attribute__((section(".l1")));
static uint32_t captured_done __attribute__((section(".l1")));
static uint32_t false_completion __attribute__((section(".l1")));
static uint32_t target_was_pending __attribute__((section(".l1")));
static uint32_t overlap_observed __attribute__((section(".l1")));
static uint32_t total_errors __attribute__((section(".l1")));

int main() {
  const uint32_t core = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  const tnc_task_key_t key = {
      .task_id = 0u,
      .generation = 1u,
      .base_group = 0u,
      .num_groups = 1u,
      .tile_mask = 1u,
      .granularity = TNC_GRANULARITY_TILE,
      .reserved = 0u,
  };

  mempool_barrier_init(core);
  tnc_task_context_init(&task_context);
  tnc_agent_context_init(&agent_context);
  for (uint32_t i = core; i < RM_SMOKE_JOBS * RM_SMOKE_X_WORDS; i += num_cores)
    ((uint16_t *)x)[i] = FP16_ONE;
  for (uint32_t i = core; i < RM_SMOKE_JOBS * RM_SMOKE_W_WORDS; i += num_cores)
    ((uint16_t *)w)[i] = FP16_ONE;
  for (uint32_t i = core; i < RM_SMOKE_JOBS * RM_SMOKE_Z_WORDS; i += num_cores)
    ((uint16_t *)z)[i] = 0u;
  if (core == 0u) {
    targets[0] = 0u;
    targets[1] = 0u;
    captured_done = 0u;
    false_completion = 0u;
    target_was_pending = 0u;
    overlap_observed = 0u;
    total_errors = 0u;
  }
  mempool_barrier(num_cores);

  if (core == 0u) {
    if (tnc_task_open(&task_context, &key) != TNC_OK)
      ++total_errors;
    captured_done = tnc_redmule_completed();
    false_completion =
        tnc_counter8_done((uint8_t)captured_done, (uint8_t)captured_done);

    for (uint32_t job = 0u; job < RM_SMOKE_JOBS; ++job) {
      const int32_t target = tnc_redmule_acquire_target();
      targets[job] = (uint32_t)target;
      if (job == 0u)
        target_was_pending = !tnc_redmule_done((uint8_t)target);
      else
        overlap_observed = !tnc_redmule_done((uint8_t)targets[0]);
      redmule_cfg((unsigned int)x[job], (unsigned int)w[job],
                  (unsigned int)z[job], RM_SMOKE_M, RM_SMOKE_N, RM_SMOKE_K, 0,
                  GEMM, Float16);
      mempool_wait(10u);
      tnc_redmule_trigger();
    }
    tnc_redmule_wait((uint8_t)targets[RM_SMOKE_JOBS - 1u]);
    const tnc_token_t token = {
        .kind = TNC_TOKEN_REDMULE,
        .reserved = 0u,
        .unit = 0u,
        .target = targets[RM_SMOKE_JOBS - 1u],
    };
    if (tnc_redmule_agent_publish(&agent_context, &key,
                                  (uint8_t)targets[RM_SMOKE_JOBS - 1u]) !=
            TNC_OK ||
        tnc_redmule_agent_progress(&agent_context, &key) != TNC_OK ||
        tnc_task_publish_token(&task_context, &key, &token) != TNC_OK ||
        tnc_task_complete_local(&task_context, &key, &agent_context) !=
            TNC_OK ||
        tnc_task_progress_ready(&task_context, &key) != TNC_OK ||
        tnc_task_wait_ready(&task_context, &key) != TNC_OK)
      ++total_errors;
    if (!false_completion || !target_was_pending || !overlap_observed ||
        (uint8_t)targets[1] != (uint8_t)(targets[0] + 1u) ||
        !tnc_redmule_done((uint8_t)targets[0]) ||
        !tnc_redmule_done((uint8_t)targets[1]))
      ++total_errors;
  }
  mempool_barrier(num_cores);

  uint32_t errors = 0u;
  for (uint32_t i = core; i < RM_SMOKE_JOBS * RM_SMOKE_Z_WORDS; i += num_cores)
    if (((uint16_t *)z)[i] != FP16_SIXTY_FOUR)
      ++errors;
  if (errors != 0u)
    __atomic_fetch_add(&total_errors, errors, __ATOMIC_RELAXED);
  mempool_barrier(num_cores);

  if (core == 0u)
    printf("[rm_async_smoke] captured=%u false=%u pending=%u overlap=%u "
           "targets=%u,%u completed=%u errors=%u\n",
           (unsigned)captured_done, (unsigned)false_completion,
           (unsigned)target_was_pending, (unsigned)overlap_observed,
           (unsigned)targets[0], (unsigned)targets[1],
           (unsigned)tnc_redmule_completed(), (unsigned)total_errors);
  mempool_barrier(num_cores);
  return (int)total_errors;
}
