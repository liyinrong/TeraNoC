// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Shared RedMulE projection machinery for the mixers: the dependency-counter
// full-K projection and the RMSNorm of row blocks. Included by
// layer_resident.c after its common definitions; not a translation unit.

#if Q38_WEIGHT_PANELS != 0 && Q38_GATEUP_PIPELINE && Q38_ASYNC
#if Q38_PROJ_DATAFLOW
// A plain full-K projection, out[rows x n] = in[rows x k] * W, synchronized
// by dependency counters like the dataflow gate/up: engines claim group g's
// rectangles once its weights landed, the last one to compute a group
// publishes group g+2 into the slot it frees, and a finished Z tile is
// copied out while the engine runs its next job.
#define Q38_DP_MAX (Q38_MAX_WIDTH / NT + 1u)
static uint32_t dp_first[Q38_DP_MAX] LOCAL;
static uint32_t dp_done[Q38_DP_MAX] LOCAL;
static uint32_t dp_ready[Q38_DP_MAX] LOCAL;
static uint32_t dp_next[Q38_DP_MAX] LOCAL;
static struct {
  const __fp16 *in;
  const __fp16 *weight;
  __fp16 *out;
  uint32_t k;
  uint32_t columns;
  uint32_t stride;
  uint32_t rows;
  uint32_t panels;
  uint32_t groups;
  uint32_t epoch;
} dp_state LOCAL;

static void dp_idle(void) {
  if (mempool_is_group_coordinator())
    (void)q38_dma_stage_progress(&gateup_dma_queue);
  mempool_wait(32u);
}

static uint32_t dp_count(uint32_t g) {
  return Q38_MIN(dp_state.panels, dp_state.columns - dp_first[g]);
}

// Projection weight ring: Q38_DP_SLOTS slots over both halves of the mixer
// weight area. Two keep the historical halves; a deeper ring of smaller slots
// keeps one loading while the others give every engine a job.
#ifndef Q38_DP_SLOTS
#define Q38_DP_SLOTS 2u
#endif
// Whole 256-byte beats per slot: the DMA splits a misaligned slot badly.
#define Q38_DP_SLOT_ELEMS                                                      \
  ((2u * mixer_buffers.slot_elems / Q38_DP_SLOTS) & ~127u)
_Static_assert(Q38_DP_SLOTS >= 2u && Q38_DP_SLOTS <= Q38_DMA_QUEUE_SLOTS,
               "The projection ring must fit the DMA queue");

static __fp16 *dp_slot(uint32_t g) {
#if Q38_DP_SLOTS == 2u
  return mixer_buffers.slot[g & 1u];
#else
  return mixer_buffers.slot[0] + g % Q38_DP_SLOTS * Q38_DP_SLOT_ELEMS;
#endif
}

static void dp_publish(uint32_t g) {
  if (q38_dma_stage_publish_any(&gateup_dma_queue, dp_state.epoch + g,
                                dp_slot(g),
                                dp_state.weight + dp_first[g] * dp_state.k * NT,
                                dp_count(g) * dp_state.k * NT * sizeof(__fp16),
                                0u) != Q38_DMA_QUEUE_OK)
    __builtin_trap();
}

static uint32_t dp_weights_ready(uint32_t g) {
  if (__atomic_load_n(&dp_ready[g], __ATOMIC_ACQUIRE))
    return 1u;
  const uint32_t status =
      q38_dma_stage_ready(&gateup_dma_queue, dp_state.epoch + g);
  if (status != Q38_DMA_QUEUE_OK && status != Q38_DMA_QUEUE_PENDING)
    __builtin_trap();
  if (status == Q38_DMA_QUEUE_OK)
    __atomic_store_n(&dp_ready[g], 1u, __ATOMIC_RELEASE);
  return status == Q38_DMA_QUEUE_OK;
}

static NOSTACK void dp_retire(uint32_t g, uint32_t jobs) {
  if (__atomic_add_fetch(&dp_done[g], 1u, __ATOMIC_ACQ_REL) != jobs)
    return;
  if (q38_dma_stage_release_any(&gateup_dma_queue, dp_state.epoch + g) !=
      Q38_DMA_QUEUE_OK)
    __builtin_trap();
  if (g + Q38_DP_SLOTS < dp_state.groups)
    dp_publish(g + Q38_DP_SLOTS);
}

static NOSTACK void dp_copy_out(const __fp16 *tile, uint32_t g, uint32_t task) {
  const uint32_t count = dp_count(g);
  const uint32_t row = task / count * Q38_PROJECTION_ROWS;
  __fp16 *const out =
      dp_state.out + row * dp_state.stride + (dp_first[g] + task % count) * NT;
  for (uint32_t r = 0; r < Q38_PROJECTION_ROWS; ++r)
    q38_copy16(out + r * dp_state.stride, tile + r * NT, NT);
}

static NOSTACK void dp_agent(void) {
  const uint32_t engine = mempool_get_redmule_id();
  __fp16 *const z = mixer_buffers.z + engine * 2 * Q38_PROJECTION_Z_SLOT;
  uint32_t zs = 0u, pending = 0u, pending_g = 0u, pending_task = 0u;
  uint32_t g = 0u;
  while (engine < NUM_REDMULE_TILES && g < dp_state.groups) {
    const uint32_t count = dp_count(g);
    const uint32_t jobs = dp_state.rows / Q38_PROJECTION_ROWS * count;
    const uint32_t task = __atomic_fetch_add(&dp_next[g], 1u, __ATOMIC_RELAXED);
    if (task >= jobs) {
      ++g;
      continue;
    }
    while (!dp_weights_ready(g)) {
      if (pending) {
        dp_copy_out(z + (zs ^ 1u) * Q38_PROJECTION_Z_SLOT, pending_g,
                    pending_task);
        pending = 0u;
      }
      dp_idle();
    }
    const uint8_t target = start_job(
        dp_state.in + task / count * Q38_PROJECTION_ROWS * dp_state.k,
        dp_slot(g) + task % count * dp_state.k * NT,
        z + zs * Q38_PROJECTION_Z_SLOT, Q38_PROJECTION_ROWS, dp_state.k, NT);
    if (pending)
      dp_copy_out(z + (zs ^ 1u) * Q38_PROJECTION_Z_SLOT, pending_g,
                  pending_task);
    while (!tnc_redmule_done(target))
      dp_idle();
    dp_retire(g, jobs);
    pending = 1u;
    pending_g = g;
    pending_task = task;
    zs ^= 1u;
  }
  if (pending)
    dp_copy_out(z + (zs ^ 1u) * Q38_PROJECTION_Z_SLOT, pending_g, pending_task);
  if (mempool_is_group_coordinator())
    while (q38_dma_stage_ready(&gateup_dma_queue,
                               dp_state.epoch + dp_state.groups - 1u) ==
           Q38_DMA_QUEUE_PENDING)
      dp_idle();
}

static NOSTACK void project_dataflow(const __fp16 *in, const __fp16 *weight,
                                     __fp16 *out, uint32_t k, uint32_t n,
                                     uint32_t rows, uint32_t panels) {
  const uint32_t core = mempool_get_core_id();
  const uint32_t columns = (n + NT - 1u) / NT;
#if Q38_DP_SLOTS > 2u
  // A ring slot holds fewer panels than a half: stage what fits.
  panels = Q38_MIN(panels, Q38_DP_SLOT_ELEMS / (k * NT));
#endif
  if (panels == 0u || panels * k * NT > Q38_DP_SLOT_ELEMS)
    __builtin_trap();
  if (core == 0u) {
    uint32_t g = 0u;
    for (uint32_t first = 0u; first < columns; ++g) {
      dp_first[g] = first;
      dp_done[g] = 0u;
      dp_ready[g] = 0u;
      dp_next[g] = 0u;
      first = next_panel_group(first, Q38_MIN(panels, columns - first), columns,
                               panels);
    }
    dp_state.in = in;
    dp_state.weight = weight;
    dp_state.out = out;
    dp_state.k = k;
    dp_state.columns = columns;
    dp_state.stride = columns * NT;
    dp_state.rows = rows;
    dp_state.panels = panels;
    dp_state.groups = g;
    dp_state.epoch = gateup_dma_next[0];
#if Q38_SCHEDULE_PROBE
    probe_projection_groups += g;
#endif
    for (uint32_t p = 0u; p < Q38_MIN(g, Q38_DP_SLOTS); ++p)
      dp_publish(p);
  }
  sync_all();
  if (tnc_is_tile_agent())
    dp_agent();
  sync_all();
  if (q38_dma_stage_ready(&gateup_dma_queue, dp_state.epoch + dp_state.groups -
                                                 1u) != Q38_DMA_QUEUE_OK)
    __builtin_trap();
  gateup_dma_next[core] = dp_state.epoch + dp_state.groups;
  sync_all();
}
#endif
#endif

static NOSTACK void norm_rows(const __fp16 *in, const __fp16 *scale,
                              __fp16 *out, uint32_t rows, uint32_t width,
                              int gated, __fp16 *gamma_buf) {
  transfer(gamma_buf, scale, width * 2);
  for (uint32_t row = q38_vec_id(); row < rows; row += Q38_VEC_CORES)
    q38_norm16(in + row * width, gamma_buf, out + row * width, width, gated);
  sync_all();
}
