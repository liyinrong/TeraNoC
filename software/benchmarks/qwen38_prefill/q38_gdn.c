// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// The gated-delta-net mixer (Q38_KIND != 0): chunked recurrence, its
// projections and the output projection. Included by layer_resident.c; not a
// translation unit.

static float recurrent[Q38_NV * Q38_KD * Q38_VD] LOCAL;
static __fp16 history[4 * Q38_QKVDIM] LOCAL;
static __fp16 conv_weight[4 * Q38_QKVDIM] LOCAL;
#if Q38_GDN_RECUR_PROJECTION_OVERLAP
_Static_assert(Q38_QKVDIM >= Q38_D,
               "GDN spare convolution slot must hold normalized input");
#if Q38_GDIM >= Q38_D
#define Q38_GDN_NEXT_RAW(slot) ((slot)->context)
#else
static __fp16 gdn_next_residual[Q38_GDN_SLOTS]
                               [Q38_GDN_STAGE_ROWS * Q38_D] LOCAL;
#define Q38_GDN_NEXT_RAW(slot) gdn_next_residual[(slot)-stage.mixer.slot]
#endif
#if Q38_GDN_TRIPLE_PIPELINE
static q38_dma_stage_queue_t gdn_input_dma_queue LOCAL;
static uint32_t gdn_dma_epoch[NUM_CORES] LOCAL;
static uint32_t gdn_raw_epoch[NUM_CORES] LOCAL;
#endif
#endif

#if Q38_GDN_READY_QUEUE
// Staged element (t, col) of the concatenated [QKV | Z | control] columns.
// Group g of global panels [pg0, pg0+count) sits at pg0*chunk*NT, laid out
// [row tile][panel][PR][NT] exactly as spilled, so one panel row of NT
// elements is contiguous.
static const __fp16 *q38_staged_at(const q38_gdn_slot_t *slot, uint32_t t,
                                   uint32_t col) {
  const uint32_t panel = col / NT;
  uint32_t base = 0u, panels = Q38_P_QKV;
  if (panel >= Q38_P_QKV + Q38_P_Z) {
    base = Q38_P_QKV + Q38_P_Z;
    panels = Q38_P_CTRL;
  } else if (panel >= Q38_P_QKV) {
    base = Q38_P_QKV;
    panels = Q38_P_Z;
  }
  const uint32_t first = base + (panel - base) / Q38_QP * Q38_QP;
  const uint32_t count = Q38_MIN(Q38_QP, base + panels - first);
  return slot->projected + first * Q38_GDN_STAGE_ROWS * NT +
         t / Q38_PR * (count * Q38_PR * NT) + (panel - first) * (Q38_PR * NT) +
         t % Q38_PR * NT + col % NT;
}
#endif

// Projected row `token` of the current or the immediately preceding window.
static const __fp16 *projected_row(const q38_gdn_slot_t *slot, uint32_t base,
                                   int32_t token, uint32_t channel) {
  const int32_t local = token - (int32_t)base;
  if (local >= 0)
#if Q38_GDN_READY_QUEUE
    return q38_staged_at(slot, (uint32_t)local, channel);
#else
    return slot->projected + (uint32_t)local * Q38_QKVDIM + channel;
#endif
  return stage.mixer.tail + (uint32_t)(local + 4) * Q38_QKVDIM + channel;
}

#if Q38_GDN_READY_QUEUE
static __fp16 gdn_spill[2][Q38_GROUPS * Q38_SPILL_GROUP]
    __attribute__((section(".l2_bss"), aligned(4 * NUM_BANKS)));
// Weight epoch s+1 fills mixer slot s%2 for group sequence s; spill epoch s+1
// drains landing buffer s%2. Sequences count up across supertiles.
static q38_dma_stage_queue_t gdn_weight_queue LOCAL;
static q38_dma_stage_queue_t gdn_spill_queue LOCAL;

typedef struct {
  volatile uint32_t active;
  uint32_t first;
  uint32_t end;
  uint32_t rows;
  uint32_t gen;
  uint32_t next;
  uint32_t done[4];
  volatile uint32_t completed;
} q38_gdn_queue_t;
static q38_gdn_queue_t gdn_queue LOCAL;

// Depth 2 queues a successor behind the running job in the HWPE job queue.
// Depth 1 starts the successor only once the engine is idle, into the other
// Z slot, before copying out the finished one -- the order project_panel_jobs
// uses. With engines free to start the next group while others still drain,
// GVSoC (timing-only and numerical alike) aborted at real geometry on L1
// bank-interconnect wide-retry assertions at the first group turnover, at
// either depth; see Q38_GDN_GROUP_SYNC.
typedef struct {
  uint32_t seq;
  uint32_t task;
  uint32_t count;
  uint32_t head;
  uint32_t job_seq[2];
  uint32_t job_task[2];
  uint32_t target[2];
} q38_engine_t;
static q38_engine_t gdn_engine[NUM_REDMULE_TILES] LOCAL;

#if Q38_SCHEDULE_PROBE
static uint32_t probe_queue_groups LOCAL;
static uint32_t probe_queue_jobs LOCAL;
static uint32_t probe_overlap_start LOCAL;
static uint32_t probe_overlap_end LOCAL;
static uint32_t probe_prologue_cycles LOCAL;
static uint32_t probe_recur_cycles LOCAL;
static uint32_t probe_drain_cycles LOCAL;
static uint32_t probe_output_cycles LOCAL;
#endif

static const __fp16 *q38_control_at(const q38_gdn_slot_t *slot, uint32_t t,
                                    uint32_t column) {
  return q38_staged_at(slot, t, Q38_QKVDIM + Q38_GDIM + column);
}

// Packed weights and global panel range of supertile group `g`.
static const __fp16 *q38_group(uint32_t g, uint32_t *first, uint32_t *count) {
  const __fp16 *weight = w_qkv;
  uint32_t base = 0u, panels = Q38_P_QKV;
  if (g >= Q38_G_QKV + Q38_G_Z) {
    g -= Q38_G_QKV + Q38_G_Z;
    weight = w_ab;
    base = Q38_P_QKV + Q38_P_Z;
    panels = Q38_P_CTRL;
  } else if (g >= Q38_G_QKV) {
    g -= Q38_G_QKV;
    weight = w_z;
    base = Q38_P_QKV;
    panels = Q38_P_Z;
  }
  *first = base + g * Q38_QP;
  *count = Q38_MIN(Q38_QP, panels - g * Q38_QP);
  return weight + g * Q38_QP * Q38_D * NT;
}

static void q38_queue_check(uint32_t status, const char *tag) {
  if (status != Q38_DMA_QUEUE_OK) {
    report_fatal(tag, status);
    __builtin_trap();
  }
}

// Ready: 1, pending: 0; a failed transfer traps.
static uint32_t q38_queue_ready(const q38_dma_stage_queue_t *queue,
                                uint32_t epoch) {
  const uint32_t status = q38_dma_stage_ready(queue, epoch);
  if (status == Q38_DMA_QUEUE_PENDING)
    return 0u;
  q38_queue_check(status, "[qwen38_queue_error] ready=");
  return 1u;
}

static void q38_publish_weights(uint32_t seq) {
  uint32_t first, count;
  const __fp16 *const weight = q38_group(seq - gdn_queue.first, &first, &count);
  q38_queue_check(q38_dma_stage_publish_any(
                      &gdn_weight_queue, seq + 1u, mixer_buffers.slot[seq & 1u],
                      weight, count * Q38_D * NT * sizeof(__fp16), 0u),
                  "[qwen38_queue_error] weights=");
}

// The last engine to retire a group frees its weight slot for the group two
// ahead and drains its landing buffer to the L2 ring.
static NOSTACK void q38_group_done(uint32_t seq, uint32_t jobs) {
  gdn_queue.done[seq % 4u] = 0u;
  q38_queue_check(q38_dma_stage_release_any(&gdn_weight_queue, seq + 1u),
                  "[qwen38_queue_error] release_weights=");
  if (seq + 2u - !Q38_GDN_WEIGHT_PREFETCH < gdn_queue.end)
    q38_publish_weights(seq + 2u - !Q38_GDN_WEIGHT_PREFETCH);
  if (seq >= 2u)
    q38_queue_check(q38_dma_stage_release_any(&gdn_spill_queue, seq - 1u),
                    "[qwen38_queue_error] release_spill=");
  q38_queue_check(
      q38_dma_stage_publish_any(&gdn_spill_queue, seq + 1u,
                                gdn_spill[gdn_queue.gen] +
                                    (seq - gdn_queue.first) * Q38_SPILL_GROUP,
                                stage.mixer.landing[seq & 1u],
                                jobs * Q38_PR * NT * sizeof(__fp16), 0u),
      "[qwen38_queue_error] spill=");
#if Q38_SCHEDULE_PROBE
  __atomic_fetch_add(&probe_queue_groups, 1u, __ATOMIC_RELAXED);
  __atomic_fetch_add(&probe_queue_jobs, jobs, __ATOMIC_RELAXED);
#endif
  __atomic_store_n(&gdn_queue.completed, seq + 1u, __ATOMIC_RELEASE);
}

static NOSTACK void q38_retire(q38_engine_t *engine, uint32_t id) {
  const uint32_t z = engine->head;
  const uint32_t seq = engine->job_seq[z], task = engine->job_task[z];
  uint32_t first, count;
  (void)q38_group(seq - gdn_queue.first, &first, &count);
  __fp16 *const landing = stage.mixer.landing[seq & 1u] +
                          task / count * (count * Q38_PR * NT) +
                          task % count * (Q38_PR * NT);
  const __fp16 *const tile =
      mixer_buffers.z + (id * 2u + z) * Q38_PROJECTION_Z_SLOT;
#if Q38_GDN_RETIRE_COPY == 1
  q38_copy16(landing, tile, Q38_PR * NT);
#elif Q38_GDN_RETIRE_COPY == 2
  for (uint32_t r = 0u; r < Q38_PR; ++r)
    q38_copy16(landing + r * NT, tile + r * NT, NT);
#else
  (void)landing;
  (void)tile;
#endif
  engine->head = Q38_GDN_Z_FIXED ? 0u : z ^ 1u;
  --engine->count;
  const uint32_t jobs = gdn_queue.rows / Q38_PR * count;
  // Publish the landing stores before counting this rectangle complete.
  __sync_synchronize();
  if (__atomic_add_fetch(&gdn_queue.done[seq % 4u], 1u, __ATOMIC_ACQ_REL) ==
      jobs)
    q38_group_done(seq, jobs);
}

static NOSTACK uint32_t q38_start(q38_engine_t *engine, uint32_t id) {
  while (engine->seq < gdn_queue.end) {
    uint32_t first, count;
    (void)q38_group(engine->seq - gdn_queue.first, &first, &count);
    if (engine->task < gdn_queue.rows / Q38_PR * count) {
      const uint32_t landed = Q38_GDN_SPILL_SERIAL ? 1u : 2u;
      if ((Q38_GDN_GROUP_SYNC &&
           __atomic_load_n(&gdn_queue.completed, __ATOMIC_ACQUIRE) <
               engine->seq) ||
          !q38_queue_ready(&gdn_weight_queue, engine->seq + 1u) ||
          (engine->seq >= landed &&
           !q38_queue_ready(&gdn_spill_queue, engine->seq + 1u - landed)))
        return 0u;
      const uint32_t z =
          Q38_GDN_Z_FIXED ? 0u : (engine->head + engine->count) & 1u;
      const uint32_t task = engine->task;
      engine->target[z] = start_job(
          stage.mixer.xin + task / count * Q38_PR * Q38_D,
          mixer_buffers.slot[engine->seq & 1u] + task % count * Q38_D * NT,
          mixer_buffers.z + (id * 2u + z) * Q38_PROJECTION_Z_SLOT, Q38_PR,
          Q38_D, NT);
      engine->job_seq[z] = engine->seq;
      engine->job_task[z] = task;
      ++engine->count;
      engine->task = task + NUM_REDMULE_TILES;
      return 1u;
    }
    ++engine->seq;
    engine->task = id;
  }
  return 0u;
}

// Tile agents retire and refill their engine; group coordinators also advance
// both DMA queues. Cheap when nothing is ready, and a no-op without a queue.
static NOSTACK void q38_service(void) {
  if (!__atomic_load_n(&gdn_queue.active, __ATOMIC_ACQUIRE))
    return;
  if (mempool_is_group_coordinator()) {
    (void)q38_dma_stage_progress(&gdn_weight_queue);
    (void)q38_dma_stage_progress(&gdn_spill_queue);
  }
  const uint32_t id = mempool_get_redmule_id();
  if (id >= NUM_REDMULE_TILES)
    return;
  q38_engine_t *const engine = &gdn_engine[id];
#if Q38_GDN_QUEUE_DEPTH == 1
  if (engine->count &&
      tnc_redmule_done((uint8_t)engine->target[engine->head])) {
    if (!Q38_GDN_Z_FIXED)
      (void)q38_start(engine, id);
    q38_retire(engine, id);
  }
  if (!engine->count)
    (void)q38_start(engine, id);
#else
  while (engine->count &&
         tnc_redmule_done((uint8_t)engine->target[engine->head]))
    q38_retire(engine, id);
  while (engine->count < 2u && q38_start(engine, id))
    ;
#endif
}

// Every core; `stage.mixer.xin` must already hold the supertile's input.
static NOSTACK void q38_queue_open(uint32_t rows, uint32_t gen) {
  if (mempool_get_core_id() == 0u) {
    gdn_queue.first = gdn_queue.next;
    gdn_queue.end = gdn_queue.first + Q38_GROUPS;
    gdn_queue.next = gdn_queue.end;
    gdn_queue.rows = rows;
    gdn_queue.gen = gen;
    q38_publish_weights(gdn_queue.first);
    if (Q38_GDN_WEIGHT_PREFETCH && Q38_GROUPS > 1u)
      q38_publish_weights(gdn_queue.first + 1u);
  }
  sync_all();
  const uint32_t id = mempool_get_redmule_id();
  if (id < NUM_REDMULE_TILES) {
    gdn_engine[id].seq = gdn_queue.first;
    gdn_engine[id].task = id;
    gdn_engine[id].count = 0u;
    gdn_engine[id].head = 0u;
  }
  sync_all();
  if (mempool_get_core_id() == 0u)
    __atomic_store_n(&gdn_queue.active, 1u, __ATOMIC_RELEASE);
  sync_all();
}

// Spill epochs complete in order in every group, so the final one being
// ready means every rectangle of the supertile is in L2. Only coordinators
// poll it, with backoff, to keep 128 spinning cores off those few progress
// words; other agents poll their own engine and the rest sleep.
static NOSTACK void q38_queue_drain(void) {
  const uint32_t id = mempool_get_redmule_id();
  const q38_engine_t *const engine =
      id < NUM_REDMULE_TILES ? &gdn_engine[id] : 0;
  if (mempool_is_group_coordinator()) {
    while (!q38_queue_ready(&gdn_spill_queue, gdn_queue.end)) {
      if (Q38_GDN_DRAIN_SLEEP && engine && engine->count)
        tnc_redmule_wait((uint8_t)engine->target[engine->head]);
      q38_service();
      mempool_wait(128u);
    }
  } else if (engine) {
    while (engine->count || engine->seq < gdn_queue.end) {
      if (Q38_GDN_DRAIN_SLEEP && engine->count)
        tnc_redmule_wait((uint8_t)engine->target[engine->head]);
      q38_service();
      mempool_wait(128u);
    }
  }
  sync_all();
  if (mempool_get_core_id() == 0u)
    __atomic_store_n(&gdn_queue.active, 0u, __ATOMIC_RELEASE);
  sync_all();
}

// Stage one recurrent chunk's projected rows back from its L2 generation.
static NOSTACK void q38_chunk_load(q38_gdn_slot_t *slot, uint32_t gen,
                                   uint32_t offset) {
  dma_collective_op_t op;
  for (uint32_t g = 0u; g < Q38_GROUPS; ++g) {
    uint32_t first, count;
    (void)q38_group(g, &first, &count);
    op = dma_collective_memcpy_nonblocking(
        slot->projected + first * Q38_GDN_STAGE_ROWS * NT,
        gdn_spill[gen] + g * Q38_SPILL_GROUP +
            offset / Q38_PR * (count * Q38_PR * NT),
        Q38_GDN_STAGE_ROWS * count * NT * sizeof(__fp16));
  }
  if (dma_collective_wait(&op) != 0u || op.status != 0u)
    __builtin_trap();
}
#else
static const __fp16 *q38_control_at(const q38_gdn_slot_t *slot, uint32_t t,
                                    uint32_t column) {
  return slot->controls + t * Q38_CTRL + column;
}
#define q38_service() ((void)0)
#endif

#if !Q38_GDN_READY_QUEUE
// Projection of one token window is independent of the preceding window's
// recurrent update once its input is ready.
static NOSTACK void gdn_project(uint32_t base, uint32_t rows, uint32_t last,
                                q38_gdn_slot_t *slot, uint32_t primed) {
  __fp16 *const projected = slot->projected;
  __fp16 *const zgate = slot->zgate;
  __fp16 *const controls = slot->controls;
#if Q38_GDN_RECUR_PROJECTION_OVERLAP
  if (primed) {
    // The spare slot held normalized input in convolution while RedMulE
    // projected the first QKV columns. Restore it before prepare reuses that
    // array for convolution; retain the raw rows for residual addition.
    copy_all16(residual, Q38_GDN_NEXT_RAW(slot), rows * Q38_D);
    copy_all16(a, slot->convolution, rows * Q38_D);
  } else {
#else
  (void)primed;
#endif
    transfer(residual, input_init + base * Q38_D, rows * Q38_D * 2);
    norm_rows(residual, w_input_norm, a, rows, Q38_D, 0, gamma);
#if Q38_GDN_RECUR_PROJECTION_OVERLAP
  }
  if (primed) {
    const uint32_t columns = Q38_QKVDIM / NT;
    const uint32_t capacity = mixer_buffers.slot_elems / (Q38_D * NT);
    uint32_t panels = Q38_ADAPTIVE_PANELS ? capacity : Q38_WEIGHT_PANELS;
    const uint32_t engine_panels =
        NUM_REDMULE_TILES / (rows / Q38_PROJECTION_ROWS);
    if (panels > columns)
      panels = columns;
    if (panels > engine_panels)
      panels = engine_panels;
    project_rows_from_panel(a, w_qkv, projected, Q38_D, Q38_QKVDIM, rows,
                            panels, panels, &mixer_buffers);
  } else {
#endif
    project_rows(a, w_qkv, projected, Q38_D, Q38_QKVDIM, w_z, Q38_D, Q38_GDIM,
                 rows, &mixer_buffers);
#if Q38_GDN_RECUR_PROJECTION_OVERLAP
  }
#endif
  project_rows(a, w_z, zgate, Q38_D, Q38_GDIM, w_ab, Q38_D, 2 * Q38_NV, rows,
               &mixer_buffers);
  project_rows(a, w_ab, controls, Q38_D, 2 * Q38_NV,
               last ? w_out : (const __fp16 *)0, Q38_GDIM, Q38_D, rows,
               &mixer_buffers);
}

#if Q38_GDN_RECUR_PROJECTION_OVERLAP
// One full-K QKV rectangle per available engine runs during the preceding
// chunk's ordered recurrence. The next slot's context/zgate storage is dead
// until its preparation, avoiding another sequence- or chunk-sized L1 array.
static NOSTACK uint32_t gdn_prime_qkv(uint32_t base, uint32_t rows,
                                      q38_gdn_slot_t *slot,
                                      uint32_t raw_ready) {
  if (!raw_ready)
    transfer(Q38_GDN_NEXT_RAW(slot), input_init + base * Q38_D,
             rows * Q38_D * 2);
  norm_rows(Q38_GDN_NEXT_RAW(slot), w_input_norm, slot->convolution, rows,
            Q38_D, 0, gamma);
  const uint32_t columns = Q38_QKVDIM / NT;
  const uint32_t capacity = mixer_buffers.slot_elems / (Q38_D * NT);
  uint32_t panels = Q38_ADAPTIVE_PANELS ? capacity : Q38_WEIGHT_PANELS;
  const uint32_t engine_panels =
      NUM_REDMULE_TILES / (rows / Q38_PROJECTION_ROWS);
  if (panels > columns)
    panels = columns;
  if (panels > engine_panels)
    panels = engine_panels;
  if (panels == 0u)
    __builtin_trap();
  transfer(mixer_buffers.slot[0], w_qkv, panels * Q38_D * NT * 2);
  uint32_t target = 256u;
  if (tnc_is_tile_agent()) {
    const uint32_t engine = mempool_get_redmule_id();
    const uint32_t tasks = (rows / Q38_PROJECTION_ROWS) * panels;
    if (engine < tasks) {
      __fp16 *z = mixer_buffers.z + engine * 2 * Q38_PROJECTION_Z_SLOT;
      target = start_job(slot->convolution +
                             (engine / panels * Q38_PROJECTION_ROWS) * Q38_D,
                         mixer_buffers.slot[0] + (engine % panels) * Q38_D * NT,
                         z, Q38_PROJECTION_ROWS, Q38_D, NT);
    }
  }
  sync_all();
  return target;
}

static NOSTACK void gdn_retire_qkv(uint32_t rows, q38_gdn_slot_t *slot,
                                   uint32_t target) {
  if (target < 256u) {
    const uint32_t engine = mempool_get_redmule_id();
    const uint32_t columns = Q38_QKVDIM / NT;
    const uint32_t capacity = mixer_buffers.slot_elems / (Q38_D * NT);
    uint32_t panels = Q38_ADAPTIVE_PANELS ? capacity : Q38_WEIGHT_PANELS;
    const uint32_t engine_panels =
        NUM_REDMULE_TILES / (rows / Q38_PROJECTION_ROWS);
    if (panels > columns)
      panels = columns;
    if (panels > engine_panels)
      panels = engine_panels;
    tnc_redmule_wait((uint8_t)target);
    const uint32_t row = engine / panels * Q38_PROJECTION_ROWS;
    const uint32_t column = engine % panels;
    const __fp16 *z = mixer_buffers.z + engine * 2 * Q38_PROJECTION_Z_SLOT;
    for (uint32_t r = 0; r < Q38_PROJECTION_ROWS; ++r)
      q38_copy16(slot->projected + (row + r) * Q38_QKVDIM + column * NT,
                 z + r * NT, NT);
  }
  sync_all();
}

#if Q38_GDN_TRIPLE_PIPELINE
static NOSTACK void gdn_dma_publish(uint32_t epoch, uint32_t base,
                                    uint32_t rows, q38_gdn_slot_t *slot) {
  if (mempool_get_core_id() == 0u &&
      q38_dma_stage_publish(&gdn_input_dma_queue, epoch, Q38_GDN_NEXT_RAW(slot),
                            input_init + base * Q38_D, rows * Q38_D * 2,
                            0u) != Q38_DMA_QUEUE_OK)
    __builtin_trap();
  if (mempool_is_group_coordinator()) {
    q38_dma_group_progress_t *const state =
        &gdn_input_dma_queue.slot[epoch % Q38_DMA_QUEUE_SLOTS]
             .group[mempool_get_group_id()];
    while (__atomic_load_n(&state->issued_epoch, __ATOMIC_ACQUIRE) != epoch)
      (void)q38_dma_stage_progress(&gdn_input_dma_queue);
  }
}

static NOSTACK void gdn_dma_wait_release(uint32_t epoch) {
  uint32_t status;
  do {
    if (mempool_is_group_coordinator())
      (void)q38_dma_stage_progress(&gdn_input_dma_queue);
    else
      mempool_wait(4u);
    status = q38_dma_stage_ready(&gdn_input_dma_queue, epoch);
  } while (status == Q38_DMA_QUEUE_PENDING);
  if (status != Q38_DMA_QUEUE_OK)
    __builtin_trap();
  if (mempool_get_core_id() == 0u &&
      q38_dma_stage_release(&gdn_input_dma_queue, epoch) != Q38_DMA_QUEUE_OK)
    __builtin_trap();
  sync_all();
}

// Keep lookahead bookkeeping out of the frame that calls gdn_finish and its
// deep projection/DMA chain on a 512-byte per-core stack.
static NOSTACK uint32_t gdn_start_next(uint32_t next) {
  if (next >= Q38_MIXER_EXEC_END)
    return 256u;
  const uint32_t core = mempool_get_core_id();
  q38_gdn_slot_t *const next_slot =
      &stage.mixer.slot[(next / Q38_GDN_STAGE_ROWS) % Q38_GDN_SLOTS];
  const uint32_t ready = gdn_raw_epoch[core] != 0u;
  if (ready)
    gdn_dma_wait_release(gdn_raw_epoch[core]);
  gdn_raw_epoch[core] = 0u;
  const uint32_t rows = block_rows(next, Q38_GDN_STAGE_ROWS);
  const uint32_t target = gdn_prime_qkv(next, rows, next_slot, ready);
  const uint32_t after_next = next + rows;
  if (after_next < Q38_MIXER_EXEC_END) {
    q38_gdn_slot_t *const future_slot =
        &stage.mixer.slot[(after_next / Q38_GDN_STAGE_ROWS) % Q38_GDN_SLOTS];
    const uint32_t epoch = ++gdn_dma_epoch[core];
    gdn_raw_epoch[core] = epoch;
    gdn_dma_publish(epoch, after_next,
                    block_rows(after_next, Q38_GDN_STAGE_ROWS), future_slot);
  }
  return target;
}
#endif
#endif

#endif

static NOSTACK void gdn_prepare(uint32_t base, uint32_t rows,
                                q38_gdn_slot_t *slot) {
  const uint32_t core = mempool_get_core_id();
  __fp16 *const convolution = slot->convolution;
  __fp16 *const context = slot->context;
  __fp16 *const tail = stage.mixer.tail;
  float *const q32 = slot->q32;
  float *const k32 = slot->k32;
  float *const decay = slot->decay;
  float *const beta = slot->beta;
  const uint32_t left = Q38_S > base ? Q38_S - base : 0u;
  const uint32_t tokens = left < rows ? left : rows;
  const float zh = llm_f16box(0.0f);
  q38_service();
  clear_all(convolution, rows * Q38_QKVDIM * 2);
  clear_all(context, Q38_NV * rows * Q38_VD * 2);
  for (uint32_t channel = q38_vec_id() * 32;
       q38_vec_id() != ~0u && channel < Q38_QKVDIM;
       channel += Q38_VEC_CORES * 32) {
    q38_service();
    for (uint32_t t = 0; t < tokens; ++t) {
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(32));
      asm volatile("vmv.v.i v8, 0");
      for (uint32_t tap = 0; tap < 4; ++tap) {
        const int32_t source = (int32_t)(base + t) + (int32_t)tap - 3;
        if (source < 0)
          continue;
        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(32));
        asm volatile("vle16.v v0, (%0)"
                     :
                     : "r"(projected_row(slot, base, source, channel))
                     : "memory");
        asm volatile("vfwadd.vf v16, v0, %0" : : "f"(zh));
        asm volatile("vle16.v v0, (%0)"
                     :
                     : "r"(conv_weight + tap * Q38_QKVDIM + channel)
                     : "memory");
        asm volatile("vfwadd.vf v24, v0, %0" : : "f"(zh));
        asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(32));
        asm volatile("vfmacc.vv v8, v16, v24");
      }
      __fp16 *out = convolution + t * Q38_QKVDIM + channel;
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(32));
      asm volatile("vfncvt.f.f.w v0, v8");
      asm volatile("vse16.v v0, (%0)" : : "r"(out) : "memory");
      q38_activation16(out, 0, out, 32, 1);
    }
    // The returned convolution history is the last four real token rows, so
    // it is refreshed while this window's projections are still readable.
    if (tokens)
      for (uint32_t tap = 0; tap < 4; ++tap) {
        const int32_t source = (int32_t)(base + tokens) + (int32_t)tap - 4;
        if (source >= 0)
          q38_copy16(history + tap * Q38_QKVDIM + channel,
                     projected_row(slot, base, source, channel), 32);
      }
    for (uint32_t tap = 0; tap < 4; ++tap)
      q38_copy16(
          tail + tap * Q38_QKVDIM + channel,
          projected_row(slot, base, (int32_t)(base + rows + tap - 4), channel),
          32);
  }
  q38_service();
  sync_all();
  for (uint32_t i = q38_vec_id(); i < tokens * Q38_NK; i += Q38_VEC_CORES) {
    const uint32_t t = i / Q38_NK, h = i % Q38_NK;
    for (uint32_t which = 0; which < 2; ++which) {
      const __fp16 *src =
          convolution + t * Q38_QKVDIM + (which * Q38_NK + h) * Q38_KD;
      float *dst = (which ? k32 : q32) + i * Q38_KD;
      float sum = 0.0f;
      for (uint32_t k = 0; k < Q38_KD; ++k) {
        const float value = q38_read16(src + k);
        sum += value * value;
      }
      float scale = llm_rsqrt(sum + 1e-6f);
      if (!which) {
        // -ffast-math lets the compiler re-associate this product, and which
        // grouping it picks moves with unrelated code, changing the last bit
        // of every query and so the layer output. Spell the multiply out.
        const float head = llm_rsqrt((float)Q38_KD);
        asm volatile("fmul.s %0, %1, %2" : "=f"(scale) : "f"(scale), "f"(head));
      }
      for (uint32_t k = 0; k < Q38_KD; ++k)
        dst[k] = q38_read16(src + k) * scale;
    }
  }
  q38_service();
  for (uint32_t i = q38_vec_id(); i < tokens * Q38_NV; i += Q38_VEC_CORES) {
    const uint32_t t = i / Q38_NV, h = i % Q38_NV;
    const float aa = q38_read16(q38_control_at(slot, t, h));
    const float bb = q38_read16(q38_control_at(slot, t, Q38_NV + h));
    decay[i] =
        llm_expf(-llm_expf(w_a_log[h]) * q38_softplus(aa + w_dt_bias[h]));
    beta[i] = llm_recip_pos(1.0f + llm_expf(-bb));
  }
  q38_service();
  sync_all();
}

// Each value-head/state-column owner advances tokens in order, carrying its
// FP32 state into the next window. Different owners remain independent.
static NOSTACK void gdn_recur(uint32_t base, uint32_t rows,
                              q38_gdn_slot_t *slot) {
  const uint32_t core = mempool_get_core_id();
  const uint32_t left = Q38_S > base ? Q38_S - base : 0u;
  const uint32_t tokens = left < rows ? left : rows;
  const float zh = llm_f16box(0.0f);
  const __fp16 *const convolution = slot->convolution;
  __fp16 *const context = slot->context;
  const float *const q32 = slot->q32;
  const float *const k32 = slot->k32;
  const float *const decay = slot->decay;
  const float *const beta = slot->beta;
  for (uint32_t first = 0u; first < tokens; first += Q38_GDN_CHUNK_ROWS) {
    uint32_t end = first + Q38_GDN_CHUNK_ROWS;
    if (end > tokens)
      end = tokens;
    for (uint32_t task = q38_vec_id(); task < Q38_NV * (Q38_VD / 16);
         task += Q38_VEC_CORES) {
      const uint32_t h = task / (Q38_VD / 16), col = task % (Q38_VD / 16) * 16;
      float *state = recurrent + h * Q38_KD * Q38_VD + col;
      for (uint32_t t = first; t < end; ++t) {
        q38_service();
        const float *query = q32 + (t * Q38_NK + h / 3) * Q38_KD;
        const float *key = k32 + (t * Q38_NK + h / 3) * Q38_KD;
        const float lambda = decay[t * Q38_NV + h],
                    strength = beta[t * Q38_NV + h];
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(16));
        asm volatile("vmv.v.i v8, 0");
        for (uint32_t k = 0; k < Q38_RECUR_K; ++k) {
          asm volatile("vle32.v v0, (%0)"
                       :
                       : "r"(state + k * Q38_VD)
                       : "memory");
          asm volatile("vfmul.vf v0, v0, %0" : : "f"(lambda));
          asm volatile("vse32.v v0, (%0)"
                       :
                       : "r"(state + k * Q38_VD)
                       : "memory");
          asm volatile("vfmacc.vf v8, %0, v0" : : "f"(key[k]));
        }
        const __fp16 *value = convolution + t * Q38_QKVDIM +
                              2 * Q38_NK * Q38_KD + h * Q38_VD + col;
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(16));
        asm volatile("vle16.v v16, (%0)" : : "r"(value) : "memory");
        asm volatile("vfwadd.vf v0, v16, %0" : : "f"(zh));
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(16));
        asm volatile("vfsub.vv v8, v0, v8");
        asm volatile("vfmul.vf v8, v8, %0" : : "f"(strength));
        asm volatile("vmv.v.i v16, 0");
        for (uint32_t k = 0; k < Q38_RECUR_K; ++k) {
          asm volatile("vle32.v v0, (%0)"
                       :
                       : "r"(state + k * Q38_VD)
                       : "memory");
          asm volatile("vfmacc.vf v0, %0, v8" : : "f"(key[k]));
          asm volatile("vse32.v v0, (%0)"
                       :
                       : "r"(state + k * Q38_VD)
                       : "memory");
          asm volatile("vfmacc.vf v16, %0, v0" : : "f"(query[k]));
        }
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(16));
        asm volatile("vfncvt.f.f.w v0, v16");
        asm volatile("vse16.v v0, (%0)"
                     :
                     : "r"(context + (h * rows + t) * Q38_VD + col)
                     : "memory");
      }
    }
    q38_service();
    sync_all();
  }
}

// Head-local gated normalization of one chunk's raw context into `dst`.
static NOSTACK void gdn_gate(uint32_t base, uint32_t rows,
                             const q38_gdn_slot_t *slot, __fp16 *dst) {
  const uint32_t core = mempool_get_core_id();
  const uint32_t left = Q38_S > base ? Q38_S - base : 0u;
  const uint32_t tokens = left < rows ? left : rows;
  const __fp16 *const context = slot->context;
  transfer(gamma, w_gated_norm, Q38_VD * 2);
  clear_all(dst, rows * Q38_GDIM * 2);
  for (uint32_t i = q38_vec_id(); i < tokens * Q38_NV; i += Q38_VEC_CORES) {
    const uint32_t t = i / Q38_NV, h = i % Q38_NV;
    __fp16 *out = dst + t * Q38_GDIM + h * Q38_VD;
    q38_norm16(context + (h * rows + t) * Q38_VD, gamma, out, Q38_VD, 1);
#if Q38_GDN_READY_QUEUE
    // A head's Z columns may straddle staged panels; gate them piecewise.
    for (uint32_t c = 0u; c < Q38_VD;) {
      const uint32_t column = h * Q38_VD + c;
      const uint32_t n = Q38_MIN(Q38_VD - c, NT - column % NT);
      q38_activation16(q38_staged_at(slot, t, Q38_QKVDIM + column), out + c,
                       out + c, n, 2);
      c += n;
    }
    q38_service();
#else
    q38_activation16(slot->zgate + i * Q38_VD, out, out, Q38_VD, 2);
#endif
  }
  q38_service();
  sync_all();
#if Q38_SAMPLED_CHECKS
  gather_context_window(context, base, rows, Q38_VD);
#else
  for (uint32_t h = 0; h < Q38_NV; ++h)
    export_tensor(result_context + (h * Q38_R + base) * Q38_VD,
                  context + h * rows * Q38_VD, rows * Q38_VD * 2);
#endif
}

#if !Q38_GDN_READY_QUEUE
static NOSTACK void gdn_finish(uint32_t base, uint32_t rows, uint32_t last,
                               q38_gdn_slot_t *slot) {
  const uint32_t core = mempool_get_core_id();
  gdn_gate(base, rows, slot, a);
  project_rows(a, w_out, mixed, Q38_GDIM, Q38_D,
               last ? w_gate : (const __fp16 *)0, Q38_D, Q38_I, rows,
               &mixer_buffers);
  if (q38_vec_id() != ~0u)
    llm_residual_add_f16(mixed, residual, rows * Q38_D, q38_vec_id(),
                         Q38_VEC_CORES);
  sync_all();
  transfer(result_mixer + base * Q38_D, mixed, rows * Q38_D * 2);
#if Q38_SCHEDULE_PROBE
  if (core == 0u) {
    ++probe_gdn_chunks;
    if ((base + rows) % Q38_WIN == 0u)
      ++probe_mixer_windows;
  }
#endif
}

// Projection, convolution and ordered recurrence are one phase. Its lookahead
// locals must retire before the outer loop calls the deep output projection.
static NOSTACK void gdn_before_finish(uint32_t base, uint32_t rows,
                                      uint32_t last, q38_gdn_slot_t *slot,
                                      uint32_t primed) {
  gdn_project(base, rows, last, slot, primed);
  gdn_prepare(base, rows, slot);
#if Q38_GDN_RECUR_PROJECTION_OVERLAP
  const uint32_t next = base + rows;
  uint32_t target = 256u;
#if Q38_GDN_TRIPLE_PIPELINE
  target = gdn_start_next(next);
#else
  if (next < Q38_MIXER_EXEC_END) {
    q38_gdn_slot_t *const next_slot =
        &stage.mixer.slot[(next / Q38_GDN_STAGE_ROWS) % Q38_GDN_SLOTS];
    target = gdn_prime_qkv(next, block_rows(next, Q38_GDN_STAGE_ROWS),
                           next_slot, 0u);
  }
#endif
#endif
  gdn_recur(base, rows, slot);
#if Q38_GDN_RECUR_PROJECTION_OVERLAP
  if (next < Q38_MIXER_EXEC_END) {
    q38_gdn_slot_t *const next_slot =
        &stage.mixer.slot[(next / Q38_GDN_STAGE_ROWS) % Q38_GDN_SLOTS];
    gdn_retire_qkv(block_rows(next, Q38_GDN_STAGE_ROWS), next_slot, target);
  }
#endif
}

static NOSTACK void gdn(void) {
  transfer(conv_weight, w_conv, sizeof(conv_weight));
#if !(Q38_SCHEDULE_PROBE && Q38_PROBE_BASE > 0)
  clear_all(history, sizeof(history));
  clear_all(recurrent, sizeof(recurrent));
  clear_all(stage.mixer.tail, sizeof(stage.mixer.tail));
#endif
#if Q38_GDN_RECUR_PROJECTION_OVERLAP
  uint32_t primed = 0u;
#endif
  for (uint32_t base = Q38_MIXER_EXEC_START; base < Q38_MIXER_EXEC_END;
       base += Q38_WIN) {
    const uint32_t rows = block_rows(base, Q38_WIN);
    for (uint32_t offset = 0u; offset < rows; offset += Q38_GDN_STAGE_ROWS) {
      const uint32_t chunk_base = base + offset;
#if Q38_GDN_DOUBLE_CHUNK
      const uint32_t chunk_rows = Q38_GDN_STAGE_ROWS;
#else
      const uint32_t chunk_rows = block_rows(chunk_base, Q38_GDN_STAGE_ROWS);
#endif
      const uint32_t final = chunk_base + chunk_rows >= Q38_MIXER_EXEC_END;
      q38_gdn_slot_t *const slot =
          &stage.mixer.slot[(chunk_base / Q38_GDN_STAGE_ROWS) % Q38_GDN_SLOTS];
#if Q38_GDN_RECUR_PROJECTION_OVERLAP
      gdn_before_finish(chunk_base, chunk_rows, final, slot, primed);
#else
      gdn_before_finish(chunk_base, chunk_rows, final, slot, 0u);
#endif
      gdn_finish(chunk_base, chunk_rows, final, slot);
#if Q38_GDN_RECUR_PROJECTION_OVERLAP
      primed = chunk_base + chunk_rows < Q38_MIXER_EXEC_END;
#endif
    }
#if Q38_FFN_NORM_STREAM
    ffn_norm_window(base, rows);
#endif
  }
}
#else
#if Q38_SCHEDULE_PROBE
// Core-0 phase timers; every phase is entered and left by all cores together.
#define Q38_PHASE_BEGIN(t)                                                     \
  const uint32_t t = mempool_get_core_id() == 0u ? mempool_get_timer() : 0u
#define Q38_PHASE_END(t, sum)                                                  \
  do {                                                                         \
    if (mempool_get_core_id() == 0u)                                           \
      sum += mempool_get_timer() - t;                                          \
  } while (0)
#else
#define Q38_PHASE_BEGIN(t) (void)0
#define Q38_PHASE_END(t, sum) (void)0
#endif

static NOSTACK void q38_stage_input(uint32_t base, uint32_t rows) {
  transfer(stage.mixer.out, input_init + base * Q38_D, rows * Q38_D * 2);
  norm_rows(stage.mixer.out, w_input_norm, stage.mixer.xin, rows, Q38_D, 0,
            gamma);
}

// Output projection of a whole supertile: (rows/PR)*QP full-K rectangles.
static NOSTACK void q38_output(uint32_t base, uint32_t rows) {
  transfer(stage.mixer.xin, input_init + base * Q38_D, rows * Q38_D * 2);
  project_rows_from_panel(stage.mixer.gated, w_out, stage.mixer.out, Q38_GDIM,
                          Q38_D, rows, 0u, Q38_QP, &mixer_buffers);
  sync_all();
  if (q38_vec_id() != ~0u)
    llm_residual_add_f16(stage.mixer.out, stage.mixer.xin, rows * Q38_D,
                         q38_vec_id(), Q38_VEC_CORES);
  sync_all();
  transfer(result_mixer + base * Q38_D, stage.mixer.out, rows * Q38_D * 2);
#if Q38_SCHEDULE_PROBE
  if (mempool_get_core_id() == 0u) {
    probe_gdn_chunks += rows / Q38_GDN_STAGE_ROWS;
    probe_mixer_windows += rows / Q38_WIN;
  }
#endif
}

static NOSTACK void q38_chunk(uint32_t base, uint32_t offset, uint32_t gen) {
  q38_gdn_slot_t *const slot = &stage.mixer.slot[0];
  const uint32_t chunk = base + offset;
  q38_chunk_load(slot, gen, offset);
  gdn_prepare(chunk, Q38_GDN_STAGE_ROWS, slot);
  {
    Q38_PHASE_BEGIN(start);
    gdn_recur(chunk, Q38_GDN_STAGE_ROWS, slot);
    Q38_PHASE_END(start, probe_recur_cycles);
  }
  gdn_gate(chunk, Q38_GDN_STAGE_ROWS, slot,
           stage.mixer.gated + offset * Q38_GDIM);
}

// Supertile n+1's projections are queued while n advances its recurrence in
// token order, one chunk at a time; the queue then drains before n's output
// projection reuses the weight slots and the input buffer.
static NOSTACK void gdn(void) {
  transfer(conv_weight, w_conv, sizeof(conv_weight));
#if !(Q38_SCHEDULE_PROBE && Q38_PROBE_BASE > 0)
  clear_all(history, sizeof(history));
  clear_all(recurrent, sizeof(recurrent));
  clear_all(stage.mixer.tail, sizeof(stage.mixer.tail));
#endif
  uint32_t gen = 0u;
  {
    Q38_PHASE_BEGIN(start);
    const uint32_t rows =
        Q38_MIN(Q38_GB, Q38_MIXER_EXEC_END - Q38_MIXER_EXEC_START);
    q38_stage_input(Q38_MIXER_EXEC_START, rows);
    q38_queue_open(rows, gen);
    q38_queue_drain();
    Q38_PHASE_END(start, probe_prologue_cycles);
  }
  for (uint32_t base = Q38_MIXER_EXEC_START; base < Q38_MIXER_EXEC_END;
       base += Q38_GB) {
    const uint32_t rows = Q38_MIN(Q38_GB, Q38_MIXER_EXEC_END - base);
    const uint32_t next = base + rows;
#if Q38_SCHEDULE_PROBE
    if (mempool_get_core_id() == 0u && next < Q38_MIXER_EXEC_END &&
        probe_overlap_start == 0u)
      probe_overlap_start = mempool_get_timer();
#endif
    if (next < Q38_MIXER_EXEC_END) {
      q38_stage_input(next, Q38_MIN(Q38_GB, Q38_MIXER_EXEC_END - next));
      q38_queue_open(Q38_MIN(Q38_GB, Q38_MIXER_EXEC_END - next), gen ^ 1u);
    }
    for (uint32_t offset = 0u; offset < rows; offset += Q38_GDN_STAGE_ROWS)
      q38_chunk(base, offset, gen);
    if (next < Q38_MIXER_EXEC_END) {
      Q38_PHASE_BEGIN(start);
      q38_queue_drain();
      Q38_PHASE_END(start, probe_drain_cycles);
#if Q38_SCHEDULE_PROBE
      if (mempool_get_core_id() == 0u && probe_overlap_end == 0u)
        probe_overlap_end = mempool_get_timer();
#endif
    }
    {
      Q38_PHASE_BEGIN(start);
      q38_output(base, rows);
      Q38_PHASE_END(start, probe_output_cycles);
    }
    gen ^= 1u;
  }
}
#endif
