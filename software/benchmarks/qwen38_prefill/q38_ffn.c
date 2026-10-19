// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// The FFN phase: gate/up, activation and down projection of every row block,
// with the residual addition. Included by layer_resident.c after
// q38_projection.c; not a translation unit.

#if !Q38_FFN_L2
#if Q38_WEIGHT_PANELS != 0

// The FFN is RedMulE-bound: one sibling per tile does its Spatz work (the
// activation and the residual add), as with two cores per tile. Further cores
// sleep at the next barrier instead of adding pollers and counter updates.
static uint32_t q38_ffn_worker(void) {
  return mempool_get_core_id() % NUM_CORES_PER_TILE == 1u;
}
#define Q38_FFN_WORKERS (NUM_CORES / NUM_CORES_PER_TILE)

#if Q38_GATEUP_PIPELINE && Q38_ASYNC
_Static_assert(2 * Q38_D <= Q38_MAX_K,
               "Gate/up weight pairs must fit one staged slot");

// Element (row, col) of a block's FFN intermediate: row-major, or slice-major
// [slice][rows][Q38_KC] for ordered down slices.
static __fp16 *ffn_gate_at(__fp16 *gate, uint32_t rows, uint32_t row,
                           uint32_t col) {
#if Q38_FFN_DOWN_SLICES > 1
  return gate + col / Q38_KC * rows * Q38_KC + row * Q38_KC + col % Q38_KC;
#else
  (void)rows;
  return gate + row * Q38_I + col;
#endif
}

static NOSTACK void activate_prior_columns(__fp16 *gate, const __fp16 *up_panel,
                                           uint32_t rows, uint32_t first,
                                           uint32_t count) {
  if (count == 0u || !q38_ffn_worker())
    return;
  for (uint32_t row = mempool_get_core_id() / NUM_CORES_PER_TILE; row < rows;
       row += Q38_FFN_WORKERS) {
    // Elementwise, so a group straddling a K slice splits at its boundary.
    for (uint32_t c = first * NT; c < (first + count) * NT;) {
      const uint32_t n = Q38_MIN((first + count) * NT - c, Q38_KC - c % Q38_KC);
      __fp16 *const g = ffn_gate_at(gate, rows, row, c);
      q38_activation16(g, up_panel + row * count * NT + (c - first * NT), g, n,
                       1);
      c += n;
    }
  }
}

#if Q38_COORDINATOR_DMA
static NOSTACK void gateup_dma_publish(__fp16 *dest, const __fp16 *gate_src,
                                       const __fp16 *up_src, uint32_t bytes,
                                       uint32_t epoch) {
  if (mempool_get_core_id() == 0u) {
    if (q38_dma_stage_publish(&gateup_dma_queue, epoch, dest, gate_src, bytes,
                              0u) != Q38_DMA_QUEUE_OK ||
        q38_dma_stage_publish(&gateup_dma_queue, epoch + 1u,
                              dest + bytes / sizeof(__fp16), up_src, bytes,
                              0u) != Q38_DMA_QUEUE_OK)
      __builtin_trap();
  }
  if (mempool_is_group_coordinator()) {
    q38_dma_group_progress_t *const state =
        &gateup_dma_queue.slot[(epoch + 1u) % Q38_DMA_QUEUE_SLOTS]
             .group[mempool_get_group_id()];
    while (__atomic_load_n(&state->issued_epoch, __ATOMIC_ACQUIRE) !=
           epoch + 1u)
      (void)q38_dma_stage_progress(&gateup_dma_queue);
  }
}

static NOSTACK void gateup_dma_wait(uint32_t epoch) {
  uint32_t status;
  do {
    if (mempool_is_group_coordinator())
      (void)q38_dma_stage_progress(&gateup_dma_queue);
    else
      mempool_wait(4u);
    status = q38_dma_stage_ready(&gateup_dma_queue, epoch + 1u);
  } while (status == Q38_DMA_QUEUE_PENDING);
  if (status != Q38_DMA_QUEUE_OK)
    __builtin_trap();
}

static NOSTACK void gateup_dma_release(uint32_t epoch) {
  if (mempool_get_core_id() == 0u &&
      (q38_dma_stage_release(&gateup_dma_queue, epoch) != Q38_DMA_QUEUE_OK ||
       q38_dma_stage_release(&gateup_dma_queue, epoch + 1u) !=
           Q38_DMA_QUEUE_OK))
    __builtin_trap();
}
#endif

#if Q38_FFN_WIDE
static uint8_t start_gateup_job(const __fp16 *in, const __fp16 *staged,
                                uint32_t task, uint32_t half, uint32_t count,
                                __fp16 *z) {
  const uint32_t up = task >= half, local = task - up * half;
  return start_job(in + local / count * Q38_PROJECTION_ROWS * Q38_D,
                   staged + (up * count + local % count) * Q38_D * NT, z,
                   Q38_PROJECTION_ROWS, Q38_D, NT);
}

// One job list over the group's gate then up rectangles; every job and its
// output placement are those of the two separate project_panel_jobs calls.
// The next group's weight prefetch, published only after the engines started:
// a coordinator issuing descriptors first would delay its own engine.
typedef struct {
  __fp16 *dest;
  const __fp16 *gate;
  const __fp16 *up;
  uint32_t bytes;
  uint32_t epoch;
} q38_gateup_prefetch_t;

static NOSTACK void project_gateup_jobs(const __fp16 *in, const __fp16 *staged,
                                        __fp16 *gate, __fp16 *up, uint32_t rows,
                                        uint32_t first, uint32_t count,
                                        const q38_gateup_prefetch_t *next) {
  if (!tnc_is_tile_agent())
    return;
  const uint32_t engine = mempool_get_redmule_id();
  const uint32_t half = rows / Q38_PROJECTION_ROWS * count;
  __fp16 *const z = ffn_buffers.z + engine * 2 * Q38_PROJECTION_Z_SLOT;
  uint32_t z_slot = 0;
  uint8_t target = 0;
  if (engine < 2u * half)
    target = start_gateup_job(in, staged, engine, half, count, z);
#if Q38_COORDINATOR_DMA
  if (next->bytes)
    gateup_dma_publish(next->dest, next->gate, next->up, next->bytes,
                       next->epoch);
#else
  (void)next;
#endif
  for (uint32_t task = engine; task < 2u * half; task += NUM_REDMULE_TILES) {
    __fp16 *const completed = z + z_slot * Q38_PROJECTION_Z_SLOT;
    tnc_redmule_wait(target);
    const uint32_t next = task + NUM_REDMULE_TILES;
    if (next < 2u * half)
      target = start_gateup_job(in, staged, next, half, count,
                                z + (z_slot ^ 1u) * Q38_PROJECTION_Z_SLOT);
    const uint32_t is_up = task >= half, local = task - is_up * half;
    const uint32_t row = local / count * Q38_PROJECTION_ROWS;
    const uint32_t column = local % count;
    for (uint32_t r = 0; r < Q38_PROJECTION_ROWS; ++r)
      q38_copy16(is_up
                     ? up + (row + r) * count * NT + column * NT
                     : ffn_gate_at(gate, rows, row + r, (first + column) * NT),
                 completed + r * NT, NT);
    schedule_delay(first + task + 1);
    z_slot ^= 1u;
  }
}
#endif

#if Q38_FFN_DOWN_SLICES > 1
_Static_assert(Q38_I % (Q38_FFN_DOWN_SLICES * NT) == 0 &&
                   2u * Q38_FFN_DOWN_PANELS * Q38_KC * NT <=
                       sizeof(weights) / sizeof(weights[0][0]),
               "Down slices must hold whole panels and two slots in L1");
#define Q38_DOWN_SLOT (Q38_FFN_DOWN_PANELS * Q38_KC * NT)
_Static_assert(Q38_DOWN_SLOT <= sizeof(weights[0]) / sizeof(weights[0][0]),
               "Down slice slots are the two gate/up weight slots");
// Down slice slot s, as ffn_down_sliced addresses it.
#define Q38_DOWN_SLOT_AT(s) (&weights[0][0] + (s)*Q38_DOWN_SLOT)
static dma_collective_op_t ffn_down_prefetch[NUM_CORES] LOCAL;
_Static_assert(Q38_FFN_ROWS / Q38_PROJECTION_ROWS * Q38_FFN_DOWN_PANELS <=
                   2u * NUM_REDMULE_TILES,
               "Each engine keeps at most two sliced rectangles' Z tiles");

static dma_collective_op_t ffn_down_stage(__fp16 *slot, uint32_t first,
                                          uint32_t count, uint32_t slice) {
  dma_collective_op_t op;
  for (uint32_t p = 0u; p < count; ++p)
    op = dma_collective_memcpy_nonblocking(
        slot + p * Q38_KC * NT,
        w_down + ((first + p) * Q38_I + slice * Q38_KC) * NT,
        Q38_KC * NT * sizeof(__fp16));
  return op;
}
#endif

#if Q38_FFN_DATAFLOW
#define Q38_GU_MAX (Q38_I / NT)
#define Q38_GU_WORKERS Q38_FFN_WORKERS
static uint32_t gu_first[Q38_GU_MAX] LOCAL;
// Rectangles computed (frees the weight slot), copied out (starts the
// activation), and activation workers finished (frees the up buffer).
static uint32_t gu_done[Q38_GU_MAX] LOCAL;
static uint32_t gu_copied[Q38_GU_MAX] LOCAL;
static uint32_t gu_act[Q38_GU_MAX] LOCAL;
// Set once a group's weights are known to have landed: one load instead of
// polling every group's DMA progress words.
static uint32_t gu_ready[Q38_GU_MAX] LOCAL;
// Next unclaimed rectangle of each group: engines take work dynamically, so a
// slow engine does fewer rectangles instead of pacing the machine.
static uint32_t gu_next[Q38_GU_MAX] LOCAL;
static struct {
  uint32_t groups;
  uint32_t epoch;
  uint32_t rows;
  uint32_t retired;
} gu_state LOCAL;

static uint32_t gu_count(uint32_t g) {
  return Q38_MIN(Q38_GU_PANELS, Q38_I / NT - gu_first[g]);
}

_Static_assert(Q38_GU_SLOTS >= 2u && Q38_GU_PANELS > 0u &&
                   Q38_GU_PANELS <= Q38_MAX(Q38_WEIGHT_PANELS, Q38_GU_PANELS) &&
                   (Q38_FFN_OWN_LAYOUT ||
                    Q38_GU_SLOTS * Q38_GU_SLOT_ELEMS <=
                        sizeof(weights) / sizeof(weights[0][0])) &&
                   2u * Q38_GU_SLOTS <= Q38_DMA_QUEUE_SLOTS,
               "The gate/up ring must fit the weight area and the DMA queue");

// Weight slot of group g: the historical halves for a two-deep ring,
// otherwise a carved slot of the ring.
static __fp16 *gu_slot(uint32_t g) {
#if Q38_GU_SLOTS == 2u
  return ffn_buffers.slot[g & 1u];
#else
  return &weights[0][0] + g % Q38_GU_SLOTS * Q38_GU_SLOT_ELEMS;
#endif
}

// Gate then up epochs of group g, into weight slot g % Q38_GU_SLOTS.
static void gu_publish(uint32_t g) {
  const uint32_t bytes = gu_count(g) * Q38_D * NT * sizeof(__fp16);
  __fp16 *const dest = gu_slot(g);
  const uint32_t epoch = gu_state.epoch + 2u * g;
  if (q38_dma_stage_publish_any(&gateup_dma_queue, epoch, dest,
                                w_gate + gu_first[g] * Q38_D * NT, bytes,
                                0u) != Q38_DMA_QUEUE_OK ||
      q38_dma_stage_publish_any(
          &gateup_dma_queue, epoch + 1u, dest + bytes / sizeof(__fp16),
          w_up + gu_first[g] * Q38_D * NT, bytes, 0u) != Q38_DMA_QUEUE_OK)
    __builtin_trap();
}

// Coordinators keep the weight queue moving whenever they wait.
static void gu_idle(void) {
  if (mempool_is_group_coordinator())
    (void)q38_dma_stage_progress(&gateup_dma_queue);
  mempool_wait(32u);
}

static uint32_t gu_weights_ready(uint32_t g) {
  if (__atomic_load_n(&gu_ready[g], __ATOMIC_ACQUIRE))
    return 1u;
  const uint32_t status =
      q38_dma_stage_ready(&gateup_dma_queue, gu_state.epoch + 2u * g + 1u);
  if (status != Q38_DMA_QUEUE_OK && status != Q38_DMA_QUEUE_PENDING)
    __builtin_trap();
  if (status == Q38_DMA_QUEUE_OK)
    __atomic_store_n(&gu_ready[g], 1u, __ATOMIC_RELEASE);
  return status == Q38_DMA_QUEUE_OK;
}

#if Q38_FFN_DOWN_SLICES > 1
#define Q38_GU_DOWN_PIECES Q38_MIN(Q38_FFN_DOWN_PANELS, Q38_D / NT)
_Static_assert(2u * Q38_GU_SLOTS + Q38_GU_DOWN_PIECES <= Q38_DMA_QUEUE_SLOTS,
               "The DMA queue must hold the ring and the first down slice");
// The first down slice's panels, into the slot group G-2 frees, as the epochs
// after the last gate/up group's.
static void gu_publish_down(void) {
  const uint32_t g = gu_state.groups;
  for (uint32_t p = 0u; p < Q38_GU_DOWN_PIECES; ++p)
    if (q38_dma_stage_publish_any(
            &gateup_dma_queue, gu_state.epoch + 2u * g + p,
            Q38_DOWN_SLOT_AT(g & 1u) + p * Q38_KC * NT, w_down + p * Q38_I * NT,
            Q38_KC * NT * sizeof(__fp16), 0u) != Q38_DMA_QUEUE_OK)
      __builtin_trap();
}
#define Q38_GU_LAST_EPOCH                                                      \
  (gu_state.epoch + 2u * gu_state.groups + Q38_GU_DOWN_PIECES - 1u)
#else
#define Q38_GU_LAST_EPOCH (gu_state.epoch + 2u * gu_state.groups - 1u)
#endif

static NOSTACK void gu_retire(uint32_t g, uint32_t jobs) {
  if (__atomic_add_fetch(&gu_done[g], 1u, __ATOMIC_ACQ_REL) != jobs)
    return;
  const uint32_t epoch = gu_state.epoch + 2u * g;
  if (q38_dma_stage_release_any(&gateup_dma_queue, epoch) != Q38_DMA_QUEUE_OK ||
      q38_dma_stage_release_any(&gateup_dma_queue, epoch + 1u) !=
          Q38_DMA_QUEUE_OK)
    __builtin_trap();
  if (g + Q38_GU_SLOTS < gu_state.groups)
    gu_publish(g + Q38_GU_SLOTS);
#if Q38_FFN_DOWN_SLICES > 1
#if Q38_GU_SLOTS == 2u
  if (g + 2u == gu_state.groups)
    gu_publish_down();
#else
  // Carved slots overlap the down slice slots: stage the first down slice
  // once the last group to finish frees the area.
  if (__atomic_add_fetch(&gu_state.retired, 1u, __ATOMIC_ACQ_REL) ==
      gu_state.groups)
    gu_publish_down();
#endif
#endif
}

// Copy a finished tile out while the engine already runs the next one.
static NOSTACK void gu_copy_out(__fp16 *gate, const __fp16 *tile, uint32_t g,
                                uint32_t task) {
  const uint32_t rows = gu_state.rows, count = gu_count(g);
  const uint32_t half = rows / Q38_PROJECTION_ROWS * count;
  const uint32_t is_up = task >= half, local = task - is_up * half;
  const uint32_t row = local / count * Q38_PROJECTION_ROWS;
  const uint32_t column = local % count;
  for (uint32_t r = 0; r < Q38_PROJECTION_ROWS; ++r)
    q38_copy16(
        is_up ? stage.ffn.up_panel[g % Q38_GU_SLOTS] + (row + r) * count * NT +
                    column * NT
              : ffn_gate_at(gate, rows, row + r, (gu_first[g] + column) * NT),
        tile + r * NT, NT);
  __sync_synchronize();
  __atomic_add_fetch(&gu_copied[g], 1u, __ATOMIC_ACQ_REL);
}

static NOSTACK void gu_agent(const __fp16 *in, __fp16 *gate) {
  const uint32_t engine = mempool_get_redmule_id();
  const uint32_t rows = gu_state.rows;
  __fp16 *const z = ffn_buffers.z + engine * 2 * Q38_PROJECTION_Z_SLOT;
  uint32_t zs = 0u, pending = 0u, pending_g = 0u, pending_task = 0u;
  uint32_t g = 0u;
  while (engine < NUM_REDMULE_TILES && g < gu_state.groups) {
    const uint32_t count = gu_count(g);
    const uint32_t half = rows / Q38_PROJECTION_ROWS * count;
    const uint32_t task = __atomic_fetch_add(&gu_next[g], 1u, __ATOMIC_RELAXED);
    if (task >= 2u * half) {
      ++g;
      continue;
    }
    const uint32_t is_up = task >= half;
    // An activation may be waiting for this engine's own pending tile, so
    // copy it out before blocking on anything.
    while (!gu_weights_ready(g) ||
           (is_up && g >= Q38_GU_SLOTS &&
            __atomic_load_n(&gu_act[g - Q38_GU_SLOTS], __ATOMIC_ACQUIRE) !=
                Q38_GU_WORKERS)) {
      if (pending) {
        gu_copy_out(gate, z + (zs ^ 1u) * Q38_PROJECTION_Z_SLOT, pending_g,
                    pending_task);
        pending = 0u;
      }
      gu_idle();
    }
    const uint8_t target = start_gateup_job(in, gu_slot(g), task, half, count,
                                            z + zs * Q38_PROJECTION_Z_SLOT);
    if (pending)
      gu_copy_out(gate, z + (zs ^ 1u) * Q38_PROJECTION_Z_SLOT, pending_g,
                  pending_task);
    while (!tnc_redmule_done(target))
      gu_idle();
    gu_retire(g, 2u * half);
    pending = 1u;
    pending_g = g;
    pending_task = task;
    zs ^= 1u;
  }
  if (pending)
    gu_copy_out(gate, z + (zs ^ 1u) * Q38_PROJECTION_Z_SLOT, pending_g,
                pending_task);
  // Later groups' (and the first down slice's) weights may still be in
  // flight for other engines.
  if (mempool_is_group_coordinator())
    while (q38_dma_stage_ready(&gateup_dma_queue, Q38_GU_LAST_EPOCH) ==
           Q38_DMA_QUEUE_PENDING)
      gu_idle();
}

static NOSTACK void gu_sibling(__fp16 *gate) {
  const uint32_t rows = gu_state.rows;
  for (uint32_t g = 0u; g < gu_state.groups; ++g) {
    const uint32_t jobs = 2u * (rows / Q38_PROJECTION_ROWS) * gu_count(g);
    while (__atomic_load_n(&gu_copied[g], __ATOMIC_ACQUIRE) != jobs)
      mempool_wait(32u);
    activate_prior_columns(gate, stage.ffn.up_panel[g % Q38_GU_SLOTS], rows,
                           gu_first[g], gu_count(g));
    __sync_synchronize();
    __atomic_add_fetch(&gu_act[g], 1u, __ATOMIC_ACQ_REL);
  }
}

// Returns the weight slot left for the first down slice.
static NOSTACK uint32_t ffn_gateup_dataflow(const __fp16 *in, __fp16 *gate,
                                            uint32_t rows) {
  const uint32_t core = mempool_get_core_id();
  const uint32_t columns = Q38_I / NT;
  if (core == 0u) {
    uint32_t g = 0u;
    for (uint32_t first = 0u; first < columns; ++g) {
      gu_first[g] = first;
      gu_done[g] = 0u;
      gu_copied[g] = 0u;
      gu_ready[g] = 0u;
      gu_next[g] = 0u;
      gu_act[g] = 0u;
      first = next_panel_group(first, Q38_MIN(Q38_GU_PANELS, columns - first),
                               columns, Q38_GU_PANELS);
    }
    gu_state.groups = g;
    gu_state.rows = rows;
    gu_state.retired = 0u;
    gu_state.epoch = gateup_dma_next[0];
#if Q38_SCHEDULE_PROBE
    probe_gateup_groups += g;
#endif
    for (uint32_t p = 0u; p < Q38_MIN(g, Q38_GU_SLOTS); ++p)
      gu_publish(p);
#if Q38_FFN_DOWN_SLICES > 1 && Q38_GU_SLOTS == 2u
    if (g == 1u)
      gu_publish_down();
#endif
  }
  sync_all();
  if (tnc_is_tile_agent())
    gu_agent(in, gate);
  else if (q38_ffn_worker())
    gu_sibling(gate);
  sync_all();
  if (q38_dma_stage_ready(&gateup_dma_queue, Q38_GU_LAST_EPOCH) !=
      Q38_DMA_QUEUE_OK)
    __builtin_trap();
  gateup_dma_next[core] = Q38_GU_LAST_EPOCH + 1u;
#if Q38_FFN_DOWN_SLICES > 1
  if (core == 0u)
    for (uint32_t p = 0u; p < Q38_GU_DOWN_PIECES; ++p)
      if (q38_dma_stage_release_any(
              &gateup_dma_queue, gu_state.epoch + 2u * gu_state.groups + p) !=
          Q38_DMA_QUEUE_OK)
        __builtin_trap();
  sync_all();
  return gu_state.groups & 1u;
#else
  return 0u;
#endif
}
#endif

// Tile agents project the current gate/up columns while sibling Spatz cores
// activate the preceding columns and iDMA fetches the next pair of panels.
// Returns the weight slot the last group left idle.
static NOSTACK uint32_t ffn_gateup_pairs(const __fp16 *in, __fp16 *gate,
                                         uint32_t rows) {
  const uint32_t columns = Q38_I / NT;
  const uint32_t panel_elems = Q38_D * NT;
  uint32_t count = columns < Q38_WEIGHT_PANELS ? columns : Q38_WEIGHT_PANELS;
#if Q38_COORDINATOR_DMA
  uint32_t epoch = gateup_dma_next[mempool_get_core_id()];
  gateup_dma_publish(ffn_buffers.slot[0], w_gate, w_up, count * panel_elems * 2,
                     epoch);
  gateup_dma_wait(epoch);
#else
  transfer(ffn_buffers.slot[0], w_gate, count * panel_elems * 2);
  transfer(ffn_buffers.slot[0] + count * panel_elems, w_up,
           count * panel_elems * 2);
#endif
  uint32_t slot = 0u;
  uint32_t prior_first = 0u, prior_count = 0u;
  for (uint32_t first = 0u; first < columns;) {
#if Q38_SCHEDULE_PROBE
    if (mempool_get_core_id() == 0u)
      ++probe_gateup_groups;
#endif
    count = columns - first;
    if (count > Q38_WEIGHT_PANELS)
      count = Q38_WEIGHT_PANELS;
    const uint32_t next =
        next_panel_group(first, count, columns, Q38_WEIGHT_PANELS);
    uint32_t next_count = columns - next;
    if (next_count > Q38_WEIGHT_PANELS)
      next_count = Q38_WEIGHT_PANELS;
#if !Q38_COORDINATOR_DMA
    dma_collective_op_t pending;
#endif
#if Q38_FFN_WIDE
    q38_gateup_prefetch_t prefetch_next = {0, 0, 0, 0u, 0u};
#endif
    if (next_count) {
#if Q38_COORDINATOR_DMA && Q38_FFN_WIDE
      prefetch_next.dest = ffn_buffers.slot[slot ^ 1u];
      prefetch_next.gate = w_gate + next * panel_elems;
      prefetch_next.up = w_up + next * panel_elems;
      prefetch_next.bytes = next_count * panel_elems * 2;
      prefetch_next.epoch = epoch + 2u;
#elif Q38_COORDINATOR_DMA
      gateup_dma_publish(ffn_buffers.slot[slot ^ 1u],
                         w_gate + next * panel_elems, w_up + next * panel_elems,
                         next_count * panel_elems * 2, epoch + 2u);
#else
      (void)dma_collective_memcpy_nonblocking(ffn_buffers.slot[slot ^ 1u],
                                              w_gate + next * panel_elems,
                                              next_count * panel_elems * 2);
      pending = dma_collective_memcpy_nonblocking(
          ffn_buffers.slot[slot ^ 1u] + next_count * panel_elems,
          w_up + next * panel_elems, next_count * panel_elems * 2);
#endif
    }
#if Q38_FFN_WIDE
#if Q38_FFN_DOWN_SLICES > 1
    // The last group leaves the other slot idle: stage the first down slice.
    if (!next_count)
      ffn_down_prefetch[mempool_get_core_id()] =
          ffn_down_stage(ffn_buffers.slot[slot ^ 1u], 0u,
                         Q38_MIN(Q38_FFN_DOWN_PANELS, Q38_D / NT), 0u);
#endif
    project_gateup_jobs(in, ffn_buffers.slot[slot], gate,
                        stage.ffn.up_panel[slot], rows, first, count,
                        &prefetch_next);
#else
    project_panel_jobs(in, ffn_buffers.slot[slot], gate, Q38_D, Q38_I, rows,
                       first, count, &ffn_buffers);
    project_panel_jobs(in, ffn_buffers.slot[slot] + count * panel_elems,
                       stage.ffn.up_panel[slot], Q38_D, count * NT, rows, 0u,
                       count, &ffn_buffers);
#endif
    activate_prior_columns(gate, stage.ffn.up_panel[slot ^ 1u], rows,
                           prior_first, prior_count);
    sync_all();
#if Q38_COORDINATOR_DMA
    if (next_count)
      gateup_dma_wait(epoch + 2u);
    gateup_dma_release(epoch);
    epoch += 2u;
#else
    if (next_count &&
        (dma_collective_wait(&pending) != 0u || pending.status != 0u))
      __builtin_trap();
#endif
    prior_first = first;
    prior_count = count;
    slot ^= 1u;
    first = next;
  }
  activate_prior_columns(gate, stage.ffn.up_panel[slot ^ 1u], rows, prior_first,
                         prior_count);
  sync_all();
#if Q38_COORDINATOR_DMA
  gateup_dma_next[mempool_get_core_id()] = epoch;
#endif
  return slot;
}
#endif
#endif

#if Q38_FFN_NORM_STREAM
// result_output holds normalized FFN input until the same rows are overwritten
// by their final output. This adds no sequence-scaled L2 allocation.
static NOSTACK void ffn_norm_window(uint32_t base, uint32_t rows) {
  transfer(Q38_FFN_RESIDUAL, result_mixer + base * Q38_D, rows * Q38_D * 2);
  norm_rows(Q38_FFN_RESIDUAL, w_ffn_norm, Q38_FFN_A, rows, Q38_D, 0,
            Q38_FFN_GAMMA);
  transfer(result_output + base * Q38_D, Q38_FFN_A, rows * Q38_D * 2);
}
#endif

#if Q38_FFN_WIDE
// Only the unsliced path stages full-K panels; sliced down checks its own.
_Static_assert(Q38_FFN_DOWN_PANELS > 0 &&
                   (Q38_FFN_DOWN_SLICES > 1 ||
                    Q38_FFN_DOWN_PANELS * Q38_I * NT <=
                        sizeof(weights) / sizeof(weights[0][0])),
               "Wide FFN down panels must fit the whole weight area");

// Single-buffered: the group's DMA is exposed, in exchange for twice the
// independent output rectangles of a double-buffered half.
static NOSTACK void ffn_down_wide(const __fp16 *in, __fp16 *out,
                                  uint32_t rows) {
  const uint32_t columns = Q38_D / NT;
  for (uint32_t first = 0u; first < columns;) {
    const uint32_t count = Q38_MIN(Q38_FFN_DOWN_PANELS, columns - first);
#if Q38_SCHEDULE_PROBE
    if (mempool_get_core_id() == 0u)
      ++probe_projection_groups;
#endif
    transfer(&weights[0][0], w_down + first * Q38_I * NT,
             count * Q38_I * NT * sizeof(__fp16));
    project_panel_jobs(in, &weights[0][0], out, Q38_I, Q38_D, rows, first,
                       count, &ffn_buffers);
    sync_all();
    first = next_panel_group(first, count, columns, Q38_FFN_DOWN_PANELS);
  }
}
#endif

#if Q38_FFN_DOWN_SLICES > 1
// Sibling Spatz cores: residual addition of a completed down group's columns.
static NOSTACK void ffn_residual_columns(__fp16 *out, uint32_t rows,
                                         uint32_t first, uint32_t count) {
  if (count == 0u || !q38_ffn_worker())
    return;
  for (uint32_t row = mempool_get_core_id() / NUM_CORES_PER_TILE; row < rows;
       row += Q38_FFN_WORKERS)
    llm_residual_add_f16(out + row * Q38_D + first * NT,
                         Q38_FFN_RESIDUAL + row * Q38_D + first * NT,
                         count * NT, 0u, 1u);
}

static uint8_t ffn_down_start(const __fp16 *gate, const __fp16 *slot,
                              uint32_t rows, uint32_t count, uint32_t slice,
                              uint32_t task) {
  // A rectangle keeps its own Z slot across all of its slices.
  __fp16 *const tile = ffn_buffers.z +
                       mempool_get_redmule_id() * 2 * Q38_PROJECTION_Z_SLOT +
                       task / NUM_REDMULE_TILES * Q38_PROJECTION_Z_SLOT;
  return start_job_y(gate + slice * rows * Q38_KC +
                         task / count * Q38_PROJECTION_ROWS * Q38_KC,
                     slot + task % count * Q38_KC * NT, tile,
                     Q38_PROJECTION_ROWS, Q38_KC, NT, slice == 0u);
}

// Completes the job started for this engine's first task, then runs the rest.
static NOSTACK void ffn_down_finish(const __fp16 *gate, const __fp16 *slot,
                                    __fp16 *out, uint32_t rows, uint32_t first,
                                    uint32_t count, uint32_t slice,
                                    uint32_t target) {
  if (!tnc_is_tile_agent())
    return;
  const uint32_t engine = mempool_get_redmule_id();
  for (uint32_t task = engine; task < rows / Q38_PROJECTION_ROWS * count;
       task += NUM_REDMULE_TILES) {
    tnc_redmule_wait(
        task == engine ? (uint8_t)target
                       : ffn_down_start(gate, slot, rows, count, slice, task));
    if (slice + 1u < Q38_FFN_DOWN_SLICES)
      continue;
    const __fp16 *const tile = ffn_buffers.z +
                               engine * 2 * Q38_PROJECTION_Z_SLOT +
                               task / NUM_REDMULE_TILES * Q38_PROJECTION_Z_SLOT;
    const uint32_t row = task / count * Q38_PROJECTION_ROWS;
    for (uint32_t r = 0; r < Q38_PROJECTION_ROWS; ++r)
      q38_copy16(out + (row + r) * Q38_D + (first + task % count) * NT,
                 tile + r * NT, NT);
  }
}

// Steps are (panel group, slice) in order. Each step starts its engines, then
// stages step n+1's weights into the other slot, and the sibling cores add the
// residual of the previous group's columns; the first step's weights were
// staged during the last gate/up group. The caller skips its own residual
// addition.
static NOSTACK void ffn_down_sliced(const __fp16 *gate, __fp16 *out,
                                    uint32_t rows, uint32_t parity) {
  __fp16 *const base = &weights[0][0];
  const uint32_t columns = Q38_D / NT;
  uint32_t first = 0u, slice = 0u, step = parity, done_first = 0u,
           done_count = 0u;
  uint32_t count = Q38_MIN(Q38_FFN_DOWN_PANELS, columns);
  dma_collective_op_t op = ffn_down_prefetch[mempool_get_core_id()];
#if !Q38_FFN_DATAFLOW
  // With dataflow gate/up the first slice already landed through the queue.
  if (dma_collective_wait(&op) != 0u || op.status != 0u)
    __builtin_trap();
#endif
  while (first < columns) {
    uint32_t next = first, next_slice = slice + 1u;
    if (next_slice == Q38_FFN_DOWN_SLICES) {
      next = next_panel_group(first, count, columns, Q38_FFN_DOWN_PANELS);
      next_slice = 0u;
    }
    const uint32_t next_count =
        next < columns ? Q38_MIN(Q38_FFN_DOWN_PANELS, columns - next) : 0u;
    const __fp16 *const slot = base + (step & 1u) * Q38_DOWN_SLOT;
#if Q38_SCHEDULE_PROBE
    if (mempool_get_core_id() == 0u && slice == 0u)
      ++probe_projection_groups;
#endif
    uint32_t target = 0u;
    if (tnc_is_tile_agent() &&
        mempool_get_redmule_id() < rows / Q38_PROJECTION_ROWS * count)
      target = ffn_down_start(gate, slot, rows, count, slice,
                              mempool_get_redmule_id());
    if (next_count)
      op = ffn_down_stage(base + ((step + 1u) & 1u) * Q38_DOWN_SLOT, next,
                          next_count, next_slice);
    ffn_down_finish(gate, slot, out, rows, first, count, slice, target);
    if (slice == 0u) {
      ffn_residual_columns(out, rows, done_first, done_count);
      done_count = 0u;
    }
    sync_all();
    if (next_count && (dma_collective_wait(&op) != 0u || op.status != 0u))
      __builtin_trap();
    if (next_slice == 0u) {
      done_first = first;
      done_count = count;
    }
    first = next;
    slice = next_slice;
    count = next_count;
    ++step;
  }
  ffn_residual_columns(out, rows, done_first, done_count);
  sync_all();
}
#endif

#endif // !Q38_FFN_L2

#if Q38_FFN_GU_SLICES > 1 || Q38_FFN_OWN_LAYOUT
_Static_assert(Q38_FFN_WIDE && (Q38_FFN_DOWN_SLICES > 1 || Q38_FFN_L2) &&
                   Q38_FFN_DATAFLOW && !Q38_FFN_NORM_STREAM &&
                   !Q38_STAGE_SEPARATE &&
                   Q38_D % (Q38_FFN_GU_SLICES * 128u) == 0u &&
                   (Q38_FFN_OWN_LAYOUT ||
                    2u * Q38_GU_SLOT_ELEMS <=
                        sizeof(weights) / sizeof(weights[0][0])) &&
                   (Q38_FFN_VC ||
                    Q38_FFN_ROWS / Q38_PROJECTION_ROWS * Q38_GU_PANELS * 2u <=
                        NUM_REDMULE_TILES),
               "K-sliced gate/up: whole slices, two slots, one job per engine");

#ifndef Q38_NORM_SPLIT_REGS
#define Q38_NORM_SPLIT_REGS 0
#endif
// q38_norm16 with its output stored slice-major, [slice][rows][Q38_GUK]: the
// same instructions, only the store addresses and chunk bounds differ.
static inline void q38_norm16_sliced(const __fp16 *x, const __fp16 *gamma,
                                     __fp16 *out, uint32_t n, uint32_t row,
                                     uint32_t rows) {
  uint32_t capacity;
  const float zero = 0.0f, zh = llm_f16box(0.0f);
  asm volatile("vsetvli %0, %1, e32, m8, tu, ma"
               : "=r"(capacity)
               : "r"(UINT32_MAX));
  asm volatile("vmv.v.i v8, 0");
  for (uint32_t offset = 0; offset < n;) {
    uint32_t vl, requested = n - offset;
    if (requested > capacity)
      requested = capacity;
    asm volatile("vsetvli %0, %1, e16, m4, tu, ma" : "=r"(vl) : "r"(requested));
#if Q38_NORM_SPLIT_REGS
    // Alternate the load register so the next chunk's (remote) load is not
    // held behind this chunk's vfwmacc; the accumulation order is unchanged.
    if (offset / capacity & 1u) {
      asm volatile("vle16.v v4, (%0)" : : "r"(x + offset) : "memory");
      asm volatile("vfwmacc.vv v8, v4, v4");
    } else
#endif
    {
      asm volatile("vle16.v v0, (%0)" : : "r"(x + offset) : "memory");
      asm volatile("vfwmacc.vv v8, v0, v0");
    }
    offset += vl;
  }
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(capacity));
  asm volatile("vfmv.s.f v24, %0" : : "f"(zero));
  asm volatile("vfredusum.vs v24, v8, v24");
  float sum;
  asm volatile("vfmv.f.s %0, v24" : "=f"(sum));
  const float inverse = llm_rsqrt(sum * llm_recip_pos((float)n) + 1e-6f);
  for (uint32_t offset = 0; offset < n;) {
    uint32_t vl, requested = n - offset;
    if (requested > capacity)
      requested = capacity;
    if (requested > Q38_GUK - offset % Q38_GUK)
      requested = Q38_GUK - offset % Q38_GUK;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(requested));
    __fp16 *const dst = out + offset / Q38_GUK * rows * Q38_GUK +
                        row * Q38_GUK + offset % Q38_GUK;
#if Q38_NORM_SPLIT_REGS
    // x, gamma and the result in separate groups: both loads can be in
    // flight, and the next chunk's loads do not wait for this store.
    asm volatile("vle16.v v0, (%0)" : : "r"(x + offset) : "memory");
    asm volatile("vle16.v v4, (%0)" : : "r"(gamma + offset) : "memory");
    asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
    asm volatile("vfwadd.vf v16, v4, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    asm volatile("vfmul.vf v8, v8, %0" : : "f"(inverse));
    asm volatile("vfmul.vv v8, v8, v16");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
    asm volatile("vfncvt.f.f.w v24, v8");
    asm volatile("vse16.v v24, (%0)" : : "r"(dst) : "memory");
#else
    asm volatile("vle16.v v0, (%0)" : : "r"(x + offset) : "memory");
    asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
    asm volatile("vle16.v v0, (%0)" : : "r"(gamma + offset) : "memory");
    asm volatile("vfwadd.vf v16, v0, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    asm volatile("vfmul.vf v8, v8, %0" : : "f"(inverse));
    asm volatile("vfmul.vv v8, v8, v16");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
    asm volatile("vfncvt.f.f.w v0, v8");
    asm volatile("vse16.v v0, (%0)" : : "r"(dst) : "memory");
#endif
    offset += vl;
  }
}

#if Q38_FFN_L2
static NOSTACK void f2_sync_all(void);
static NOSTACK void f2_early_weight(void);
#endif

static NOSTACK void norm_rows_sliced(const __fp16 *in, const __fp16 *scale,
                                     __fp16 *out, uint32_t rows,
                                     __fp16 *gamma_buf) {
  transfer(gamma_buf, scale, Q38_D * 2);
#if Q38_FFN_L2
  f2_early_weight();
#endif
  for (uint32_t row = q38_vec_id(); row < rows; row += Q38_VEC_CORES)
    q38_norm16_sliced(in + row * Q38_D, gamma_buf, out, Q38_D, row, rows);
#if Q38_FFN_L2
  f2_sync_all();
#else
  sync_all();
#endif
}

// One step's weights: K slice `slice` of `count` gate panels, then of the same
// up panels, each panel slice contiguous in L2.
#if !Q38_FFN_L2
static dma_collective_op_t gu_slice_stage(__fp16 *slot, uint32_t first,
                                          uint32_t count, uint32_t slice) {
  dma_collective_op_t op;
  for (uint32_t m = 0u; m < 2u; ++m) {
    const __fp16 *const src = m ? w_up : w_gate;
    for (uint32_t p = 0u; p < count; ++p)
      op = dma_collective_memcpy_nonblocking(
          slot + (m * count + p) * Q38_GUK * NT,
          src + ((first + p) * Q38_D + slice * Q38_GUK) * NT,
          Q38_GUK * NT * sizeof(__fp16));
  }
  return op;
}

// Steps are (panel group, K slice) in order; every step offers one job per
// engine, each engine keeping its Z tile across the group's slices. A step
// starts its jobs, stages step n+1 into the other slot, and workers activate
// the previous group's columns. `in` is slice-major (norm_rows_sliced).
// Returns the down slot the first down slice was staged into.
static NOSTACK uint32_t ffn_gateup_ksliced(const __fp16 *in, __fp16 *gate,
                                           uint32_t rows) {
  __fp16 *const base = &weights[0][0];
  const uint32_t columns = Q38_I / NT;
  uint32_t first = 0u, slice = 0u, step = 0u, group = 0u, prior_first = 0u,
           prior_count = 0u;
  uint32_t count = Q38_MIN(Q38_GU_PANELS, columns);
  dma_collective_op_t op = gu_slice_stage(base, 0u, count, 0u);
  if (dma_collective_wait(&op) != 0u || op.status != 0u)
    __builtin_trap();
  __fp16 *const z =
      ffn_buffers.z + mempool_get_redmule_id() * 2 * Q38_PROJECTION_Z_SLOT;
  while (first < columns) {
    uint32_t next = first, next_slice = slice + 1u;
    if (next_slice == Q38_FFN_GU_SLICES) {
      next = next_panel_group(first, count, columns, Q38_GU_PANELS);
      next_slice = 0u;
    }
    const uint32_t next_count =
        next < columns ? Q38_MIN(Q38_GU_PANELS, columns - next) : 0u;
    const __fp16 *const slot = base + (step & 1u) * Q38_GU_SLOT_ELEMS;
#if Q38_SCHEDULE_PROBE
    if (mempool_get_core_id() == 0u && slice == 0u)
      ++probe_gateup_groups;
#endif
    const uint32_t half = rows / Q38_PROJECTION_ROWS * count;
    const uint32_t task = mempool_get_redmule_id();
    const uint32_t mine = tnc_is_tile_agent() && task < 2u * half;
    const uint32_t is_up = task >= half, local = task - is_up * half;
    __fp16 *const tile = z + (group & 1u) * Q38_PROJECTION_Z_SLOT;
    uint8_t target = 0u;
    if (mine)
      target =
          start_job_y(in + slice * rows * Q38_GUK +
                          local / count * Q38_PROJECTION_ROWS * Q38_GUK,
                      slot + (is_up * count + local % count) * Q38_GUK * NT,
                      tile, Q38_PROJECTION_ROWS, Q38_GUK, NT, slice == 0u);
    if (next_count)
      op = gu_slice_stage(base + ((step + 1u) & 1u) * Q38_GU_SLOT_ELEMS, next,
                          next_count, next_slice);
    if (slice == 0u)
      activate_prior_columns(gate, stage.ffn.up_panel[(group + 1u) & 1u], rows,
                             prior_first, prior_count);
    if (mine) {
      tnc_redmule_wait(target);
      if (slice + 1u == Q38_FFN_GU_SLICES) {
        const uint32_t row = local / count * Q38_PROJECTION_ROWS;
        const uint32_t column = local % count;
        for (uint32_t r = 0; r < Q38_PROJECTION_ROWS; ++r)
          q38_copy16(
              is_up ? stage.ffn.up_panel[group & 1u] + (row + r) * count * NT +
                          column * NT
                    : ffn_gate_at(gate, rows, row + r, (first + column) * NT),
              tile + r * NT, NT);
      }
    }
    sync_all();
    if (next_count && (dma_collective_wait(&op) != 0u || op.status != 0u))
      __builtin_trap();
    if (next_slice == 0u) {
      prior_first = first;
      prior_count = count;
      ++group;
    }
    first = next;
    slice = next_slice;
    count = next_count;
    ++step;
  }
  activate_prior_columns(gate, stage.ffn.up_panel[(group + 1u) & 1u], rows,
                         prior_first, prior_count);
  // The first down slice, once no gate/up slot is live.
  op = ffn_down_stage(Q38_DOWN_SLOT_AT(0u), 0u,
                      Q38_MIN(Q38_FFN_DOWN_PANELS, Q38_D / NT), 0u);
  if (dma_collective_wait(&op) != 0u || op.status != 0u)
    __builtin_trap();
  sync_all();
  return 0u;
}
#endif

#endif // !Q38_FFN_L2

#if Q38_FFN_L2
// A full attention layer keeps NT for its projections (their L1 arena does
// not fit wider panels) and may give the L2 FFN wider ones. Only
// q38_ffn_l2.c sees the override; nothing after it depends on f2's NT.
#ifndef Q38_FFN_COLUMNS
#define Q38_FFN_COLUMNS Q38_PROJECTION_COLUMNS
#endif
_Static_assert(Q38_FFN_COLUMNS == Q38_PROJECTION_COLUMNS || Q38_ATTN_L2_LAYER,
               "Separate FFN panel widths need a full attention layer");
#pragma push_macro("NT")
#undef NT
#define NT Q38_FFN_COLUMNS
#include "q38_ffn_l2.c"
#pragma pop_macro("NT")
#elif Q38_FFN_VC
// The FFN with the synchronization scope of a 2x2 virtual cluster (VC).
//
// Pairs: every engine owns one gate/up pair per pair-step -- the gate tile
// and then the up tile of the same (row tile, column panel) -- and its own
// tile's worker activates the pair from the two Z tiles into G, so the
// RedMulE -> Spatz dependency stays inside the tile.
//
// Weights: each VC double-buffers its own column panels in two VC-local
// slots. A job-step is one K slice of the step's gate panels, of its up
// panels, or (down) of its output panels; job-step t reads slot t % 2. The
// VC's 16 tiles meet at one fractal_sync barrier per buffer switch: a tile
// arrives once its engine finished job-step t, and a group's DMA issuer (its
// coordinator, the group's only iDMA issuer) additionally once its stripe of
// job-step t + 1 landed. The response therefore means "slot t % 2 is free
// and slot (t + 1) % 2 is full": everyone starts job-step t + 1 and the
// issuers load job-step t + 2 into the freed slot. A job-step's load
// overlaps the previous job.
//
// Down: output K slice s reads G columns produced by every engine, so a down
// job waits until the pair-steps covering slice s are activated cluster-wide.
// A (normalized input), G and the residual rows stay global.
//
// Waits sleep on the tile's events (RedMulE end, group DMA end, barrier
// response); a core leaving the phase drops its pending events before it
// restores the barrier subscription the rest of the program expects.
// Row tiles per block, panels per VC and per step.
#define Q38_VC_RT (Q38_FFN_ROWS / Q38_PROJECTION_ROWS)
#define Q38_VC_PV (Q38_VC_ENGINES / Q38_VC_RT)
#define Q38_VC_P (Q38_VC_PV * Q38_VC_COUNT)
#define Q38_VC_GU_MAX ((Q38_I / NT + Q38_VC_P - 1u) / Q38_VC_P)
#define Q38_VC_DN_MAX ((Q38_D / NT + Q38_VC_P - 1u) / Q38_VC_P)
#define Q38_VC_BLOCKS ((Q38_EXEC_END - Q38_EXEC_START) / Q38_FFN_ROWS)
#define Q38_VC_PIECE_ELEMS Q38_MAX(Q38_GUK *NT, Q38_KC *NT)
#define Q38_VC_SLOT_BYTES                                                      \
  ((Q38_VC_PV * Q38_VC_PIECE_ELEMS * 2u + Q38_VC_ROW - 1u) / Q38_VC_ROW *      \
   Q38_VC_ROW)
_Static_assert(Q38_FFN_VC == 2u && !Q38_FFN_PIPE && VIRTUAL_CLUSTER &&
                   FSYNC_TPG_SIDE != 0 && NUM_GROUPS % 4u == 0u &&
                   NUM_REDMULE_TILES == NUM_GROUPS * NUM_TILES_PER_GROUP &&
                   Q38_VC_ENGINES == 4u * NUM_TILES_PER_GROUP &&
                   Q38_VC_ENGINES % Q38_VC_RT == 0u && Q38_KC % NT == 0u &&
                   Q38_GUK * NT * 2u % Q38_VC_ROW == 0u &&
                   Q38_KC * NT * 2u % Q38_VC_ROW == 0u &&
                   (Q38_EXEC_END - Q38_EXEC_START) % Q38_FFN_ROWS == 0u &&
                   Q38_REDMULE_MATMUL && !Q38_DELAY_SEED,
               "VC FFN: two slots, whole VCs, pairs per engine, row-aligned "
               "slices");

// One job-step: gate/up (pair-step n, matrix m, K slice s) or down (output
// step n, K slice s) of block b. A short final step leaves a VC fewer panels
// than Q38_VC_PV, or none: engines without a panel only take part in the
// barriers.
typedef struct {
  uint32_t b, down, n, m, s;
} q38_vc_step_t;

static struct {
  uint32_t vc_offset; // level-1 VC-local offset of slot 0
  __fp16 *gate;       // G, slice-major [Q38_FFN_DOWN_SLICES][rows][Q38_KC]
  __fp16 *z;          // group window: four Z tiles per local engine
} vc_buf Q38_VC_SMALL;
static struct {
  uint32_t gu_steps, dn_steps, block_steps, steps, rows;
  // Last pair-step whose columns reach into down K slice s.
  uint16_t need[Q38_FFN_DOWN_SLICES];
} vc_state Q38_VC_SMALL;
// First column panel of each gate/up pair-step and down step (the probe
// keeps its leading steps and the final one), and the pairs per pair-step.
static uint16_t vc_gu_first[Q38_VC_GU_MAX] Q38_VC_SMALL;
static uint16_t vc_dn_first[Q38_VC_DN_MAX] Q38_VC_SMALL;
static uint16_t vc_gu_pairs[Q38_VC_GU_MAX] Q38_VC_SMALL;
// Monotone over blocks. Per engine, written by one core and read by its
// sibling: pairs computed / activated, down tiles computed / added to the
// residual. Per pair-step, cluster-wide: pairs activated.
typedef struct {
  uint32_t pair_done, act_done, dnz_done, dn_done;
  uint32_t padding[NUM_BANKS_PER_TILE - 4u];
} q38_ffn_vc_progress_t;
_Static_assert(sizeof(q38_ffn_vc_progress_t) == NUM_BANKS_PER_TILE * 4u,
               "FFN progress must occupy one owner-tile bank stripe");
static q38_ffn_vc_progress_t vc_progress[NUM_REDMULE_TILES] Q38_VC_TILE_LOCAL;
static uint32_t vc_act_step[Q38_VC_GU_MAX] Q38_VC_SMALL;
// Per-engine agent cycles (probe only): waiting for its group's weight DMA,
// for its own worker, for the cluster-wide down hand-off, starting and
// running jobs, and waiting at the VC barrier.
enum {
  VCP_WEIGHTS,
  VCP_WORKER,
  VCP_HANDOFF,
  VCP_START,
  VCP_RUN,
  VCP_BARRIER,
  VCP_COUNT
};
#if Q38_SCHEDULE_PROBE && Q38_VC_PROFILE
static uint32_t vc_prof[VCP_COUNT][NUM_REDMULE_TILES] Q38_VC_SMALL;
#define VCP_NOW() mempool_get_timer()
#define VCP_ADD(k, t0) (vc_prof[k][engine] += mempool_get_timer() - (t0))
#else
#define VCP_NOW() 0u
#define VCP_ADD(k, t0) ((void)(k), (void)(t0))
#endif
extern uint32_t __heap_end;

static void vc_step(uint32_t t, q38_vc_step_t *st) {
  st->b = t / vc_state.block_steps;
  uint32_t j = t - st->b * vc_state.block_steps;
  const uint32_t gu = vc_state.gu_steps * 2u * Q38_FFN_GU_SLICES;
  st->down = j >= gu;
  if (!st->down) {
    st->n = j / (2u * Q38_FFN_GU_SLICES);
    j -= st->n * 2u * Q38_FFN_GU_SLICES;
    st->m = j / Q38_FFN_GU_SLICES;
    st->s = j - st->m * Q38_FFN_GU_SLICES;
  } else {
    j -= gu;
    st->n = j / Q38_FFN_DOWN_SLICES;
    st->m = 0u;
    st->s = j - st->n * Q38_FFN_DOWN_SLICES;
  }
}

// Panels of VC v in a step: Q38_VC_PV, fewer at a short tail, or zero.
static uint32_t vc_panels(uint32_t v, const q38_vc_step_t *st) {
  const uint32_t columns = st->down ? Q38_D / NT : Q38_I / NT;
  const uint32_t first =
      (st->down ? vc_dn_first[st->n] : vc_gu_first[st->n]) + v * Q38_VC_PV;
  return first >= columns ? 0u : Q38_MIN(Q38_VC_PV, columns - first);
}

static uint32_t vc_slot_offset(uint32_t t) {
  return vc_buf.vc_offset + (t & 1u) * Q38_VC_SLOT_BYTES;
}

// This group's stripe of job-step t's weights into its VC slot: one
// VC_ROW_BYTES chunk per bank row (a VC-local row is its four groups'
// stripes, contiguous in L2), row after row, panel after panel. Called by
// the group's coordinator only. Returns 0 when the VC has no panel.
static dma_id_t vc_load(uint32_t v, uint32_t t) {
  q38_vc_step_t st;
  vc_step(t, &st);
  const uint32_t panels = vc_panels(v, &st);
  if (panels == 0u)
    return 0u;
  const __fp16 *src;
  uint32_t piece, stride;
  if (st.down) {
    src = w_down +
          ((vc_dn_first[st.n] + v * Q38_VC_PV) * Q38_I + st.s * Q38_KC) * NT;
    piece = Q38_KC * NT * sizeof(__fp16);
    stride = Q38_I * NT * sizeof(__fp16);
  } else {
    src = (st.m ? w_up : w_gate) +
          ((vc_gu_first[st.n] + v * Q38_VC_PV) * Q38_D + st.s * Q38_GUK) * NT;
    piece = Q38_GUK * NT * sizeof(__fp16);
    stride = Q38_D * NT * sizeof(__fp16);
  }
  if ((uint32_t)src % VC_ROW_BYTES != 0u)
    q38_fatal("[qwen38_fatal] vc weights misaligned step=", t);
  const uint32_t group = mempool_get_group_id();
  const uint32_t rows = piece / Q38_VC_ROW;
  char *const dest =
      (char *)(vc_slot_offset(t) / Q38_VC_ROW * VC_GLOBAL_ROW_BYTES +
               group * VC_ROW_BYTES);
  return dma_local_memcpy_3d_nonblocking(
      0u, dest, (const char *)src + (group - v * Q38_VC_GROUPS) * VC_ROW_BYTES,
      VC_ROW_BYTES, VC_GLOBAL_ROW_BYTES, Q38_VC_ROW, rows, VC_GLOBAL_ROW_BYTES,
      stride - (rows - 1u) * Q38_VC_ROW, panels);
}

// Steps and buffers (core 0). The FFN owns the phase area above the common
// statics: the two VC slots on whole global rows (each VC uses its own
// stripe), G and the Z tiles.
static __attribute__((noinline)) void vc_setup(void) {
  const uint32_t columns = Q38_I / NT, out_columns = Q38_D / NT;
  uint32_t n = 0u;
  for (uint32_t first = 0u; first < columns; ++n) {
    vc_gu_first[n] = (uint16_t)first;
    vc_gu_pairs[n] = (uint16_t)(Q38_MIN(Q38_VC_P, columns - first) * Q38_VC_RT);
    vc_act_step[n] = 0u;
    first = next_panel_group(first, Q38_VC_P, columns, Q38_VC_P);
  }
  vc_state.gu_steps = n;
  for (uint32_t s = 0u; s < Q38_FFN_DOWN_SLICES; ++s) {
    vc_state.need[s] = 0u;
    for (uint32_t i = 0u; i < n; ++i)
      if (vc_gu_first[i] * NT < (s + 1u) * Q38_KC)
        vc_state.need[s] = (uint16_t)i;
  }
  n = 0u;
  for (uint32_t first = 0u; first < out_columns; ++n) {
    vc_dn_first[n] = (uint16_t)first;
    first = next_panel_group(first, Q38_VC_P, out_columns, Q38_VC_P);
  }
  vc_state.dn_steps = n;
  vc_state.block_steps = vc_state.gu_steps * 2u * Q38_FFN_GU_SLICES +
                         vc_state.dn_steps * Q38_FFN_DOWN_SLICES;
  vc_state.steps = vc_state.block_steps * Q38_VC_BLOCKS;
  vc_state.rows = Q38_FFN_ROWS;
  uint32_t p = (uint32_t)&stage < (uint32_t)&weights[0][0]
                   ? (uint32_t)&stage
                   : (uint32_t)&weights[0][0];
  p = (p + VC_GLOBAL_ROW_BYTES - 1u) / VC_GLOBAL_ROW_BYTES *
      VC_GLOBAL_ROW_BYTES;
  vc_buf.vc_offset = VC_GLOBAL_ROW(p) * Q38_VC_ROW;
  p += 2u * Q38_VC_SLOT_BYTES / Q38_VC_ROW * VC_GLOBAL_ROW_BYTES;
  vc_buf.gate = (__fp16 *)p;
  p += Q38_FFN_ROWS * Q38_I * sizeof(__fp16);
  p = (p + VC_GLOBAL_ROW_BYTES - 1u) / VC_GLOBAL_ROW_BYTES *
      VC_GLOBAL_ROW_BYTES;
  vc_buf.z = vc_group_window(p);
  const uint32_t z_rows =
      (NUM_TILES_PER_GROUP * 4u * Q38_PROJECTION_Z_SLOT * sizeof(__fp16) +
       VC_ROW_BYTES - 1u) /
      VC_ROW_BYTES;
  p += z_rows * VC_GLOBAL_ROW_BYTES;
  if (p > (uint32_t)&__heap_end)
    q38_fatal("[qwen38_fatal] vc ffn layout bytes over L1=",
              p - (uint32_t)&__heap_end);
  for (uint32_t e = 0u; e < NUM_REDMULE_TILES; ++e)
    vc_progress[e].pair_done = vc_progress[e].act_done =
        vc_progress[e].dnz_done = vc_progress[e].dn_done = 0u;
#if Q38_SCHEDULE_PROBE
#if Q38_VC_PROFILE
  for (uint32_t k = 0u; k < VCP_COUNT; ++k)
    for (uint32_t e = 0u; e < NUM_REDMULE_TILES; ++e)
      vc_prof[k][e] = 0u;
#endif
  probe_gateup_groups += vc_state.gu_steps * Q38_VC_BLOCKS;
  probe_projection_groups += vc_state.dn_steps * Q38_VC_BLOCKS;
  probe_ffn_blocks += Q38_VC_BLOCKS;
#endif
}

// Whether this engine may start job-step t: the Z tile it writes is free
// and, for down, the G columns of its K slice are activated cluster-wide.
static uint32_t vc_ready(uint32_t e, const q38_vc_step_t *st, uint32_t *why) {
  *why = VCP_WORKER;
  if (!st->down) {
    // Pair gp overwrites pair gp - 2's Z tiles, and a block's first pairs
    // the previous block's down tiles.
    const uint32_t gp = st->b * vc_state.gu_steps + st->n;
    if (gp >= 2u &&
        __atomic_load_n(&vc_progress[e].act_done, __ATOMIC_ACQUIRE) < gp - 1u)
      return 0u;
    return st->b == 0u || st->n >= 2u ||
           __atomic_load_n(&vc_progress[e].dn_done, __ATOMIC_ACQUIRE) >=
               st->b * vc_state.dn_steps;
  }
  // Down tiles reuse the gate Z tiles: this engine's own pairs of the block
  // must all be activated, and down tile bd - 2 added to the residual.
  const uint32_t bd = st->b * vc_state.dn_steps + st->n;
  if (__atomic_load_n(&vc_progress[e].act_done, __ATOMIC_ACQUIRE) <
      (st->b + 1u) * vc_state.gu_steps)
    return 0u;
  if (st->s == 0u && bd >= 2u &&
      __atomic_load_n(&vc_progress[e].dn_done, __ATOMIC_ACQUIRE) < bd - 1u)
    return 0u;
  *why = VCP_HANDOFF;
  for (uint32_t i = 0u; i <= vc_state.need[st->s]; ++i)
    if (__atomic_load_n(&vc_act_step[i], __ATOMIC_ACQUIRE) <
        (st->b + 1u) * vc_gu_pairs[i])
      return 0u;
  return 1u;
}

// Program job-step t into a free RedMulE context. Nothing is read or written
// until the context is triggered, so this runs while the previous job does,
// even before the step's weights have landed.
static uint8_t vc_configure(uint32_t t, const q38_vc_step_t *st,
                            const __fp16 *in, __fp16 *z, uint32_t r,
                            uint32_t q) {
  const uint32_t rows = vc_state.rows;
  const __fp16 *const slot =
      (const __fp16 *)(VC_WINDOW_BASE_1 + vc_slot_offset(t));
  const __fp16 *x, *w;
  __fp16 *out;
  uint32_t k;
  if (st->down) {
    const uint32_t bd = st->b * vc_state.dn_steps + st->n;
    x = vc_buf.gate + st->s * rows * Q38_KC + r * Q38_PROJECTION_ROWS * Q38_KC;
    w = slot + q * Q38_KC * NT;
    out = z + (bd & 1u) * Q38_PROJECTION_Z_SLOT;
    k = Q38_KC;
  } else {
    const uint32_t gp = st->b * vc_state.gu_steps + st->n;
    x = in + st->s * rows * Q38_GUK + r * Q38_PROJECTION_ROWS * Q38_GUK;
    w = slot + q * Q38_GUK * NT;
    out = z + (2u * st->m + (gp & 1u)) * Q38_PROJECTION_Z_SLOT;
    k = Q38_GUK;
  }
  const int32_t target = tnc_redmule_acquire_target();
  // A first K slice overwrites Z (MATMUL); later ones add Z (Y = Z, GEMM).
  redmule_cfg((uint32_t)x, (uint32_t)w, (uint32_t)out,
              (uint16_t)Q38_PROJECTION_ROWS, (uint16_t)k, (uint16_t)NT, 0,
              st->s == 0u ? MATMUL : GEMM, Float16);
  redmule_x_shift_set(0);
  return (uint8_t)target;
}

// Tile agent: every job-step of block b on this tile's RedMulE, and for a
// group coordinator its group's share of the VC's weight loads. Job-step
// t + 1 is programmed while job t runs and only triggered after the buffer
// switch; the issuer watches its load of t + 1 during job t too, so both
// are usually ready the moment job t ends.
static NOSTACK void vc_agent(uint32_t b, const __fp16 *in) {
  const uint32_t engine = mempool_get_redmule_id();
  if (engine >= NUM_REDMULE_TILES) {
    fsync_subscribe_self(0u);
    return;
  }
  const uint32_t v = engine / Q38_VC_ENGINES, ev = engine % Q38_VC_ENGINES;
  const uint32_t r = ev / Q38_VC_PV, q = ev % Q38_VC_PV;
  const uint32_t issuer = mempool_is_group_coordinator();
  __fp16 *const z =
      vc_buf.z + engine % NUM_TILES_PER_GROUP * 4u * Q38_PROJECTION_Z_SLOT;
  vc_subscribe(FSYNC_EVT_FSYNC | FSYNC_EVT_ENGINE |
               (issuer ? FSYNC_EVT_DMA : 0u));
  // The load of the next job-step is in flight on the issuer from the
  // previous switch (or started here for the very first one).
  static dma_id_t vc_pending[NUM_GROUPS] Q38_VC_SMALL;
  dma_id_t *const pending = &vc_pending[mempool_get_group_id()];
  uint32_t t0 = VCP_NOW();
  if (b == 0u) {
    if (issuer) {
      const dma_id_t first = vc_load(v, 0u);
      while (first && !dma_local_done_id(0u, first))
        vc_sleep(1u, 0u);
      VCP_ADD(VCP_WEIGHTS, t0);
      t0 = VCP_NOW();
    }
    vc_barrier();
    VCP_ADD(VCP_BARRIER, t0);
  }
  // Step whose load the issuer submits right after its next trigger, so its
  // own engine never waits behind the descriptor writes.
  uint32_t load_t = b == 0u && issuer && vc_state.steps > 1u ? 1u : ~0u;
  const uint32_t first = b * vc_state.block_steps;
  const uint32_t last = first + vc_state.block_steps;
  q38_vc_step_t st;
  vc_step(first, &st);
  uint8_t next =
      q < vc_panels(v, &st) ? vc_configure(first, &st, in, z, r, q) : 0u;
  for (uint32_t t = first; t < last; ++t) {
    vc_step(t, &st);
    const uint32_t mine = q < vc_panels(v, &st);
    const uint8_t target = next;
    t0 = VCP_NOW();
    if (mine) {
      uint32_t why;
      while (!vc_ready(engine, &st, &why)) {
#if Q38_VC_POLL
        vc_watch(7u + why, t);
#endif
        mempool_wait(32u);
        VCP_ADD(why, t0);
        t0 = VCP_NOW();
      }
      tnc_redmule_trigger();
    }
    if (load_t != ~0u) {
      *pending = vc_load(v, load_t);
      load_t = ~0u;
    }
    if (t + 1u < last) {
      q38_vc_step_t after;
      vc_step(t + 1u, &after);
      if (q < vc_panels(v, &after))
        next = vc_configure(t + 1u, &after, in, z, r, q);
    }
    VCP_ADD(VCP_START, t0);
#if Q38_VC_DMA_STRESS
    // Debug: keep an L1 -> L2 DMA (this block's input rows) running under
    // the job, the pattern that aborts the GVSoC bank interconnect.
    if (issuer && mine) {
      static dma_id_t vc_stress[NUM_GROUPS] Q38_VC_SMALL;
      const uint32_t g = mempool_get_group_id();
      if (vc_stress[g] == 0u || dma_local_done_id(0u, vc_stress[g]))
        vc_stress[g] = dma_coordinator_submit_linear(
                           0u, result_output + Q38_EXEC_START * Q38_D, in,
                           vc_state.rows * Q38_D * sizeof(__fp16))
                           .id;
    }
#endif
    // While the job runs, the issuer also watches its load of step t + 1.
    uint32_t landed = !issuer || *pending == 0u;
    t0 = VCP_NOW();
    if (mine) {
      while (!tnc_redmule_done(target)) {
        if (!landed)
          landed = dma_local_done_id(0u, *pending);
        vc_sleep(3u, t);
      }
      VCP_ADD(VCP_RUN, t0);
      if (!st.down && st.m == 1u && st.s + 1u == Q38_FFN_GU_SLICES)
        __atomic_store_n(&vc_progress[engine].pair_done,
                         st.b * vc_state.gu_steps + st.n + 1u,
                         __ATOMIC_RELEASE);
      if (st.down && st.s + 1u == Q38_FFN_DOWN_SLICES)
        __atomic_store_n(&vc_progress[engine].dnz_done,
                         st.b * vc_state.dn_steps + st.n + 1u,
                         __ATOMIC_RELEASE);
    }
    if (t + 1u >= vc_state.steps)
      break;
    // Buffer switch: slot t % 2 is free once every engine finished, slot
    // (t + 1) % 2 full once every issuer's stripe landed.
    t0 = VCP_NOW();
    while (!landed) {
      landed = dma_local_done_id(0u, *pending);
      if (!landed)
        vc_sleep(4u, t);
    }
    VCP_ADD(VCP_WEIGHTS, t0);
    t0 = VCP_NOW();
    vc_barrier();
    if (issuer) {
      *pending = 0u;
      if (t + 2u < vc_state.steps)
        load_t = t + 2u;
      // At the block end no trigger follows soon: load now.
      if (load_t != ~0u && t + 1u == last) {
        *pending = vc_load(v, load_t);
        load_t = ~0u;
      }
    }
    VCP_ADD(VCP_BARRIER, t0);
  }
  // GVSoC workaround (see av_drain in q38_attn_vc.c): the next block's first
  // load must land before the block-end L1 -> L2 output store starts.
  if (issuer)
    while (*pending && !dma_local_done_id(0u, *pending))
      mempool_wait(16u);
  fsync_subscribe_self(0u);
}

// Tile worker: activate this tile's pairs into G, then add the residual to
// its finished down tiles, in the order its agent produces them.
static NOSTACK void vc_worker(uint32_t b, __fp16 *residual) {
  // The engine of this core's tile (mempool_get_redmule_id's numbering).
  const uint32_t engine = mempool_get_group_id() * NUM_TILES_PER_GROUP +
                          mempool_get_tile_id() % NUM_TILES_PER_GROUP;
  const uint32_t v = engine / Q38_VC_ENGINES, ev = engine % Q38_VC_ENGINES;
  const uint32_t r = ev / Q38_VC_PV, q = ev % Q38_VC_PV;
  const uint32_t rows = vc_state.rows, row = r * Q38_PROJECTION_ROWS;
  const __fp16 *const z =
      vc_buf.z + engine % NUM_TILES_PER_GROUP * 4u * Q38_PROJECTION_Z_SLOT;
  const uint32_t gu_columns = Q38_I / NT, out_columns = Q38_D / NT;
  vc_subscribe(FSYNC_EVT_ENGINE);
  for (uint32_t n = 0u; n < vc_state.gu_steps; ++n) {
    const uint32_t gp = b * vc_state.gu_steps + n;
    const uint32_t panel = vc_gu_first[n] + v * Q38_VC_PV + q;
    if (panel < gu_columns) {
      vc_worker_wait(&vc_progress[engine].pair_done, gp + 1u);
      const __fp16 *const gate = z + (gp & 1u) * Q38_PROJECTION_Z_SLOT;
      const __fp16 *const up = z + (2u + (gp & 1u)) * Q38_PROJECTION_Z_SLOT;
      for (uint32_t i = 0u; i < Q38_PROJECTION_ROWS; ++i)
        q38_activation16(
            Q38_VC_GLOBAL(gate + i * NT), Q38_VC_GLOBAL(up + i * NT),
            ffn_gate_at(vc_buf.gate, rows, row + i, panel * NT), NT, 1);
      __sync_synchronize();
      __atomic_add_fetch(&vc_act_step[n], 1u, __ATOMIC_ACQ_REL);
    }
    // Kept monotone also where this engine has no pair (short final step).
    __atomic_store_n(&vc_progress[engine].act_done, gp + 1u, __ATOMIC_RELEASE);
  }
  for (uint32_t d = 0u; d < vc_state.dn_steps; ++d) {
    const uint32_t bd = b * vc_state.dn_steps + d;
    const uint32_t panel = vc_dn_first[d] + v * Q38_VC_PV + q;
    if (panel < out_columns) {
      vc_worker_wait(&vc_progress[engine].dnz_done, bd + 1u);
      const __fp16 *const tile = z + (bd & 1u) * Q38_PROJECTION_Z_SLOT;
      for (uint32_t i = 0u; i < Q38_PROJECTION_ROWS; ++i)
        llm_residual_add_f16(residual + (row + i) * Q38_D + panel * NT,
                             Q38_VC_GLOBAL(tile + i * NT), NT, 0u, 1u);
      __sync_synchronize();
    }
    __atomic_store_n(&vc_progress[engine].dn_done, bd + 1u, __ATOMIC_RELEASE);
  }
  fsync_subscribe_self(0u);
}

#if Q38_VC_POLL
#define VC_MARK(tag, step) vc_watch((tag), (step))
#else
#define VC_MARK(tag, step) ((void)0)
#endif

static NOSTACK void ffn(void) {
  VC_MARK(0x10u, 0u);
  if (mempool_get_core_id() == 0u)
    vc_setup();
  sync_all();
  for (uint32_t b = 0u; b < Q38_VC_BLOCKS; ++b) {
    const uint32_t first = Q38_EXEC_START + b * Q38_FFN_ROWS;
    VC_MARK(0x11u, b);
    transfer(Q38_FFN_RESIDUAL, result_mixer + first * Q38_D,
             Q38_FFN_ROWS * Q38_D * 2);
    VC_MARK(0x12u, b);
    norm_rows_sliced(Q38_FFN_RESIDUAL, w_ffn_norm, Q38_FFN_A, Q38_FFN_ROWS,
                     Q38_FFN_GAMMA);
    VC_MARK(0x13u, b);
    if (tnc_is_tile_agent())
      vc_agent(b, Q38_FFN_A);
    else if (q38_ffn_worker())
      vc_worker(b, Q38_FFN_RESIDUAL);
    else
      fsync_subscribe_self(0u);
    VC_MARK(0x14u, b);
    sync_all();
    fsync_subscribe_self(FSYNC_EVT_FSYNC);
    VC_MARK(0x15u, b);
    transfer(result_output + first * Q38_D, Q38_FFN_RESIDUAL,
             Q38_FFN_ROWS * Q38_D * 2);
  }
  VC_MARK(0x16u, 0u);
}
#elif Q38_FFN_PIPE
// The whole FFN as one dataflow sequence of steps over every row block: per
// block, every K slice of every gate/up panel group, then every K slice of
// every down panel group. Each step offers one job per engine, with a fixed
// task -> engine mapping so an engine keeps its Z tile across a group's K
// slices. Steps alternate between two full-wave weight slots. No barrier
// between steps: an engine starts step n once its weights landed (and, for
// down, once the gate/up columns of that K slice are activated); the last
// engine to finish step n frees its slot and publishes step n + 2, so each
// load has a step of slack.
// Nor between blocks, except for the write-back. Residual rows ping-pong
// between two buffers: down copy-outs add into the block's residual in place;
// the last of them queues its write-back to L2, which the next block's first
// step waits for (an iDMA read of L1 under RedMulE traffic aborts the GVSoC
// bank interconnect, and Spatz stores to L2 are far slower); then the block
// after next is loaded into the freed buffer. Tile workers activate gate/up
// columns and, once a block's gate/up input is dead, normalize the next
// block's residual into it.
// Q38_FP_SLOTS weight slots: step n + Q38_FP_SLOTS is published once step n
// retired, so a load has Q38_FP_SLOTS - 1 steps of slack.
#define Q38_FP_GU_STEP (2u * Q38_GU_PANELS * Q38_GUK * NT)
#define Q38_FP_STEP Q38_MAX(Q38_FP_GU_STEP, Q38_DOWN_SLOT)
#define Q38_FP_GU_GROUPS ((Q38_I / NT + Q38_GU_PANELS - 1u) / Q38_GU_PANELS)
#define Q38_FP_DN_GROUPS                                                       \
  ((Q38_D / NT + Q38_FFN_DOWN_PANELS - 1u) / Q38_FFN_DOWN_PANELS)
#define Q38_FP_PER_BLOCK                                                       \
  (Q38_FP_GU_GROUPS * Q38_FFN_GU_SLICES +                                      \
   Q38_FP_DN_GROUPS * Q38_FFN_DOWN_SLICES)
#define Q38_FP_BLOCKS ((Q38_EXEC_END - Q38_EXEC_START) / Q38_FFN_ROWS)
// Queue epochs per step: gate and up, or down.
#define Q38_FP_PIECES 2u
#ifndef Q38_FP_SLOTS
#define Q38_FP_SLOTS 2u
#endif
// Per-step counters, recycled every Q38_FP_RING steps.
#define Q38_FP_RING 8u
#define Q38_FP_NONE 0xffffu
_Static_assert(Q38_FFN_ROWS / Q38_PROJECTION_ROWS * 2u * Q38_GU_PANELS <=
                       NUM_REDMULE_TILES &&
                   Q38_FFN_ROWS / Q38_PROJECTION_ROWS * Q38_FFN_DOWN_PANELS <=
                       NUM_REDMULE_TILES &&
                   Q38_FP_SLOTS >= 2u && 2u * Q38_FP_SLOTS <= Q38_FP_RING &&
                   (Q38_FP_SLOTS + 1u) * Q38_FP_PIECES <= Q38_DMA_QUEUE_SLOTS,
               "FFN pipeline: one job per engine, DMA queue");
// The residual queue never recycles a slot.
_Static_assert((Q38_EXEC_END - Q38_EXEC_START) % Q38_FFN_ROWS == 0 &&
                   Q38_FP_BLOCKS <= Q38_DMA_QUEUE_SLOTS,
               "FFN pipeline: whole blocks, residual queue");

// The FFN's own layout, carved from the phase area at the start of the FFN:
// two wave slots, the intermediate, two up panel buffers and the odd blocks'
// residual rows, each on whole L1 rows.
#define Q38_FP_ROW_ELEMS (NUM_BANKS * sizeof(uint32_t) / sizeof(__fp16))
#define Q38_FP_ROUND(n)                                                        \
  (((n) + Q38_FP_ROW_ELEMS - 1u) / Q38_FP_ROW_ELEMS * Q38_FP_ROW_ELEMS)
#define Q38_FP_UP (Q38_FFN_ROWS * Q38_GU_PANELS * NT)
// Small pipeline state, without LOCAL's whole-row alignment.
#define Q38_FP_SMALL __attribute__((section(".l1"), aligned(8)))
static struct { __fp16 *slots, *gate, *up[2], *residual; } fp_buf Q38_FP_SMALL;
extern uint32_t __heap_end;

typedef struct {
  uint16_t down, slice, first, count, group, pieces, need, epoch;
} q38_fp_step_t;
// One block's steps; global step n is local step n % per_block of block
// n / per_block. Step n's weights are published once step n - 2 retired, so
// step n + 2's publisher recycles step n - 2's retire counter. An engine may
// poll step n before its publication: the ready slot holds the step number
// (plus one), never a stale flag.
static q38_fp_step_t fp_step[Q38_FP_PER_BLOCK] Q38_FP_SMALL;
static uint32_t fp_done[Q38_FP_RING] Q38_FP_SMALL;
static uint32_t fp_ready[Q38_FP_RING] Q38_FP_SMALL;
// Monotone over blocks: block b is complete at (b + 1) times the per-block
// count.
static uint32_t fp_copied[Q38_FP_GU_GROUPS] Q38_FP_SMALL;
static uint32_t fp_act[Q38_FP_GU_GROUPS] Q38_FP_SMALL;
// Retired down steps of the even and odd blocks.
static uint32_t fp_dn_done[2] Q38_FP_SMALL;
static uint16_t fp_first[Q38_FP_GU_GROUPS + Q38_FP_DN_GROUPS] Q38_FP_SMALL;
static uint16_t fp_count[Q38_FP_GU_GROUPS + Q38_FP_DN_GROUPS] Q38_FP_SMALL;
// Residual traffic, in epoch order: load block 1, write back block 0, load
// block 2, write back block 1, ... (fp_out_epoch; epoch 2k - 1 loads k).
static q38_dma_stage_queue_t fp_io_queue Q38_FP_SMALL;
static struct {
  uint32_t steps, per_block, blocks, gu_groups, dn_groups, rows, epoch0,
      block_epochs, normed, dn_steps, dn_tiles, dn_copied, loaded;
} fp_state Q38_FP_SMALL;

// Debug: a pipeline wait spinning 60000 times dumps every core's wait state
// and ends the run.
#ifndef Q38_FP_WATCHDOG
#define Q38_FP_WATCHDOG 0
#endif
#if Q38_FP_WATCHDOG
static uint32_t fp_dbg[NUM_CORES][4] Q38_FP_SMALL;
static uint32_t fp_dbg_owner Q38_FP_SMALL;
static NOSTACK void fp_watch(uint32_t *spins, uint32_t tag, uint32_t a,
                             uint32_t b, uint32_t c) {
  const uint32_t core = mempool_get_core_id();
  fp_dbg[core][0] = tag;
  fp_dbg[core][1] = a;
  fp_dbg[core][2] = b;
  fp_dbg[core][3] = c;
  if (++*spins < 60000u)
    return;
  if (__atomic_fetch_add(&fp_dbg_owner, 1u, __ATOMIC_ACQ_REL) != 0u)
    while (1)
      mempool_wait(1000u);
  mempool_wait(20000u);
  for (uint32_t i = 0u; i < NUM_CORES; ++i) {
    report_fatal("[fp_dog] core=", i);
    for (uint32_t j = 0u; j < 4u; ++j)
      report_fatal("[fp_dog]   v=", fp_dbg[i][j]);
  }
  for (uint32_t g = 0u; g < NUM_GROUPS; ++g) {
    report_fatal("[fp_dog] group=", g);
    report_fatal("[fp_dog]   w_issue=", gateup_dma_queue.next_issue[g]);
    report_fatal("[fp_dog]   w_done=", gateup_dma_queue.next_complete[g]);
    report_fatal("[fp_dog]   io_issue=", fp_io_queue.next_issue[g]);
    report_fatal("[fp_dog]   io_done=", fp_io_queue.next_complete[g]);
  }
  report_fatal("[fp_dog] normed=", fp_state.normed);
  report_fatal("[fp_dog] epoch0=", fp_state.epoch0);
  report_fatal("[fp_dog] per_block=", fp_state.per_block);
  q38_fatal("[fp_dog] stuck core=", core);
}
#define FP_WATCH(...) fp_watch(__VA_ARGS__)
#else
#define FP_WATCH(...) ((void)0)
#endif

// Per-engine cycles of the agent loop (probe only): waiting for weights, for
// activated columns, for the normalized input, copying out, starting jobs,
// running them, retiring.
enum {
  FPP_WEIGHTS,
  FPP_NEED,
  FPP_NORM,
  FPP_COPY,
  FPP_START,
  FPP_RUN,
  FPP_RETIRE,
  FPP_COUNT
};
#if Q38_SCHEDULE_PROBE
static uint32_t fp_prof[FPP_COUNT][NUM_REDMULE_TILES] Q38_FP_SMALL;
#define FPP_ADD(k, t0) (fp_prof[k][engine] += mempool_get_timer() - (t0))
#else
#define FPP_ADD(k, t0) ((void)(k), (void)(t0))
#endif

static __fp16 *fp_residual(uint32_t b) {
  return (b & 1u) ? fp_buf.residual : Q38_FFN_RESIDUAL;
}

static void fp_layout(void) {
  // Both phase-area objects and all of L1 above them are free in the FFN.
  __fp16 *p = (uint32_t)&stage < (uint32_t)&weights[0][0] ? (__fp16 *)&stage
                                                          : &weights[0][0];
  fp_buf.slots = p;
  p += Q38_FP_ROUND(Q38_FP_SLOTS * Q38_FP_STEP);
  fp_buf.gate = p;
  p += Q38_FP_ROUND(Q38_FFN_ROWS * Q38_I);
  for (uint32_t i = 0u; i < 2u; ++i) {
    fp_buf.up[i] = p;
    p += Q38_FP_ROUND(Q38_FP_UP);
  }
  fp_buf.residual = p;
  p += Q38_FP_ROUND(Q38_FFN_ROWS * Q38_D);
  if ((uint32_t)p > (uint32_t)&__heap_end)
    q38_fatal("[qwen38_fatal] ffn layout bytes over L1=",
              (uint32_t)p - (uint32_t)&__heap_end);
}

static uint32_t fp_jobs(const q38_fp_step_t *st) {
  const uint32_t tiles = fp_state.rows / Q38_PROJECTION_ROWS;
  return st->down ? tiles * st->count : 2u * tiles * st->count;
}

static __fp16 *fp_slot(uint32_t n) {
  return fp_buf.slots + n % Q38_FP_SLOTS * Q38_FP_STEP;
}

static uint32_t fp_epoch(uint32_t n) {
  const uint32_t b = n / fp_state.per_block;
  return fp_state.epoch0 + b * fp_state.block_epochs +
         fp_step[n - b * fp_state.per_block].epoch;
}

static void fp_idle(void) {
  if (mempool_is_group_coordinator()) {
    (void)q38_dma_stage_progress(&gateup_dma_queue);
    (void)q38_dma_stage_progress(&fp_io_queue);
  }
  mempool_wait(32u);
}

static void fp_publish(uint32_t n) {
  const q38_fp_step_t *const st = &fp_step[n % fp_state.per_block];
  __fp16 *const slot = fp_slot(n);
  const uint32_t epoch = fp_epoch(n);
  fp_done[n % Q38_FP_RING] = 0u;
  __sync_synchronize();
  // One strided epoch per matrix: the step's panel slices are `count` pieces
  // one panel apart in L2 and back to back in the slot.
  for (uint32_t m = 0u; m < st->pieces; ++m) {
    uint32_t ok;
    if (st->down)
      ok = q38_dma_stage_publish_pieces(
          &gateup_dma_queue, epoch, slot,
          w_down + (st->first * Q38_I + st->slice * Q38_KC) * NT,
          Q38_KC * NT * sizeof(__fp16), Q38_I * NT * sizeof(__fp16), st->count,
          0u);
    else
      ok = q38_dma_stage_publish_pieces(
          &gateup_dma_queue, epoch + m, slot + m * st->count * Q38_GUK * NT,
          (m ? w_up : w_gate) + (st->first * Q38_D + st->slice * Q38_GUK) * NT,
          Q38_GUK * NT * sizeof(__fp16), Q38_D * NT * sizeof(__fp16), st->count,
          0u);
    if (ok != Q38_DMA_QUEUE_OK)
      __builtin_trap();
  }
}

static uint32_t fp_weights_ready(uint32_t n) {
  if (__atomic_load_n(&fp_ready[n % Q38_FP_RING], __ATOMIC_ACQUIRE) == n + 1u)
    return 1u;
  const q38_fp_step_t *const st = &fp_step[n % fp_state.per_block];
  const uint32_t status =
      q38_dma_stage_ready(&gateup_dma_queue, fp_epoch(n) + st->pieces - 1u);
  if (status != Q38_DMA_QUEUE_OK && status != Q38_DMA_QUEUE_PENDING)
    __builtin_trap();
  if (status == Q38_DMA_QUEUE_OK) {
#if Q38_FP_CHECK
    // The staged step equals its L2 source at each piece's ends and middle.
    __sync_synchronize();
    const __fp16 *const slot = fp_slot(n);
    const uint32_t piece = st->down ? Q38_KC * NT : Q38_GUK * NT;
    for (uint32_t m = 0u; m < (st->down ? 1u : 2u); ++m)
      for (uint32_t p = 0u; p < st->count; ++p) {
        const uint16_t *const dst =
            (const uint16_t *)(slot + (m * st->count + p) * piece);
        const uint16_t *const src =
            st->down ? (const uint16_t *)(w_down + ((st->first + p) * Q38_I +
                                                    st->slice * Q38_KC) *
                                                       NT)
                     : (const uint16_t *)((m ? w_up : w_gate) +
                                          ((st->first + p) * Q38_D +
                                           st->slice * Q38_GUK) *
                                              NT);
        const uint32_t at[3] = {0u, piece / 2u + 1u, piece - 1u};
        for (uint32_t i = 0u; i < 3u; ++i)
          if (dst[at[i]] != src[at[i]])
            q38_fatal("[qwen38_fatal] ffn pipe staged step=", n);
      }
#endif
    __atomic_store_n(&fp_ready[n % Q38_FP_RING], n + 1u, __ATOMIC_RELEASE);
  }
  return status == Q38_DMA_QUEUE_OK;
}

// Queue block b's residual rows into their buffer, if block b exists.
static void fp_load(uint32_t b) {
  if (b < fp_state.blocks &&
      q38_dma_stage_publish_any(
          &fp_io_queue, 2u * b - 1u, fp_residual(b),
          result_mixer + (Q38_EXEC_START + b * Q38_FFN_ROWS) * Q38_D,
          fp_state.rows * Q38_D * sizeof(__fp16), 0u) != Q38_DMA_QUEUE_OK)
    __builtin_trap();
}

static NOSTACK void fp_retire(uint32_t n) {
  const q38_fp_step_t *const st = &fp_step[n % fp_state.per_block];
  if (__atomic_add_fetch(&fp_done[n % Q38_FP_RING], 1u, __ATOMIC_ACQ_REL) !=
      fp_jobs(st))
    return;
  const uint32_t epoch = fp_epoch(n);
  for (uint32_t p = 0u; p < st->pieces; ++p)
    if (q38_dma_stage_release_any(&gateup_dma_queue, epoch + p) !=
        Q38_DMA_QUEUE_OK)
      __builtin_trap();
  if (st->down)
    __atomic_add_fetch(&fp_dn_done[(n / fp_state.per_block) & 1u], 1u,
                       __ATOMIC_ACQ_REL);
  if (n + Q38_FP_SLOTS < fp_state.steps)
    fp_publish(n + Q38_FP_SLOTS);
}

// Queue epoch of block b's write-back: after the load of block b + 1, if any.
static uint32_t fp_out_epoch(uint32_t b) {
  return b + 1u < fp_state.blocks ? 2u * b + 2u : 2u * b + 1u;
}

// Block b's write-back completed. The first engine to see it queues the
// load of block b + 2 into the freed buffer.
static uint32_t fp_written(uint32_t b) {
  const uint32_t status = q38_dma_stage_ready(&fp_io_queue, fp_out_epoch(b));
  if (status == Q38_DMA_QUEUE_PENDING)
    return 0u;
  if (status != Q38_DMA_QUEUE_OK)
    __builtin_trap();
  uint32_t expected = b + 1u;
  if (__atomic_compare_exchange_n(&fp_state.loaded, &expected, b + 2u, 0,
                                  __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
    fp_load(b + 2u);
  return 1u;
}

// Block b's gate/up copy-outs overwrite the intermediate and up panels, so
// they wait for block b - 1's down steps (with two or more K slices, the
// first of them already follows block b - 1's last step).
static uint32_t fp_copy_ok(uint32_t n) {
  const uint32_t b = n / fp_state.per_block;
  if (b == 0u || fp_step[n - b * fp_state.per_block].down)
    return 1u;
  return __atomic_load_n(&fp_dn_done[(b - 1u) & 1u], __ATOMIC_ACQUIRE) >=
         ((b - 1u) / 2u + 1u) * fp_state.dn_steps;
}

// Copy a finished tile out of Z (after the group's last K slice).
static NOSTACK void fp_copy_out(uint32_t n, uint32_t task, const __fp16 *tile,
                                __fp16 *gate) {
  const uint32_t b = n / fp_state.per_block;
  const q38_fp_step_t *const st = &fp_step[n - b * fp_state.per_block];
  const uint32_t rows = fp_state.rows, count = st->count;
  if (st->down) {
    // Fused residual addition, in place; the block's last tile queues the
    // block's write-back.
    const uint32_t row = task / count * Q38_PROJECTION_ROWS;
    __fp16 *const out =
        fp_residual(b) + row * Q38_D + (st->first + task % count) * NT;
    for (uint32_t r = 0; r < Q38_PROJECTION_ROWS; ++r)
      llm_residual_add_f16(out + r * Q38_D, tile + r * NT, NT, 0u, 1u);
    __sync_synchronize();
    if (__atomic_add_fetch(&fp_state.dn_copied, 1u, __ATOMIC_ACQ_REL) ==
            (b + 1u) * fp_state.dn_tiles &&
        q38_dma_stage_publish_any(
            &fp_io_queue, fp_out_epoch(b),
            result_output + (Q38_EXEC_START + b * Q38_FFN_ROWS) * Q38_D,
            fp_residual(b), fp_state.rows * Q38_D * sizeof(__fp16),
            0u) != Q38_DMA_QUEUE_OK)
      __builtin_trap();
    return;
  }
  const uint32_t half = rows / Q38_PROJECTION_ROWS * count;
  const uint32_t is_up = task >= half, local = task - is_up * half;
  const uint32_t row = local / count * Q38_PROJECTION_ROWS;
  const uint32_t column = local % count;
  // The up panel buffer of group g - 2 must have been activated (of the
  // previous block for g < 2, which its down steps already waited for).
  // A plain wait: the DMA queue is driven from fp_agent's own waits, and
  // polling it here would nest the submit path one frame deeper (stack gate).
  if (is_up && st->group >= 2u)
    for (uint32_t spins = 0u;
         __atomic_load_n(&fp_act[st->group - 2u], __ATOMIC_ACQUIRE) <
         (b + 1u) * Q38_FFN_WORKERS;) {
      FP_WATCH(&spins, 3u, n, fp_act[st->group - 2u], 0u);
      mempool_wait(32u);
    }
  for (uint32_t r = 0; r < Q38_PROJECTION_ROWS; ++r)
    q38_copy16(
        is_up ? fp_buf.up[st->group & 1u] + (row + r) * count * NT + column * NT
              : ffn_gate_at(gate, rows, row + r, (st->first + column) * NT),
        tile + r * NT, NT);
  __sync_synchronize();
  __atomic_add_fetch(&fp_copied[st->group], 1u, __ATOMIC_ACQ_REL);
}

static NOSTACK void fp_agent(const __fp16 *in, __fp16 *gate) {
  const uint32_t engine = mempool_get_redmule_id();
  if (engine >= NUM_REDMULE_TILES)
    return;
  const uint32_t rows = fp_state.rows, per_block = fp_state.per_block;
  __fp16 *const z = ffn_buffers.z + engine * 2 * Q38_PROJECTION_Z_SLOT;
  uint32_t pending = Q38_FP_NONE, pending_zs = 0u;
  for (uint32_t n = 0u; n < fp_state.steps; ++n) {
    const uint32_t b = n / per_block, l = n - b * per_block;
    const q38_fp_step_t *const st = &fp_step[l];
    const uint32_t task = engine;
    if (task >= fp_jobs(st))
      continue;
    const uint32_t zs = (st->group + (st->down ? fp_state.gu_groups : 0u)) & 1u;
    uint32_t spins = 0u;
    (void)spins;
    // Copy out a pending tile before blocking: activation (and so a later
    // down step) may be waiting for it. A block's first step also waits for
    // its normalized input.
    for (;;) {
      uint32_t t0 = mempool_get_timer(), why;
      if (!fp_weights_ready(n))
        why = FPP_WEIGHTS;
      else if (st->need != Q38_FP_NONE &&
               __atomic_load_n(&fp_act[st->need], __ATOMIC_ACQUIRE) <
                   (b + 1u) * Q38_FFN_WORKERS)
        why = FPP_NEED;
      else if (l == 0u && (__atomic_load_n(&fp_state.normed, __ATOMIC_ACQUIRE) <
                               b * Q38_FFN_WORKERS ||
                           (b > 0u && !fp_written(b - 1u))))
        why = FPP_NORM;
      else
        break;
      FP_WATCH(&spins, 1u, n,
               ((uint32_t)st->need << 16) |
                   (st->need != Q38_FP_NONE ? fp_act[st->need] & 0xffffu : 0u),
               ((fp_ready[n % Q38_FP_RING] == n + 1u) << 31) | fp_state.normed);
      if (pending != Q38_FP_NONE && fp_copy_ok(pending)) {
        FPP_ADD(why, t0);
        t0 = mempool_get_timer();
        fp_copy_out(pending, task, z + pending_zs * Q38_PROJECTION_Z_SLOT,
                    gate);
        pending = Q38_FP_NONE;
        FPP_ADD(FPP_COPY, t0);
        t0 = mempool_get_timer();
      }
      fp_idle();
      FPP_ADD(why, t0);
    }
    uint32_t t0 = mempool_get_timer();
    if (pending != Q38_FP_NONE && pending_zs == zs) {
      while (!fp_copy_ok(pending))
        fp_idle();
      fp_copy_out(pending, task, z + pending_zs * Q38_PROJECTION_Z_SLOT, gate);
      pending = Q38_FP_NONE;
    }
    FPP_ADD(FPP_COPY, t0);
    t0 = mempool_get_timer();
    const __fp16 *const slot = fp_slot(n);
    __fp16 *const tile = z + zs * Q38_PROJECTION_Z_SLOT;
    uint8_t target;
    if (st->down) {
      target = start_job_y(gate + st->slice * rows * Q38_KC +
                               task / st->count * Q38_PROJECTION_ROWS * Q38_KC,
                           slot + task % st->count * Q38_KC * NT, tile,
                           Q38_PROJECTION_ROWS, Q38_KC, NT, st->slice == 0u);
    } else {
      const uint32_t half = rows / Q38_PROJECTION_ROWS * st->count;
      const uint32_t is_up = task >= half, local = task - is_up * half;
      target = start_job_y(
          in + st->slice * rows * Q38_GUK +
              local / st->count * Q38_PROJECTION_ROWS * Q38_GUK,
          slot + (is_up * st->count + local % st->count) * Q38_GUK * NT, tile,
          Q38_PROJECTION_ROWS, Q38_GUK, NT, st->slice == 0u);
    }
    FPP_ADD(FPP_START, t0);
    t0 = mempool_get_timer();
    if (pending != Q38_FP_NONE) {
      while (!fp_copy_ok(pending))
        fp_idle();
      fp_copy_out(pending, task, z + pending_zs * Q38_PROJECTION_Z_SLOT, gate);
      pending = Q38_FP_NONE;
    }
    FPP_ADD(FPP_COPY, t0);
    t0 = mempool_get_timer();
    while (!tnc_redmule_done(target)) {
      FP_WATCH(&spins, 2u, n, 0u, 0u);
      fp_idle();
    }
    FPP_ADD(FPP_RUN, t0);
    t0 = mempool_get_timer();
    fp_retire(n);
    FPP_ADD(FPP_RETIRE, t0);
    if (st->slice + 1u ==
        (st->down ? Q38_FFN_DOWN_SLICES : Q38_FFN_GU_SLICES)) {
      pending = n;
      pending_zs = zs;
    }
  }
  if (pending != Q38_FP_NONE) {
    while (!fp_copy_ok(pending))
      fp_idle();
    fp_copy_out(pending, engine, z + pending_zs * Q38_PROJECTION_Z_SLOT, gate);
  }
}

static NOSTACK void fp_worker(__fp16 *gate) {
  const uint32_t tiles = fp_state.rows / Q38_PROJECTION_ROWS;
  const uint32_t core = mempool_get_core_id();
  for (uint32_t b = 0u; b < fp_state.blocks; ++b) {
    const uint32_t next = b + 1u < fp_state.blocks;
    for (uint32_t g = 0u; g < fp_state.gu_groups; ++g) {
      uint32_t spins = 0u;
      (void)spins;
      while (__atomic_load_n(&fp_copied[g], __ATOMIC_ACQUIRE) <
             (b + 1u) * 2u * tiles * fp_count[g]) {
        FP_WATCH(&spins, 4u, b, g, fp_copied[g]);
        mempool_wait(32u);
      }
      activate_prior_columns(gate, fp_buf.up[g & 1u], fp_state.rows,
                             fp_first[g], fp_count[g]);
      __sync_synchronize();
      __atomic_add_fetch(&fp_act[g], 1u, __ATOMIC_ACQ_REL);
    }
    if (!next)
      break;
    // Every gate/up job of block b copied out, so its input is dead.
    uint32_t spins = 0u;
    (void)spins;
    for (;;) {
      const uint32_t status = q38_dma_stage_ready(&fp_io_queue, 2u * b + 1u);
      if (status == Q38_DMA_QUEUE_OK)
        break;
      FP_WATCH(&spins, 5u, b, status, 0u);
      if (status != Q38_DMA_QUEUE_PENDING)
        __builtin_trap();
      mempool_wait(32u);
    }
    __fp16 *const residual = fp_residual(b + 1u);
    for (uint32_t row = core / NUM_CORES_PER_TILE; row < fp_state.rows;
         row += Q38_FFN_WORKERS)
      q38_norm16_sliced(residual + row * Q38_D, Q38_FFN_GAMMA, Q38_FFN_A, Q38_D,
                        row, fp_state.rows);
    __sync_synchronize();
    __atomic_add_fetch(&fp_state.normed, 1u, __ATOMIC_ACQ_REL);
  }
}

// Step tables and the first loads (core 0).
static __attribute__((noinline)) void fp_setup(void) {
  {
    uint32_t n = 0u, epoch = 0u, g = 0u;
    const uint32_t columns = Q38_I / NT;
    for (uint32_t first = 0u; first < columns; ++g) {
      const uint32_t count = Q38_MIN(Q38_GU_PANELS, columns - first);
      fp_first[g] = (uint16_t)first;
      fp_count[g] = (uint16_t)count;
      fp_copied[g] = 0u;
      fp_act[g] = 0u;
      for (uint32_t s = 0u; s < Q38_FFN_GU_SLICES; ++s, ++n) {
        fp_step[n] = (q38_fp_step_t){
            0u,          (uint16_t)s, (uint16_t)first, (uint16_t)count,
            (uint16_t)g, 2u,          Q38_FP_NONE,     (uint16_t)epoch};
        epoch += 2u;
      }
      first = next_panel_group(first, count, columns, Q38_GU_PANELS);
    }
    uint32_t d = 0u;
    const uint32_t out_columns = Q38_D / NT;
    for (uint32_t first = 0u; first < out_columns; ++d) {
      const uint32_t count = Q38_MIN(Q38_FFN_DOWN_PANELS, out_columns - first);
      fp_first[g + d] = (uint16_t)first;
      fp_count[g + d] = (uint16_t)count;
      for (uint32_t s = 0u; s < Q38_FFN_DOWN_SLICES; ++s, ++n) {
        // Last gate/up group whose columns reach into K slice s.
        uint32_t need = 0u;
        for (uint32_t c = 0u; c < g; ++c)
          if (fp_first[c] * NT < (s + 1u) * Q38_KC)
            need = c;
        fp_step[n] = (q38_fp_step_t){
            1u,          (uint16_t)s, (uint16_t)first, (uint16_t)count,
            (uint16_t)d, 1u,          (uint16_t)need,  (uint16_t)epoch};
        epoch += 1u;
      }
      first = next_panel_group(first, count, out_columns, Q38_FFN_DOWN_PANELS);
    }
    fp_state.gu_groups = g;
    fp_state.dn_groups = d;
    fp_state.per_block = n;
    fp_state.blocks = Q38_FP_BLOCKS;
    fp_state.steps = n * Q38_FP_BLOCKS;
    fp_state.rows = Q38_FFN_ROWS;
    fp_state.epoch0 = gateup_dma_next[0];
    fp_state.block_epochs = epoch;
    fp_state.normed = 0u;
    fp_state.dn_steps = d * Q38_FFN_DOWN_SLICES;
    fp_dn_done[0] = fp_dn_done[1] = 0u;
#if Q38_SCHEDULE_PROBE
    for (uint32_t k = 0u; k < FPP_COUNT; ++k)
      for (uint32_t e = 0u; e < NUM_REDMULE_TILES; ++e)
        fp_prof[k][e] = 0u;
#endif
    for (uint32_t i = 0u; i < Q38_FP_RING; ++i)
      fp_ready[i] = 0u;
#if Q38_FP_WATCHDOG
    fp_dbg_owner = 0u;
    for (uint32_t i = 0u; i < NUM_CORES; ++i)
      fp_dbg[i][0] = 0u;
#endif
    q38_dma_stage_init(&fp_io_queue);
#if Q38_SCHEDULE_PROBE
    probe_gateup_groups += g * Q38_FP_BLOCKS;
    probe_projection_groups += d * Q38_FP_BLOCKS;
    probe_ffn_blocks += Q38_FP_BLOCKS;
#endif
    fp_state.dn_tiles = Q38_FFN_ROWS / Q38_PROJECTION_ROWS * (Q38_D / NT);
    fp_state.dn_copied = 0u;
    fp_state.loaded = 1u;
    fp_load(1u);
    for (uint32_t i = 0u; i < Q38_FP_SLOTS && i < fp_state.steps; ++i)
      fp_publish(i);
  }
}

// Runs the FFN of every block into result_output; block 0 must be
// normalized into `in`.
static NOSTACK void ffn_pipeline(const __fp16 *in, __fp16 *gate) {
  const uint32_t core = mempool_get_core_id();
  if (core == 0u)
    fp_setup();
  sync_all();
  if (tnc_is_tile_agent())
    fp_agent(in, gate);
  else if (q38_ffn_worker())
    fp_worker(gate);
  // Later steps' weights may still be in flight for other engines.
  const uint32_t epoch_end = fp_epoch(fp_state.steps);
  uint32_t spins = 0u;
  (void)spins;
  FP_WATCH(&spins, 7u, 0u, 0u, 0u);
  if (mempool_is_group_coordinator())
    while (
        q38_dma_stage_ready(&gateup_dma_queue, epoch_end - 1u) ==
            Q38_DMA_QUEUE_PENDING ||
        q38_dma_stage_ready(&fp_io_queue, fp_out_epoch(fp_state.blocks - 1u)) ==
            Q38_DMA_QUEUE_PENDING) {
      FP_WATCH(&spins, 6u, epoch_end, 0u, 0u);
      fp_idle();
    }
  sync_all();
  gateup_dma_next[core] = epoch_end;
  sync_all();
}

static NOSTACK void ffn(void) {
  if (mempool_get_core_id() == 0u)
    fp_layout();
  sync_all();
  __fp16 *const first_residual = fp_residual(0u);
  transfer(first_residual, result_mixer + Q38_EXEC_START * Q38_D,
           Q38_FFN_ROWS * Q38_D * 2);
  norm_rows_sliced(first_residual, w_ffn_norm, Q38_FFN_A, Q38_FFN_ROWS,
                   Q38_FFN_GAMMA);
  ffn_pipeline(Q38_FFN_A, fp_buf.gate);
}
#else
static NOSTACK void ffn(void) {
  // Token rows are independent in the FFN. Reuse intermediate buffers only
  // after down-projection completion, preserving the complete layer result.
  __fp16 *const gate = stage.ffn.gate;
#if !(Q38_GATEUP_PIPELINE && Q38_ASYNC)
  __fp16 *const up = stage.ffn.up;
#endif
  for (uint32_t first = Q38_EXEC_START; first < Q38_EXEC_END;
       first += Q38_FFN_ROWS) {
    const uint32_t rows = block_rows(first, Q38_FFN_ROWS);
    transfer(Q38_FFN_RESIDUAL, result_mixer + first * Q38_D, rows * Q38_D * 2);
#if Q38_FFN_NORM_STREAM
#if Q38_SCHEDULE_PROBE
    if (first < Q38_MIXER_EXEC_START || first + rows > Q38_MIXER_EXEC_END)
      norm_rows(Q38_FFN_RESIDUAL, w_ffn_norm, Q38_FFN_A, rows, Q38_D, 0,
                Q38_FFN_GAMMA);
    else
#endif
      transfer(Q38_FFN_A, result_output + first * Q38_D, rows * Q38_D * 2);
#else
#if Q38_FFN_GU_SLICES > 1
    norm_rows_sliced(Q38_FFN_RESIDUAL, w_ffn_norm, Q38_FFN_A, rows,
                     Q38_FFN_GAMMA);
#else
    norm_rows(Q38_FFN_RESIDUAL, w_ffn_norm, Q38_FFN_A, rows, Q38_D, 0,
              Q38_FFN_GAMMA);
#endif
#endif
#if Q38_GATEUP_PIPELINE && Q38_ASYNC
#if Q38_FFN_GU_SLICES > 1
    const uint32_t idle_slot = ffn_gateup_ksliced(Q38_FFN_A, gate, rows);
#elif Q38_FFN_DATAFLOW
    const uint32_t idle_slot = ffn_gateup_dataflow(Q38_FFN_A, gate, rows);
#else
    const uint32_t idle_slot = ffn_gateup_pairs(Q38_FFN_A, gate, rows);
#endif
    (void)idle_slot;
    __fp16 *const down_out = Q38_FFN_A;
#else
    project_rows(Q38_FFN_A, w_gate, gate, Q38_D, Q38_I, w_up, Q38_D, Q38_I,
                 rows, &ffn_buffers);
    project_rows(Q38_FFN_A, w_up, up, Q38_D, Q38_I, w_down, Q38_I, Q38_D, rows,
                 &ffn_buffers);
    const uint32_t per_core = rows * Q38_I / NUM_CORES;
    const uint32_t offset = mempool_get_core_id() * per_core;
    q38_activation16(gate + offset, up + offset, gate + offset, per_core, 1);
    sync_all();
    __fp16 *const down_out = mixed;
#endif
#if Q38_FFN_DOWN_SLICES > 1
    ffn_down_sliced(gate, down_out, rows, idle_slot);
#else
#if Q38_FFN_WIDE
    ffn_down_wide(gate, down_out, rows);
#else
    project_rows(gate, w_down, down_out, Q38_I, Q38_D, 0, 0, 0, rows,
                 &ffn_buffers);
#endif
    llm_residual_add_f16(down_out, Q38_FFN_RESIDUAL, rows * Q38_D,
                         mempool_get_core_id(), NUM_CORES);
    sync_all();
#endif
    transfer(result_output + first * Q38_D, down_out, rows * Q38_D * 2);
#if Q38_SCHEDULE_PROBE
    if (mempool_get_core_id() == 0u)
      ++probe_ffn_blocks;
#endif
  }
}
#endif
