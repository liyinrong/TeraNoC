// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Included by q38_attn_vc.c. One KV head and two Q heads per level-1 VC;
// owner operands remain in each group's level-0 window.
#if Q38_ATTN_VC_SWEEP
#define AVS_TILE Q38_AV_SWEEP_TILE
#define AVS_ROWS Q38_AV_SWEEP_ROWS
#define AVS_TILES (AVS_ROWS / AVS_TILE)
#define AVS_OWNERS (2u * AVS_TILES / Q38_VC_GROUPS)
#define AVS_GROUP_ROWS (AVS_OWNERS * AVS_TILE)
#define AVS_REPLICAS (Q38_VC_COUNT / Q38_KVH)
#define AVS_BATCHES                                                            \
  ((Q38_QH / Q38_KVH + 2u * AVS_REPLICAS - 1u) / (2u * AVS_REPLICAS))
#define AVS_KV_BYTES (2u * Q38_HD * BK * sizeof(__fp16))
// One K or V matrix of a key block; the VC holds AVS_KV_MATRICES of them.
#define AVS_MAT_ELEMS (Q38_HD * BK)
#if Q38_AV_SWEEP_OVERLAP
// The stream keeps K and V apart: K(n) is dead after QK(n), V(n) after PV(n).
// Three slots of each keep the two-job prefetch lead of four K+V slots. Three
// query banks: window w+2 loads while w finishes and w+1 runs.
#ifndef AVS_KSLOTS
#define AVS_KSLOTS 3u
#endif
#ifndef AVS_VSLOTS
#define AVS_VSLOTS 3u
#endif
#define AVS_KV_MATRICES (AVS_KSLOTS + AVS_VSLOTS)
#define AVS_QBANKS 3u
// Score slots: QK(n) may run once PV(n - SSLOTS), the slot's last reader, is
// queued ahead of it.
#ifndef AVS_SSLOTS
#define AVS_SSLOTS 2u
#endif
// 1: a ready PV goes to RedMulE before a ready QK (default: QK first).
#ifndef Q38_AV_SWEEP_PV_FIRST
#define Q38_AV_SWEEP_PV_FIRST 0
#endif
#else
#define AVS_KV_MATRICES 8u
#define AVS_QBANKS 2u
#define AVS_SSLOTS 2u
#endif
#define AVS_STATE_SLOTS (Q38_SLICE_BYTES / sizeof(qwen38_softmax_state_t))
#define AVS_ALPHA_SLOTS (Q38_SLICE_BYTES / sizeof(float))
#define AVS_STATE_COUNT                                                        \
  ((AVS_TILE + AVS_STATE_SLOTS - 1u) / AVS_STATE_SLOTS * NUM_TILES_PER_GROUP * \
   AVS_STATE_SLOTS)
#define AVS_ALPHA_COUNT                                                        \
  ((AVS_TILE + AVS_ALPHA_SLOTS - 1u) / AVS_ALPHA_SLOTS * NUM_TILES_PER_GROUP * \
   AVS_ALPHA_SLOTS)
_Static_assert(Q38_ATTN_FUSED_SOFTMAX &&Q38_ATTN_RM_ACCUM &&
                   !Q38_ATTN_RM_FOLD && !Q38_ATTN_TILE_LOCAL &&
                   !Q38_ATTN_ISOLATE && Q38_VC_COUNT % Q38_KVH == 0u &&
                   AVS_ROWS % AVS_TILE == 0u && AVS_TILE >= REDMULE_H &&
                   AVS_TILE <= UINT16_MAX && AVS_TILE % REDMULE_H == 0u &&
                   AVS_OWNERS == NUM_TILES_PER_GROUP &&
                   AVS_TILES % Q38_VC_GROUPS == 0u && Q38_R % BK == 0u &&
                   AVS_TILE * Q38_HD * 2u % VC_ROW_BYTES == 0u &&
                   BK * 2u <= VC_ROW_BYTES && Q38_BK % 128u == 0u,
               "KV-affine sweep needs one whole aligned owner per engine");

#if Q38_AV_SWEEP_OVERLAP
// 1 (with FIN_SPREAD): window w-1's finish also takes a share after the first
// block of window w+1, so three windows hold Z and softmax state at once.
#ifndef Q38_AV_SWEEP_FIN_SPILL
#define Q38_AV_SWEEP_FIN_SPILL 0
#endif
// Two windows are live at once (one finishing, one running): softmax state and
// Z per window parity `ws`, passed down as an argument. (A per-core selector in
// L1 is not safe: the model can return a reload that precedes the store.)
#define AVS_WBUFS (Q38_AV_SWEEP_FIN_SPILL ? 3u : 2u)
#else
#define AVS_WBUFS 1u
#endif

typedef struct {
  __fp16 q[AVS_QBANKS][AVS_GROUP_ROWS * Q38_HD] Q38_ROW_ALIGNED;
  __fp16 gate[AVS_QBANKS][AVS_GROUP_ROWS * Q38_HD] Q38_ROW_ALIGNED;
#if !Q38_AV_SWEEP_TILE_WIN
  __fp16 z[AVS_GROUP_ROWS * Q38_HD] Q38_ROW_ALIGNED;
#endif
#if !Q38_AV_SWEEP_REUSE_QUERY
  __fp16 gated[AVS_GROUP_ROWS * Q38_HD] Q38_ROW_ALIGNED;
#endif
#if !Q38_AV_SWEEP_TILE_WIN
  __fp16 score[AVS_SSLOTS][AVS_GROUP_ROWS * BK] Q38_ROW_ALIGNED;
#endif
  float alpha[AVS_ALPHA_COUNT] Q38_ROW_ALIGNED;
  qwen38_softmax_state_t state[AVS_WBUFS][AVS_STATE_COUNT] Q38_ROW_ALIGNED;
} q38_avs_local_t;
#define AVS_STATE(ws, i)                                                       \
  Q38_LA(avs_local->state[ws] + avs_scalar_index(i, AVS_STATE_SLOTS))

#if Q38_AV_SWEEP_TILE_WIN
// Intermediates (scores/probabilities, the Z accumulator) live in the tile
// window: one owner per tile, re-placed in the requester's own tile banks.
// Q/gate and the exported results stay in the group window, so the iDMA
// descriptors do not change. The window is a model extension (2026-10-04).
#define AVS_TILE_ROW_BYTES (NUM_BANKS_PER_TILE * 4u)
#define AVS_TILE_WINDOW_BASE VC_REGION_END
typedef struct {
  __fp16 score[AVS_SSLOTS][AVS_TILE * BK]
      __attribute__((aligned(AVS_TILE_ROW_BYTES)));
  __fp16 z[AVS_WBUFS][AVS_TILE * Q38_HD]
      __attribute__((aligned(AVS_TILE_ROW_BYTES)));
} q38_avs_tile_t;
_Static_assert(sizeof(q38_avs_tile_t) % AVS_TILE_ROW_BYTES == 0u &&
                   sizeof(q38_avs_tile_t) <=
                       VC_GROUP_BYTES / NUM_TILES_PER_GROUP &&
                   Q38_AV_SWEEP_MODE != 0u,
               "Tile-window intermediates need tile-owned rows");
static q38_avs_tile_t *avs_tile Q38_VC_SMALL;
#define AVS_SCORE(slot, i) (avs_tile->score[slot] + (i) % AVS_TILE * BK)
#define AVS_Z(ws, i) (avs_tile->z[ws] + (i) % AVS_TILE * Q38_HD)
#define AVS_TL(ptr) (ptr)
#else
#define AVS_SCORE(slot, i) (avs_local->score[slot] + (i)*BK)
#define AVS_Z(ws, i) ((void)(ws), avs_local->z + (i)*Q38_HD)
#define AVS_TL(ptr) Q38_LA(ptr)
#endif

typedef struct {
  uint32_t arrived, released;
#if Q38_AV_SWEEP_ASYNC
  // Posted by the tile agent per key block: job targets (AVS_JOB_ISSUED set
  // when a job exists), then the count of posted blocks. Only core 0 reaches
  // RedMulE's registers, so the agent relays completions as done counts.
  uint32_t qk_posted, pv_posted, qk_done, pv_done;
  uint32_t qk_target[4], pv_target[4];
  uint32_t padding[NUM_BANKS_PER_TILE - 14u];
#else
  uint32_t padding[NUM_BANKS_PER_TILE - 2u];
#endif
} q38_avs_tile_sync_t;
static q38_avs_tile_sync_t avs_sync[NUM_REDMULE_TILES] Q38_VC_TILE_LOCAL;
#if Q38_AV_SWEEP_ASYNC
// Blocks each core of a tile has completed (softmax, rescale), one word per
// core. PV(n) reads the probabilities of every core's rows, so it waits for
// the slowest core: a summed count lets fast cores stand in for a late one.
typedef struct {
  uint32_t done[NUM_CORES_PER_TILE];
  uint32_t padding[NUM_BANKS_PER_TILE - NUM_CORES_PER_TILE];
} q38_avs_lanes_t;
static q38_avs_lanes_t avs_lanes[NUM_REDMULE_TILES] Q38_VC_TILE_LOCAL;
#endif
static q38_avs_local_t *avs_local Q38_VC_SMALL;
static __fp16 *avs_kv Q38_VC_SMALL;
#if Q38_AV_SWEEP_FAST
// Softmax constants, one copy in each tile's own banks (a single shared copy
// is a bank hotspot: every core reloads them per row).
typedef struct {
  avs_fast_k_t k;
  uint32_t padding[NUM_BANKS_PER_TILE - sizeof(avs_fast_k_t) / 4u];
} avs_fast_kc_t;
static avs_fast_kc_t avs_fast_kc[NUM_REDMULE_TILES] Q38_VC_TILE_LOCAL;
// 1: blocks with causally masked rows also take the fast rows (mode 4).
#ifndef Q38_AV_SWEEP_FAST_MASK
#define Q38_AV_SWEEP_FAST_MASK 0
#endif

#if Q38_AV_SWEEP_FAST_MASK
// Element indices of one fast-row piece, per tile (FAST_MASK). Loaded, not
// vid.v: that encoding decodes as XpulpIMG pv.sll.c and writes sp.
typedef struct {
  int16_t idx[NUM_BANKS_PER_TILE * 2u];
} avs_mask_idx_t;
static avs_mask_idx_t avs_mask_idx[NUM_REDMULE_TILES] Q38_VC_TILE_LOCAL;
_Static_assert(AVS_FAST_PIECE <= NUM_BANKS_PER_TILE * 2u,
               "One index row covers a fast-row piece");
#endif
#endif
static uint32_t avs_layout_end Q38_VC_SMALL;
#if Q38_AV_PROFILE
enum { AVS_RM, AVS_SM, AVS_RESCALE, AVS_DMA, AVS_SYNC, AVS_FINISH, AVS_PARTS };
typedef struct {
  uint32_t cycles[NUM_CORES_PER_TILE][AVS_PARTS];
  uint32_t rows[NUM_CORES_PER_TILE], jobs;
  uint32_t
      padding[NUM_BANKS_PER_TILE - NUM_CORES_PER_TILE * (AVS_PARTS + 1u) - 1u];
} q38_avs_profile_t;
static q38_avs_profile_t avs_stats[NUM_REDMULE_TILES] Q38_VC_TILE_LOCAL;
static struct {
  uint32_t cycles[NUM_CORES_PER_TILE];
  uint32_t padding[NUM_BANKS_PER_TILE - NUM_CORES_PER_TILE];
} avs_wait_stats[NUM_REDMULE_TILES] Q38_VC_TILE_LOCAL;

#define AVS_ADD(part, start)                                                   \
  (avs_stats[mempool_get_core_id() / NUM_CORES_PER_TILE]                       \
       .cycles[mempool_get_core_id() % NUM_CORES_PER_TILE][part] +=            \
   mempool_get_timer() - (start))
#else
#define AVS_ADD(part, start) ((void)(start))
#endif

// 1 (needs SCHED_CORE): the 1 + 3 tile's event discipline. No read-back
// drains: a scalar store, the counter and raise stores included, already waits
// for the core's vector memory operations, and a `fence` covers them before an
// AMO. Compute cores sleep on EVT_CORE, which the scheduler raises after every
// relay and release; machine-wide joins use the epoch-checked fractal_sync
// barrier instead of mempool_barrier.
#ifndef Q38_AV_SWEEP_EVT
#define Q38_AV_SWEEP_EVT 0
#endif
// Read-back drains: only without EVT.
#define AVS_RB (!Q38_AV_SWEEP_EVT)
// Cores 1.. of the tile: the compute cores of a 1 + 3 tile.
#define AVS_COMPUTE_MASK (((1u << NUM_CORES_PER_TILE) - 1u) & ~1u)

// EVT: order the core's vector memory operations before an AMO.
static inline void avs_fence(void) {
#if Q38_AV_SWEEP_EVT
  asm volatile("fence" ::: "memory");
#endif
}

// EVT: wake the tile's compute cores once the stores before have landed.
static inline void avs_wake(void) {
#if Q38_AV_SWEEP_EVT
  asm volatile("fence" ::: "memory");
  fsync_raise(AVS_COMPUTE_MASK);
#endif
}

// A compute core's wait step: sleep until the next scheduler event, or poll.
static inline void avs_idle(void) {
#if Q38_AV_SWEEP_EVT
  mempool_wfi();
#else
  mempool_wait(16u);
#endif
}

// Cores rendezvous in their own tile banks. Group/global scopes additionally
// aggregate tile agents with fractal_sync; compute handoffs may stay
// tile-local.
static void avs_scope_sync(uint32_t *epoch, uint32_t scope) {
  const uint32_t engine = mempool_get_group_id() * NUM_TILES_PER_GROUP +
                          mempool_get_tile_id() % NUM_TILES_PER_GROUP;
  q38_avs_tile_sync_t *const s = avs_sync + engine;
  const uint32_t n = ++*epoch;
  const uint32_t start = mempool_get_timer();
  avs_fence();
  __atomic_fetch_add(&s->arrived, 1u, __ATOMIC_ACQ_REL);
  if (tnc_is_tile_agent()) {
    while (__atomic_load_n(&s->arrived, __ATOMIC_ACQUIRE) <
           n * NUM_CORES_PER_TILE)
      mempool_wait(16u);
    if (scope != 2u) {
      fsync_trigger(FSYNC_ID_H,
                    scope ? fsync_aggr_global() : fsync_aggr_of_vc_level(0u));
      while (fsync_status() & FSYNC_STATUS_BUSY_MASK)
        mempool_wfi();
    }
    __atomic_store_n(&s->released, n, __ATOMIC_RELEASE);
    avs_wake();
  } else {
    while (__atomic_load_n(&s->released, __ATOMIC_ACQUIRE) < n)
      avs_idle();
  }
  __sync_synchronize();
  AVS_ADD(AVS_SYNC, start);
}

static void avs_group_sync(uint32_t *epoch) { avs_scope_sync(epoch, 0u); }

static void avs_compute_sync(uint32_t *epoch) {
  avs_scope_sync(epoch, Q38_AV_SWEEP_MODE ? 2u : 0u);
}

// 1: core 0 of each tile only schedules (stream only); the other cores split
// the tile's rows, r mod (cores - 1). A fixed split: a row may not change core
// between blocks, since cores can be a block apart.
#ifndef Q38_AV_SWEEP_SCHED_CORE
#define Q38_AV_SWEEP_SCHED_CORE 0
#endif
// 1: QK(0) starts on window 0's queries and K(0); gates and V(0) follow.
#ifndef Q38_AV_SWEEP_EARLY_START
#define Q38_AV_SWEEP_EARLY_START 0
#endif
// k > 0 (with EARLY_START): window 0's gates load once barrier k passed,
// behind K(k) and V(k-1) (1: right behind V(0)).
#ifndef Q38_AV_SWEEP_LATE_GATE0
#define Q38_AV_SWEEP_LATE_GATE0 0
#endif
_Static_assert(!Q38_AV_SWEEP_LATE_GATE0 || Q38_AV_SWEEP_EARLY_START,
               "Late gates need the early start");
_Static_assert(!Q38_AV_SWEEP_EVT || Q38_AV_SWEEP_SCHED_CORE,
               "Core events need the scheduler core to raise them");
// Finish timing what-ifs (break the numerics; for sizing only): bit 0 no
// read-back drain, bit 1 gate read from the tile-local Z row, bit 2 output
// written to the tile-local Z row.
#ifndef Q38_AV_SWEEP_FIN_WHATIF
#define Q38_AV_SWEEP_FIN_WHATIF 0
#endif
// Scheduler-side code (tile agent, group coordinator): its own section. With
// SCHED_CORE it is all the scheduler core (rv32ima, no F/V) runs after the
// shared start-up; ctrlcheck.py rejects F/V/Zfh/Xpulp in its call closure.
#define AVS_CTRL __attribute__((noinline, section(".text.ctrl")))
// Cores that count in for a window's export: every core but the scheduler.
#define AVS_FINISHERS                                                          \
  (NUM_TILES_PER_GROUP *                                                       \
   (NUM_CORES_PER_TILE - (Q38_AV_SWEEP_SCHED_CORE ? 1u : 0u)))
// Probes: only lanes 1..SCHED_LANES compute; lane L takes the row residue
// (L - 1 + SCHED_SHIFT) mod SCHED_LANES.
#ifndef Q38_AV_SWEEP_SCHED_LANES
#define Q38_AV_SWEEP_SCHED_LANES (NUM_CORES_PER_TILE - 1u)
#endif
#ifndef Q38_AV_SWEEP_SCHED_SHIFT
#define Q38_AV_SWEEP_SCHED_SHIFT 0
#endif
_Static_assert(!Q38_AV_SWEEP_SCHED_CORE ||
                   (Q38_AV_SWEEP_OVERLAP && Q38_AV_SWEEP_MODE &&
                    NUM_CORES_PER_TILE > 1u),
               "A scheduling core needs the stream and several cores a tile");

// The core of each tile that fills its FP constants: with a scheduler core
// (rv32ima only, no F/V), the tile's first vector core.
static uint32_t avs_is_tile_filler(void) {
#if Q38_AV_SWEEP_SCHED_CORE
  return mempool_get_core_id() % NUM_CORES_PER_TILE == 1u;
#else
  return tnc_is_tile_agent();
#endif
}

static uint32_t avs_core_first(void) {
#if Q38_AV_SWEEP_SCHED_CORE
  const uint32_t lane = mempool_get_core_id() % NUM_CORES_PER_TILE;
  // Core 0 gets an empty range: it walks the stream for its scheduling only.
  return (mempool_get_core_id() / NUM_CORES_PER_TILE % NUM_TILES_PER_GROUP) *
             AVS_TILE +
         (lane && lane <= Q38_AV_SWEEP_SCHED_LANES
              ? (lane - 1u + Q38_AV_SWEEP_SCHED_SHIFT) %
                    Q38_AV_SWEEP_SCHED_LANES
              : AVS_TILE);
#elif Q38_AV_SWEEP_MODE
  return (mempool_get_core_id() / NUM_CORES_PER_TILE % NUM_TILES_PER_GROUP) *
             AVS_TILE +
         mempool_get_core_id() % NUM_CORES_PER_TILE;
#else
  return mempool_get_core_id() % Q38_LOC_CORES;
#endif
}

static uint32_t avs_core_end(void) {
#if Q38_AV_SWEEP_MODE
  return (mempool_get_core_id() / NUM_CORES_PER_TILE % NUM_TILES_PER_GROUP +
          1u) *
         AVS_TILE;
#else
  return AVS_GROUP_ROWS;
#endif
}

#if Q38_AV_SWEEP_SCHED_CORE
#define AVS_CORE_STEP Q38_AV_SWEEP_SCHED_LANES
#else
#define AVS_CORE_STEP (Q38_AV_SWEEP_MODE ? NUM_CORES_PER_TILE : Q38_LOC_CORES)
#endif

static __fp16 *avs_gated_bank(uint32_t bank) {
#if Q38_AV_SWEEP_REUSE_QUERY || Q38_AV_SWEEP_OVERLAP
  // This bank's final QK is complete; next queries occupy the other bank.
  // The stream needs it: a window's export may still read its rows while the
  // next window finishes.
  return avs_local->q[bank];
#else
  (void)bank;
  return avs_local->gated;
#endif
}

// Normalised context rows, exported by DMA. With tile-window intermediates
// they land in this bank's query rows, dead after the window's last QK.
static __fp16 *avs_ctx_base(uint32_t bank) {
#if Q38_AV_SWEEP_TILE_WIN
  return avs_local->q[bank];
#else
  (void)bank;
  return avs_local->z;
#endif
}

static uint32_t avs_head(uint32_t batch, uint32_t owner) {
  const uint32_t vc = mempool_get_group_id() / Q38_VC_GROUPS;
  return vc % Q38_KVH * (Q38_QH / Q38_KVH) +
         2u * (batch * AVS_REPLICAS + vc / Q38_KVH) +
         owner / (AVS_TILES / Q38_VC_GROUPS);
}

static uint32_t avs_row_step(void) {
#if Q38_AV_SWEEP_INTERLEAVE
  const uint32_t span = Q38_MIN(AVS_TILES, BK / AVS_TILE);
  if (span && BK % AVS_TILE == 0u && AVS_TILES % span == 0u)
    return span;
#endif
  return 1u;
}

static uint32_t avs_row_at(uint32_t owner, uint32_t row) {
  const uint32_t rank = mempool_get_group_id() % Q38_VC_GROUPS +
                        owner % (AVS_TILES / Q38_VC_GROUPS) * Q38_VC_GROUPS;
  const uint32_t span = avs_row_step();
  // Stay within an aligned key region, preserving causal owner liveness.
  return rank / span * span * AVS_TILE + rank % span + row * span;
}

static uint32_t avs_row(uint32_t owner) { return avs_row_at(owner, 0u); }

static uint32_t avs_owner_rows(uint32_t base, uint32_t owner) {
  const uint32_t first = base + avs_row(owner), step = avs_row_step();
  return first < Q38_R ? Q38_MIN(AVS_TILE, (Q38_R - first + step - 1u) / step)
                       : 0u;
}

// Each row stays with the same core through initialization, softmax and finish.
// Scalar state is in that core's tile, even when scores span several tiles.
static uint32_t avs_scalar_index(uint32_t i, uint32_t slots) {
#if Q38_AV_SWEEP_MODE
  const uint32_t tile = i / AVS_TILE, row = i % AVS_TILE;
#else
  const uint32_t tile = i % Q38_LOC_CORES / NUM_CORES_PER_TILE;
  const uint32_t row =
      i / Q38_LOC_CORES * NUM_CORES_PER_TILE + i % NUM_CORES_PER_TILE;
#endif
  return row / slots * NUM_TILES_PER_GROUP * slots + tile * slots + row % slots;
}

static uint32_t avs_exists(uint32_t base, uint32_t batch, uint32_t owner) {
  const uint32_t vc = mempool_get_group_id() / Q38_VC_GROUPS;
  const uint32_t end = (vc % Q38_KVH + 1u) * (Q38_QH / Q38_KVH);
  return avs_head(batch, owner) < end && base + avs_row(owner) < Q38_R;
}

static uint32_t avs_live(uint32_t base, uint32_t batch, uint32_t key,
                         uint32_t owner) {
  const uint32_t first = base + avs_row(owner);
  return avs_exists(base, batch, owner) && first < Q38_S &&
         first + (AVS_TILE - 1u) * avs_row_step() >= key;
}

// Scheduler-side zero fill: word stores the compiler may not vectorize (the
// group coordinator is the rv32ima scheduler core with SCHED_CORE).
static void avs_zero_ctrl(__fp16 *p, uint32_t n) {
  volatile uint32_t *const w = (volatile uint32_t *)p;
  for (uint32_t i = 0u; i < n / 2u; ++i)
    w[i] = 0u;
}

// Queries (what bit 0) and gates (bit 1) of one window, per owner.
static __attribute__((noinline)) dma_id_t
avs_query_parts(uint32_t base, uint32_t batch, uint32_t bank, uint32_t what) {
  if (!mempool_is_group_coordinator())
    return 0u;
  dma_id_t id = 0u;
  for (uint32_t j = 0; j < AVS_OWNERS; ++j) {
    if (!avs_exists(base, batch, j))
      continue;
    const uint32_t h = avs_head(batch, j), row = base + avs_row(j);
    __fp16 *const q = avs_local->q[bank] + j * AVS_TILE * Q38_HD;
    __fp16 *const g = avs_local->gate[bank] + j * AVS_TILE * Q38_HD;
    const uint32_t rows = avs_owner_rows(base, j);
    for (uint32_t r = rows; r < AVS_TILE; ++r) {
      if (what & 1u)
        avs_zero_ctrl(Q38_LA(q + r * Q38_HD), Q38_HD);
      if (what & 2u)
        avs_zero_ctrl(Q38_LA(g + r * Q38_HD), Q38_HD);
    }
    if (avs_row_step() != 1u) {
      const uint32_t lanes = VC_ROW_BYTES / (Q38_HD * 2u);
      for (uint32_t lane = 0u; lane < lanes && lane < rows; ++lane) {
        const uint32_t count = (rows - lane + lanes - 1u) / lanes;
        const uint32_t token = base + avs_row_at(j, lane);
        if (what & 1u)
          id = dma_local_memcpy_2d_nonblocking(
              0u, q38_la(q + lane * Q38_HD),
              av_q + (h * AV_QG_ROWS + token) * Q38_HD, Q38_HD * 2u,
              VC_GLOBAL_ROW_BYTES, lanes * avs_row_step() * Q38_HD * 2u, count);
        if (what & 2u)
          id = dma_local_memcpy_2d_nonblocking(
              0u, q38_la(g + lane * Q38_HD),
              av_gate + (h * AV_QG_ROWS + token) * Q38_HD, Q38_HD * 2u,
              VC_GLOBAL_ROW_BYTES, lanes * avs_row_step() * Q38_HD * 2u, count);
      }
      continue;
    }
    const uint32_t bytes = rows * Q38_HD * 2u;
    if (what & 1u)
      id = dma_local_memcpy_2d_nonblocking(
          0u, q38_la(q), av_q + (h * AV_QG_ROWS + row) * Q38_HD, VC_ROW_BYTES,
          VC_GLOBAL_ROW_BYTES, VC_ROW_BYTES, bytes / VC_ROW_BYTES);
    if (what & 2u)
      id = dma_local_memcpy_2d_nonblocking(
          0u, q38_la(g), av_gate + (h * AV_QG_ROWS + row) * Q38_HD,
          VC_ROW_BYTES, VC_GLOBAL_ROW_BYTES, VC_ROW_BYTES,
          bytes / VC_ROW_BYTES);
    const uint32_t whole = bytes / VC_ROW_BYTES * VC_ROW_BYTES;
    if (whole != bytes) {
      if (what & 1u)
        id = dma_local_memcpy_nonblocking(
            0u, q38_la(q + whole / 2u),
            av_q + (h * AV_QG_ROWS + row) * Q38_HD + whole / 2u, bytes - whole);
      if (what & 2u)
        id = dma_local_memcpy_nonblocking(
            0u, q38_la(g + whole / 2u),
            av_gate + (h * AV_QG_ROWS + row) * Q38_HD + whole / 2u,
            bytes - whole);
    }
  }
  return id;
}

static dma_id_t avs_query_load(uint32_t base, uint32_t batch, uint32_t bank) {
  return avs_query_parts(base, batch, bank, 3u);
}

static void avs_dma_wait(dma_id_t id) {
  const uint32_t start = mempool_get_timer();
  while (id && !dma_local_done_id(0u, id))
    mempool_wfi();
  __sync_synchronize();
  AVS_ADD(AVS_DMA, start);
}

// Each group's issuer loads only its stripe of this VC's K (which = 0) or V (1)
// matrix of one key block, into VC matrix m.
static dma_id_t avs_mat_load(uint32_t key, uint32_t which, uint32_t m) {
  if (!mempool_is_group_coordinator())
    return 0u;
  const uint32_t vc = mempool_get_group_id() / Q38_VC_GROUPS;
  const uint32_t kv = vc % Q38_KVH, group = mempool_get_group_id() % 4u;
  const uint32_t matrix = AVS_MAT_ELEMS * 2u;
  const __fp16 *const src = result_cache +
                            key / BK * 2u * Q38_KVH * Q38_HD * BK +
                            (which * Q38_KVH + kv) * Q38_HD * BK;
  const uint32_t offset = (uint32_t)avs_kv - VC_WINDOW_BASE_1 + m * matrix;
  const vc_rect_t rect = vc_rect(1u);
  const uint32_t dst = vc_global_addr(&rect, offset) + group * VC_ROW_BYTES;
  return dma_local_memcpy_2d_nonblocking(
      0u, (void *)dst, (const char *)src + group * VC_ROW_BYTES, VC_ROW_BYTES,
      VC_GLOBAL_ROW_BYTES, Q38_VC_ROW, matrix / Q38_VC_ROW);
}

// K and V of one block into combined slot `slot` (per-window pipeline).
static dma_id_t avs_kv_load(uint32_t key, uint32_t slot) {
  avs_mat_load(key, 0u, 2u * slot);
  return avs_mat_load(key, 1u, 2u * slot + 1u);
}

static void avs_kv_ready(dma_id_t id, uint32_t *epoch) {
  if (mempool_is_group_coordinator())
    avs_dma_wait(id);
  if (tnc_is_tile_agent())
    vc_barrier();
  avs_group_sync(epoch);
}

#if Q38_AV_SWEEP_FINISH
// One window's gated rows to L2 as a few strided descriptors per owner (group
// coordinator).
static dma_id_t avs_export(uint32_t base, uint32_t batch, uint32_t bank) {
  dma_id_t id = 0u;
  const uint32_t lanes = VC_ROW_BYTES / (Q38_HD * 2u);
  for (uint32_t j = 0; j < AVS_OWNERS; ++j) {
    if (!avs_exists(base, batch, j))
      continue;
    const uint32_t h = avs_head(batch, j), rows = avs_owner_rows(base, j);
    for (uint32_t lane = 0u; lane < lanes && lane < rows; ++lane) {
      const uint32_t token = base + avs_row_at(j, lane);
      id = dma_local_memcpy_2d_nonblocking(
          0u, AV_GATED(token, h),
          q38_la(avs_gated_bank(bank) + (j * AVS_TILE + lane) * Q38_HD),
          Q38_HD * 2u, lanes * avs_row_step() * AV_GK * 2u, VC_GLOBAL_ROW_BYTES,
          (rows - lane + lanes - 1u) / lanes);
    }
  }
  return id;
}
#endif

static uint8_t avs_job(uint32_t base, uint32_t batch, uint32_t bank,
                       uint32_t key, uint32_t mat, uint32_t slot, uint32_t pv,
                       uint32_t zb, uint32_t *issued) {
  const uint32_t j = mempool_get_redmule_id() % NUM_TILES_PER_GROUP;
  if (!avs_live(base, batch, key, j))
    return 0u;
  const __fp16 *const kv = avs_kv + mat * AVS_MAT_ELEMS;
  *issued = 1u;
#if Q38_AV_PROFILE
  ++avs_stats[mempool_get_redmule_id()].jobs;
#endif
  if (pv)
    return start_job_y(AVS_SCORE(slot, j * AVS_TILE), kv,
                       AVS_Z(zb, j * AVS_TILE), AVS_TILE, BK, Q38_HD,
                       key == 0u);
  return start_job(avs_local->q[bank] + j * AVS_TILE * Q38_HD, kv,
                   AVS_SCORE(slot, j * AVS_TILE), AVS_TILE, Q38_HD, BK);
}

static void avs_wait_job(uint32_t issued, uint8_t target) {
  const uint32_t start = mempool_get_timer();
  while (issued && !tnc_redmule_done(target))
    mempool_wfi();
#if Q38_AV_PROFILE
  avs_wait_stats[mempool_get_core_id() / NUM_CORES_PER_TILE]
      .cycles[mempool_get_core_id() % NUM_CORES_PER_TILE] +=
      mempool_get_timer() - start;
#else
  (void)start;
#endif
}

#if Q38_AV_SWEEP_ASYNC || Q38_AV_SWEEP_FINISH
// Issue loads over rows this core stored, then wait for the last: a load
// after a store to the same address returns once that store has landed.
static void avs_load16(const __fp16 *p, uint32_t n) {
  for (uint32_t off = 0u; off < n;) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(n - off));
    asm volatile("vle16.v v0, (%0)" : : "r"(p + off) : "memory");
    off += vl;
  }
}

#endif

#if Q38_AV_SWEEP_ASYNC
#define AVS_JOB_ISSUED 0x100u

static uint32_t avs_next_key(uint32_t key, uint32_t limit);

static q38_avs_tile_sync_t *avs_tile_sync(void) {
  return avs_sync + mempool_get_group_id() * NUM_TILES_PER_GROUP +
         mempool_get_tile_id() % NUM_TILES_PER_GROUP;
}

static q38_avs_lanes_t *avs_tile_lanes(void) {
  return avs_lanes + (avs_tile_sync() - avs_sync);
}

// This core has completed global block b.
static void avs_block_done(uint32_t b) {
  __atomic_store_n(
      &avs_tile_lanes()->done[mempool_get_core_id() % NUM_CORES_PER_TILE],
      b + 1u, __ATOMIC_RELEASE);
}

// Every compute core of the tile has completed global block b.
static uint32_t avs_tile_block_done(uint32_t b) {
  const q38_avs_lanes_t *const l = avs_tile_lanes();
  for (uint32_t c = Q38_AV_SWEEP_SCHED_CORE ? 1u : 0u; c < NUM_CORES_PER_TILE;
       ++c)
    if (__atomic_load_n(&l->done[c], __ATOMIC_ACQUIRE) <= b)
      return 0u;
  return 1u;
}

// Whether the RedMulE job posted for global block b has finished, given one
// read of RedMulE's completion counter (tile agent only).
static uint32_t avs_job_done(const uint32_t *posted, const uint32_t *target,
                             uint32_t b, uint8_t completed) {
  if (*posted <= b)
    return 0u;
  const uint32_t t = target[b % 4u];
  return !(t & AVS_JOB_ISSUED) || tnc_counter8_done(completed, (uint8_t)t);
}

#if Q38_AV_SWEEP_OVERLAP
// One continuous key-block stream over every window. Global block n uses KV
// slot n % 4 and score slot n % 2; window w uses query bank, Z and softmax
// state w % 2. Window w-1 finishes between blocks of window w, so neither
// RedMulE nor the KV pipeline drains at a window boundary.
static uint32_t avs_windows(void) {
  return (Q38_MIXER_EXEC_END - Q38_MIXER_EXEC_START + AVS_ROWS - 1u) /
         AVS_ROWS * AVS_BATCHES;
}

static uint32_t avs_win_base(uint32_t w) {
  return Q38_MIXER_EXEC_START + w / AVS_BATCHES * AVS_ROWS;
}

// Key blocks of window w (keys 0 .. its last row), as avs_it_t walks them.
static inline uint32_t avs_win_blocks(uint32_t w) {
  return (Q38_MIN(avs_win_base(w) + AVS_ROWS, Q38_S) - 1u) / BK + 1u;
}

// Per group, in its coordinator's tile: cores done with their finish rows
// (over all windows) and windows whose queries landed.
typedef struct {
  // Cores of the group done with window w, at w % QBANKS: a per-window count,
  // so a core a window behind cannot be covered by others a window ahead.
  uint32_t finished[AVS_QBANKS], queries;
  uint32_t gates0; // LATE_GATE0: window 0's gates landed
  uint32_t padding[NUM_BANKS_PER_TILE - 2u - AVS_QBANKS];
} q38_avs_group_sync_t;
static q38_avs_group_sync_t avs_gsync[NUM_REDMULE_TILES] Q38_VC_TILE_LOCAL;

static q38_avs_group_sync_t *avs_group_rec(void) {
  return avs_gsync + mempool_get_group_id() * NUM_TILES_PER_GROUP;
}

// A stream position: global block, window, key, the window's key limit.
typedef struct {
  uint32_t n, w, key, limit;
} avs_it_t;

static void avs_it_window(avs_it_t *it) {
  it->key = 0u;
  it->limit = Q38_MIN(avs_win_base(it->w) + AVS_ROWS, Q38_S);
}

static void avs_it_next(avs_it_t *it) {
  ++it->n;
  it->key = avs_next_key(it->key, it->limit);
  if (it->key >= it->limit) {
    ++it->w;
    avs_it_window(it);
  }
}

// The tile agent's scheduler state for the whole stream.
typedef struct {
  avs_it_t qk, pv, load;    // next QK, PV and K load
  avs_it_t vload;           // next V load
  uint32_t windows, blocks; // stream length
  uint32_t barrier, armed;  // next KV barrier block; triggered, not passed
  dma_id_t kv;              // descriptor of the last KV load
  uint32_t jobs, queries;   // jobs issued; windows seen with queries landed
  uint8_t last, before;     // targets of the two most recent jobs
  // Group coordinator: windows exported and query loads issued, and the
  // descriptors in flight.
  uint32_t exported, queried;
  dma_id_t export_id, query_id;
  // LATE_GATE0: window 0's gates, 0 not issued, 1 in flight, 2 landed.
  dma_id_t gate0_id;
  uint32_t gate0;
} avs_agent_t;
// In the agent's own tile banks, off the 512-byte stack.
static union {
  avs_agent_t a;
  uint32_t row[(sizeof(avs_agent_t) + NUM_BANKS_PER_TILE * 4u - 1u) /
               (NUM_BANKS_PER_TILE * 4u) * NUM_BANKS_PER_TILE];
} avs_agents[NUM_REDMULE_TILES] Q38_VC_TILE_LOCAL;
#else
// The tile agent's scheduler state for one window; block indices are local
// to the window, `g` is the window's first global block.
typedef struct {
  uint32_t base, batch, bank, limit, g, blocks;
  uint32_t qk, qk_key;       // next QK to issue and its key
  uint32_t pv, pv_key;       // next PV to issue and its key
  uint32_t barrier, armed;   // next KV barrier block; triggered, not passed
  uint32_t loaded, load_key; // next block the coordinator loads
  dma_id_t load;             // descriptor of the last load
  uint32_t jobs;             // jobs issued in this window
  uint8_t last, before;      // targets of the two most recent jobs
  uint32_t hold;             // no barrier activity (other joins pending)
} avs_agent_t;
#endif

static void avs_post(uint32_t *posted, uint32_t *target, uint32_t b,
                     uint32_t issued, uint8_t t, avs_agent_t *a) {
  target[b % 4u] = issued ? AVS_JOB_ISSUED | t : 0u;
  __atomic_store_n(posted, b + 1u, __ATOMIC_RELEASE);
  if (issued) {
    a->before = a->last;
    a->last = t;
    ++a->jobs;
  }
}

// Publish every posted job that has completed, in order. Only core 0 reaches
// RedMulE's registers; the tile's other cores read these counts.
static void avs_relay(uint32_t *done, const uint32_t *posted,
                      const uint32_t *target, uint8_t completed) {
  uint32_t d = *done;
  while (avs_job_done(posted, target, d, completed))
    ++d;
  if (d != *done) {
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    __atomic_store_n(done, d, __ATOMIC_RELEASE);
    avs_wake();
  }
}

#if Q38_AV_SWEEP_OVERLAP
// Group coordinator: export window `exported` once every core of the group
// stored its rows, then load the queries of window exported + 2 into the
// freed bank. An export never overlaps a load of this group (KV or queries).
static void avs_group_dma(avs_agent_t *a) {
  q38_avs_group_sync_t *const gs = avs_group_rec();
  if (a->query_id && dma_local_done_id(0u, a->query_id)) {
    a->query_id = 0u;
    __atomic_store_n(&gs->queries, a->queried, __ATOMIC_RELEASE);
  }
  if (a->export_id) {
    if (!dma_local_done_id(0u, a->export_id))
      return;
    a->export_id = 0u;
    ++a->exported;
  }
  if (a->query_id)
    return;
  if (a->queried < a->windows && a->queried < a->exported + AVS_QBANKS) {
    const uint32_t w = a->queried++;
    a->query_id =
        avs_query_load(avs_win_base(w), w % AVS_BATCHES, w % AVS_QBANKS);
    if (!a->query_id)
      __atomic_store_n(&gs->queries, a->queried, __ATOMIC_RELEASE);
  } else if (a->exported < a->windows &&
             __atomic_load_n(&gs->finished[a->exported % AVS_QBANKS],
                             __ATOMIC_ACQUIRE) == AVS_FINISHERS &&
             (!a->kv || dma_local_done_id(0u, a->kv))) {
    const uint32_t w = a->exported;
    // Window w+QBANKS counts here only after its queries, which wait for this.
    gs->finished[w % AVS_QBANKS] = 0u;
    a->export_id = avs_export(avs_win_base(w), w % AVS_BATCHES, w % AVS_QBANKS);
    if (!a->export_id)
      ++a->exported;
  }
}

static uint32_t avs_queries_ready(avs_agent_t *a, uint32_t w) {
  if (a->queries <= w)
    a->queries = __atomic_load_n(&avs_group_rec()->queries, __ATOMIC_ACQUIRE);
  return a->queries > w;
}

// Non-blocking scheduler step of the tile agent, over global blocks. The
// dependencies of the per-window pipeline, plus: QK of a window's first block
// waits for its queries; KV loads wait while the group exports.
static AVS_CTRL void avs_agent_poll(avs_agent_t *a) {
  q38_avs_tile_sync_t *const s = avs_tile_sync();
  const uint8_t completed = tnc_redmule_completed();
  const uint32_t coordinator = mempool_is_group_coordinator();
  avs_relay(&s->qk_done, &s->qk_posted, s->qk_target, completed);
  avs_relay(&s->pv_done, &s->pv_posted, s->pv_target, completed);
  if (coordinator)
    avs_group_dma(a);
  if (a->armed) {
    if (!(fsync_status() & FSYNC_STATUS_BUSY_MASK)) {
      a->armed = 0u;
      ++a->barrier;
    }
  } else if (a->barrier <= a->blocks &&
             (a->barrier >= a->blocks || a->load.n > a->barrier) &&
             a->vload.n >= a->barrier &&
             (a->barrier + 1u < AVS_KSLOTS ||
              __atomic_load_n(&s->qk_done, __ATOMIC_RELAXED) >
                  a->barrier + 1u - AVS_KSLOTS) &&
             (a->barrier < AVS_VSLOTS ||
              __atomic_load_n(&s->pv_done, __ATOMIC_RELAXED) >
                  a->barrier - AVS_VSLOTS) &&
             (!coordinator || !a->kv || dma_local_done_id(0u, a->kv))) {
    // Barrier k: K(k) and V(k-1) landed in every group, and every tile of the
    // VC is past QK(k+1-KSLOTS) and PV(k-VSLOTS), the last readers of the
    // slots K(k+1) and V(k) reuse.
    fsync_trigger(FSYNC_ID_H, fsync_aggr_of_vc_level(1u));
    a->armed = 1u;
  }
  if (!a->export_id) {
    if (a->load.n < a->blocks && a->load.n <= a->barrier) {
      a->kv = avs_mat_load(a->load.key, 0u, a->load.n % AVS_KSLOTS);
      avs_it_next(&a->load);
    }
    if (a->vload.n < a->blocks && a->vload.n < a->barrier) {
      a->kv =
          avs_mat_load(a->vload.key, 1u, AVS_KSLOTS + a->vload.n % AVS_VSLOTS);
      avs_it_next(&a->vload);
    }
  }
#if Q38_AV_SWEEP_LATE_GATE0
  // Window 0's gates go behind K(k), V(k-1); finish(0) waits for the flag.
  if (coordinator) {
    if (!a->gate0 && a->barrier >= Q38_AV_SWEEP_LATE_GATE0) {
      a->gate0_id = avs_query_parts(avs_win_base(0u), 0u, 0u, 2u);
      a->gate0 = 1u;
    }
    if (a->gate0 == 1u &&
        (!a->gate0_id || dma_local_done_id(0u, a->gate0_id))) {
      __atomic_store_n(&avs_group_rec()->gates0, 1u, __ATOMIC_RELEASE);
      a->gate0 = 2u;
    }
  }
#endif
  const uint32_t room = a->jobs < 2u || tnc_counter8_done(completed, a->before);
  if (!room)
    return;
  const uint32_t pv_ready = Q38_AV_SWEEP_PV_FIRST && a->pv.n < a->qk.n &&
                            a->barrier > a->pv.n + 1u &&
                            avs_tile_block_done(a->pv.n);
  if (!pv_ready && a->qk.n < a->blocks && a->barrier > a->qk.n &&
      a->pv.n + AVS_SSLOTS > a->qk.n &&
      (a->qk.key || avs_queries_ready(a, a->qk.w))) {
    const uint32_t w = a->qk.w;
    uint32_t issued = 0u;
    const uint8_t t =
        avs_job(avs_win_base(w), w % AVS_BATCHES, w % AVS_QBANKS, a->qk.key,
                a->qk.n % AVS_KSLOTS, a->qk.n % AVS_SSLOTS, 0u, 0u, &issued);
    avs_post(&s->qk_posted, s->qk_target, a->qk.n, issued, t, a);
    avs_it_next(&a->qk);
  } else if (pv_ready || (a->pv.n < a->qk.n && a->barrier > a->pv.n + 1u &&
                          avs_tile_block_done(a->pv.n))) {
    const uint32_t w = a->pv.w;
    uint32_t issued = 0u;
    const uint8_t t = avs_job(avs_win_base(w), w % AVS_BATCHES, w % AVS_QBANKS,
                              a->pv.key, AVS_KSLOTS + a->pv.n % AVS_VSLOTS,
                              a->pv.n % AVS_SSLOTS, 1u, w % AVS_WBUFS, &issued);
    avs_post(&s->pv_posted, s->pv_target, a->pv.n, issued, t, a);
    avs_it_next(&a->pv);
  }
}
#else
// Non-blocking scheduler step of the tile agent. Dependencies only:
// - KV barrier k (split phase): this tile's PV(k-3) is done (the last reader
//   of the slot the next load reuses) and, at the coordinator, KV(k) landed.
//   Once every agent of the VC arrived, KV(k) is complete in every group and
//   the coordinator loads KV(k+1).
// - QK(m): KV(m) passed its barrier and PV(m-2), the previous reader of score
//   slot m % 2, is queued ahead of it.
// - PV(n): every core of the tile finished block n.
// RedMulE holds two job contexts: one runs, one waits. One completion-counter
// read per step; the barrier and DMA status are read only when they matter.
static void avs_agent_poll(avs_agent_t *a) {
  q38_avs_tile_sync_t *const s = avs_tile_sync();
  const uint8_t completed = tnc_redmule_completed();
  avs_relay(&s->qk_done, &s->qk_posted, s->qk_target, completed);
  avs_relay(&s->pv_done, &s->pv_posted, s->pv_target, completed);
  if (a->armed) {
    if (!(fsync_status() & FSYNC_STATUS_BUSY_MASK)) {
      a->armed = 0u;
      ++a->barrier;
      if (a->loaded < a->blocks) {
        a->load = avs_kv_load(a->load_key, a->loaded % 4u);
        a->load_key = avs_next_key(a->load_key, a->limit);
        ++a->loaded;
      }
    }
  } else if (!a->hold && a->barrier < a->blocks &&
             (a->barrier < 3u ||
              __atomic_load_n(&s->pv_done, __ATOMIC_RELAXED) >
                  a->g + a->barrier - 3u) &&
             (!mempool_is_group_coordinator() ||
              (a->loaded > a->barrier &&
               (!a->load || dma_local_done_id(0u, a->load))))) {
    fsync_trigger(FSYNC_ID_H, fsync_aggr_of_vc_level(1u));
    a->armed = 1u;
  }
  const uint32_t room = a->jobs < 2u || tnc_counter8_done(completed, a->before);
  if (!room)
    return;
  if (a->qk < a->blocks && a->barrier > a->qk && a->pv + 2u > a->qk) {
    uint32_t issued = 0u;
    const uint8_t t = avs_job(a->base, a->batch, a->bank, a->qk_key,
                              2u * (a->qk % 4u), a->qk % 2u, 0u, 0u, &issued);
    avs_post(&s->qk_posted, s->qk_target, a->g + a->qk, issued, t, a);
    a->qk_key = avs_next_key(a->qk_key, a->limit);
    ++a->qk;
  } else if (a->pv < a->qk && avs_tile_block_done(a->g + a->pv)) {
    uint32_t issued = 0u;
    const uint8_t t =
        avs_job(a->base, a->batch, a->bank, a->pv_key, 2u * (a->pv % 4u) + 1u,
                a->pv % 2u, 1u, 0u, &issued);
    avs_post(&s->pv_posted, s->pv_target, a->g + a->pv, issued, t, a);
    a->pv_key = avs_next_key(a->pv_key, a->limit);
    ++a->pv;
  }
}
#endif

// Wait until the job of global block b is relayed as done; the agent keeps
// scheduling (and relaying) meanwhile.
static void avs_await(uint32_t *done, uint32_t b, avs_agent_t *a) {
  const uint32_t start = mempool_get_timer();
  while (__atomic_load_n(done, __ATOMIC_ACQUIRE) <= b) {
    if (a)
      avs_agent_poll(a);
    else
      avs_idle();
  }
  AVS_ADD(AVS_SYNC, start);
}

static void avs_drain_late(uint32_t base, uint32_t batch, uint32_t key,
                           uint32_t slot, uint32_t ws) {
  for (uint32_t i = avs_core_first(); i < avs_core_end(); i += AVS_CORE_STEP) {
    if (!avs_live(base, batch, key, i / AVS_TILE))
      continue;
    if (Q38_AV_SWEEP_LATE_DRAIN >= 2)
      avs_load16(AVS_TL(AVS_SCORE(slot, i)), BK);
    if (key)
      avs_load16(AVS_TL(AVS_Z(ws, i)), Q38_HD);
  }
  float sink;
  asm volatile("vfmv.f.s %0, v0" : "=f"(sink));
  (void)sink;
}
#define AVS_POLL(a)                                                            \
  do {                                                                         \
    if (a)                                                                     \
      avs_agent_poll(a);                                                       \
  } while (0)
#else
typedef void avs_agent_t;
#define AVS_POLL(a) ((void)(a))
#endif

#if Q38_AV_SWEEP_OVERLAP
// Own frames: the stream's caller frame stays within the 512-byte stack.
#define AVS_ROW_STAGE NOSTACK
#else
#define AVS_ROW_STAGE
#endif

// PV(n-1) reads the other slot before that engine's QK(n+1) overwrites it.
// Softmax(n) consumes/overwrites the current slot; the group join protects
// both transitions. Two physical slots hold scores or probabilities.
#if Q38_AV_SWEEP_FAST && Q38_AV_SWEEP_VFEXP == 4
// Every row of this core sees the whole key block: one loop, the liveness and
// width decided once, the constant in a register across rows. The exp is the
// FP16 VFEXPF (e^t, t = s - o; avs_fast_ref2), a timing stand-in unless
// Q38_AV_SWEEP_EXP_POLY.
static NOSTACK void avs_fast_rows(__fp16 *row, uint32_t i0, uint32_t i1,
                                  uint32_t ws, avs_agent_t *poll, uint32_t base,
                                  uint32_t key) {
  register float zero asm("fs8") = 0.0f;
  asm volatile("" : "+f"(zero));
#if Q38_AV_SWEEP_EXP_POLY
  const avs_fast_k_t *const k =
      &avs_fast_kc[mempool_get_core_id() / NUM_CORES_PER_TILE].k;
  register float low asm("fs1") = k->low;
  register float tiny asm("fs2") = k->tiny;
  register float c0 asm("fs4") = k->c0;
  register float c1 asm("fs5") = k->c1;
  register float c2 asm("fs6") = k->c2;
  register float c3 asm("fs7") = k->c3;
  register float l2eh asm("fs9") = k->l2eh;
  int32_t bias = k->bias;
  asm volatile(""
               : "+f"(low), "+f"(tiny), "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3),
                 "+f"(l2eh), "+r"(bias));
#endif
  for (uint32_t i = i0; i < i1; i += AVS_CORE_STEP, row += AVS_CORE_STEP * BK) {
    qwen38_softmax_state_t *const st = AVS_STATE(ws, i);
    float mh, m, sum, n16;
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(AVS_FAST_PIECE));
    asm volatile("vle16.v v24, (%0)" : : "r"(row) : "memory");
    asm volatile("vle16.v v26, (%0)" : : "r"(row + AVS_FAST_PIECE) : "memory");
    asm volatile("vle16.v v28, (%0)"
                 :
                 : "r"(row + 2u * AVS_FAST_PIECE)
                 : "memory");
    asm volatile("vle16.v v30, (%0)"
                 :
                 : "r"(row + 3u * AVS_FAST_PIECE)
                 : "memory");
#if Q38_AV_SWEEP_FAST_MASK
    // Keys from `vis` on are causally masked: their scores become -inf.
    // Integer select, no vector compare (the model traps on those):
    // (idx - lim) >> 15 is all ones where idx < lim.
    const int32_t vis =
        (int32_t)(base + avs_row_at(i0 / AVS_TILE, i % AVS_TILE) + 1u - key);
    if (vis < (int32_t)BK) {
      const uint32_t ninf = 0xfc00u;
      asm volatile(
          "vle16.v v2, (%0)"
          :
          : "r"(avs_mask_idx[mempool_get_core_id() / NUM_CORES_PER_TILE].idx)
          : "memory");
#define AVS_MASK_PIECE(vr, p)                                                  \
  asm volatile("vsub.vx v4, v2, %0\n\t"                                        \
               "vsra.vi v4, v4, 15\n\t"                                        \
               "vand.vv " vr ", " vr ", v4\n\t"                                \
               "vxor.vi v4, v4, -1\n\t"                                        \
               "vand.vx v4, v4, %1\n\t"                                        \
               "vor.vv " vr ", " vr ", v4"                                     \
               :                                                               \
               : "r"(vis - (int32_t)((p)*AVS_FAST_PIECE)), "r"(ninf))
      AVS_MASK_PIECE("v24", 0u);
      AVS_MASK_PIECE("v26", 1u);
      AVS_MASK_PIECE("v28", 2u);
      AVS_MASK_PIECE("v30", 3u);
#undef AVS_MASK_PIECE
    }
#else
    (void)base;
    (void)key;
#endif
    asm volatile("vfmax.vv v0, v24, v26");
    asm volatile("vfmax.vv v2, v28, v30");
    asm volatile("vfmax.vv v0, v0, v2");
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma"
                 :
                 : "r"(AVS_FAST_PIECE / 2u));
    asm volatile("vfmax.vv v0, v0, v1");
    asm volatile("vfredmax.vs v0, v0, v0");
    asm volatile("vfmv.f.s %0, v0" : "=f"(mh));
    asm volatile("fcvt.s.h %0, %1" : "=f"(m) : "f"(mh));
    const float correction = avs_fast_ref2(st, m, AVS_LOG2E, &n16);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(AVS_FAST_CHUNK));
    asm volatile("vfsub.vf v4, v24, %0" : : "f"(n16));
    AVS_EXPE_M4("v8", "v4", "v16", "v20");
    asm volatile("vfsub.vf v4, v28, %0" : : "f"(n16));
    AVS_EXPE_M4("v12", "v4", "v16", "v20");
    // The sum first, so the vector unit adds while the stores drain.
    asm volatile("vfwadd.vv v16, v8, v12");
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma"
                 :
                 : "r"(AVS_FAST_CHUNK / 2u));
    asm volatile("vfadd.vv v16, v16, v20");
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma"
                 :
                 : "r"(AVS_FAST_CHUNK / 4u));
    asm volatile("vfadd.vv v16, v16, v18");
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma"
                 :
                 : "r"(AVS_FAST_CHUNK / 8u));
    asm volatile("vfadd.vv v16, v16, v17");
    asm volatile("vfmv.s.f v0, %0" : : "f"(zero));
    asm volatile("vfredusum.vs v0, v16, v0");
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(AVS_FAST_PIECE));
    asm volatile("vse16.v v8, (%0)" : : "r"(row) : "memory");
    asm volatile("vse16.v v10, (%0)" : : "r"(row + AVS_FAST_PIECE) : "memory");
    asm volatile("vse16.v v12, (%0)"
                 :
                 : "r"(row + 2u * AVS_FAST_PIECE)
                 : "memory");
    asm volatile("vse16.v v14, (%0)"
                 :
                 : "r"(row + 3u * AVS_FAST_PIECE)
                 : "memory");
    asm volatile("vfmv.f.s %0, v0" : "=f"(sum));
    st->denominator = correction * st->denominator + sum;
    *Q38_LA(avs_local->alpha + avs_scalar_index(i, AVS_ALPHA_SLOTS)) =
        correction;
    if (AVS_RB &&
        (Q38_ATTN_DRAIN || Q38_AV_SWEEP_MODE || Q38_AV_SWEEP_REGISTER) &&
        Q38_AV_SWEEP_LATE_DRAIN < 2)
      q38_drain16(row, BK);
#if Q38_AV_PROFILE
    ++avs_stats[mempool_get_core_id() / NUM_CORES_PER_TILE]
          .rows[mempool_get_core_id() % NUM_CORES_PER_TILE];
#endif
    if ((i - i0) / AVS_CORE_STEP % 2u)
      AVS_POLL(poll);
  }
}
#endif

#if Q38_AV_SWEEP_FAST && Q38_AV_SWEEP_VFEXP == 4
// Whether every row of this core is live and sees the whole key block (with
// FAST_MASK: at least one key of it; the rest are masked in the fast rows).
static uint32_t avs_rows_full(uint32_t base, uint32_t batch, uint32_t key) {
  const uint32_t i0 = avs_core_first(), i1 = avs_core_end();
  const uint32_t owner = i0 / AVS_TILE;
  if (i0 >= i1 || !avs_live(base, batch, key, owner))
    return 0u;
  const uint32_t lo = base + avs_row_at(owner, i0 % AVS_TILE);
  const uint32_t hi = base + avs_row_at(owner, (i1 - AVS_CORE_STEP) % AVS_TILE);
  return (Q38_AV_SWEEP_FAST_MASK ? lo >= key : lo + 1u >= key + BK) &&
         hi < Q38_S;
}
#endif

static AVS_ROW_STAGE void avs_softmax(uint32_t base, uint32_t batch,
                                      uint32_t key, uint32_t slot,
                                      avs_agent_t *poll, uint32_t ws) {
  const uint32_t start = mempool_get_timer();
#if Q38_AV_SWEEP_FAST
  // Shared constants (core 0 fills them): nothing on the 512-byte stack.
  const avs_fast_k_t *const k =
      &avs_fast_kc[mempool_get_core_id() / NUM_CORES_PER_TILE].k;
#endif
  for (uint32_t i = avs_core_first(); i < avs_core_end(); i += AVS_CORE_STEP) {
    if (!avs_live(base, batch, key, i / AVS_TILE))
      continue;
    const uint32_t token = base + avs_row_at(i / AVS_TILE, i % AVS_TILE);
    const uint32_t visible =
        Q38_MIN(BK, token < Q38_S && token >= key ? token + 1u - key : 0u);
#if Q38_AV_SWEEP_FAST && Q38_AV_SWEEP_VFEXP == 3
    // Two full rows of the same owner at once.
    const uint32_t j = i + AVS_CORE_STEP;
    if (visible == BK && j < avs_core_end() && j / AVS_TILE == i / AVS_TILE) {
      const uint32_t tj = base + avs_row_at(j / AVS_TILE, j % AVS_TILE);
      if (tj < Q38_S && tj >= key && tj + 1u - key >= BK) {
        float aa, ab;
        avs_fast_softmax_pair(AVS_TL(AVS_SCORE(slot, i)),
                              AVS_TL(AVS_SCORE(slot, j)), AVS_STATE(ws, i),
                              AVS_STATE(ws, j), k, &aa, &ab);
        *Q38_LA(avs_local->alpha + avs_scalar_index(i, AVS_ALPHA_SLOTS)) = aa;
        *Q38_LA(avs_local->alpha + avs_scalar_index(j, AVS_ALPHA_SLOTS)) = ab;
        if (AVS_RB &&
            (Q38_ATTN_DRAIN || Q38_AV_SWEEP_MODE || Q38_AV_SWEEP_REGISTER) &&
            Q38_AV_SWEEP_LATE_DRAIN < 2) {
          q38_drain16(AVS_TL(AVS_SCORE(slot, i)), BK);
          q38_drain16(AVS_TL(AVS_SCORE(slot, j)), BK);
        }
        i = j;
        AVS_POLL(poll);
        continue;
      }
    }
#endif
#if Q38_AV_SWEEP_FAST
    *Q38_LA(avs_local->alpha + avs_scalar_index(i, AVS_ALPHA_SLOTS)) =
        avs_fast_softmax_row(AVS_TL(AVS_SCORE(slot, i)), visible,
                             AVS_STATE(ws, i), k);
#else
    *Q38_LA(avs_local->alpha + avs_scalar_index(i, AVS_ALPHA_SLOTS)) =
        softmax_row(AVS_TL(AVS_SCORE(slot, i)), AVS_TL(AVS_SCORE(slot, i)),
                    visible, AVS_STATE(ws, i));
#endif
    if (AVS_RB &&
        (Q38_ATTN_DRAIN || Q38_AV_SWEEP_MODE || Q38_AV_SWEEP_REGISTER) &&
        Q38_AV_SWEEP_LATE_DRAIN < 2)
      q38_drain16(AVS_TL(AVS_SCORE(slot, i)), BK);
#if Q38_AV_PROFILE
    ++avs_stats[mempool_get_core_id() / NUM_CORES_PER_TILE]
          .rows[mempool_get_core_id() % NUM_CORES_PER_TILE];
#endif
    AVS_POLL(poll);
  }
  AVS_ADD(AVS_SM, start);
}

static AVS_ROW_STAGE void avs_rescale(uint32_t base, uint32_t batch,
                                      uint32_t key, avs_agent_t *poll,
                                      uint32_t ws) {
  const uint32_t start = mempool_get_timer();
  if (key)
    for (uint32_t i = avs_core_first(); i < avs_core_end();
         i += AVS_CORE_STEP) {
      if (!avs_live(base, batch, key, i / AVS_TILE))
        continue;
      const float alpha =
          *Q38_LA(avs_local->alpha + avs_scalar_index(i, AVS_ALPHA_SLOTS));
#if Q38_AV_SWEEP_FAST
      // Powers of two only; nothing is stored, so nothing drains, at 1.
      if (alpha == 1.0f)
        continue;
      avs_fast_rescale_row(AVS_TL(AVS_Z(ws, i)), alpha);
#else
      if (alpha != 1.0f)
        llm_scale_f16_f32(AVS_TL(AVS_Z(ws, i)), Q38_HD, alpha);
#endif
      if (AVS_RB &&
          (Q38_ATTN_DRAIN || Q38_AV_SWEEP_MODE || Q38_AV_SWEEP_REGISTER) &&
          !Q38_AV_SWEEP_LATE_DRAIN)
        q38_drain16(AVS_TL(AVS_Z(ws, i)), Q38_HD);
    }
  AVS_POLL(poll);
  AVS_ADD(AVS_RESCALE, start);
}

static uint32_t avs_next_key(uint32_t key, uint32_t limit) {
  uint32_t next = key + BK;
#if Q38_SCHEDULE_PROBE
  const uint32_t last = (limit - 1u) / BK * BK;
  if (next >= 2u * BK && next < last)
    next = last;
#else
  (void)limit;
#endif
  return next;
}

static void avs_pipe(uint32_t base, uint32_t batch, uint32_t bank,
                     uint32_t nextbase, uint32_t nextbatch, dma_id_t *queries,
                     uint32_t *epoch) {
  const uint32_t limit = Q38_MIN(base + AVS_ROWS, Q38_S);
  avs_kv_ready(avs_kv_load(0u, 0u), epoch);
  uint32_t prev = 0u, cur = 0u, next = avs_next_key(0u, limit);
  dma_id_t loading = next < limit ? avs_kv_load(next, 1u) : 0u;
  uint32_t issued = 0u;
  uint8_t target = 0u;
  if (tnc_is_tile_agent()) {
    target = avs_job(base, batch, bank, cur, 0u, 0u, 0u, 0u, &issued);
  }
  // Queue next queries behind KV(1), while QK(0) is already running.
  *queries = nextbase < Q38_MIXER_EXEC_END
                 ? avs_query_load(nextbase, nextbatch, bank ^ 1u)
                 : 0u;
  if (tnc_is_tile_agent()) {
    avs_wait_job(issued, target);
  }
  avs_compute_sync(epoch);
  uint32_t n = 0u;
  for (; cur < limit; ++n) {
    avs_kv_ready(loading, epoch);
    const uint32_t after = next < limit ? avs_next_key(next, limit) : limit;
    loading = after < limit ? avs_kv_load(after, (n + 2u) % 4u) : 0u;
    const uint32_t start = mempool_get_timer();
    issued = 0u;
    uint32_t pv_issued = 0u;
    uint8_t pv_target = 0u;
    if (tnc_is_tile_agent()) {
      if (n) {
        pv_target = avs_job(base, batch, bank, prev, 2u * ((n + 3u) % 4u) + 1u,
                            (n + 1u) % 2u, 1u, 0u, &pv_issued);
        target = pv_target;
        issued = pv_issued;
      }
      if (next < limit) {
        uint32_t qk = 0u;
        const uint8_t qkt =
            avs_job(base, batch, bank, next, 2u * ((n + 1u) % 4u),
                    (n + 1u) % 2u, 0u, 0u, &qk);
        if (qk)
          target = qkt;
        issued |= qk;
      }
    }
    avs_softmax(base, batch, cur, n % 2u, 0, 0u);
    if (tnc_is_tile_agent()) {
#if Q38_AV_SWEEP_MODE == 2
      // QK(next) touches scores, not C: rescale once PV(previous) finishes.
      avs_wait_job(pv_issued, pv_target);
#else
      avs_wait_job(issued, target);
#endif
      AVS_ADD(AVS_RM, start);
    }
    avs_compute_sync(epoch);
    avs_rescale(base, batch, cur, 0, 0u);
#if Q38_AV_SWEEP_MODE == 2
    if (tnc_is_tile_agent())
      avs_wait_job(issued, target);
#endif
    avs_compute_sync(epoch);
#if Q38_SCHEDULE_PROBE
    if (mempool_get_core_id() == 0u)
      ++probe_key_blocks;
#endif
    prev = cur;
    cur = next;
    next = after;
  }
  if (tnc_is_tile_agent()) {
    issued = 0u;
    target = avs_job(base, batch, bank, prev, 2u * ((n + 3u) % 4u) + 1u,
                     (n + 1u) % 2u, 1u, 0u, &issued);
    avs_wait_job(issued, target);
  }
  avs_compute_sync(epoch);
}

#if Q38_AV_SWEEP_ASYNC && !Q38_AV_SWEEP_OVERLAP
// Dependency-only window: every core waits for the QK of its block and the
// PV before its rescale; nobody joins inside a block. `blocks` is this core's
// running count of key blocks over all windows (the tile's posted counters
// are monotone).
static NOSTACK void avs_pipe_async(uint32_t base, uint32_t batch, uint32_t bank,
                                   uint32_t nextbase, uint32_t nextbatch,
                                   dma_id_t *queries, uint32_t *epoch,
                                   uint32_t *blocks) {
  q38_avs_tile_sync_t *const s = avs_tile_sync();
  avs_agent_t state = {0};
  avs_agent_t *const a = tnc_is_tile_agent() ? &state : 0;
  const uint32_t limit = Q38_MIN(base + AVS_ROWS, Q38_S), g = *blocks;
  uint32_t count = 0u;
  for (uint32_t key = 0u; key < limit; key = avs_next_key(key, limit))
    ++count;
  // KV(0) is complete in every group before the first job (blocking).
  avs_kv_ready(avs_kv_load(0u, 0u), epoch);
  if (a) {
    state = (avs_agent_t){.base = base,
                          .batch = batch,
                          .bank = bank,
                          .limit = limit,
                          .g = g,
                          .blocks = count,
                          .barrier = 1u,
                          .loaded = 1u};
    state.load_key = avs_next_key(0u, limit);
    if (count > 1u) {
      state.load = avs_kv_load(state.load_key, 1u);
      state.load_key = avs_next_key(state.load_key, limit);
      state.loaded = 2u;
    }
    avs_agent_poll(a); // QK(0)
  }
  *queries = nextbase < Q38_MIXER_EXEC_END
                 ? avs_query_load(nextbase, nextbatch, bank ^ 1u)
                 : 0u;
  uint32_t key = 0u;
  for (uint32_t n = 0u; n < count; ++n, key = avs_next_key(key, limit)) {
    avs_await(&s->qk_done, g + n, a);
    avs_softmax(base, batch, key, n % 2u, a, 0u);
    if (n) {
      avs_await(&s->pv_done, g + n - 1u, a);
      avs_rescale(base, batch, key, a, 0u);
    }
    if (Q38_AV_SWEEP_LATE_DRAIN)
      avs_drain_late(base, batch, key, n % 2u, 0u);
    avs_block_done(g + n);
    AVS_POLL(a);
  }
  // Finish normalizes Z: the last PV must be done.
  avs_await(&s->pv_done, g + count - 1u, a);
  *blocks = g + count;
  avs_compute_sync(epoch);
}
#endif

#if Q38_AV_SWEEP_TILE_WIN
// llm_scale_f16_f32 out of place: dst = fp16(fp32(src) * scale), same rounding.
static void avs_scale_to(__fp16 *dst, const __fp16 *src, uint32_t n,
                         float scale) {
  const float zh = llm_f16box(0.0f);
  for (uint32_t avl = n; avl;) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle16.v v0, (%0)" : : "r"(src) : "memory");
    asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    asm volatile("vfmul.vf v8, v8, %0" : : "f"(scale));
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
    asm volatile("vfncvt.f.f.w v0, v8");
    asm volatile("vse16.v v0, (%0)" : : "r"(dst) : "memory");
    src += vl;
    dst += vl;
    avl -= vl;
  }
}
#endif

// Reuse Z for normalized context, export it by DMA, then gate it. No direct
// Spatz L2 stores, and no exports overlap this group's pending loads.
static NOSTACK void avs_finish(uint32_t base, uint32_t batch, uint32_t bank,
                               uint32_t *epoch) {
  const uint32_t ws = 0u;
  const uint32_t start = mempool_get_timer();
  for (uint32_t i = avs_core_first(); i < avs_core_end(); i += AVS_CORE_STEP)
    if (avs_exists(base, batch, i / AVS_TILE)) {
      const uint32_t token = base + avs_row_at(i / AVS_TILE, i % AVS_TILE);
      __fp16 *const ctx = Q38_LA(avs_ctx_base(bank) + i * Q38_HD);
      if (token >= Q38_S) {
        llm_zero_f16(ctx, Q38_HD);
        if (AVS_RB &&
            (Q38_ATTN_DRAIN || Q38_AV_SWEEP_REUSE_QUERY || Q38_AV_SWEEP_MODE))
          q38_drain16(ctx, Q38_HD);
        continue;
      }
      const float inverse = llm_recip_pos(AVS_STATE(ws, i)->denominator);
#if Q38_AV_SWEEP_TILE_WIN
      avs_scale_to(ctx, AVS_Z(ws, i), Q38_HD, inverse);
#else
      llm_scale_f16_f32(ctx, Q38_HD, inverse);
#endif
      if (AVS_RB &&
          (Q38_ATTN_DRAIN || Q38_AV_SWEEP_REUSE_QUERY || Q38_AV_SWEEP_MODE))
        q38_drain16(ctx, Q38_HD);
    }
  // Baseline response routing is not qualified for another VC's loads while
  // this VC exports. Isolate export boundaries, keeping the inner loop local.
  avs_scope_sync(epoch, 1u);
  if (mempool_is_group_coordinator()) {
    dma_id_t id = 0u;
    for (uint32_t j = 0; j < AVS_OWNERS; ++j) {
      if (!avs_exists(base, batch, j))
        continue;
      const uint32_t h = avs_head(batch, j), row = base + avs_row(j);
#if Q38_SAMPLED_CHECKS
      uint32_t runs = Q38_RUNS_CONTEXT;
      asm volatile("" : "+r"(runs));
      for (uint32_t r = 0; r < runs; ++r) {
        const uint32_t flat = run_start_context[r];
        if (flat / (Q38_R * Q38_HD) != h)
          continue;
        const uint32_t token = flat / Q38_HD % Q38_R;
        if (token < row || (token - row) % avs_row_step() ||
            (token - row) / avs_row_step() >= AVS_TILE)
          continue;
        id = dma_local_memcpy_nonblocking(
            0u, result_sample_context + r * Q38_RUNLEN_CONTEXT,
            q38_la(avs_ctx_base(bank) +
                   (j * AVS_TILE + (token - row) / avs_row_step()) * Q38_HD +
                   flat % Q38_HD),
            Q38_RUNLEN_CONTEXT * 2u);
      }
#else
      if (avs_row_step() != 1u) {
        const uint32_t rows = avs_owner_rows(base, j);
        const uint32_t lanes = VC_ROW_BYTES / (Q38_HD * 2u);
        for (uint32_t lane = 0u; lane < lanes && lane < rows; ++lane) {
          const uint32_t token = base + avs_row_at(j, lane);
          id = dma_local_memcpy_2d_nonblocking(
              0u, result_context + (h * Q38_R + token) * Q38_HD,
              q38_la(avs_ctx_base(bank) + (j * AVS_TILE + lane) * Q38_HD),
              Q38_HD * 2u, lanes * avs_row_step() * Q38_HD * 2u,
              VC_GLOBAL_ROW_BYTES, (rows - lane + lanes - 1u) / lanes);
        }
        continue;
      }
      const uint32_t bytes = avs_owner_rows(base, j) * Q38_HD * 2u;
      const uint32_t whole = bytes / VC_ROW_BYTES * VC_ROW_BYTES;
      if (whole)
        id = dma_local_memcpy_2d_nonblocking(
            0u, result_context + (h * Q38_R + row) * Q38_HD,
            q38_la(avs_ctx_base(bank) + j * AVS_TILE * Q38_HD), VC_ROW_BYTES,
            VC_ROW_BYTES, VC_GLOBAL_ROW_BYTES, whole / VC_ROW_BYTES);
      if (whole != bytes)
        id = dma_local_memcpy_nonblocking(
            0u, result_context + (h * Q38_R + row) * Q38_HD + whole / 2u,
            q38_la(avs_ctx_base(bank) + j * AVS_TILE * Q38_HD + whole / 2u),
            bytes - whole);
#endif
    }
    avs_dma_wait(id);
  }
  avs_group_sync(epoch);
  for (uint32_t i = avs_core_first(); i < avs_core_end(); i += AVS_CORE_STEP)
    if (avs_exists(base, batch, i / AVS_TILE)) {
      q38_activation16(Q38_LA(avs_local->gate[bank] + i * Q38_HD),
                       Q38_LA(avs_ctx_base(bank) + i * Q38_HD),
                       Q38_LA(avs_gated_bank(bank) + i * Q38_HD), Q38_HD, 0);
      if (AVS_RB &&
          (Q38_ATTN_DRAIN || Q38_AV_SWEEP_REUSE_QUERY || Q38_AV_SWEEP_MODE))
        q38_drain16(Q38_LA(avs_gated_bank(bank) + i * Q38_HD), Q38_HD);
    }
  avs_group_sync(epoch);
  if (mempool_is_group_coordinator()) {
    dma_id_t id = 0u;
    for (uint32_t j = 0; j < AVS_OWNERS; ++j) {
      if (!avs_exists(base, batch, j))
        continue;
      const uint32_t h = avs_head(batch, j);
#if Q38_AV_SWEEP_ROW_DMA
      const uint32_t rows = avs_owner_rows(base, j);
      const uint32_t lanes = VC_ROW_BYTES / (Q38_HD * 2u);
      for (uint32_t lane = 0u; lane < lanes && lane < rows; ++lane) {
        const uint32_t token = base + avs_row_at(j, lane);
        id = dma_local_memcpy_2d_nonblocking(
            0u, AV_GATED(token, h),
            q38_la(avs_gated_bank(bank) + (j * AVS_TILE + lane) * Q38_HD),
            Q38_HD * 2u, lanes * avs_row_step() * AV_GK * 2u,
            VC_GLOBAL_ROW_BYTES, (rows - lane + lanes - 1u) / lanes);
      }
#else
      for (uint32_t r = 0; r < avs_owner_rows(base, j); ++r)
        id = dma_local_memcpy_nonblocking(
            0u, AV_GATED(base + avs_row_at(j, r), h),
            q38_la(avs_gated_bank(bank) + (j * AVS_TILE + r) * Q38_HD),
            Q38_HD * 2u);
#endif
    }
    avs_dma_wait(id);
  }
  avs_scope_sync(epoch, 1u);
  AVS_ADD(AVS_FINISH, start);
}

#if Q38_AV_SWEEP_FINISH
// Gate fused into the normalisation; only the gated rows leave L1, as a few
// strided descriptors per owner. The full context is not materialised (the
// sampled context checks of the baseline finish need it).
static NOSTACK void avs_finish_fast(uint32_t base, uint32_t batch,
                                    uint32_t bank, uint32_t *epoch) {
  const uint32_t ws = 0u;
  const uint32_t start = mempool_get_timer();
  avs_gate_k_t k;
  avs_gate_init(&k);
  for (uint32_t i = avs_core_first(); i < avs_core_end(); i += AVS_CORE_STEP)
    if (avs_exists(base, batch, i / AVS_TILE)) {
      const uint32_t token = base + avs_row_at(i / AVS_TILE, i % AVS_TILE);
      __fp16 *const out = Q38_LA(avs_gated_bank(bank) + i * Q38_HD);
      if (token >= Q38_S) {
        llm_zero_f16(out, Q38_HD);
        continue;
      }
      const float inverse = llm_recip_pos(AVS_STATE(ws, i)->denominator);
      avs_gate_row(out, AVS_TL(AVS_Z(ws, i)),
                   Q38_LA(avs_local->gate[bank] + i * Q38_HD), inverse, &k);
    }
  // One drain for all rows: a load after a store returns once it landed.
  if (AVS_RB) {
    for (uint32_t i = avs_core_first(); i < avs_core_end(); i += AVS_CORE_STEP)
      if (avs_exists(base, batch, i / AVS_TILE))
        avs_load16(Q38_LA(avs_gated_bank(bank) + i * Q38_HD), Q38_HD);
    float sink;
    asm volatile("vfmv.f.s %0, v0" : "=f"(sink));
    (void)sink;
  }
  avs_group_sync(epoch);
  if (mempool_is_group_coordinator())
    avs_dma_wait(avs_export(base, batch, bank));
  avs_scope_sync(epoch, 1u);  // fast finish: exports isolated
  AVS_ADD(AVS_FINISH, start); // fast finish
}
#endif

#if Q38_AV_SWEEP_OVERLAP
static void avs_state_init(uint32_t w) {
  const uint32_t ws = w % AVS_WBUFS;
  for (uint32_t i = avs_core_first(); i < avs_core_end(); i += AVS_CORE_STEP)
    qwen38_softmax_init(AVS_STATE(ws, i));
}

// This core's rows of window w, gated and normalised into the window's query
// bank; the group coordinator exports once every core of the group counted in.
// Rows p of this core with p % parts == part; the last part counts the core in.
static NOSTACK void avs_stream_finish(uint32_t w, avs_agent_t *poll,
                                      uint32_t part, uint32_t parts) {
  const uint32_t start = mempool_get_timer();
#if Q38_AV_SWEEP_LATE_GATE0
  while (!w && !__atomic_load_n(&avs_group_rec()->gates0, __ATOMIC_ACQUIRE))
    mempool_wait(16u);
#endif
  const uint32_t base = avs_win_base(w), batch = w % AVS_BATCHES;
  const uint32_t bank = w % AVS_QBANKS, ws = w % AVS_WBUFS;
  // A core without rows (the scheduler core) runs no F/V instruction here.
  const uint32_t rows = avs_core_first() < avs_core_end();
  avs_gate_k_t k;
  if (rows)
    avs_gate_init(&k);
  uint32_t p = 0u;
  for (uint32_t i = avs_core_first(); i < avs_core_end();
       i += AVS_CORE_STEP, ++p)
    if (p % parts == part && avs_exists(base, batch, i / AVS_TILE)) {
      const uint32_t token = base + avs_row_at(i / AVS_TILE, i % AVS_TILE);
      __fp16 *const out = Q38_AV_SWEEP_FIN_WHATIF & 4u
                              ? AVS_TL(AVS_Z(ws, i))
                              : Q38_LA(avs_gated_bank(bank) + i * Q38_HD);
      if (token >= Q38_S) {
        llm_zero_f16(out, Q38_HD);
        continue;
      }
      const float inverse = llm_recip_pos(AVS_STATE(ws, i)->denominator);
      avs_gate_row(out, AVS_TL(AVS_Z(ws, i)),
                   Q38_AV_SWEEP_FIN_WHATIF & 2u
                       ? AVS_TL(AVS_Z(ws, i))
                       : Q38_LA(avs_local->gate[bank] + i * Q38_HD),
                   inverse, &k);
      AVS_POLL(poll);
    }
  p = 0u;
  for (uint32_t i = avs_core_first(); i < avs_core_end();
       i += AVS_CORE_STEP, ++p)
    if (AVS_RB && !(Q38_AV_SWEEP_FIN_WHATIF & 1u) && p % parts == part &&
        avs_exists(base, batch, i / AVS_TILE))
      avs_load16(Q38_AV_SWEEP_FIN_WHATIF & 4u
                     ? AVS_TL(AVS_Z(ws, i))
                     : Q38_LA(avs_gated_bank(bank) + i * Q38_HD),
                 Q38_HD);
  if (AVS_RB && rows) {
    float sink;
    asm volatile("vfmv.f.s %0, v0" : "=f"(sink));
    (void)sink;
  }
  avs_fence();
  if (part + 1u == parts)
    __atomic_fetch_add(&avs_group_rec()->finished[w % AVS_QBANKS], 1u,
                       __ATOMIC_RELEASE); // rows
  AVS_ADD(AVS_FINISH, start);
}

#ifndef Q38_AV_SWEEP_FIN_DELAY
#define Q38_AV_SWEEP_FIN_DELAY 0u
#endif
// 1: spread window w-1's finish over the blocks of window w.
#ifndef Q38_AV_SWEEP_FIN_SPREAD
#define Q38_AV_SWEEP_FIN_SPREAD 0
#endif
_Static_assert(!Q38_AV_SWEEP_FIN_SPILL || Q38_AV_SWEEP_FIN_SPREAD,
               "The finish spill extends the spread finish");

static __attribute__((noinline)) uint32_t avs_stream_blocks(uint32_t windows) {
  avs_it_t it = {0};
  avs_it_window(&it);
  while (it.w < windows)
    avs_it_next(&it);
  return it.n;
}

// Every core: the stream's first loads, ready in every group of the VC.
static AVS_CTRL void avs_stream_start(uint32_t *epoch) {
#if Q38_AV_SWEEP_EARLY_START
  // QK(0) needs window 0's queries and K(0) only. The DMA queue is in order:
  // queries, K(0), then the gates; V(0) is the agent's first V load, and
  // barrier 1 (K(1), V(0)) gates QK(1) and PV(0) as usual.
  const dma_id_t k0 = avs_mat_load(0u, 0u, 0u);
  if (!Q38_AV_SWEEP_LATE_GATE0)
    avs_query_parts(avs_win_base(0u), 0u, 0u, 2u);
  if (mempool_is_group_coordinator()) {
    avs_dma_wait(k0);
    __atomic_store_n(&avs_group_rec()->queries, 1u, __ATOMIC_RELEASE);
  }
  avs_kv_ready(k0, epoch);
#else
  // K(0) and V(0) are complete in every group before the first job.
  avs_mat_load(0u, 0u, 0u);
  avs_kv_ready(avs_mat_load(0u, 1u, AVS_KSLOTS), epoch);
#endif
}

// The tile agent's scheduler state at the stream's start, then QK(0).
static AVS_CTRL void avs_agent_start(avs_agent_t *a, uint32_t windows,
                                     uint32_t blocks) {
  avs_it_t it = {0};
  avs_it_window(&it);
  *a = (avs_agent_t){.qk = it,
                     .pv = it,
                     .load = it,
                     .vload = it,
                     .windows = windows,
                     .blocks = blocks,
                     .barrier = 1u,
                     .queries = 1u,
                     .queried = 1u};
  avs_it_next(&a->load);
  if (!Q38_AV_SWEEP_EARLY_START)
    avs_it_next(&a->vload);
  if (blocks > 1u) {
    a->kv = avs_mat_load(a->load.key, 0u, 1u);
    avs_it_next(&a->load);
  }
  avs_agent_poll(a); // QK(0)
}

#if Q38_AV_SWEEP_SCHED_CORE
// The scheduler core (rv32ima, no rows): the tile agent alone, until the
// tile's last PV is relayed and, on the group coordinator, every window is
// exported. It takes no part in the blocks, so nothing waits for it there.
static AVS_CTRL void avs_sched_stream(uint32_t *epoch) {
  q38_avs_tile_sync_t *const s = avs_tile_sync();
  avs_agent_t *const a = &avs_agents[s - avs_sync].a;
  const uint32_t windows = avs_windows();
  const uint32_t blocks = avs_stream_blocks(windows);
  avs_stream_start(epoch);
  avs_agent_start(a, windows, blocks);
  while (__atomic_load_n(&s->pv_done, __ATOMIC_ACQUIRE) < blocks ||
         (mempool_is_group_coordinator() && a->exported < windows))
    avs_agent_poll(a);
}
#endif

// Every compute core: its rows of each block in stream order and, after each
// block of window w, its share of window w-1's finish. With 4 vector cores
// the tile agent is one of them and schedules in its waits and between rows.
static NOSTACK void avs_stream(uint32_t *epoch) {
  q38_avs_tile_sync_t *const s = avs_tile_sync();
  const uint32_t windows = avs_windows();
  avs_agent_t *const a = !Q38_AV_SWEEP_SCHED_CORE && tnc_is_tile_agent()
                             ? &avs_agents[avs_tile_sync() - avs_sync].a
                             : 0;
  avs_it_t it = {0};
  avs_it_window(&it);
  const uint32_t blocks = avs_stream_blocks(windows);
  avs_stream_start(epoch);
  if (a)
    avs_agent_start(a, windows, blocks);
  for (; it.w < windows; avs_it_next(&it)) {
    const uint32_t w = it.w, base = avs_win_base(w), batch = w % AVS_BATCHES;
    avs_await(&s->qk_done, it.n, a);
#if Q38_AV_SWEEP_FAST && Q38_AV_SWEEP_VFEXP == 4
    if (avs_rows_full(base, batch, it.key)) {
      const uint32_t s0 = mempool_get_timer();
      avs_fast_rows(AVS_TL(AVS_SCORE(it.n % AVS_SSLOTS, avs_core_first())),
                    avs_core_first(), avs_core_end(), w % AVS_WBUFS, a, base,
                    it.key);
      AVS_ADD(AVS_SM, s0);
    } else {
      // Blocks with masked rows; counted again under the stream-unused
      // AVS_RM slot (printed as rm_interval).
      const uint32_t s1 = mempool_get_timer();
      avs_softmax(base, batch, it.key, it.n % AVS_SSLOTS, a, w % AVS_WBUFS);
      AVS_ADD(AVS_RM, s1);
    }
#else
    avs_softmax(base, batch, it.key, it.n % AVS_SSLOTS, a, w % AVS_WBUFS);
#endif
    if (it.key) {
      avs_await(&s->pv_done, it.n - 1u, a);
      avs_rescale(base, batch, it.key, a, w % AVS_WBUFS);
    }
    if (Q38_AV_SWEEP_LATE_DRAIN)
      avs_drain_late(base, batch, it.key, it.n % AVS_SSLOTS, w % AVS_WBUFS);
    avs_block_done(it.n);
    AVS_POLL(a);
#if Q38_AV_SWEEP_FIN_SPREAD
    // Window w-1's rows finish in shares, one after each block of window w
    // (FIN_SPILL: and a last one after the first block of window w+1). The
    // state and Z parity of window w+1 are free at the end of window w.
    if (w) {
      const uint32_t part = it.key / BK, nb = avs_win_blocks(w);
#if Q38_AV_SWEEP_FIN_SPILL
      if (!part && w >= 2u)
        avs_stream_finish(w - 2u, a, avs_win_blocks(w - 1u),
                          avs_win_blocks(w - 1u) + 1u);
#endif
      if (!part)
        avs_await(&s->pv_done, it.n - 1u, a);
      avs_stream_finish(w - 1u, a, part, nb + Q38_AV_SWEEP_FIN_SPILL);
      if (part + 1u == nb && w + 1u < windows)
        avs_state_init(w + 1u);
    }
#else
    // Window w-1 finishes after block FIN_DELAY of window w (or its last).
    if (w && it.key == Q38_MIN(Q38_AV_SWEEP_FIN_DELAY * BK,
                               (it.limit - 1u) / BK * BK)) {
      avs_await(&s->pv_done, it.n - it.key / BK - 1u, a);
      avs_stream_finish(w - 1u, a, 0u, 1u);
      // Window w+1 reuses the parity window w-1 just released.
      if (w + 1u < windows)
        avs_state_init(w + 1u);
    }
#endif
  }
  avs_await(&s->pv_done, blocks - 1u, a);
#if Q38_AV_SWEEP_FIN_SPREAD && Q38_AV_SWEEP_FIN_SPILL
  if (windows >= 2u)
    avs_stream_finish(windows - 2u, a, avs_win_blocks(windows - 1u),
                      avs_win_blocks(windows - 1u) + 1u);
#endif
  avs_stream_finish(windows - 1u, a, 0u, 1u);
  while (a && mempool_is_group_coordinator() && a->exported < windows)
    avs_agent_poll(a);
}
#endif

// Every core of the machine meets. EVT: the epoch-checked fractal_sync barrier,
// which a core may enter with other events pending; else mempool_barrier.
static void avs_sync_all(void) {
#if Q38_AV_SWEEP_EVT
  __sync_synchronize();
  fsync_barrier_all(fsync_tile_counters, mempool_get_tile_id());
#else
  sync_all();
#endif
}

static NOSTACK void avs_run(void) {
  const uint32_t core = mempool_get_core_id();
  if (!core) {
    uint32_t p = Q38_MIN((uint32_t)&stage, (uint32_t)&weights[0][0]);
    p = (p + VC_GLOBAL_ROW_BYTES - 1u) / VC_GLOBAL_ROW_BYTES *
        VC_GLOBAL_ROW_BYTES;
    avs_local = (q38_avs_local_t *)vc_group_window(p);
    p += sizeof(q38_avs_local_t) * NUM_GROUPS;
#if Q38_AV_SWEEP_TILE_WIN
    // A tile-window row is each tile's bank stripe of one global row.
    p = (p + VC_GLOBAL_ROW_BYTES - 1u) / VC_GLOBAL_ROW_BYTES *
        VC_GLOBAL_ROW_BYTES;
    avs_tile = (q38_avs_tile_t *)(AVS_TILE_WINDOW_BASE +
                                  VC_GLOBAL_ROW(p) * AVS_TILE_ROW_BYTES);
    p += sizeof(q38_avs_tile_t) / AVS_TILE_ROW_BYTES * VC_GLOBAL_ROW_BYTES;
#endif
    avs_kv = (__fp16 *)(VC_WINDOW_BASE_1 + VC_GLOBAL_ROW(p) * Q38_VC_ROW);
    p += AVS_KV_MATRICES * AVS_MAT_ELEMS * 2u * Q38_VC_COUNT;
    avs_layout_end = p;
    if (p > (uint32_t)&__heap_end)
      q38_fatal("[qwen38_fatal] sweep layout over L1=",
                p - (uint32_t)&__heap_end);
    for (uint32_t e = 0; e < NUM_REDMULE_TILES; ++e)
      avs_sync[e].arrived = avs_sync[e].released = 0u;
  }
#if Q38_AV_SWEEP_FAST
  // One core per tile fills its tile's copy (local stores); the first
  // window's group join orders it before any softmax.
  if (avs_is_tile_filler())
    avs_fast_init(&avs_fast_kc[core / NUM_CORES_PER_TILE].k);
#if Q38_AV_SWEEP_FAST_MASK
  if (avs_is_tile_filler())
    for (uint32_t e = 0u; e < NUM_BANKS_PER_TILE * 2u; ++e)
      avs_mask_idx[core / NUM_CORES_PER_TILE].idx[e] = (int16_t)e;
#endif
#endif
  avs_sync_all();
  vc_subscribe(tnc_is_tile_agent()
                   ? FSYNC_EVT_FSYNC | FSYNC_EVT_ENGINE | FSYNC_EVT_DMA
                   : (Q38_AV_SWEEP_EVT ? FSYNC_EVT_CORE : 0u));
  uint32_t epoch = 0u, bank = 0u;
#if Q38_AV_SWEEP_ASYNC
  uint32_t blocks = 0u;
  if (tnc_is_tile_agent()) {
    // Each agent clears its own tile's counters (local stores); the first
    // group join in the window loop orders them before any use.
    q38_avs_tile_sync_t *const s = avs_tile_sync();
    s->qk_posted = s->pv_posted = s->qk_done = s->pv_done = 0u;
    for (uint32_t i = 0; i < 4u; ++i)
      s->qk_target[i] = s->pv_target[i] = 0u;
    for (uint32_t i = 0; i < NUM_CORES_PER_TILE; ++i)
      avs_tile_lanes()->done[i] = 0u;
  }
#endif
#if Q38_AV_SWEEP_OVERLAP
  (void)bank;
  (void)blocks;
  if (mempool_is_group_coordinator()) {
    for (uint32_t b = 0u; b < AVS_QBANKS; ++b)
      avs_group_rec()->finished[b] = 0u;
    avs_group_rec()->gates0 = 0u;
#if Q38_AV_SWEEP_EARLY_START
    // Window 0's queries now; its gates follow K(0) (avs_stream).
    avs_query_parts(avs_win_base(0u), 0u, 0u, 1u);
#else
    avs_dma_wait(avs_query_load(avs_win_base(0u), 0u, 0u));
    avs_group_rec()->queries = 1u;
#endif
  }
  for (uint32_t w = 0u; w < 2u && w < avs_windows(); ++w)
    avs_state_init(w);
  avs_group_sync(&epoch);
#if Q38_AV_SWEEP_SCHED_CORE
  if (tnc_is_tile_agent())
    avs_sched_stream(&epoch);
  else
#endif
    avs_stream(&epoch);
#else
  dma_id_t queries = avs_query_load(Q38_MIXER_EXEC_START, 0u, bank);
  for (uint32_t base = Q38_MIXER_EXEC_START; base < Q38_MIXER_EXEC_END;
       base += AVS_ROWS)
    for (uint32_t batch = 0; batch < AVS_BATCHES; ++batch) {
      avs_dma_wait(queries);
      avs_group_sync(&epoch);
      const uint32_t nb = batch + 1u < AVS_BATCHES ? batch + 1u : 0u;
      const uint32_t nextbase = nb ? base : base + AVS_ROWS;
      for (uint32_t i = avs_core_first(); i < avs_core_end();
           i += AVS_CORE_STEP)
        qwen38_softmax_init(AVS_STATE(0u, i));
      avs_group_sync(&epoch);
#if Q38_AV_SWEEP_ASYNC
      avs_pipe_async(base, batch, bank, nextbase, nb, &queries, &epoch,
                     &blocks);
#else
      avs_pipe(base, batch, bank, nextbase, nb, &queries, &epoch);
#endif
      // Finish only after the next-query load has landed: mixed-direction DMA
      // from one group is deliberately serialized until model qualification.
      avs_dma_wait(queries);
#if Q38_AV_SWEEP_FINISH
      avs_finish_fast(base, batch, bank, &epoch);
#else
      avs_finish(base, batch, bank, &epoch);
#endif
      bank ^= 1u;
#if Q38_SCHEDULE_PROBE
      if (!core)
        ++probe_mixer_windows;
#endif
    }
#endif
#if Q38_AV_SWEEP_EVT
  // The barrier sleeps on its epoch; afterwards drop whatever it left pending
  // (mempool_barrier, used after this, sleeps once without a re-check).
  fsync_subscribe_self(FSYNC_EVT_FSYNC);
  avs_sync_all();
  fsync_subscribe_self(0u);
#else
  fsync_subscribe_self(0u);
  sync_all();
#endif
  fsync_subscribe_self(FSYNC_EVT_FSYNC);
}

#if Q38_AV_PROFILE
static void avs_profile_init(void) {
  const uint32_t core = mempool_get_core_id();
  for (uint32_t part = 0u; part < AVS_PARTS; ++part)
    avs_stats[core / NUM_CORES_PER_TILE]
        .cycles[core % NUM_CORES_PER_TILE][part] = 0u;
  avs_stats[core / NUM_CORES_PER_TILE].rows[core % NUM_CORES_PER_TILE] = 0u;
  avs_wait_stats[core / NUM_CORES_PER_TILE].cycles[core % NUM_CORES_PER_TILE] =
      0u;
  if (tnc_is_tile_agent())
    avs_stats[mempool_get_redmule_id()].jobs = 0u;
}

static void avs_report(void) {
  printf("[qwen38_avs_layout] rows=%u tile=%u key_block=%u owners=%u "
         "group_bytes=%u kv_bytes_per_vc=%u end=%u free=%u\n",
         AVS_ROWS, AVS_TILE, BK, AVS_OWNERS * NUM_GROUPS,
         (uint32_t)sizeof(q38_avs_local_t),
         AVS_KV_MATRICES * AVS_MAT_ELEMS * 2u, avs_layout_end,
         (uint32_t)&__heap_end - avs_layout_end);
  for (uint32_t e = 0; e < NUM_REDMULE_TILES; ++e)
    printf("[qwen38_avs_engine] engine=%u jobs=%u\n", e, avs_stats[e].jobs);
  for (uint32_t c = 0; c < NUM_CORES; ++c)
    printf("[qwen38_avs_core] core=%u rows=%u rm_interval=%u softmax=%u "
           "rescale=%u dma_wait=%u sync=%u finish=%u\n",
           c, avs_stats[c / NUM_CORES_PER_TILE].rows[c % NUM_CORES_PER_TILE],
           avs_stats[c / NUM_CORES_PER_TILE]
               .cycles[c % NUM_CORES_PER_TILE][AVS_RM],
           avs_stats[c / NUM_CORES_PER_TILE]
               .cycles[c % NUM_CORES_PER_TILE][AVS_SM],
           avs_stats[c / NUM_CORES_PER_TILE]
               .cycles[c % NUM_CORES_PER_TILE][AVS_RESCALE],
           avs_stats[c / NUM_CORES_PER_TILE]
               .cycles[c % NUM_CORES_PER_TILE][AVS_DMA],
           avs_stats[c / NUM_CORES_PER_TILE]
               .cycles[c % NUM_CORES_PER_TILE][AVS_SYNC],
           avs_stats[c / NUM_CORES_PER_TILE]
               .cycles[c % NUM_CORES_PER_TILE][AVS_FINISH]);
  for (uint32_t c = 0; c < NUM_CORES; ++c)
    printf(
        "[qwen38_avs_wait] core=%u cycles=%u\n", c,
        avs_wait_stats[c / NUM_CORES_PER_TILE].cycles[c % NUM_CORES_PER_TILE]);
}
#endif
#endif
