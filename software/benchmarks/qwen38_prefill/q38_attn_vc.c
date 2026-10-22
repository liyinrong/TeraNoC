// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// The phased attention mixer (Q38_ATTN_VC): three phases over all mixer rows,
// each with the whole L1 to itself. Included by layer_resident.c after
// q38_attention.c; not a translation unit.
//
// 1. Q/gate/K/V projection of every row block as one GEMM on the VC scheme of
//    the FFN (q38_ffn.c): weights double-buffered per 2x2 VC, one fractal_sync
//    barrier per buffer switch, the next job programmed while the current one
//    runs. A unit is one head's columns (Q38_HD / NT panels) of one row tile,
//    computed by one engine on consecutive job-steps, so its tile's worker
//    finishes it -- Q and K head norm and rotary -- from group-local Z,
//    and stores it to L2: Q and the gate head-major, K and V into the cache.
// 2. The causal key sweep of every window (query_sweep_pipe), its queries and
//    gates loaded from L2, the gated context stored to L2.
// 3. Output projection of every row block, one column panel per unit, its
//    worker adding the residual.

#if Q38_ATTN_VC
extern uint32_t __heap_end;
// Rows per block: Q/gate/K/V and output projection.
#define Q38_AV_R Q38_ATTN_VC_ROWS
#define Q38_AV_OR Q38_ATTN_VC_OROWS
#define Q38_AV_U (Q38_HD / NT)
#define Q38_AV_QKV_UNITS (2u * Q38_QH + 2u * Q38_KVH)
#define Q38_AV_O_UNITS (Q38_D / NT)
#define Q38_AV_MIXER_ROWS (Q38_MIXER_EXEC_END - Q38_MIXER_EXEC_START)
// A phase's row block: whole row tiles, VC engines split into row tiles and
// units, whole mixer windows and blocks.
#define Q38_AV_ROWS_OK(rows)                                                   \
  ((rows) % Q38_PROJECTION_ROWS == 0u &&                                       \
   Q38_VC_ENGINES % ((rows) / Q38_PROJECTION_ROWS) == 0u &&                    \
   (rows) % Q38_WIN == 0u && Q38_AV_MIXER_ROWS % (rows) == 0u)
#define Q38_AV_TILE (Q38_PROJECTION_ROWS * NT)
#define Q38_AV_MAX_STEPS 64u
_Static_assert(Q38_KIND == 0 && Q38_ATTN_LOCAL && Q38_ATTN_LOCAL_PIPE &&
                   Q38_ATTN_RM_ACCUM && !Q38_ATTN_RM_FOLD &&
                   !Q38_ATTN_SUPERTILE && (Q38_FFN_VC || Q38_FFN_PIPE),
               "Phased attention runs the pipelined local sweep");
_Static_assert(VIRTUAL_CLUSTER &&FSYNC_TPG_SIDE != 0 &&
                   NUM_GROUPS % Q38_VC_GROUPS == 0u &&
                   NUM_REDMULE_TILES == NUM_GROUPS * NUM_TILES_PER_GROUP &&
                   Q38_VC_ENGINES == 4u * NUM_TILES_PER_GROUP &&
                   Q38_AV_ROWS_OK(Q38_AV_R) && Q38_AV_ROWS_OK(Q38_AV_OR) &&
                   Q38_HD % NT == 0u && Q38_D % NT == 0u &&
                   BK % Q38_PROJECTION_ROWS == 0u &&
                   Q38_D * NT * 2u % Q38_VC_ROW == 0u &&
                   Q38_QDIM * NT * 2u % Q38_VC_ROW == 0u &&
                   Q38_PROJECTION_ROWS * Q38_HD * 2u % VC_ROW_BYTES == 0u &&
                   VC_ROW_BYTES % (Q38_HD * 2u) == 0u &&
                   VC_ROW_BYTES % (Q38_PROJECTION_ROWS * 2u) == 0u &&
                   Q38_HD % (VC_ROW_BYTES / (Q38_PROJECTION_ROWS * 2u)) == 0u &&
                   Q38_REDMULE_MATMUL && !Q38_DELAY_SEED,
               "Phased attention: whole VCs, head units, aligned row blocks");

// Attention-only handshakes, reclaimed by FFN after the final attention join.
// Persistent timing records use Q38_VC_TILE_LOCAL and remain outside this area.
// Global alignment and the tile/group window mappings are unchanged.
#define Q38_AV_STATE                                                           \
  __attribute__((section(".l1_attn"), aligned(NUM_BANKS * sizeof(uint32_t))))

// Projected Q and gates (head-major, [head][row][Q38_HD]) and the gated context
// ([row][Q38_QDIM]) between the phases. They are too large for the program's
// L2 image, which ends below the data sidecar, so they take the top of L2,
// above it (run_layers.py checks the two are disjoint).
#define Q38_AV_HEADS_ELEMS (Q38_QH * Q38_R * Q38_HD)
#define Q38_AV_L2_BYTES ((2u * Q38_AV_HEADS_ELEMS + Q38_R * Q38_QDIM) * 2u)
#if DOMAINS > 1 && DOMAIN_L2_HOME && Q38_L2_TILED
// Home builds: the attention scratch in home segment 0 (below the wrap),
// from its top down: gated context (+ Q/gates), P, the K/V cache, the mixer
// output (AV_CACHE, AV_MIXER: not result_* below the home base). Row aligned.
#define AV_L2_END (DMA_DOMAIN_HOME_BASE + DMA_DOMAIN_HOME_WRAP / DOMAINS)
#define av_gated ((__fp16 *)(AV_L2_END - Q38_AV_L2_BYTES))
#else
#define av_gated ((__fp16 *)(L2_BASE + L2_SIZE - Q38_AV_L2_BYTES))
#endif
#define av_q (av_gated + Q38_R * Q38_QDIM)
#define av_gate (av_q + Q38_AV_HEADS_ELEMS)
_Static_assert(Q38_R *Q38_QDIM * 2u % 16384u == 0u &&
                   Q38_AV_HEADS_ELEMS * 2u % 16384u == 0u,
               "Phased attention L2 buffers keep 16 KiB alignment");
// Out-projection on the f2 down pass (as gdn2_output) instead of the VC
// projection: the gated context is then [slices][R][QDIM / slices]. RedMulE
// reads X at a row stride of its K, and K = QDIM (12 KiB) puts the 32 rows of
// a job on only two offsets of the 8 KiB global L1 row.
#ifndef Q38_AV_OUT_F2
#define Q38_AV_OUT_F2 0
#endif
#ifndef Q38_AV_OUT_SLICES
#define Q38_AV_OUT_SLICES (Q38_AV_OUT_F2 ? 2u : 1u)
#endif
// Out-projection: three weight slots (engines meet one step behind).
#ifndef Q38_AV_OUT_LAG
#define Q38_AV_OUT_LAG 1u
#endif
#define AV_GK (Q38_QDIM / Q38_AV_OUT_SLICES)
// Head h's columns of token's gated row; AV_GPITCH(h): its row pitch.
#if Q38_AV_OUT_SLICES == 1
#define AV_GATED(token, h) (av_gated + (token)*Q38_QDIM + (h)*Q38_HD)
#define AV_GPITCH(h) Q38_QDIM
#elif Q38_L2_TILED
// Two slices, the first of whole 8 KiB L2 rows (AV_GK0 columns, as F2_KC),
// the second the rest (f2.k_last): gated rows stay inside L2 rows.
#define AV_GK0 4096u
_Static_assert(
    Q38_AV_OUT_SLICES == 2u && Q38_QDIM > AV_GK0 && AV_GK0 % Q38_HD == 0u &&
        8192u % ((Q38_QDIM - AV_GK0) * 2u) == 0u,
    "Tiled gated context: [4096 | rest] columns, rest divides a row");
#define AV_GPITCH(h) ((h)*Q38_HD < AV_GK0 ? AV_GK0 : Q38_QDIM - AV_GK0)
#define AV_GATED(token, h)                                                     \
  ((h)*Q38_HD < AV_GK0                                                         \
       ? av_gated + (token)*AV_GK0 + (h)*Q38_HD                                \
       : av_gated + Q38_R * AV_GK0 + (token) * (Q38_QDIM - AV_GK0) +           \
             (h)*Q38_HD - AV_GK0)
#else
#define AV_GATED(token, h)                                                     \
  (av_gated + ((h)*Q38_HD / AV_GK * Q38_R + (token)) * AV_GK +                 \
   (h)*Q38_HD % AV_GK)
#define AV_GPITCH(h) AV_GK
#endif
_Static_assert(AV_GK *Q38_AV_OUT_SLICES == Q38_QDIM && AV_GK % Q38_HD == 0u,
               "A head's gated columns lie in one out-projection slice");
_Static_assert(Q38_AV_OUT_F2 ? Q38_FFN_L2 != 0 : Q38_AV_OUT_SLICES == 1u,
               "Sliced gated context needs the f2 out-projection");
// Q/gate/K/V projection on the f2 up pass (as gdn2_project) into P = [R][Q
// gate | K | V] below the gated context, then a separate epilogue pass.
#ifndef Q38_AV_QKV_F2
#define Q38_AV_QKV_F2 0
#endif
// Debug (timing probe): skip the epilogue pass; Q, gates and the caches stay
// unwritten.
#ifndef Q38_AV_QKV_F2_EPI
#define Q38_AV_QKV_F2_EPI 1
#endif
// Debug (timing probe): skip the up pass; the epilogue runs alone on whatever
// P holds.
#ifndef Q38_AV_QKV_F2_UP
#define Q38_AV_QKV_F2_UP 1
#endif
// Debug: the epilogue's busy and wait cycles per core kind, printed by core 0.
#ifndef Q38_AV_QKV_F2_PROF
#define Q38_AV_QKV_F2_PROF 0
#endif
// P head-major, [unit][R][HD] (f2's sliced output with a head-wide slice):
// every epilogue load reads one contiguous slab.
#ifndef Q38_AV_QKV_F2_HEADS
#define Q38_AV_QKV_F2_HEADS 0
#endif
// Epilogue rows with the same arithmetic but fewer stalls (avq_epi_row_fast).
#ifndef Q38_AV_QKV_F2_FAST
#define Q38_AV_QKV_F2_FAST 0
#endif
#define AV_PCOLS (2u * Q38_QDIM + 2u * Q38_KVH * Q38_HD)
// P row stride: an odd number of 512 B L2 granules. The 16 L2 banks
// interleave by granule, and 56 granules (14336 columns) put every row of a
// head column on two banks. Measured: no effect on the epilogue (its loads
// are latency-bound), kept since it costs nothing.
#ifndef Q38_AV_QKV_F2_PAD
#define Q38_AV_QKV_F2_PAD 1
#endif
#define AV_PSTRIDE                                                             \
  ((AV_PCOLS / 256u + (Q38_AV_QKV_F2_PAD && AV_PCOLS / 256u % 2u == 0u)) * 256u)
#define AV_P_L2 (av_gated - Q38_R * AV_PSTRIDE)
// K/V cache ([R / BK][K | V][KVH] blocks) and mixer output of this path.
#if DOMAINS > 1 && DOMAIN_L2_HOME && Q38_L2_TILED
#define AV_CACHE (AV_P_L2 - 2u * Q38_R * Q38_KVH * Q38_HD)
#define AV_MIXER (AV_CACHE - Q38_R * Q38_D)
_Static_assert(Q38_R *AV_PSTRIDE * 2u % 8192u == 0u &&
                   Q38_R * Q38_KVH * Q38_HD * 4u % 8192u == 0u &&
                   Q38_R * Q38_D * 2u % 8192u == 0u,
               "Home attention scratch: whole 8 KiB L2 rows");
#else
#define AV_CACHE result_cache
#define AV_MIXER result_mixer
#endif
// Q and gates stay in P (head-major): the slabs of head h are 2h and 2h + 1,
// so the epilogue rotates Q in place and leaves the gates alone.
#ifndef Q38_AV_QKV_F2_DIRECT
#define Q38_AV_QKV_F2_DIRECT 0
#endif
_Static_assert(!Q38_AV_QKV_F2_DIRECT ||
                   (Q38_AV_QKV_F2 && Q38_AV_QKV_F2_HEADS && Q38_AV_QKV_F2_EPI),
               "Q and gates in P need the head-major f2 projection");
#if Q38_AV_QKV_F2_DIRECT
#undef av_q
#undef av_gate
#define av_q AV_P_L2
#define av_gate (AV_P_L2 + Q38_R * Q38_HD)
#define AV_QG_ROWS (2u * Q38_R)
#else
#define AV_QG_ROWS Q38_R
#endif
// f2's panel width (q38_ffn.c): the three weights are packed at it.
#define AV_FNT Q38_FFN_COLUMNS
_Static_assert(!Q38_AV_QKV_F2 || (Q38_FFN_L2 != 0 && AV_PCOLS % 256u == 0u &&
                                  Q38_KVH * Q38_HD % AV_FNT == 0u),
               "QKV f2: P rows are whole L2 granules and whole panels");

// n bursts of len bytes, dest and src stepping d2 and s2 (dma 2-D). With
// Q38_L2_TILED, as planes of k bursts (k * the L2 pitch = one 8 KiB L2 row)
// stepping d2 and s2 from a plane's last burst: the same bytes, each plane in
// one L2 row once the L2 side starts row aligned. A short last plane goes as
// its own 2-D walk.
static inline __attribute__((always_inline)) dma_id_t
av_dma_rows(void *dst, const void *src, uint32_t len, uint32_t d2, uint32_t s2,
            uint32_t n, uint32_t k) {
#if Q38_L2_TILED
  if (n > k) {
    const uint32_t full = n / k * k;
    dma_id_t id = dma_local_memcpy_3d_nonblocking(0u, dst, src, len, d2, s2, k,
                                                  d2, s2, n / k);
    if (full == n)
      return id;
    return dma_local_memcpy_2d_nonblocking(0u, (char *)dst + full * d2,
                                           (const char *)src + full * s2, len,
                                           d2, s2, n - full);
  }
#else
  (void)k;
#endif
  return dma_local_memcpy_2d_nonblocking(0u, dst, src, len, d2, s2, n);
}

// A projection phase: kind 0 is Q/gate/K/V, kind 1 the output projection.
static struct {
  __fp16 *x;   // input rows [rows][k]
  __fp16 *z;   // group window: two units of `upanels` Z tiles per local engine
  __fp16 *epi; // group window: one row of Q38_HD per local engine (kind 0)
  __fp16 *res; // residual rows [rows][Q38_D] (kind 1)
  __fp16 *cos, *sin;
  uint32_t vc_offset, slot_bytes, kind, units, upanels, k;
  uint32_t rows, pv, p, blocks;
  uint32_t unit_steps, block_steps, steps, base, stage_row;
} av Q38_VC_SMALL;
// Finished Q/gate/K/V units leave through the group's iDMA, whose single
// issuer (the coordinator) may only address its own group's L1: each worker
// hands its unit over in one of two group-local staging slots, and the
// issuer ships them between its own jobs.
#define Q38_AV_UNIT_BYTES (Q38_PROJECTION_ROWS * Q38_HD * 2u)
#define Q38_AV_STAGE_ROW VC_ROW_BYTES
#define Q38_AV_STAGE_ROWS                                                      \
  (NUM_TILES_PER_GROUP * 2u * Q38_AV_UNIT_BYTES / VC_ROW_BYTES)
// Per engine, monotone over the phase's units: shipped (issued) and landed
// (sent), and the ids of the two slots in flight.
typedef struct {
  uint32_t unit_done, epi_done, issued, sent;
  dma_id_t ids[2];
#if Q38_AV_DUAL
  uint32_t worker_done, unit_armed;
  uint32_t claimed[2], rows_done[2];
  uint32_t completed[2];
  uint32_t padding[NUM_BANKS_PER_TILE - 14u];
#else
  uint32_t padding[NUM_BANKS_PER_TILE - 6u];
#endif
} q38_av_progress_t;
typedef struct {
  uint32_t block, unit, row, elapsed, worst;
} q38_av_work_t;
_Static_assert(sizeof(q38_av_progress_t) == NUM_BANKS_PER_TILE * 4u,
               "Attention progress must occupy one owner-tile bank stripe");
static q38_av_progress_t av_progress[NUM_REDMULE_TILES] Q38_AV_STATE;

static uint32_t av_copy_unit(uint32_t u) {
  return u < 2u * Q38_QH ? u & 1u : u >= 2u * Q38_QH + Q38_KVH;
}
// First unit of each unit-step (the probe keeps its leading steps and the
// final one).
static uint16_t av_first[Q38_AV_MAX_STEPS] Q38_VC_SMALL;
static dma_id_t av_pending[NUM_GROUPS] Q38_VC_SMALL;
// DMA targets: a beat must not straddle a tile, so they are row aligned.
static __fp16 av_qg[Q38_HD] LOCAL;
static __fp16 av_kg[Q38_HD] LOCAL;
// Probe: per-engine agent cycles by phase and part, and the phase bounds.
enum { AVP_WEIGHTS, AVP_WORKER, AVP_START, AVP_RUN, AVP_BARRIER, AVP_COUNT };
#if Q38_SCHEDULE_PROBE
static uint32_t av_prof[2][AVP_COUNT][NUM_REDMULE_TILES] Q38_VC_SMALL;
#define AVP_ADD(k, t0)                                                         \
  (av_prof[av.kind][k][engine] += mempool_get_timer() - (t0))
#else
#define AVP_ADD(k, t0) ((void)(k), (void)(t0))
#endif

#if Q38_AV_PROFILE
static uint32_t av_bounds[4] Q38_VC_SMALL;
static struct {
  uint32_t rows, tile, columns, weight_bytes, z_bytes, stage_bytes, end, free;
} av_layout[2] Q38_VC_SMALL;
enum { AVE_K, AVE_V, AVE_Q, AVE_GATE, AVE_O, AVE_COUNT };
typedef struct {
  uint32_t count, sum, maximum;
} q38_av_timing_t;
typedef struct {
  q38_av_timing_t epi[AVE_COUNT], panel[AVE_COUNT];
  uint32_t padding[NUM_BANKS_PER_TILE - 6u * AVE_COUNT];
} q38_av_pair_timing_t;
typedef struct {
  q38_av_timing_t unit[AVE_COUNT];
  uint32_t rows[AVE_COUNT];
  uint32_t padding[NUM_BANKS_PER_TILE - 4u * AVE_COUNT];
} q38_av_unit_timing_t;
static q38_av_pair_timing_t av_times[NUM_REDMULE_TILES] Q38_VC_TILE_LOCAL;
static q38_av_unit_timing_t av_units[NUM_REDMULE_TILES] Q38_VC_TILE_LOCAL;
#if Q38_AV_DUAL
static q38_av_unit_timing_t av_agent_times[NUM_REDMULE_TILES] Q38_VC_TILE_LOCAL;
#if Q38_AV_STATIC_CORES > 2
static q38_av_unit_timing_t av_extra_times[Q38_AV_STATIC_CORES - 2]
                                          [NUM_REDMULE_TILES] Q38_VC_TILE_LOCAL;
#endif
#endif
static void av_time_add(q38_av_timing_t *p, uint32_t duration) {
  ++p->count;
  p->sum += duration;
  p->maximum = Q38_MAX(p->maximum, duration);
}
static void av_profile_init(void) {
  if (tnc_is_tile_agent()) {
    const uint32_t e = mempool_get_redmule_id();
    for (uint32_t k = 0u; k < AVE_COUNT; ++k) {
      av_times[e].epi[k] = (q38_av_timing_t){0u, 0u, 0u};
      av_times[e].panel[k] = (q38_av_timing_t){0u, 0u, 0u};
      av_units[e].unit[k] = (q38_av_timing_t){0u, 0u, 0u};
      av_units[e].rows[k] = 0u;
#if Q38_AV_DUAL
      av_agent_times[e].unit[k] = (q38_av_timing_t){0u, 0u, 0u};
      av_agent_times[e].rows[k] = 0u;
#if Q38_AV_STATIC_CORES > 2
      for (uint32_t c = 0u; c < Q38_AV_STATIC_CORES - 2u; ++c) {
        av_extra_times[c][e].unit[k] = (q38_av_timing_t){0u, 0u, 0u};
        av_extra_times[c][e].rows[k] = 0u;
      }
#endif
#endif
    }
  }
}
#endif

static __attribute__((noinline)) void av_setup(uint32_t kind) {
  av.kind = kind;
  av.units = kind ? Q38_AV_O_UNITS : Q38_AV_QKV_UNITS;
  av.upanels = kind ? 1u : Q38_AV_U;
  av.k = kind ? Q38_QDIM : Q38_D;
  av.rows = kind ? Q38_AV_OR : Q38_AV_R;
  av.pv = Q38_VC_ENGINES / (av.rows / Q38_PROJECTION_ROWS);
  av.p = av.pv * Q38_VC_COUNT;
  av.blocks = Q38_AV_MIXER_ROWS / av.rows;
  uint32_t n = 0u;
  for (uint32_t first = 0u; first < av.units; ++n) {
    if (n >= Q38_AV_MAX_STEPS)
      q38_fatal("[qwen38_fatal] attn vc unit-steps=", n);
    av_first[n] = (uint16_t)first;
    first = next_panel_group(first, av.p, av.units, av.p);
  }
  av.unit_steps = n;
  av.block_steps = n * av.upanels;
  av.steps = av.block_steps * av.blocks;
  uint32_t p = (uint32_t)&stage < (uint32_t)&weights[0][0]
                   ? (uint32_t)&stage
                   : (uint32_t)&weights[0][0];
  p = (p + VC_GLOBAL_ROW_BYTES - 1u) / VC_GLOBAL_ROW_BYTES *
      VC_GLOBAL_ROW_BYTES;
  av.vc_offset = VC_GLOBAL_ROW(p) * Q38_VC_ROW;
  av.slot_bytes =
      (av.pv * av.k * NT * 2u + Q38_VC_ROW - 1u) / Q38_VC_ROW * Q38_VC_ROW;
  p += 2u * av.slot_bytes / Q38_VC_ROW * VC_GLOBAL_ROW_BYTES;
  av.x = (__fp16 *)p;
  p += av.rows * av.k * sizeof(__fp16);
  if (kind) {
    av.res = (__fp16 *)p;
    p += av.rows * Q38_D * sizeof(__fp16);
  }
  p = (p + VC_GLOBAL_ROW_BYTES - 1u) / VC_GLOBAL_ROW_BYTES *
      VC_GLOBAL_ROW_BYTES;
  av.z = vc_group_window(p);
  const uint32_t z_rows =
      (NUM_TILES_PER_GROUP * 2u * av.upanels * Q38_AV_TILE * sizeof(__fp16) +
       VC_ROW_BYTES - 1u) /
      VC_ROW_BYTES;
  p += z_rows * VC_GLOBAL_ROW_BYTES;
  if (!kind) {
#if !Q38_AV_PANEL_AWARE
    av.epi = vc_group_window(p);
    const uint32_t epi_rows =
        (NUM_TILES_PER_GROUP * Q38_HD * sizeof(__fp16) + VC_ROW_BYTES - 1u) /
        VC_ROW_BYTES;
    p += epi_rows * VC_GLOBAL_ROW_BYTES;
#endif
    av.cos = (__fp16 *)p;
    p += (av.rows * Q38_HALF * sizeof(__fp16) + 255u) & ~255u;
    av.sin = (__fp16 *)p;
    p += (av.rows * Q38_HALF * sizeof(__fp16) + 255u) & ~255u;
    p = (p + VC_GLOBAL_ROW_BYTES - 1u) / VC_GLOBAL_ROW_BYTES *
        VC_GLOBAL_ROW_BYTES;
    av.stage_row = p / VC_GLOBAL_ROW_BYTES;
    p += Q38_AV_STAGE_ROWS * VC_GLOBAL_ROW_BYTES;
  }
  if (p > (uint32_t)&__heap_end)
    q38_fatal("[qwen38_fatal] attn vc layout bytes over L1=",
              p - (uint32_t)&__heap_end);
#if Q38_AV_PROFILE
  av_layout[kind].rows = av.rows;
  av_layout[kind].tile = Q38_PROJECTION_ROWS;
  av_layout[kind].columns = NT;
  av_layout[kind].weight_bytes =
      2u * av.slot_bytes / Q38_VC_ROW * VC_GLOBAL_ROW_BYTES;
  av_layout[kind].z_bytes = z_rows * VC_GLOBAL_ROW_BYTES;
  av_layout[kind].stage_bytes =
      kind ? 0u : Q38_AV_STAGE_ROWS * VC_GLOBAL_ROW_BYTES;
  av_layout[kind].end = p;
  av_layout[kind].free = (uint32_t)&__heap_end - p;
#endif
  for (uint32_t e = 0u; e < NUM_REDMULE_TILES; ++e) {
    av_progress[e].unit_done = av_progress[e].epi_done = av_progress[e].issued =
        av_progress[e].sent = 0u;
#if Q38_AV_DUAL
    av_progress[e].worker_done = 0u;
    av_progress[e].unit_armed = 0u;
    for (uint32_t s = 0; s < 2u; ++s) {
      av_progress[e].claimed[s] = av_progress[e].rows_done[s] = 0u;
      av_progress[e].completed[s] = 0u;
    }
#endif
  }
  for (uint32_t g = 0u; g < NUM_GROUPS; ++g)
    av_pending[g] = 0u;
#if Q38_SCHEDULE_PROBE
  for (uint32_t k = 0u; k < AVP_COUNT; ++k)
    for (uint32_t e = 0u; e < NUM_REDMULE_TILES; ++e)
      av_prof[kind][k][e] = 0u;
  probe_projection_groups += av.unit_steps * av.blocks;
#endif
}

static void av_step(uint32_t t, uint32_t *b, uint32_t *n, uint32_t *p) {
  *b = t / av.block_steps;
  const uint32_t j = t - *b * av.block_steps;
  *n = j / av.upanels;
  *p = j - *n * av.upanels;
}

// Units of VC v in unit-step n: av.pv, fewer at the tail, or none.
static uint32_t av_count(uint32_t v, uint32_t n) {
  const uint32_t first = av_first[n] + v * av.pv;
  return first >= av.units ? 0u : Q38_MIN(av.pv, av.units - first);
}

static uint32_t av_slot_offset(uint32_t t) {
  return av.vc_offset + (t & 1u) * av.slot_bytes;
}

// Byte `o` of engine e's staging slot, in its group's stripe.
static char *av_stage_at(uint32_t e, uint32_t slot, uint32_t o) {
  o += (e % NUM_TILES_PER_GROUP * 2u + slot) * Q38_AV_UNIT_BYTES;
  return (char *)((av.stage_row + o / VC_ROW_BYTES) * VC_GLOBAL_ROW_BYTES +
                  e / NUM_TILES_PER_GROUP * VC_ROW_BYTES + o % VC_ROW_BYTES);
}

// Unit of engine e in unit-step n (~0 without one), and its row tile.
static uint32_t av_unit_of(uint32_t e, uint32_t n, uint32_t *r) {
  const uint32_t v = e / Q38_VC_ENGINES, ev = e % Q38_VC_ENGINES;
  const uint32_t q = ev % av.pv;
  *r = ev / av.pv;
  return q < av_count(v, n) ? av_first[n] + v * av.pv + q : ~0u;
}

// Q/gate/K/V units in schedule order: K and V heads first, then Q heads,
// then the gates, so a block ends on the units whose finish is a plain copy.
// Returns the head unit: 2h is Q of head h, 2h + 1 its gate, then K, then V.
static uint32_t av_head_unit(uint32_t unit) {
  if (unit < 2u * Q38_KVH)
    return 2u * Q38_QH + unit;
  unit -= 2u * Q38_KVH;
  return unit < Q38_QH ? 2u * unit : 2u * (unit - Q38_QH) + 1u;
}

#if Q38_AV_PROFILE
static uint32_t av_time_kind(uint32_t u) {
  if (av.kind)
    return AVE_O;
  u = av_head_unit(u);
  return u < 2u * Q38_QH ? (u & 1u ? AVE_GATE : AVE_Q)
                         : (u < 2u * Q38_QH + Q38_KVH ? AVE_K : AVE_V);
}

static NOSTACK void av_timing_report(void) {
  for (uint32_t k = 0; k < 2u; ++k)
    printf("[qwen38_av_layout] kind=%u rows=%u tile=%u columns=%u "
           "weight_bytes=%u z_bytes=%u stage_bytes=%u end=%u free=%u "
           "kv_bytes=0\n",
           k, av_layout[k].rows, av_layout[k].tile, av_layout[k].columns,
           av_layout[k].weight_bytes, av_layout[k].z_bytes,
           av_layout[k].stage_bytes, av_layout[k].end, av_layout[k].free);
  printf("[qwen38_av_bounds] qkv=%u sweep=%u out=%u "
         "start=%u qkv_end=%u sweep_end=%u end=%u\n",
         av_bounds[1] - av_bounds[0], av_bounds[2] - av_bounds[1],
         av_bounds[3] - av_bounds[2], av_bounds[0], av_bounds[1], av_bounds[2],
         av_bounds[3]);
  for (uint32_t k = 0; k < AVE_COUNT; ++k)
    for (uint32_t e = 0; e < NUM_REDMULE_TILES; ++e)
      printf("[qwen38_av_timing] kind=%u engine=%u epi_count=%u epi_sum=%u "
             "epi_max=%u panel_count=%u panel_sum=%u panel_max=%u "
             "unit_count=%u unit_sum=%u unit_max=%u\n",
             k, e, av_times[e].epi[k].count, av_times[e].epi[k].sum,
             av_times[e].epi[k].maximum, av_times[e].panel[k].count,
             av_times[e].panel[k].sum, av_times[e].panel[k].maximum,
             av_units[e].unit[k].count, av_units[e].unit[k].sum,
             av_units[e].unit[k].maximum);
#if Q38_AV_DUAL
  for (uint32_t k = 0; k < AVE_COUNT; ++k)
    for (uint32_t e = 0; e < NUM_REDMULE_TILES; ++e) {
      printf("[qwen38_av_epi_core] kind=%u engine=%u core=0 count=%u sum=%u "
             "max=%u rows=%u\n",
             k, e, av_agent_times[e].unit[k].count,
             av_agent_times[e].unit[k].sum, av_agent_times[e].unit[k].maximum,
             av_agent_times[e].rows[k]);
      printf("[qwen38_av_epi_core] kind=%u engine=%u core=1 count=%u sum=%u "
             "max=%u rows=%u\n",
             k, e, av_times[e].epi[k].count, av_times[e].epi[k].sum,
             av_times[e].epi[k].maximum, av_units[e].rows[k]);
#if Q38_AV_STATIC_CORES > 2
      for (uint32_t c = 2u; c < Q38_AV_STATIC_CORES; ++c)
        printf("[qwen38_av_epi_core] kind=%u engine=%u core=%u count=%u "
               "sum=%u max=%u rows=%u\n",
               k, e, c, av_extra_times[c - 2u][e].unit[k].count,
               av_extra_times[c - 2u][e].unit[k].sum,
               av_extra_times[c - 2u][e].unit[k].maximum,
               av_extra_times[c - 2u][e].rows[k]);
#endif
    }
#endif
}
#endif

#include "q38_attn_vc_epi.c"
#if Q38_AV_DUAL
#define AV_PUMP(work) avp_service(work, 1u)
#define AV_CONTROL(work) avp_service(work, 0u)
#else
#define AV_PUMP(work) ((void)(work), 0u)
#define AV_CONTROL(work) ((void)(work), 0u)
#endif

// Issuer: ship engine e's staged unit `unit` to L2. Returns 0 without data.
static dma_id_t av_ship(uint32_t e, uint32_t unit) {
  const uint32_t n = unit % av.unit_steps, b = unit / av.unit_steps;
  uint32_t r;
  const uint32_t su = av_unit_of(e, n, &r);
  if (su == ~0u)
    return 0u;
  const uint32_t u = av_head_unit(su);
  const uint32_t token =
      Q38_MIXER_EXEC_START + b * av.rows + r * Q38_PROJECTION_ROWS;
  const char *const src = av_stage_at(e, unit & 1u, 0u);
  const uint32_t heads = 2u * Q38_QH, half = Q38_KVH * Q38_HD * BK;
  __fp16 *const block = AV_CACHE + token / BK * 2u * half;
  if (u >= heads && u < heads + Q38_KVH) {
    // K transposed: Q38_HD runs of Q38_PROJECTION_ROWS keys, BK apart.
    const uint32_t run = Q38_PROJECTION_ROWS * 2u;
    const uint32_t per = Q38_AV_STAGE_ROW / run;
    return dma_local_memcpy_3d_nonblocking(
        0u, block + (u - heads) * Q38_HD * BK + token % BK, src, run, BK * 2u,
        run, per, BK * 2u, VC_GLOBAL_ROW_BYTES - (per - 1u) * run,
        Q38_AV_UNIT_BYTES / Q38_AV_STAGE_ROW);
  }
  __fp16 *const dst =
      u < heads
          ? (u & 1u ? av_gate : av_q) + ((u / 2u) * AV_QG_ROWS + token) * Q38_HD
          : block + half + ((u - heads - Q38_KVH) * BK + token % BK) * Q38_HD;
#if Q38_AV_DIRECT_DMA
  if (av_copy_unit(u)) {
    const __fp16 *const z =
        av.z +
        (e % NUM_TILES_PER_GROUP * 2u + (unit & 1u)) * av.upanels * Q38_AV_TILE;
    const uint32_t len = NT * 2u, per = VC_ROW_BYTES / len;
    const uint32_t rows = Q38_MIN(per, Q38_PROJECTION_ROWS);
    dma_id_t id = 0u;
    // DONE_ID is monotone: the last panel's completion covers every earlier
    // descriptor on this stream. Each chunk remains in this group's stripe.
    for (uint32_t p = 0u; p < av.upanels; ++p)
      id = dma_local_memcpy_3d_nonblocking(
          0u, dst + p * NT, Q38_VC_GLOBAL(z + p * Q38_AV_TILE), len,
          Q38_HD * 2u, len, rows, Q38_HD * 2u,
          VC_GLOBAL_ROW_BYTES - (rows - 1u) * len, Q38_PROJECTION_ROWS / rows);
    return id;
  }
#endif
  return dma_local_memcpy_2d_nonblocking(0u, dst, src, Q38_AV_STAGE_ROW,
                                         Q38_AV_STAGE_ROW, VC_GLOBAL_ROW_BYTES,
                                         Q38_AV_UNIT_BYTES / Q38_AV_STAGE_ROW);
}

// Issuer: retire its group's landed units and, with `ship`, ship the newly
// staged ones. Returns whether none is in flight and every unit below `until`
// has landed.
//
// GVSoC workaround: the model's DMA-to-TCDM beat adapter cannot carry an
// L1-read and an L1-write transfer of one group at once (it violates the bank
// interconnect's retry contract and aborts), so these L1 -> L2 ships never
// overlap the group's L2 -> L1 weight loads: the issuer ships only once its
// pending load landed, and lets its ships land before the next load.
static uint32_t av_drain(uint32_t ship, uint32_t until) {
  uint32_t all = 1u;
  const uint32_t e0 = mempool_get_group_id() * NUM_TILES_PER_GROUP;
  for (uint32_t e = e0; e < e0 + NUM_TILES_PER_GROUP; ++e) {
    uint32_t sent = av_progress[e].sent;
    while (sent < av_progress[e].issued &&
           (av_progress[e].ids[sent & 1u] == 0u ||
            dma_local_done_id(0u, av_progress[e].ids[sent & 1u])))
      ++sent;
    const uint32_t ready =
        __atomic_load_n(&av_progress[e].epi_done, __ATOMIC_ACQUIRE);
    while (ship && av_progress[e].issued < ready &&
           av_progress[e].issued < sent + 2u) {
      av_progress[e].ids[av_progress[e].issued & 1u] =
          av_ship(e, av_progress[e].issued);
      ++av_progress[e].issued;
    }
    while (sent < av_progress[e].issued && av_progress[e].ids[sent & 1u] == 0u)
      ++sent;
    __atomic_store_n(&av_progress[e].sent, sent, __ATOMIC_RELEASE);
    all &= sent >= until && sent == av_progress[e].issued;
  }
  return all;
}

// Panel p of a unit in L2, [panel][k][NT].
static const char *av_panel(uint32_t unit, uint32_t p) {
  if (av.kind)
    return (const char *)(w_out + unit * Q38_QDIM * NT);
  uint32_t h = av_head_unit(unit);
  const __fp16 *w = w_q_gate;
  if (h >= 2u * Q38_QH) {
    h -= 2u * Q38_QH;
    w = w_key;
    if (h >= Q38_KVH) {
      h -= Q38_KVH;
      w = w_value;
    }
  }
  return (const char *)(w + (h * Q38_AV_U + p) * Q38_D * NT);
}

// This group's stripe of job-step t's panels into its VC slot, one 3-D
// descriptor per run of equally spaced panels (see vc_load). Returns the last
// id, or 0 when the VC has no unit.
static dma_id_t av_load(uint32_t v, uint32_t t) {
  uint32_t b, n, p;
  av_step(t, &b, &n, &p);
  const uint32_t count = av_count(v, n);
  const uint32_t piece = av.k * NT * sizeof(__fp16);
  const uint32_t rows = piece / Q38_VC_ROW;
  const uint32_t group = mempool_get_group_id();
  dma_id_t id = 0u;
  for (uint32_t q = 0u; q < count;) {
    const uint32_t unit = av_first[n] + v * av.pv + q;
    const char *const src = av_panel(unit, p);
    uint32_t run = 1u, stride = piece;
    if (q + 1u < count) {
      stride = (uint32_t)(av_panel(unit + 1u, p) - src);
      run = 2u;
      while (q + run < count &&
             (uint32_t)(av_panel(unit + run, p) -
                        av_panel(unit + run - 1u, p)) == stride)
        ++run;
    }
    if ((uint32_t)src % VC_ROW_BYTES != 0u)
      q38_fatal("[qwen38_fatal] attn vc weights misaligned step=", t);
    char *const dest = (char *)((av_slot_offset(t) + q * piece) / Q38_VC_ROW *
                                    VC_GLOBAL_ROW_BYTES +
                                group * VC_ROW_BYTES);
    id = dma_local_memcpy_3d_nonblocking(
        0u, dest, src + (group - v * Q38_VC_GROUPS) * VC_ROW_BYTES,
        VC_ROW_BYTES, VC_GLOBAL_ROW_BYTES, Q38_VC_ROW, rows,
        VC_GLOBAL_ROW_BYTES, stride - (rows - 1u) * Q38_VC_ROW, run);
    q += run;
  }
  return id;
}

// Program job-step t into a free RedMulE context (see vc_configure).
static uint8_t av_configure(uint32_t t, __fp16 *z, uint32_t r, uint32_t q) {
  uint32_t b, n, p;
  av_step(t, &b, &n, &p);
  const uint32_t unit = b * av.unit_steps + n;
  const __fp16 *const slot =
      (const __fp16 *)(VC_WINDOW_BASE_1 + av_slot_offset(t));
  const int32_t target = tnc_redmule_acquire_target();
  redmule_cfg((uint32_t)(av.x + r * Q38_PROJECTION_ROWS * av.k),
              (uint32_t)(slot + q * av.k * NT),
              (uint32_t)(z + ((unit & 1u) * av.upanels + p) * Q38_AV_TILE),
              (uint16_t)Q38_PROJECTION_ROWS, (uint16_t)av.k, (uint16_t)NT, 0,
              MATMUL, Float16);
  redmule_x_shift_set(0);
  return (uint8_t)target;
}

// The VC barrier; an issuer with staged units to ship polls instead of
// sleeping, since a worker's handover raises no event.
static void av_barrier(uint32_t drain, q38_av_work_t *work) {
  fsync_trigger(FSYNC_ID_H, fsync_aggr_of_vc_level(1u));
  while (fsync_status() & FSYNC_STATUS_BUSY_MASK) {
    if (AV_CONTROL(work)) {
      if (drain)
        av_drain(1u, 0u);
      continue;
    }
    if (drain) {
      av_drain(1u, 0u);
      mempool_wait(64u);
    } else {
      vc_sleep(2u, 0u);
    }
  }
}

// Tile agent: every job-step of block b on this tile's RedMulE, and for a
// group coordinator its group's share of the VC's weight loads (as vc_agent)
// and the shipping of its group's finished Q/gate/K/V units.
static inline uint32_t av_z_dma_busy(uint32_t engine, uint32_t unit) {
#if Q38_AV_DIRECT_DMA
  if (!av.kind && unit >= 2u) {
    uint32_t tile;
    const uint32_t su = av_unit_of(engine, (unit - 2u) % av.unit_steps, &tile);
    return su != ~0u && av_copy_unit(av_head_unit(su)) &&
           __atomic_load_n(&av_progress[engine].sent, __ATOMIC_ACQUIRE) <
               unit - 1u;
  }
#else
  (void)engine;
  (void)unit;
#endif
  return 0u;
}

static NOSTACK void av_agent(uint32_t b) {
  q38_av_work_t work = {b, 0u, 0u, 0u, Q38_HD * 16u + 256u};
  const uint32_t engine = mempool_get_redmule_id();
  if (engine >= NUM_REDMULE_TILES) {
    fsync_subscribe_self(0u);
    return;
  }
  const uint32_t v = engine / Q38_VC_ENGINES, ev = engine % Q38_VC_ENGINES;
  const uint32_t r = ev / av.pv, q = ev % av.pv;
  const uint32_t issuer = mempool_is_group_coordinator();
  const uint32_t ships = issuer && !av.kind;
  __fp16 *const z =
      av.z + engine % NUM_TILES_PER_GROUP * 2u * av.upanels * Q38_AV_TILE;
  dma_id_t *const pending = &av_pending[mempool_get_group_id()];
  vc_subscribe(FSYNC_EVT_FSYNC | FSYNC_EVT_ENGINE |
               (issuer ? FSYNC_EVT_DMA : 0u));
  uint32_t t0 = mempool_get_timer();
  if (b == 0u) {
    if (issuer) {
      // av_first_load issued it.
      const dma_id_t first = *pending;
      while (first && !dma_local_done_id(0u, first))
        vc_sleep(1u, 0u);
      AVP_ADD(AVP_WEIGHTS, t0);
      t0 = mempool_get_timer();
    }
    vc_barrier();
    AVP_ADD(AVP_BARRIER, t0);
  }
  uint32_t load_t = b == 0u && issuer && av.steps > 1u ? 1u : ~0u;
  const uint32_t first = b * av.block_steps, last = first + av.block_steps;
  uint32_t sb, sn, sp;
  av_step(first, &sb, &sn, &sp);
  uint8_t next = q < av_count(v, sn) ? av_configure(first, z, r, q) : 0u;
#if Q38_AV_PROFILE
  uint32_t unit_start = 0u;
#endif
  for (uint32_t t = first; t < last; ++t) {
    av_step(t, &sb, &sn, &sp);
    const uint32_t mine = q < av_count(v, sn);
    const uint32_t unit = sb * av.unit_steps + sn;
    const uint8_t target = next;
    t0 = mempool_get_timer();
#if Q38_AV_PROFILE || Q38_AV_DUAL
    uint32_t panel_start = 0u;
#endif
    if (mine) {
      // A unit's first job overwrites unit - 2's Z tiles.
      if (sp == 0u && unit >= 2u)
        while (__atomic_load_n(&av_progress[engine].epi_done,
                               __ATOMIC_ACQUIRE) < unit - 1u ||
               av_z_dma_busy(engine, unit)) {
          if (!AV_PUMP(&work))
            mempool_wait(32u);
          if (ships)
            av_drain(!*pending || dma_local_done_id(0u, *pending), 0u);
          AVP_ADD(AVP_WORKER, t0);
          t0 = mempool_get_timer();
        }
#if Q38_AV_PROFILE || Q38_AV_DUAL
      panel_start = mempool_get_timer();
#endif
#if Q38_AV_PROFILE
      if (sp == 0u)
        unit_start = panel_start;
#endif
#if Q38_AV_DUAL
      if (sp == 0u)
        av_progress[engine].claimed[unit & 1u] =
            av_progress[engine].rows_done[unit & 1u] = 0u;
      if (sp == 0u)
        av_progress[engine].completed[unit & 1u] = 0u;
      if (sp + 1u == av.upanels) {
        __atomic_store_n(&av_progress[engine].unit_armed, unit + 1u,
                         __ATOMIC_RELEASE);
      }
#endif
      tnc_redmule_trigger();
    }
    if (load_t != ~0u) {
      while (ships && !av_drain(0u, 0u))
        if (!AV_CONTROL(&work))
          mempool_wait(16u);
      *pending = av_load(v, load_t);
      load_t = ~0u;
    }
    if (t + 1u < last) {
      uint32_t nb, nn, np;
      av_step(t + 1u, &nb, &nn, &np);
      if (q < av_count(v, nn))
        next = av_configure(t + 1u, z, r, q);
    }
    AVP_ADD(AVP_START, t0);
    uint32_t landed = !issuer || *pending == 0u;
    t0 = mempool_get_timer();
    if (mine) {
      while (!tnc_redmule_done(target)) {
        if (!landed)
          landed = dma_local_done_id(0u, *pending);
        if (ships)
          av_drain(landed, 0u);
#if Q38_AV_DUAL
        const uint32_t floor =
            Q38_PROJECTION_ROWS * av.k * NT / (REDMULE_H * 32u);
        const uint32_t elapsed = mempool_get_timer() - panel_start;
        const uint32_t progressed = elapsed + work.worst + 64u < floor
                                        ? AV_PUMP(&work)
                                        : AV_CONTROL(&work);
#else
        const uint32_t progressed = 0u;
#endif
        if (!progressed)
          vc_sleep(3u, t);
      }
      AVP_ADD(AVP_RUN, t0);
#if Q38_AV_PROFILE
      const uint32_t end = mempool_get_timer();
      const uint32_t kind = av_time_kind(av_first[sn] + v * av.pv + q);
      av_time_add(&av_times[engine].panel[kind], end - panel_start);
      if (sp + 1u == av.upanels)
        av_time_add(&av_units[engine].unit[kind], end - unit_start);
#endif
      if (sp + 1u == av.upanels)
        __atomic_store_n(&av_progress[engine].unit_done, unit + 1u,
                         __ATOMIC_RELEASE);
    }
    if (t + 1u >= av.steps)
      break;
    t0 = mempool_get_timer();
    while (!landed) {
      landed = dma_local_done_id(0u, *pending);
      if (!landed && !AV_CONTROL(&work))
        vc_sleep(4u, t);
    }
    AVP_ADD(AVP_WEIGHTS, t0);
    t0 = mempool_get_timer();
    av_barrier(ships, &work);
    if (issuer) {
      *pending = 0u;
      if (t + 2u < av.steps)
        load_t = t + 2u;
      if (load_t != ~0u && t + 1u == last) {
        while (ships && !av_drain(0u, 0u))
          if (!AV_CONTROL(&work))
            mempool_wait(16u);
        *pending = av_load(v, load_t);
        load_t = ~0u;
      }
    }
    AVP_ADD(AVP_BARRIER, t0);
  }
  // The next block's first load is in flight (see av_drain): let it land
  // before the block's last units ship, or the phase's block-end L1 -> L2
  // transfer starts.
  if (issuer)
    while (*pending && !dma_local_done_id(0u, *pending))
      if (!AV_PUMP(&work))
        mempool_wait(16u);
#if Q38_AV_DUAL
  while (work.unit < av.unit_steps) {
    const uint32_t progressed = AV_PUMP(&work);
    if (ships)
      av_drain(1u, 0u);
    if (!progressed)
      mempool_wait(16u);
  }
#endif
  if (ships)
    while (!av_drain(1u, (b + 1u) * av.unit_steps))
      if (!AV_PUMP(&work))
        mempool_wait(64u);
  fsync_subscribe_self(0u);
}

static void av_rotary(__fp16 *values, const __fp16 *cos, const __fp16 *sin) {
  const uint32_t half = Q38_HALF;
  const float zh = llm_f16box(0.0f);
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(half));
  asm volatile("vle16.v v0, (%0)" : : "r"(values) : "memory");
  asm volatile("vle16.v v1, (%0)" : : "r"(values + half) : "memory");
  asm volatile("vfwadd.vf v4, v0, %0" : : "f"(zh));
  asm volatile("vfwadd.vf v6, v1, %0" : : "f"(zh));
  asm volatile("vle16.v v0, (%0)" : : "r"(cos) : "memory");
  asm volatile("vle16.v v1, (%0)" : : "r"(sin) : "memory");
  asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
  asm volatile("vfwadd.vf v10, v1, %0" : : "f"(zh));
  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(half));
  asm volatile("vfmul.vv v12, v4, v8");
  asm volatile("vfmul.vv v14, v6, v10");
  asm volatile("vfsub.vv v12, v12, v14");
  asm volatile("vfmul.vv v14, v6, v8");
  asm volatile("vfmacc.vv v14, v4, v10");
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(half));
  asm volatile("vfncvt.f.f.w v0, v12");
  asm volatile("vfncvt.f.f.w v1, v14");
  asm volatile("vse16.v v0, (%0)" : : "r"(values) : "memory");
  asm volatile("vse16.v v1, (%0)" : : "r"(values + half) : "memory");
}

// Finish a Q/gate/K/V unit row by row: gather it from the panel tiles,
// normalize and rotate Q and K heads, and stage it for the issuer -- K
// transposed ([Q38_HD][rows], q38_pack_kv_to_l2's layout), the rest
// row-major.
static NOSTACK void av_epi_qkv(const __fp16 *zu, uint32_t u, uint32_t row0,
                               __fp16 *e, uint32_t engine, uint32_t slot) {
  const uint32_t heads = 2u * Q38_QH;
  const uint32_t key = u >= heads && u < heads + Q38_KVH;
  const uint32_t rotate = u < heads ? (u & 1u) == 0u : key;
  const __fp16 *const g = u < heads ? av_qg : av_kg;
  // Transposed K: the Q38_HD values of a row land Q38_PROJECTION_ROWS apart,
  // `run` of them per stripe chunk.
  const uint32_t run = Q38_AV_STAGE_ROW / (Q38_PROJECTION_ROWS * 2u);
  for (uint32_t i = 0u; i < Q38_PROJECTION_ROWS; ++i) {
    for (uint32_t j = 0u; j < Q38_AV_U; ++j)
      q38_copy16(e + j * NT, Q38_VC_GLOBAL(zu + j * Q38_AV_TILE + i * NT), NT);
    if (rotate) {
      q38_norm16(e, g, e, Q38_HD, 0);
      av_rotary(e, av.cos + (row0 + i) * Q38_HALF,
                av.sin + (row0 + i) * Q38_HALF);
    }
    if (!key) {
      const uint32_t chunk = Q38_MIN(Q38_HD, Q38_AV_STAGE_ROW / 2u);
      for (uint32_t d = 0u; d < Q38_HD; d += chunk)
        q38_copy16((__fp16 *)av_stage_at(engine, slot, (i * Q38_HD + d) * 2u),
                   e + d, chunk);
      continue;
    }
    for (uint32_t d = 0u; d < Q38_HD; d += run) {
      __fp16 *const kt = (__fp16 *)av_stage_at(
          engine, slot, (d * Q38_PROJECTION_ROWS + i) * 2u);
      for (uint32_t o = 0u; o < run;) {
        uint32_t vl;
        asm volatile("vsetvli %0, %1, e16, m2, ta, ma"
                     : "=r"(vl)
                     : "r"(run - o));
        asm volatile("vle16.v v0, (%0)" : : "r"(e + d + o) : "memory");
        asm volatile("vsse16.v v0, (%0), %1"
                     :
                     : "r"(kt + o * Q38_PROJECTION_ROWS),
                       "r"(Q38_PROJECTION_ROWS * 2u)
                     : "memory");
        o += vl;
      }
    }
  }
}

// Tile worker: finish this tile's units in the order its agent computes them.
static NOSTACK void av_worker(uint32_t b) {
  const uint32_t engine = mempool_get_group_id() * NUM_TILES_PER_GROUP +
                          mempool_get_tile_id() % NUM_TILES_PER_GROUP;
  const uint32_t v = engine / Q38_VC_ENGINES, ev = engine % Q38_VC_ENGINES;
  const uint32_t r = ev / av.pv, q = ev % av.pv;
  const uint32_t row0 = r * Q38_PROJECTION_ROWS;
  const __fp16 *const z =
      av.z + engine % NUM_TILES_PER_GROUP * 2u * av.upanels * Q38_AV_TILE;
  vc_subscribe(FSYNC_EVT_ENGINE);
  for (uint32_t n = 0u; n < av.unit_steps; ++n) {
    const uint32_t unit = b * av.unit_steps + n;
    if (q < av_count(v, n)) {
#if Q38_AV_DIRECT_DMA
      if (!av.kind && av_copy_unit(av_head_unit(av_first[n] + v * av.pv + q)))
        continue;
#endif
#if Q38_AV_DUAL
      avp_wait_unit(engine, unit + 1u);
#else
      vc_worker_wait(&av_progress[engine].unit_done, unit + 1u);
#endif
      // Staging slot unit & 1 is free once unit - 2 has landed.
      if (!av.kind && unit >= 2u)
#if Q38_AV_DUAL
        while (__atomic_load_n(&av_progress[engine].sent, __ATOMIC_ACQUIRE) <
               unit - 1u)
          mempool_wait(16u);
#else
        vc_worker_wait(&av_progress[engine].sent, unit - 1u);
#endif
      const __fp16 *const zu = z + (unit & 1u) * av.upanels * Q38_AV_TILE;
      const uint32_t u = av_first[n] + v * av.pv + q;
#if Q38_AV_PROFILE
      const uint32_t epi_start = mempool_get_timer();
#endif
      uint32_t completed_rows = 0u;
#if Q38_AV_STATIC_CORES
      const uint32_t core = mempool_get_core_id() % NUM_CORES_PER_TILE;
      uint32_t first, last;
      avp_range(av_head_unit(u), core, &first, &last);
      if (!av.kind)
        avp_epi_rows(zu, av_head_unit(u), row0, engine, unit & 1u, first, last);
      else
        for (uint32_t base = first; base < last; base += Q38_AV_ROW_UNROLL) {
#pragma clang loop unroll(full)
          for (uint32_t j = 0u; j < Q38_AV_ROW_UNROLL; ++j) {
            const uint32_t row = base + j;
            if (row >= last)
              break;
            llm_residual_add_f16(av.res + (row0 + row) * Q38_D + u * NT,
                                 Q38_VC_GLOBAL(zu + row * NT), NT, 0u, 1u);
          }
        }
      completed_rows = last - first;
      avp_publish(engine, unit & 1u, core, zu, u, row0, last);
#if Q38_AV_PROFILE && Q38_AV_STATIC_CORES > 2
      if (core >= 2u) {
        av_time_add(&av_extra_times[core - 2u][engine].unit[av_time_kind(u)],
                    mempool_get_timer() - epi_start);
        av_extra_times[core - 2u][engine].rows[av_time_kind(u)] +=
            completed_rows;
      }
#endif
#else
      for (uint32_t i = 0u;;) {
#if Q38_AV_DUAL
        i = __atomic_fetch_add(&av_progress[engine].claimed[unit & 1u], 1u,
                               __ATOMIC_ACQ_REL);
#endif
        if (i >= Q38_PROJECTION_ROWS)
          break;
        if (av.kind) {
          llm_residual_add_f16(av.res + (row0 + i) * Q38_D + u * NT,
                               Q38_VC_GLOBAL(zu + i * NT), NT, 0u, 1u);
        } else {
#if Q38_AV_PANEL_AWARE
          avp_epi_row(zu, av_head_unit(u), row0, engine, unit & 1u, i);
#else
          av_epi_qkv(
              zu, av_head_unit(u), row0,
              Q38_VC_GLOBAL(av.epi + engine % NUM_TILES_PER_GROUP * Q38_HD),
              engine, unit & 1u);
          completed_rows = Q38_PROJECTION_ROWS;
          break;
#endif
        }
        ++completed_rows;
#if Q38_AV_DUAL
        __sync_synchronize();
        __atomic_fetch_add(&av_progress[engine].rows_done[unit & 1u], 1u,
                           __ATOMIC_ACQ_REL);
#else
        ++i;
#endif
      }
      __sync_synchronize();
#endif
#if Q38_AV_PROFILE
      if (mempool_get_core_id() % NUM_CORES_PER_TILE == 1u) {
        av_time_add(&av_times[engine].epi[av_time_kind(u)],
                    mempool_get_timer() - epi_start);
        av_units[engine].rows[av_time_kind(u)] += completed_rows;
      }
#endif
    }
#if Q38_AV_DUAL && !Q38_AV_STATIC_CORES
    __atomic_store_n(&av_progress[engine].worker_done, unit + 1u,
                     __ATOMIC_RELEASE);
#elif !Q38_AV_DUAL
    __atomic_store_n(&av_progress[engine].epi_done, unit + 1u,
                     __ATOMIC_RELEASE);
#endif
  }
  fsync_subscribe_self(0u);
}

static NOSTACK void av_block(uint32_t b) {
  if (tnc_is_tile_agent())
    av_agent(b);
  else if (mempool_get_core_id() % NUM_CORES_PER_TILE <
           (Q38_AV_STATIC_CORES ? Q38_AV_STATIC_CORES : 2u))
    av_worker(b);
  else
    fsync_subscribe_self(0u);
  sync_all();
  fsync_subscribe_self(FSYNC_EVT_FSYNC);
}

// The issuer starts the phase's first weight load early, so it overlaps the
// block's input staging (both are L2 -> L1).
static void av_first_load(void) {
  if (tnc_is_tile_agent() && mempool_is_group_coordinator() &&
      mempool_get_redmule_id() < NUM_REDMULE_TILES)
    av_pending[mempool_get_group_id()] =
        av_load(mempool_get_group_id() / Q38_VC_GROUPS, 0u);
}

#if Q38_AV_QKV_F2
#if Q38_AV_QKV_F2_EPI
// Epilogue pass over P. Engine e finishes row tile e (Q38_PROJECTION_ROWS
// rows) for every head unit, through a ring of AVQ_SLOTS unit slots in its
// group's stripe: a head row per stripe row, then a K unit's transposed copy.
// The group coordinator issues every DMA, loads and ships never at once (see
// av_drain); the tile's vector cores norm and rotate Q and K rows in place,
// as avp_epi_row_pipe does. Gates and V only pass through.
#define AVQ_SLOTS 4u
#define AVQ_SLOT_BYTES (2u * Q38_AV_UNIT_BYTES)
#define AVQ_ROPE_BYTES (Q38_PROJECTION_ROWS * Q38_HALF * 2u)
#define AVQ_ROPE (AVQ_SLOTS * AVQ_SLOT_BYTES)
#define AVQ_TILE_BYTES (AVQ_ROPE + 2u * AVQ_ROPE_BYTES)
#define AVQ_ROW_TILES (Q38_AV_MIXER_ROWS / Q38_PROJECTION_ROWS)
#define AVQ_WORKERS (NUM_CORES_PER_TILE - 1u)
_Static_assert(AVQ_ROW_TILES <= NUM_REDMULE_TILES &&
                   VC_ROW_BYTES == Q38_HD * 2u &&
                   AVQ_ROPE_BYTES % VC_ROW_BYTES == 0u &&
                   NUM_SCALAR_CORES_PER_TILE == 1u && AVQ_WORKERS >= 1u,
               "QKV f2 epilogue: a row tile per engine, a head row per "
               "stripe row, a scalar agent and vector workers per tile");

typedef struct {
  uint32_t landed;            // units in their slots (coordinator)
  uint32_t done[AVQ_WORKERS]; // units each vector core finished
} avq_progress_t;
static avq_progress_t avq_progress[NUM_REDMULE_TILES] Q38_AV_STATE;
static uint32_t avq_row Q38_VC_SMALL;
#if Q38_AV_QKV_F2_PROF
// Per engine: worker Q busy, K busy, wait; coordinator (first engine of its
// group) load wait, ship wait, idle.
enum { AVQ_QBUSY, AVQ_KBUSY, AVQ_WWAIT, AVQ_LWAIT, AVQ_SWAIT, AVQ_IDLE, AVQ_N };
static uint32_t avq_prof[NUM_REDMULE_TILES][AVQ_N] Q38_VC_SMALL;
#define AVQ_T(x) const uint32_t x = mempool_get_timer()
#define AVQ_ADD(e, k, v)                                                       \
  __atomic_fetch_add(&avq_prof[e][k], v, __ATOMIC_RELAXED)
#else
#define AVQ_T(x)
#define AVQ_ADD(e, k, v)
#endif

#if Q38_AV_QKV_F2_DIRECT
// Unit j of a row tile: Q of head j, then K (even) and V (odd) of KV head
// (j - Q38_QH) / 2. Only V passes through.
#define AVQ_UNITS (Q38_QH + 2u * Q38_KVH)
#define avq_copy(j) ((j) >= Q38_QH && ((j)-Q38_QH) % 2u)
static uint32_t avq_unit(uint32_t j) {
  if (j < Q38_QH)
    return 2u * j;
  const uint32_t i = j - Q38_QH;
  return 2u * Q38_QH + i % 2u * Q38_KVH + i / 2u;
}
#else
// Unit j of a row tile: Q of head j / 2 (even j) and its gate (odd j), then
// K (even) and V (odd) of KV head j / 2 - Q38_QH. Even units rotate.
#define AVQ_UNITS Q38_AV_QKV_UNITS
#define avq_copy(j) ((j)&1u)
static uint32_t avq_unit(uint32_t j) {
  const uint32_t i = j / 2u;
  if (i < Q38_QH)
    return 2u * i + (j & 1u);
  return 2u * Q38_QH + (j & 1u) * Q38_KVH + (i - Q38_QH);
}
#endif

// Byte o of engine e's region, in its group's stripe.
static char *avq_at(uint32_t e, uint32_t o) {
  o += e % NUM_TILES_PER_GROUP * AVQ_TILE_BYTES;
  return (char *)((avq_row + o / VC_ROW_BYTES) * VC_GLOBAL_ROW_BYTES +
                  e / NUM_TILES_PER_GROUP * VC_ROW_BYTES + o % VC_ROW_BYTES);
}

// RMSNorm (Q also scaled, see avp_epi_row_pipe) and rotary of one Q or K row
// in place; a K row also goes transposed to the slot's second half.
static void avq_epi_row(__fp16 *x, uint32_t u, uint32_t e, uint32_t kt,
                        uint32_t row) {
  const uint32_t heads = 2u * Q38_QH, key = u >= heads;
  const float zero = 0.0f, zh = llm_f16box(0.0f);
  const __fp16 *const gamma = key ? av_kg : av_qg;
  const uint32_t chunk = Q38_MIN(Q38_HD, AVP_REDUCE_ELEMS);
  asm volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"(AVP_REDUCE_ELEMS));
  asm volatile("vmv.v.i v8, 0");
  asm volatile("vsetvli zero, %0, e16, m4, tu, ma" : : "r"(chunk));
  asm volatile("vle16.v v0, (%0)" : : "r"(x) : "memory");
  asm volatile("vfwmacc.vv v8, v0, v0");
#if Q38_HD > AVP_REDUCE_ELEMS
  asm volatile("vle16.v v4, (%0)" : : "r"(x + AVP_REDUCE_ELEMS) : "memory");
  asm volatile("vsetvli zero, %0, e16, m4, tu, ma" : : "r"(AVP_REDUCE_ELEMS));
  asm volatile("vfwmacc.vv v8, v4, v4");
#endif
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(AVP_REDUCE_ELEMS));
  float sum;
  asm volatile("vfmv.s.f v24, %0" : : "f"(zero));
  asm volatile("vfredusum.vs v24, v8, v24");
  asm volatile("vfmv.f.s %0, v24" : "=f"(sum));
#if Q38_AV_SWEEP_QSCALE
  const float inverse = llm_rsqrt(sum * llm_recip_pos((float)Q38_HD) + 1e-6f) *
                        (key ? 1.0f : Q38_QK_SCALE);
#else
  const float inverse = llm_rsqrt(sum * llm_recip_pos((float)Q38_HD) + 1e-6f);
#endif
#pragma clang loop unroll(full)
  for (uint32_t off = 0; off < Q38_HD; off += AVP_REDUCE_ELEMS) {
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(chunk));
    if (off)
      asm volatile("vfwadd.vf v8, v4, %0" : : "f"(zh));
    else
      asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
    asm volatile("vle16.v v24, (%0)" : : "r"(gamma + off) : "memory");
    asm volatile("vfwadd.vf v16, v24, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(chunk));
    asm volatile("vfmul.vf v8, v8, %0" : : "f"(inverse));
    asm volatile("vfmul.vv v8, v8, v16");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(chunk));
    if (off)
      asm volatile("vfncvt.f.f.w v4, v8");
    else
      asm volatile("vfncvt.f.f.w v0, v8");
  }
  const __fp16 *const cos =
      (const __fp16 *)avq_at(e, AVQ_ROPE + row * Q38_HALF * 2u);
  const __fp16 *const sin = (const __fp16 *)avq_at(
      e, AVQ_ROPE + AVQ_ROPE_BYTES + row * Q38_HALF * 2u);
  asm volatile("vsetvli zero, %0, e16, m8, tu, ma" : : "r"(Q38_HALF));
  asm volatile("vslidedown.vx v24, v0, %0" : : "r"(Q38_HALF));
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(Q38_HALF));
  asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
  asm volatile("vfwadd.vf v10, v24, %0" : : "f"(zh));
  asm volatile("vle16.v v24, (%0)" : : "r"(cos) : "memory");
  asm volatile("vfwadd.vf v12, v24, %0" : : "f"(zh));
  asm volatile("vle16.v v24, (%0)" : : "r"(sin) : "memory");
  asm volatile("vfwadd.vf v14, v24, %0" : : "f"(zh));
  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(Q38_HALF));
  asm volatile("vfmul.vv v16, v8, v12");
  asm volatile("vfmul.vv v18, v10, v14");
  asm volatile("vfsub.vv v16, v16, v18");
  asm volatile("vfmul.vv v18, v10, v12");
  asm volatile("vfmacc.vv v18, v8, v14");
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(Q38_HALF));
  asm volatile("vfncvt.f.f.w v24, v16");
  asm volatile("vfncvt.f.f.w v16, v18");
  asm volatile("vsetvli zero, %0, e16, m8, tu, ma" : : "r"(Q38_HALF));
  asm volatile("vmv.v.v v0, v24");
  asm volatile("vsetvli zero, %0, e16, m8, tu, ma" : : "r"(Q38_ROT));
  asm volatile("vslideup.vx v0, v16, %0" : : "r"(Q38_HALF));
  if (!key) {
    asm volatile("vsetvli zero, %0, e16, m8, ta, ma" : : "r"(Q38_HD));
    asm volatile("vse16.v v0, (%0)" : : "r"(x) : "memory");
    return;
  }
  const uint32_t run = VC_ROW_BYTES / (Q38_PROJECTION_ROWS * 2u);
  for (uint32_t d = 0; d < Q38_HD; d += run) {
    asm volatile("vsetvli zero, %0, e16, m8, tu, ma" : : "r"(run));
    asm volatile("vslidedown.vx v24, v0, %0" : : "r"(d));
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(run));
    asm volatile("vsse16.v v24, (%0), %1"
                 :
                 : "r"(avq_at(e, kt + (d * Q38_PROJECTION_ROWS + row) * 2u)),
                   "r"(Q38_PROJECTION_ROWS * 2u)
                 : "memory");
  }
}

#if Q38_AV_QKV_F2_FAST
_Static_assert(Q38_HD == 2u * AVP_REDUCE_ELEMS,
               "avq_epi_row_fast: a head row is two reduction chunks");
// avq_at with the region's first stripe row given.
static char *avq_at_b(uint32_t base, uint32_t e, uint32_t o) {
  o += e % NUM_TILES_PER_GROUP * AVQ_TILE_BYTES;
  return (char *)((base + o / VC_ROW_BYTES) * VC_GLOBAL_ROW_BYTES +
                  e / NUM_TILES_PER_GROUP * VC_ROW_BYTES + o % VC_ROW_BYTES);
}

// avq_epi_row, bit for bit, on resolved addresses (the compiler re-derives
// avq_row from an L1 flag on every avq_at). The sum of squares starts with a
// product (x * x is exact in FP32) instead of a cleared accumulator, and a Q
// row is stored as 32-bit elements: a vector store costs per element.
static void avq_epi_row_fast(__fp16 *x, uint32_t key, const __fp16 *cos,
                             const __fp16 *sin, char *kt) {
  const float zero = 0.0f, zh = llm_f16box(0.0f);
  const __fp16 *const gamma = key ? av_kg : av_qg;
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(AVP_REDUCE_ELEMS));
  asm volatile("vle16.v v0, (%0)" : : "r"(x) : "memory");
  asm volatile("vle16.v v4, (%0)" : : "r"(x + AVP_REDUCE_ELEMS) : "memory");
  asm volatile("vfwmul.vv v8, v0, v0");
  asm volatile("vfwmacc.vv v8, v4, v4");
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(AVP_REDUCE_ELEMS));
  float sum;
  asm volatile("vfmv.s.f v24, %0" : : "f"(zero));
  asm volatile("vfredusum.vs v24, v8, v24");
  asm volatile("vfmv.f.s %0, v24" : "=f"(sum));
#if Q38_AV_SWEEP_QSCALE
  const float inverse = llm_rsqrt(sum * llm_recip_pos((float)Q38_HD) + 1e-6f) *
                        (key ? 1.0f : Q38_QK_SCALE);
#else
  const float inverse = llm_rsqrt(sum * llm_recip_pos((float)Q38_HD) + 1e-6f);
#endif
#pragma clang loop unroll(full)
  for (uint32_t off = 0; off < Q38_HD; off += AVP_REDUCE_ELEMS) {
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(AVP_REDUCE_ELEMS));
    if (off)
      asm volatile("vfwadd.vf v8, v4, %0" : : "f"(zh));
    else
      asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
    asm volatile("vle16.v v24, (%0)" : : "r"(gamma + off) : "memory");
    asm volatile("vfwadd.vf v16, v24, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(AVP_REDUCE_ELEMS));
    asm volatile("vfmul.vf v8, v8, %0" : : "f"(inverse));
    asm volatile("vfmul.vv v8, v8, v16");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(AVP_REDUCE_ELEMS));
    if (off)
      asm volatile("vfncvt.f.f.w v4, v8");
    else
      asm volatile("vfncvt.f.f.w v0, v8");
  }
  asm volatile("vsetvli zero, %0, e16, m8, tu, ma" : : "r"(Q38_HALF));
  asm volatile("vslidedown.vx v24, v0, %0" : : "r"(Q38_HALF));
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(Q38_HALF));
  asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
  asm volatile("vfwadd.vf v10, v24, %0" : : "f"(zh));
  asm volatile("vle16.v v24, (%0)" : : "r"(cos) : "memory");
  asm volatile("vfwadd.vf v12, v24, %0" : : "f"(zh));
  asm volatile("vle16.v v24, (%0)" : : "r"(sin) : "memory");
  asm volatile("vfwadd.vf v14, v24, %0" : : "f"(zh));
  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(Q38_HALF));
  asm volatile("vfmul.vv v16, v8, v12");
  asm volatile("vfmul.vv v18, v10, v14");
  asm volatile("vfsub.vv v16, v16, v18");
  asm volatile("vfmul.vv v18, v10, v12");
  asm volatile("vfmacc.vv v18, v8, v14");
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(Q38_HALF));
  asm volatile("vfncvt.f.f.w v24, v16");
  asm volatile("vfncvt.f.f.w v16, v18");
  asm volatile("vsetvli zero, %0, e16, m8, tu, ma" : : "r"(Q38_HALF));
  asm volatile("vmv.v.v v0, v24");
  asm volatile("vsetvli zero, %0, e16, m8, tu, ma" : : "r"(Q38_ROT));
  asm volatile("vslideup.vx v0, v16, %0" : : "r"(Q38_HALF));
  if (!key) {
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(Q38_HD / 2u));
    asm volatile("vse32.v v0, (%0)" : : "r"(x) : "memory");
    return;
  }
  // Column d of this row: kt + d * Q38_PROJECTION_ROWS * 2 in the region, so
  // each run of columns fills one stripe row.
  const uint32_t run = VC_ROW_BYTES / (Q38_PROJECTION_ROWS * 2u);
  for (uint32_t d = 0; d < Q38_HD; d += run, kt += VC_GLOBAL_ROW_BYTES) {
    asm volatile("vsetvli zero, %0, e16, m8, tu, ma" : : "r"(run));
    asm volatile("vslidedown.vx v24, v0, %0" : : "r"(d));
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(run));
    asm volatile("vsse16.v v24, (%0), %1"
                 :
                 : "r"(kt), "r"(Q38_PROJECTION_ROWS * 2u)
                 : "memory");
  }
}
#endif

// Vector core w of engine e: its share of every Q and K unit's rows.
static NOSTACK __attribute__((noinline)) void avq_worker(uint32_t e,
                                                         uint32_t w) {
  if (e >= AVQ_ROW_TILES)
    return;
#if Q38_AV_QKV_F2_PROF
  uint32_t busy[2] = {0u, 0u}, waited = 0u;
#endif
#if Q38_AV_QKV_F2_FAST
  uint32_t base = avq_row;
  asm volatile("" : "+r"(base));
#endif
  for (uint32_t j = 0u; j < AVQ_UNITS; ++j) {
    if (!avq_copy(j)) {
      AVQ_T(c0);
      while (__atomic_load_n(&avq_progress[e].landed, __ATOMIC_ACQUIRE) <= j)
        mempool_wait(16u);
      AVQ_T(c1);
      const uint32_t slot = j % AVQ_SLOTS * AVQ_SLOT_BYTES;
      for (uint32_t i = w; i < Q38_PROJECTION_ROWS; i += AVQ_WORKERS)
#if Q38_AV_QKV_F2_FAST
        avq_epi_row_fast(
            (__fp16 *)avq_at_b(base, e, slot + i * Q38_HD * 2u),
            avq_unit(j) >= 2u * Q38_QH,
            (const __fp16 *)avq_at_b(base, e, AVQ_ROPE + i * Q38_HALF * 2u),
            (const __fp16 *)avq_at_b(
                base, e, AVQ_ROPE + AVQ_ROPE_BYTES + i * Q38_HALF * 2u),
            avq_at_b(base, e, slot + Q38_AV_UNIT_BYTES + i * 2u));
#else
        avq_epi_row((__fp16 *)avq_at(e, slot + i * Q38_HD * 2u), avq_unit(j), e,
                    slot + Q38_AV_UNIT_BYTES, i);
#endif
      __sync_synchronize();
#if Q38_AV_QKV_F2_PROF
      waited += c1 - c0;
      busy[avq_unit(j) >= 2u * Q38_QH] += mempool_get_timer() - c1;
#endif
    }
    __atomic_store_n(&avq_progress[e].done[w], j + 1u, __ATOMIC_RELEASE);
  }
  AVQ_ADD(e, AVQ_QBUSY, busy[0]);
  AVQ_ADD(e, AVQ_KBUSY, busy[1]);
  AVQ_ADD(e, AVQ_WWAIT, waited);
}

static uint32_t avq_finished(uint32_t e, uint32_t j) {
  for (uint32_t w = 0u; w < AVQ_WORKERS; ++w)
    if (__atomic_load_n(&avq_progress[e].done[w], __ATOMIC_ACQUIRE) <= j)
      return 0u;
  return 1u;
}

// Unit j of engine e from its slot to L2: Q and gates head-major, K and V
// into the cache (as av_ship).
static dma_id_t avq_ship(uint32_t e, uint32_t j) {
  const uint32_t u = avq_unit(j), slot = j % AVQ_SLOTS * AVQ_SLOT_BYTES;
  const uint32_t token = Q38_MIXER_EXEC_START + e * Q38_PROJECTION_ROWS;
  const uint32_t heads = 2u * Q38_QH, half = Q38_KVH * Q38_HD * BK;
  __fp16 *const block = AV_CACHE + token / BK * 2u * half;
  if (u >= heads && u < heads + Q38_KVH) {
    const uint32_t run = Q38_PROJECTION_ROWS * 2u;
    const uint32_t per = VC_ROW_BYTES / run;
#if Q38_L2_TILED
    // One 2-D walk per staging row: its `per` key runs, BK * 2 apart, stay in
    // one L2 row; the staging rows' 1 KiB plane step is no whole L2 row.
    dma_id_t id = 0u;
    for (uint32_t i = 0u; i < Q38_AV_UNIT_BYTES / VC_ROW_BYTES; ++i)
      id = dma_local_memcpy_2d_nonblocking(
          0u, block + (u - heads) * Q38_HD * BK + token % BK + i * per * BK,
          (const char *)avq_at(e, slot + Q38_AV_UNIT_BYTES) +
              i * VC_GLOBAL_ROW_BYTES,
          run, BK * 2u, run, per);
    return id;
#else
    return dma_local_memcpy_3d_nonblocking(
        0u, block + (u - heads) * Q38_HD * BK + token % BK,
        avq_at(e, slot + Q38_AV_UNIT_BYTES), run, BK * 2u, run, per, BK * 2u,
        VC_GLOBAL_ROW_BYTES - (per - 1u) * run,
        Q38_AV_UNIT_BYTES / VC_ROW_BYTES);
#endif
  }
  __fp16 *const dst =
      u < heads
          ? (u & 1u ? av_gate : av_q) + ((u / 2u) * AV_QG_ROWS + token) * Q38_HD
          : block + half + ((u - heads - Q38_KVH) * BK + token % BK) * Q38_HD;
  return av_dma_rows(dst, avq_at(e, slot), Q38_HD * 2u, Q38_HD * 2u,
                     VC_GLOBAL_ROW_BYTES, Q38_PROJECTION_ROWS,
                     8192u / (Q38_HD * 2u));
}

static void avq_wait(dma_id_t id) {
  while (!dma_local_done_id(0u, id))
    mempool_wait(16u);
  __sync_synchronize();
}

// Group coordinator: alternate load rounds (the rope rows with the first
// unit, then up to AVQ_SLOTS units ahead of the ships) and ship rounds (landed
// gates and V, finished Q and K, in order) until every unit is out.
static NOSTACK __attribute__((noinline)) void avq_coordinator(void) {
  const uint32_t e0 = mempool_get_group_id() * NUM_TILES_PER_GROUP;
  uint32_t loaded[NUM_TILES_PER_GROUP], shipped[NUM_TILES_PER_GROUP];
  uint32_t left = 0u;
  for (uint32_t t = 0u; t < NUM_TILES_PER_GROUP; ++t) {
    loaded[t] = shipped[t] = 0u;
    if (e0 + t < AVQ_ROW_TILES)
      left += AVQ_UNITS;
  }
  while (left) {
    dma_id_t id = 0u;
    for (uint32_t t = 0u; t < NUM_TILES_PER_GROUP; ++t) {
      const uint32_t e = e0 + t;
      if (e >= AVQ_ROW_TILES)
        continue;
      const uint32_t token = Q38_MIXER_EXEC_START + e * Q38_PROJECTION_ROWS;
      if (loaded[t] == 0u) {
        id = dma_local_memcpy_2d_nonblocking(
            0u, avq_at(e, AVQ_ROPE), rope_cos + token * Q38_HALF, VC_ROW_BYTES,
            VC_GLOBAL_ROW_BYTES, VC_ROW_BYTES, AVQ_ROPE_BYTES / VC_ROW_BYTES);
        id = dma_local_memcpy_2d_nonblocking(
            0u, avq_at(e, AVQ_ROPE + AVQ_ROPE_BYTES),
            rope_sin + token * Q38_HALF, VC_ROW_BYTES, VC_GLOBAL_ROW_BYTES,
            VC_ROW_BYTES, AVQ_ROPE_BYTES / VC_ROW_BYTES);
      }
      while (loaded[t] < AVQ_UNITS && loaded[t] < shipped[t] + AVQ_SLOTS) {
        const uint32_t j = loaded[t]++;
#if Q38_AV_QKV_F2_HEADS
        id = av_dma_rows(avq_at(e, j % AVQ_SLOTS * AVQ_SLOT_BYTES),
                         AV_P_L2 + (avq_unit(j) * Q38_R + token) * Q38_HD,
                         Q38_HD * 2u, VC_GLOBAL_ROW_BYTES, Q38_HD * 2u,
                         Q38_PROJECTION_ROWS, 8192u / (Q38_HD * 2u));
#else
        id = dma_local_memcpy_2d_nonblocking(
            0u, avq_at(e, j % AVQ_SLOTS * AVQ_SLOT_BYTES),
            AV_P_L2 + token * AV_PSTRIDE + avq_unit(j) * Q38_HD, Q38_HD * 2u,
            VC_GLOBAL_ROW_BYTES, AV_PSTRIDE * 2u, Q38_PROJECTION_ROWS);
#endif
      }
    }
    if (id) {
      AVQ_T(c0);
      avq_wait(id);
      AVQ_ADD(e0, AVQ_LWAIT, mempool_get_timer() - c0);
      for (uint32_t t = 0u; t < NUM_TILES_PER_GROUP; ++t)
        if (e0 + t < AVQ_ROW_TILES)
          __atomic_store_n(&avq_progress[e0 + t].landed, loaded[t],
                           __ATOMIC_RELEASE);
    }
    dma_id_t sid = 0u;
    for (uint32_t t = 0u; t < NUM_TILES_PER_GROUP; ++t) {
      const uint32_t e = e0 + t;
      if (e >= AVQ_ROW_TILES)
        continue;
      while (shipped[t] < loaded[t] &&
             (avq_copy(shipped[t]) || avq_finished(e, shipped[t]))) {
        sid = avq_ship(e, shipped[t]++);
        --left;
      }
    }
    AVQ_T(c1);
    if (sid)
      avq_wait(sid);
    else if (!id)
      mempool_wait(64u);
    AVQ_ADD(e0, sid ? AVQ_SWAIT : AVQ_IDLE, mempool_get_timer() - c1);
  }
}

static NOSTACK void av_qkv_epilogue(void) {
  const uint32_t core = mempool_get_core_id();
  if (core == 0u) {
    uint32_t p = (uint32_t)&stage < (uint32_t)&weights[0][0]
                     ? (uint32_t)&stage
                     : (uint32_t)&weights[0][0];
    avq_row = (p + VC_GLOBAL_ROW_BYTES - 1u) / VC_GLOBAL_ROW_BYTES;
    p = (avq_row + NUM_TILES_PER_GROUP * AVQ_TILE_BYTES / VC_ROW_BYTES) *
        VC_GLOBAL_ROW_BYTES;
    if (p > (uint32_t)&__heap_end)
      q38_fatal("[qwen38_fatal] qkv epilogue bytes over L1=",
                p - (uint32_t)&__heap_end);
  }
  if (tnc_is_tile_agent()) {
    const uint32_t e = mempool_get_redmule_id();
    avq_progress[e].landed = 0u;
    for (uint32_t w = 0u; w < AVQ_WORKERS; ++w)
      avq_progress[e].done[w] = 0u;
#if Q38_AV_QKV_F2_PROF
    for (uint32_t k = 0u; k < AVQ_N; ++k)
      avq_prof[e][k] = 0u;
#endif
  }
  sync_all();
  transfer(av_qg, w_q_norm, Q38_HD * 2);
  transfer(av_kg, w_k_norm, Q38_HD * 2);
  if (mempool_is_group_coordinator())
    avq_coordinator();
  else if (!tnc_is_tile_agent())
    avq_worker(mempool_get_group_id() * NUM_TILES_PER_GROUP +
                   mempool_get_tile_id() % NUM_TILES_PER_GROUP,
               core % NUM_CORES_PER_TILE - 1u);
  sync_all();
#if Q38_AV_QKV_F2_PROF
  if (core == 0u) {
    uint32_t sum[AVQ_N] = {0u}, peak[AVQ_N] = {0u};
    for (uint32_t e = 0u; e < AVQ_ROW_TILES; ++e)
      for (uint32_t k = 0u; k < AVQ_N; ++k) {
        sum[k] += avq_prof[e][k];
        peak[k] = Q38_MAX(peak[k], avq_prof[e][k]);
      }
    const uint32_t nw = AVQ_ROW_TILES * AVQ_WORKERS;
    const uint32_t ng = AVQ_ROW_TILES / NUM_TILES_PER_GROUP;
    printf("[qwen38_avq_prof] worker avg qbusy=%u kbusy=%u wait=%u; "
           "coord avg lwait=%u swait=%u idle=%u max lwait=%u swait=%u\n",
           sum[AVQ_QBUSY] / nw, sum[AVQ_KBUSY] / nw, sum[AVQ_WWAIT] / nw,
           sum[AVQ_LWAIT] / ng, sum[AVQ_SWAIT] / ng, sum[AVQ_IDLE] / ng,
           peak[AVQ_LWAIT], peak[AVQ_SWAIT]);
  }
#endif
}
#endif

// Q/gate/K/V projection of all mixer rows on the f2 up pass (as
// gdn2_project): P = RMSNorm(x) [W_q_gate | W_k | W_v] to L2, P's columns in
// head-unit order. Nothing ran f2 before it in this layer.
static NOSTACK void av_qkv(void) {
  const uint32_t core = mempool_get_core_id();
  const uint32_t t0 = mempool_get_timer();
  if (core == 0u) {
    f2.peak = 0u, f2.q_base = 0u;
    f2.pair = 0u, f2.k_up = Q38_D, f2.n_up = AV_PCOLS;
    f2.rows_up = Q38_FFN_ROWS, f2.nseg = 3u;
    f2.seg_w[0] = w_q_gate, f2.seg_panels[0] = 2u * Q38_QDIM / AV_FNT;
    f2.seg_w[1] = w_key, f2.seg_panels[1] = Q38_KVH * Q38_HD / AV_FNT;
    f2.seg_w[2] = w_value, f2.seg_panels[2] = Q38_KVH * Q38_HD / AV_FNT;
#if Q38_AV_QKV_F2_HEADS
    f2.g_out = AV_P_L2, f2.g_kc = Q38_HD, f2.g_rows = Q38_R;
#else
    f2.g_out = AV_P_L2, f2.g_kc = AV_PSTRIDE, f2.g_rows = Q38_R;
#endif
#if Q38_L2_TILED
    f2.g_main = ~0u; // one pitch
#endif
  }
  if (tnc_is_tile_agent())
    f2_progress[mempool_get_redmule_id()].step_done = 0u;
  if (mempool_is_group_coordinator()) {
    const uint32_t g = mempool_get_group_id();
    f2_down[g].weight_valid = f2_down[g].weight_pending = 0u;
    f2_down[g].x_pending = 0u;
  }
  sync_all();
#pragma clang loop unroll(disable)
  for (uint32_t first = Q38_MIXER_EXEC_START;
       Q38_AV_QKV_F2_UP && first < Q38_MIXER_EXEC_END; first += Q38_FFN_ROWS) {
    f2_layout(0u);
    f2_reset_progress();
#if Q38_F2_NORM_HOME
    f2_home_setup(gamma, first, Q38_MIXER_EXEC_START);
#endif
    if (core == 0u)
      f2.first = first;
    f2_sync_all();
#if Q38_F2_NORM_HOME
    f2_norm_batch(input_init, w_input_norm, gamma, first, Q38_MIXER_EXEC_START);
#else
    f2_transfer(f2.residual, input_init + first * Q38_D,
                Q38_FFN_ROWS * Q38_D * sizeof(__fp16));
    norm_rows_sliced(f2.residual, w_input_norm, f2.input, Q38_FFN_ROWS, gamma);
#endif
    f2_execute();
  }
  const uint32_t t1 = mempool_get_timer();
#if Q38_AV_QKV_F2_EPI
  av_qkv_epilogue();
#endif
  if (core == 0u)
    printf("[qwen38_av_qkv] up=%u epilogue=%u\n", t1 - t0,
           mempool_get_timer() - t1);
}
#else
static NOSTACK void av_qkv(void) {
  if (mempool_get_core_id() == 0u)
    av_setup(0u);
  sync_all();
  transfer(av_qg, w_q_norm, Q38_HD * 2);
  transfer(av_kg, w_k_norm, Q38_HD * 2);
  for (uint32_t b = 0u; b < av.blocks; ++b) {
    const uint32_t base = Q38_MIXER_EXEC_START + b * av.rows;
    if (mempool_get_core_id() == 0u)
      av.base = base;
    transfer(av.x, input_init + base * Q38_D, av.rows * Q38_D * 2);
    transfer(av.cos, rope_cos + base * Q38_HALF, av.rows * Q38_HALF * 2);
    transfer(av.sin, rope_sin + base * Q38_HALF, av.rows * Q38_HALF * 2);
    if (b == 0u)
      av_first_load();
    norm_rows(av.x, w_input_norm, av.x, av.rows, Q38_D, 0, gamma);
    av_block(b);
  }
}
#endif

#if Q38_AV_OUT_F2
// Out-projection with the residual on the f2 down pass, K-sliced (see
// gdn2_output): mixer = x + gated W_out, to AV_MIXER. Nothing ran f2
// before it in this layer, so its queue counters start here (as in ffn).
static NOSTACK void av_out(void) {
  const uint32_t core = mempool_get_core_id();
  _Static_assert(Q38_AV_MIXER_ROWS % Q38_FFN_DN_ROWS == 0u,
                 "Out-projection rows are whole down batches");
  if (core == 0u) {
    f2.k_dn = AV_GK, f2.k_full = Q38_QDIM, f2.n_dn = Q38_D;
    f2.rows_dn = Q38_FFN_DN_ROWS, f2.slices = Q38_AV_OUT_SLICES;
#if Q38_L2_TILED && Q38_AV_OUT_SLICES > 1
    // The gated context's slices (AV_GATED): whole L2 rows, then the rest.
    f2.k_dn = AV_GK0, f2.k_last = Q38_QDIM - AV_GK0;
#elif Q38_L2_TILED
    f2.k_last = 0u;
#endif
    f2.dn_lag = Q38_AV_OUT_LAG;
    f2.end = Q38_MIXER_EXEC_END;
    f2.w_dn = w_out, f2.dn_in = av_gated, f2.dn_rows = Q38_R;
    f2.res_src = input_init, f2.dn_out = AV_MIXER;
    f2.q_base = 0u, f2.peak = 0u;
#if Q38_FFN_STAMPS
    f2_nst = 0u; // .l1 powers up undefined
#endif
  }
  if (tnc_is_tile_agent())
    f2_progress[mempool_get_redmule_id()].step_done = 0u;
  if (mempool_is_group_coordinator()) {
    const uint32_t g = mempool_get_group_id();
    f2_down[g].weight_valid = f2_down[g].weight_pending = 0u;
    f2_down[g].x_pending = 0u;
  }
  sync_all();
#pragma clang loop unroll(disable)
  for (uint32_t first = Q38_MIXER_EXEC_START; first < Q38_MIXER_EXEC_END;
       first += Q38_FFN_DN_ROWS) {
    f2_layout(1u);
    f2_reset_progress();
    if (core == 0u)
      f2.first = first;
    f2_sync_all();
#pragma clang loop unroll(disable)
    for (uint32_t slice = 0u; slice < Q38_AV_OUT_SLICES; ++slice) {
      if (core == 0u) {
        f2.slice = slice;
        f2.weight_epoch = (first - Q38_MIXER_EXEC_START) / Q38_FFN_DN_ROWS *
                              Q38_AV_OUT_SLICES +
                          slice;
        f2.input = f2.x_buf[f2.weight_epoch & 1u];
      }
      f2_slice_reset();
      f2_sync_all();
      f2_down_input();
      f2_execute();
    }
  }
#if Q38_FFN_STAMPS
  f2_stamps_print();
#endif
}
#else
static NOSTACK void av_out(void) {
  if (mempool_get_core_id() == 0u)
    av_setup(1u);
  sync_all();
  for (uint32_t b = 0u; b < av.blocks; ++b) {
    const uint32_t base = Q38_MIXER_EXEC_START + b * av.rows;
    if (mempool_get_core_id() == 0u)
      av.base = base;
    if (b == 0u)
      av_first_load();
    transfer(av.res, input_init + base * Q38_D, av.rows * Q38_D * 2);
    transfer(av.x, av_gated + base * Q38_QDIM, av.rows * Q38_QDIM * 2);
    sync_all();
    av_block(b);
    transfer(AV_MIXER + base * Q38_D, av.res, av.rows * Q38_D * 2);
  }
}
#endif

// One owner block [Q38_PROJECTION_ROWS][Q38_HD] from L2 into a level-0
// arena field, whose window rows are VC_GLOBAL_ROW_BYTES apart globally.
static dma_id_t av_owner_load(__fp16 *window, const __fp16 *l2) {
  return dma_local_memcpy_2d_nonblocking(
      0u, q38_la(window), l2, VC_ROW_BYTES, VC_GLOBAL_ROW_BYTES, VC_ROW_BYTES,
      Q38_PROJECTION_ROWS * Q38_HD * 2u / VC_ROW_BYTES);
}

// Queries and gates of this group's owners from L2; softmax state reset.
static NOSTACK void av_query_prep(uint32_t base) {
  q38_local_sweep_t *const loc = q38_loc;
  if (mempool_is_group_coordinator()) {
    dma_id_t id = 0u;
    for (uint32_t j = 0u; j < Q38_LOC_OWNERS; ++j) {
      const uint32_t t = q38_loc_owner(j);
      if (t >= Q38_LOC_TOTAL)
        continue;
      const uint32_t h = t / Q38_LOC_TILES;
      const uint32_t row = base + t % Q38_LOC_TILES * Q38_PROJECTION_ROWS;
      __fp16 *const at = loc->q + j * Q38_PROJECTION_ROWS * Q38_HD;
      __fp16 *const gt = loc->gate + j * Q38_PROJECTION_ROWS * Q38_HD;
      id = av_owner_load(at, av_q + (h * AV_QG_ROWS + row) * Q38_HD);
      id = av_owner_load(gt, av_gate + (h * AV_QG_ROWS + row) * Q38_HD);
    }
    while (id && !dma_local_done_id(0u, id))
      mempool_wait(16u);
    __sync_synchronize();
  }
  const uint32_t lc = mempool_get_core_id() % Q38_LOC_CORES;
  for (uint32_t i = lc; i < Q38_LOC_ROWS; i += Q38_LOC_CORES)
    if (q38_loc_owner(i / Q38_PROJECTION_ROWS) < Q38_LOC_TOTAL)
      qwen38_softmax_init(Q38_LA(loc->states + Q38_ST(i)));
  q38_group_sync();
}

// Normalize and gate the group's owner rows, and store them to the gated
// context in L2; the raw context goes out for the checks as before.
static NOSTACK void av_query_finish(uint32_t base) {
  q38_local_sweep_t *const loc = q38_loc;
  const uint32_t lc = mempool_get_core_id() % Q38_LOC_CORES;
  for (uint32_t i = lc; i < Q38_LOC_ROWS; i += Q38_LOC_CORES) {
    const uint32_t t = q38_loc_owner(i / Q38_PROJECTION_ROWS);
    if (t >= Q38_LOC_TOTAL)
      continue;
    const uint32_t h = t / Q38_LOC_TILES;
    const uint32_t row =
        t % Q38_LOC_TILES * Q38_PROJECTION_ROWS + i % Q38_PROJECTION_ROWS;
    const float inverse =
        base + row < Q38_S
            ? llm_recip_pos(Q38_LA(loc->states + Q38_ST(i))->denominator)
            : 0.0f;
    __fp16 *const context = query_stage.context + (h * Q38_WIN + row) * Q38_HD;
    const __fp16 *const z = Q38_LA(loc->z + i * Q38_HD);
    const float zh = llm_f16box(0.0f);
    for (uint32_t o = 0; o < Q38_HD;) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                   : "=r"(vl)
                   : "r"(Q38_HD - o));
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
      asm volatile("vle16.v v0, (%0)" : : "r"(z + o) : "memory");
      asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
      asm volatile("vfmul.vf v8, v8, %0" : : "f"(inverse));
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
      asm volatile("vfncvt.f.f.w v0, v8");
      asm volatile("vse16.v v0, (%0)" : : "r"(context + o) : "memory");
      o += vl;
    }
    q38_activation16(Q38_LA(loc->gate + i * Q38_HD), context,
                     Q38_LA(loc->gated + i * Q38_HD), Q38_HD, 0);
  }
  q38_group_sync();
  if (mempool_is_group_coordinator()) {
    // Each owner row goes to its own head's columns of the gated context.
    const uint32_t per =
        Q38_HD * 2u <= VC_ROW_BYTES ? VC_ROW_BYTES / (Q38_HD * 2u) : 1u;
    dma_id_t id = 0u;
    for (uint32_t j = 0u; j < Q38_LOC_OWNERS; ++j) {
      const uint32_t t = q38_loc_owner(j);
      if (t >= Q38_LOC_TOTAL)
        continue;
      const uint32_t h = t / Q38_LOC_TILES;
      const uint32_t row = base + t % Q38_LOC_TILES * Q38_PROJECTION_ROWS;
      const char *const src =
          (const char *)q38_la(loc->gated + j * Q38_PROJECTION_ROWS * Q38_HD);
      char *const dst = (char *)AV_GATED(row, h);
      if (Q38_HD * 2u <= VC_ROW_BYTES)
        // `per` rows per window row.
        id = dma_local_memcpy_3d_nonblocking(
            0u, dst, src, Q38_HD * 2u, AV_GPITCH(h) * 2u, Q38_HD * 2u, per,
            AV_GPITCH(h) * 2u, VC_GLOBAL_ROW_BYTES - (per - 1u) * Q38_HD * 2u,
            Q38_PROJECTION_ROWS / per);
      else
        // A row over several window rows.
        id = dma_local_memcpy_3d_nonblocking(
            0u, dst, src, VC_ROW_BYTES, VC_ROW_BYTES, VC_GLOBAL_ROW_BYTES,
            Q38_HD * 2u / VC_ROW_BYTES,
            AV_GPITCH(h) * 2u -
                (Q38_HD * 2u / VC_ROW_BYTES - 1u) * VC_ROW_BYTES,
            VC_GLOBAL_ROW_BYTES, Q38_PROJECTION_ROWS);
    }
    while (id && !dma_local_done_id(0u, id))
      mempool_wait(16u);
  }
  sync_all();
#if Q38_SAMPLED_CHECKS
  gather_context_window(query_stage.context, base, Q38_WIN, Q38_HD);
#else
  for (uint32_t h = 0; h < Q38_QH; ++h)
    export_tensor(result_context + (h * Q38_R + base) * Q38_HD,
                  query_stage.context + h * Q38_WIN * Q38_HD,
                  Q38_WIN * Q38_HD * 2);
#endif
}

#include "q38_attn_vc_sweep.c"

static NOSTACK void av_sweep(void) {
#if Q38_ATTN_VC_SWEEP
  avs_run();
#else
  q38_local_init();
  for (uint32_t base = Q38_MIXER_EXEC_START; base < Q38_MIXER_EXEC_END;
       base += Q38_WIN) {
    KP(KP_COUNT);
    av_query_prep(base);
    KP(KP_PREP);
    query_sweep_pipe(base);
    KP(KP_COUNT);
    av_query_finish(base);
    KP(KP_FINISH);
#if Q38_SCHEDULE_PROBE
    if (mempool_get_core_id() == 0u)
      ++probe_mixer_windows;
#endif
  }
#endif
}

static NOSTACK void attention_phased(void) {
#if Q38_AV_PROFILE
  if (mempool_get_core_id() == 0u)
    av_bounds[0] = mempool_get_timer();
#endif
  av_qkv();
#if Q38_AV_PROFILE
  if (mempool_get_core_id() == 0u)
    av_bounds[1] = mempool_get_timer();
#endif
#if Q38_AV_STOP_PHASE
  if (q38_av_stop == 1u) {
    sync_all();
    if (mempool_get_core_id() == 0u)
      av_bounds[2] = av_bounds[3] = av_bounds[1];
    return;
  }
#endif
  av_sweep();
#if Q38_AV_PROFILE
  sync_all();
  if (mempool_get_core_id() == 0u)
    av_bounds[2] = mempool_get_timer();
#endif
#if Q38_AV_STOP_PHASE
  if (q38_av_stop == 2u) {
    if (mempool_get_core_id() == 0u)
      av_bounds[3] = av_bounds[2];
    return;
  }
#endif
  av_out();
#if Q38_AV_PROFILE
  if (mempool_get_core_id() == 0u)
    av_bounds[3] = mempool_get_timer();
#endif
}
#endif
