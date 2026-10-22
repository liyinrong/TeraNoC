// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Phase-owned streamed GEMMs on RedMulE (the L2 FFN engine). An "up" pass
// multiplies X (rows x k_up, L1) by a list of weight panels -- the FFN's
// gate/up pairs, or up to three single-matrix segments -- and ships each panel
// group, activated or copied by the tile's workers, to an L2 tensor. A "down"
// pass multiplies K-sliced X from L2 by full-K weight panels, keeps M32 partial
// outputs across the slices and adds a residual. The shapes are runtime
// fields of `f2`, set by the phase driver (ffn(), gdn2()).

// Debug: skip the gate/up pass (the down pass reads whatever G holds).
#ifndef Q38_FFN_DEBUG_DOWN_ONLY
#define Q38_FFN_DEBUG_DOWN_ONLY 0
#endif
// Up passes: three weight slots, so a VC's engines meet one step behind and
// a slow engine no longer stalls the others at every step (down passes keep
// two unless f2.dn_lag: the FFN's K slices do not fit three times).
#ifndef Q38_FFN_LAG
#define Q38_FFN_LAG 1
#endif
// Up passes: weight slots, Z and ship overlay the residual (f2_layout).
#ifndef Q38_FFN_UP_OVERLAY
#define Q38_FFN_UP_OVERLAY 0
#endif
#ifndef Q38_FFN_NEXT_WEIGHT_PREFETCH
#define Q38_FFN_NEXT_WEIGHT_PREFETCH 1
#endif
// Down passes: two X buffers; the issuers load the next K slice's X during
// this one (after step 1's weights) instead of between the slices.
#ifndef Q38_FFN_X_PREFETCH
#define Q38_FFN_X_PREFETCH 0
#endif
// Queue-ahead agents: each engine holds two steps (its two HWPE contexts), so
// a step starts as its predecessor ends, without a VC barrier between them.
#ifndef Q38_FFN_QUEUE
#define Q38_FFN_QUEUE 0
#endif
// Zero every f2 record once at program start (.l1 is not zeroed: X in RTL).
// A GDN layer's first f2 user (gdn2_project) reads f2.q_base and step_done,
// which only the FFN and attention entries reset.
#ifndef Q38_GDN_ENTRY_INIT
#define Q38_GDN_ENTRY_INIT 1
#endif
// Debug (with Q38_FFN_QUEUE): each agent samples its weight panel in the slot
// before triggering a step and after it ends, against the L2 source.
#ifndef Q38_FFN_W_CHECK
#define Q38_FFN_W_CHECK 0
#endif
// Debug: core 0 timestamps the pass boundaries (tags: 1 own last job done,
// 2 agent drained, 3 pass barrier passed, 4 driver barrier passed, 5 first
// job triggered), printed by f2_stamps_print.
#ifndef Q38_FFN_STAMPS
#define Q38_FFN_STAMPS 0
#endif
#if Q38_FFN_STAMPS
static uint32_t f2_st[256] Q38_VC_SMALL;
static uint32_t f2_nst Q38_VC_SMALL;
#define F2_STAMP(tag)                                                          \
  do {                                                                         \
    if (mempool_get_core_id() == 0u && f2_nst < 256u)                          \
      f2_st[f2_nst++] = mempool_get_timer() << 4 | (tag);                      \
  } while (0)
static void f2_stamps_print(void) {
  if (mempool_get_core_id() == 0u)
    for (uint32_t i = 0u; i < f2_nst; ++i)
      printf("[f2_stamp] %u %u\n", f2_st[i] & 15u, f2_st[i] >> 4);
}
#else
#define F2_STAMP(tag)                                                          \
  do {                                                                         \
  } while (0)
#endif
// Debug: a wait loop still spinning after this many rounds prints its group's
// progress (groups in turn) and ends the run.
#ifndef Q38_FFN_WATCHDOG
#define Q38_FFN_WATCHDOG 0
#endif
// Debug: tag sampled X elements in L2 with their address before the down
// passes and check each slice's copy in L1 (timing-only runs carry no data).
#ifndef Q38_FFN_X_CHECK
#define Q38_FFN_X_CHECK 0
#endif
#define F2_X_CHECK_STRIDE 997u
// Up passes with the overlay: the residual sits at the overlay's end, clear
// of weight slot 0, so the issuers load step 0's weights under the RMSNorm.
#ifndef Q38_FFN_UP_EARLY_WEIGHT
#define Q38_FFN_UP_EARLY_WEIGHT 0
#endif
// Up-pass batch loops that opt in (f2_home_setup, f2_norm_batch): each group
// coordinator loads the group's share of the batch's residual rows into its
// own stripe (where the global residual was), and the group's workers RMSNorm
// them from local L1 instead of rows spread over eight groups.
#ifndef Q38_F2_NORM_HOME
#define Q38_F2_NORM_HOME 0
#endif
// C1 placement ladder (timing-only, default off): f2_c1 runs one pass of kind
// Q38_F2_C1 for Q38_F2_C1_STEPS steps with the weights resident (no weight or
// X DMA, values unchecked): 1 up K=D, 2 the same for one step, 3 up K=D
// gate/up pairs, 4 a down K slice of an out-projection (K=Q38_F2_C1_KO in
// two slices, as av_out / gdn2_output), 5 an FFN down K slice; down kinds run
// a middle slice (GEMM accumulation, no residual or output DMA).
// The placement switches move one operand off the deployment placement:
// W 0 = flat global (1 = VC level 1), Z 1 = flat global (0 = group window),
// X 2 = one copy per level-2 rectangle (0 = flat global). A last down slice
// (never run by f2_c1) keeps its group-window Z.
#ifndef Q38_F2_C1
#define Q38_F2_C1 0
#endif
#ifndef Q38_F2_C1_STEPS
#define Q38_F2_C1_STEPS 6u
#endif
#ifndef Q38_F2_C1_KO
#define Q38_F2_C1_KO 6144u
#endif
#ifndef Q38_F2_W_PLACE
#define Q38_F2_W_PLACE 1
#endif
#ifndef Q38_F2_Z_PLACE
#define Q38_F2_Z_PLACE 0
#endif
#ifndef Q38_F2_X_PLACE
#define Q38_F2_X_PLACE 0
#endif
_Static_assert(Q38_F2_C1 || (Q38_F2_W_PLACE == 1 && Q38_F2_Z_PLACE == 0 &&
                             Q38_F2_X_PLACE == 0),
               "Placement switches are C1-only");
_Static_assert(Q38_F2_X_PLACE != 2 || VC_LEVELS > 2,
               "X copies need a level-2 window");
// f2 state replicas (default off, C1-only until every writer is audited): each
// group reads a copy of f2 in its own banks; core 0 writes group 0's copy and
// f2_sync_all pushes it to the other groups before the barrier.
#ifndef Q38_F2_REP
#define Q38_F2_REP 0
#endif
_Static_assert(!Q38_F2_REP || Q38_F2_C1, "f2 replicas are C1-only");
// Z tiles: per tile of a group (window), or per engine of the chip (global).
#define F2_ZN (Q38_F2_Z_PLACE ? NUM_REDMULE_TILES : NUM_TILES_PER_GROUP)
#define F2_ZE(e) (Q38_F2_Z_PLACE ? (e) : (e) % NUM_TILES_PER_GROUP)
#if Q38_F2_Z_PLACE
#define F2_ZG(ptr) (ptr)
#else
#define F2_ZG(ptr) Q38_VC_GLOBAL(ptr)
#endif
// Worker cores per tile (1 .. cores - 1; the agent is core 0): each takes rows
// w, w + WORKERS, ... of the tile's activation and residual tiles.
#ifndef Q38_FFN_WORKERS
#define Q38_FFN_WORKERS 1
#endif
// 1: workers sleep on EVT_CORE, raised by their agent after each RedMulE
// completion it publishes; the phase join is the epoch-checked fsync barrier.
#ifndef Q38_FFN_EVT
#define Q38_FFN_EVT 0
#endif
#define F2_LANES (NUM_CORES_PER_TILE - 1u)
_Static_assert(Q38_FFN_WORKERS >= 1 && Q38_FFN_WORKERS <= F2_LANES,
               "FFN workers are the tile's cores after the agent");
#define F2_WORKER_MASK (((1u << Q38_FFN_WORKERS) - 1u) << 1)
#define F2_GROUP_WORDS (NUM_TILES_PER_GROUP * NUM_BANKS_PER_TILE)
#define F2_Z_ELEMS (Q38_FFN_GU_ROWS * NT)
#if Q38_X8_PIPE && Q38_X8_SPAN
// Layer pipeline: each domain's scratch in its own span below the L2 top.
#define F2_G_L2                                                                \
  ((__fp16 *)(Q38_L2_TOP - mempool_get_domain_id() * Q38_X8_SPAN -             \
              Q38_R * Q38_I * sizeof(__fp16)))
#else
#define F2_G_L2 ((__fp16 *)(Q38_L2_TOP - Q38_R * Q38_I * sizeof(__fp16)))
#endif
#ifndef Q38_GDN_V2
#define Q38_GDN_V2 0
#endif
_Static_assert((Q38_PROBE_PHASE == 2 || Q38_GDN_V2 || Q38_ATTN_L2_LAYER) &&
                   Q38_FFN_VC == 2 && VIRTUAL_CLUSTER &&
                   Q38_FFN_GU_SLICES == 1 && Q38_FFN_DOWN_SLICES > 0 &&
                   !Q38_STAGE_SEPARATE && !Q38_FFN_NORM_STREAM &&
                   !Q38_PROBE_BURST_ONLY && Q38_FFN_GU_ROWS >= 32u &&
                   Q38_FFN_GU_ROWS <= MT && Q38_FFN_GU_ROWS % 32u == 0 &&
                   Q38_FFN_ROWS % Q38_FFN_GU_ROWS == 0 &&
                   Q38_FFN_DN_ROWS % Q38_FFN_DN_TILE == 0 &&
                   Q38_FFN_DN_TILE % 32u == 0 &&
                   Q38_VC_ENGINES % (Q38_FFN_ROWS / Q38_FFN_GU_ROWS) == 0 &&
                   Q38_VC_ENGINES % (Q38_FFN_DN_ROWS / Q38_FFN_DN_TILE) == 0 &&
                   (Q38_EXEC_END - Q38_EXEC_START) % Q38_FFN_ROWS == 0 &&
                   (Q38_EXEC_END - Q38_EXEC_START) % Q38_FFN_DN_ROWS == 0 &&
                   Q38_D * NT * 2u % Q38_VC_ROW == 0 &&
                   Q38_KC * NT * 2u % Q38_VC_ROW == 0,
               "L2 FFN requires independent aligned VC batches and K slices");

// The tiled attention out pass keeps its original 4096+2048 reductions.
// At its large slice's tail, slot 2 has retired and holds the last four Z
// steps. The small slice uses only each slot's lower half. Its X and residual
// occupy slot 1's upper half, after the large slice has completely drained.
#if Q38_KIND == 0 && Q38_ATTN_VC && Q38_L2_TILED && Q38_FFN_QUEUE &&           \
    Q38_FFN_X_PREFETCH &&                                                      \
    (!defined(Q38_F2_W_PLACE) || (Q38_F2_W_PLACE == 1 && !Q38_F2_Z_PLACE &&    \
                                  !Q38_F2_X_PLACE && !Q38_F2_C1))
#define F2_AV_PACK 1
#else
#define F2_AV_PACK 0
#endif
#if F2_AV_PACK
#define F2_PACKED_OUT (f2.packed_out)
#else
#define F2_PACKED_OUT 0u
#endif

// act/res: per worker, so a worker behind is never covered by one ahead.
typedef struct {
  uint32_t pair_done, down_done, act[F2_LANES], res[F2_LANES];
  // Q38_FFN_QUEUE: steps this engine has finished since FFN entry, never
  // reset per pass, so a stale read is low (safe), never a previous pass's
  // high count.
  uint32_t step_done;
  uint32_t pad[NUM_BANKS_PER_TILE - 3u - 2u * F2_LANES];
} f2_progress_t;
static f2_progress_t f2_progress[NUM_REDMULE_TILES] Q38_VC_TILE_LOCAL;

// Load each counter once: Q38_MIN evaluates its arguments twice, and a second
// load of a counter that moved on since the compare returns more than the
// minimum already found.
// Out of line, like f2_wake, f2_panels and f2_q_idle: inlined into f2_agent_q
// they cost its frame 32 B of the lock-step driver's 512-byte stack.
#if Q38_KIND != 0
static NOSTACK uint32_t f2_min_act(uint32_t e);
#endif
static uint32_t f2_min_act(uint32_t e) {
  uint32_t m = ~0u;
  for (uint32_t w = 0u; w < Q38_FFN_WORKERS; ++w) {
    const uint32_t x =
        __atomic_load_n(&f2_progress[e].act[w], __ATOMIC_ACQUIRE);
    m = Q38_MIN(m, x);
  }
  return m;
}

static uint32_t f2_min_res(uint32_t e) {
  uint32_t m = ~0u;
  for (uint32_t w = 0u; w < Q38_FFN_WORKERS; ++w) {
    const uint32_t x =
        __atomic_load_n(&f2_progress[e].res[w], __ATOMIC_ACQUIRE);
    m = Q38_MIN(m, x);
  }
  return m;
}

// The agent published a RedMulE completion: wake the tile's workers.
#if Q38_KIND != 0
static NOSTACK void f2_wake(void);
#endif
static void f2_wake(void) {
#if Q38_FFN_EVT
  asm volatile("fence" ::: "memory");
  fsync_raise(F2_WORKER_MASK);
#endif
}

// A worker waits for its agent's monotone flag.
static void f2_worker_wait(const uint32_t *flag, uint32_t value) {
#if Q38_FFN_EVT
  while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) < value)
    mempool_wfi();
#else
  vc_worker_wait(flag, value);
#endif
}
typedef struct {
  uint32_t issued, released;
  dma_id_t pending;
  uint32_t pad[F2_GROUP_WORDS - 3u];
} f2_ship_t;
static f2_ship_t f2_ship[NUM_GROUPS] Q38_VC_TILE_LOCAL;
#if Q38_F2_REP
struct f2_state {
#else
static struct {
#endif
  __fp16 *residual, *input, *z, *ship, *x_buf[2];
  uint32_t down, rows, tile_rows, pv, steps, slot_bytes, w_offset, slots;
  uint32_t ship_rows, ship_columns, ship_elems, peak, first, slice, arena_base;
  uint32_t weight_epoch, early;
#if F2_AV_PACK
  __fp16 *z_tail;
  uint32_t packed_out;
#endif
  uint32_t q_base; // Q38_FFN_QUEUE: step_done at this pass's start
  // Up pass: X width, output columns; gate/up pairs (pair) or single-matrix
  // segments; shipped to g_out[slice][g_rows][g_kc].
  uint32_t pair, k_up, n_up, rows_up, nseg, seg_panels[3];
  const __fp16 *seg_w[3];
  __fp16 *g_out;
  uint32_t g_kc, g_rows;
  // Down pass: K slice width, full-K weight stride, output columns, slices;
  // X slices from dn_in[slice][dn_rows][k_dn]; output = residual + X W.
  uint32_t k_dn, k_full, n_dn, rows_dn, slices, end, dn_rows, dn_lag;
  const __fp16 *w_dn, *dn_in, *res_src;
  __fp16 *dn_out;
#if Q38_L2_TILED
  // Uneven K slices: the last is k_last wide (0: as the others); g_out's
  // columns from g_main on take pitch g_kc_last.
  uint32_t k_last, g_main, g_kc_last;
#endif
#if G2_O_PADDED
  uint32_t dn_pitch; // dn_in's row pitch in elements (0: packed, F2_K)
#endif
}
#if Q38_F2_REP
;
static union {
  struct f2_state s;
  uint32_t pad[F2_GROUP_WORDS];
} f2_rep[NUM_GROUPS] Q38_VC_TILE_LOCAL;
#define f2 (f2_rep[mempool_get_group_id()].s)
#else
f2 Q38_VC_SMALL;
#endif
#if Q38_F2_C1
static struct { uint32_t w_flat, x_win; } f2c1 Q38_VC_SMALL;
#endif

#if Q38_L2_TILED
// FFN down K slices stay Q38_FFN_DOWN_SLICES even ones: an extra short slice
// costs about a whole slice (S1 A/B 10-19, down +22.6%). G keeps its pitch
// and its ships are cut into row-local runs (f2_ship_progress).
#define F2_KC Q38_KC
#define F2_DN_SLICES Q38_FFN_DOWN_SLICES
static uint32_t f2_k(uint32_t slice) {
  return f2.k_last && slice + 1u == f2.slices ? f2.k_last : f2.k_dn;
}
#define F2_K(slice) f2_k(slice)
#else
#define F2_KC Q38_KC
#define F2_DN_SLICES Q38_FFN_DOWN_SLICES
#define F2_K(slice) f2.k_dn
#endif

#if Q38_F2_NORM_HOME
// Group g's share of a batch: rows g * F2_HOME_ROWS + k, row k at byte
// k * Q38_D * 2 of the group's stripe from global L1 row f2h.res_row on.
#define F2_HOME_ROWS (Q38_FFN_ROWS / NUM_GROUPS)
#define F2_HOME_WORKERS (NUM_TILES_PER_GROUP * Q38_FFN_WORKERS)
_Static_assert(Q38_FFN_ROWS % NUM_GROUPS == 0u &&
                   Q38_D * 2u % VC_ROW_BYTES == 0u,
               "Home-bank norm: whole rows per group, rows of whole stripe "
               "rows");
typedef struct {
  uint32_t ready; // batches of this pass whose share has landed
  uint32_t pad[F2_GROUP_WORDS - 1u];
} f2_home_t;
static f2_home_t f2_home[NUM_GROUPS] Q38_VC_TILE_LOCAL;
_Static_assert(sizeof(f2_home_t) == VC_ROW_BYTES,
               "Group control records must occupy one owner-group stripe");
static struct {
  const __fp16 *gamma; // RMSNorm weights in L1
  uint32_t res_row;    // first global L1 row of the shares (f2.residual's)
} f2h Q38_VC_SMALL;

// Byte o of group g's residual share.
static char *f2_home_at(uint32_t g, uint32_t o) {
  return (char *)(VC_GLOBAL_BASE +
                  (f2h.res_row + o / VC_ROW_BYTES) * VC_GLOBAL_ROW_BYTES +
                  g * VC_ROW_BYTES + o % VC_ROW_BYTES);
}

// Group coordinator: its share of the batch whose rows start at `rows` (L2).
static dma_id_t f2_home_issue(uint32_t g, const __fp16 *rows) {
  return dma_local_memcpy_2d_nonblocking(
      0u, f2_home_at(g, 0u), rows + g * F2_HOME_ROWS * Q38_D, VC_ROW_BYTES,
      VC_GLOBAL_ROW_BYTES, VC_ROW_BYTES,
      F2_HOME_ROWS * Q38_D * 2u / VC_ROW_BYTES);
}

// q38_norm16_sliced, bit for bit (same chunks, same instructions), on a row
// laid out one stripe row (VC_ROW_BYTES) per global L1 row, with the result
// stored as 32-bit elements (the same bytes).
static void q38_norm16_striped(const __fp16 *x, const __fp16 *gamma,
                               __fp16 *out, uint32_t row) {
  const uint32_t se = VC_ROW_BYTES / 2u, ge = VC_GLOBAL_ROW_BYTES / 2u;
  uint32_t capacity;
  const float zero = 0.0f, zh = llm_f16box(0.0f);
  asm volatile("vsetvli %0, %1, e32, m8, tu, ma"
               : "=r"(capacity)
               : "r"(UINT32_MAX));
  if (se % capacity)
    q38_fatal("[qwen38_fatal] norm-home chunk=", capacity);
  asm volatile("vmv.v.i v8, 0");
  for (uint32_t offset = 0; offset < Q38_D;) {
    uint32_t vl, requested = Q38_D - offset;
    if (requested > capacity)
      requested = capacity;
    asm volatile("vsetvli %0, %1, e16, m4, tu, ma" : "=r"(vl) : "r"(requested));
    asm volatile("vle16.v v0, (%0)"
                 :
                 : "r"(x + offset / se * ge + offset % se)
                 : "memory");
    asm volatile("vfwmacc.vv v8, v0, v0");
    offset += vl;
  }
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(capacity));
  asm volatile("vfmv.s.f v24, %0" : : "f"(zero));
  asm volatile("vfredusum.vs v24, v8, v24");
  float sum;
  asm volatile("vfmv.f.s %0, v24" : "=f"(sum));
  const float inverse = llm_rsqrt(sum * llm_recip_pos((float)Q38_D) + 1e-6f);
  for (uint32_t offset = 0; offset < Q38_D;) {
    uint32_t vl, requested = Q38_D - offset;
    if (requested > capacity)
      requested = capacity;
    if (requested > Q38_GUK - offset % Q38_GUK)
      requested = Q38_GUK - offset % Q38_GUK;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(requested));
    __fp16 *const dst = out + offset / Q38_GUK * Q38_FFN_ROWS * Q38_GUK +
                        row * Q38_GUK + offset % Q38_GUK;
    asm volatile("vle16.v v0, (%0)"
                 :
                 : "r"(x + offset / se * ge + offset % se)
                 : "memory");
    asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
    asm volatile("vle16.v v0, (%0)" : : "r"(gamma + offset) : "memory");
    asm volatile("vfwadd.vf v16, v0, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    asm volatile("vfmul.vf v8, v8, %0" : : "f"(inverse));
    asm volatile("vfmul.vv v8, v8, v16");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
    asm volatile("vfncvt.f.f.w v0, v8");
    if (vl & 1u) {
      asm volatile("vse16.v v0, (%0)" : : "r"(dst) : "memory");
    } else {
      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl / 2u));
      asm volatile("vse32.v v0, (%0)" : : "r"(dst) : "memory");
    }
    offset += vl;
  }
}

// Row k of group g's share into input buffer `out`.
static void f2_home_row(uint32_t g, uint32_t k, __fp16 *out) {
  q38_norm16_striped((const __fp16 *)f2_home_at(g, k * Q38_D * 2u), f2h.gamma,
                     out, g * F2_HOME_ROWS + k);
}
#endif

// Panel group (pair: gate/up pair) index of step t.
static uint32_t f2_n(uint32_t t) { return f2.down ? t : f2.pair ? t / 2u : t; }

// Panel groups shipped by an up pass.
static uint32_t f2_groups(void) { return f2.pair ? f2.steps / 2u : f2.steps; }
typedef struct {
  uint32_t res_issued, res_ready, out_issued;
  dma_id_t res_pending, out_pending;
  dma_id_t weight_pending, x_pending;
  uint32_t weight_epoch, weight_slot, weight_valid;
  uint32_t pad[F2_GROUP_WORDS - 10u];
} f2_down_t;
static f2_down_t f2_down[NUM_GROUPS] Q38_VC_TILE_LOCAL;
_Static_assert(sizeof(f2_ship_t) == VC_ROW_BYTES &&
                   sizeof(f2_down_t) == VC_ROW_BYTES,
               "Group control records must occupy one owner-group stripe");
extern uint32_t __heap_end;
#if Q38_KIND == 0 && Q38_ATTN_VC
extern uint32_t __l1_attn_reclaim;
#endif

static uint32_t f2_align(uint32_t bytes) {
  return (bytes + VC_GLOBAL_ROW_BYTES - 1u) / VC_GLOBAL_ROW_BYTES *
         VC_GLOBAL_ROW_BYTES;
}

static NOSTACK void f2_layout(uint32_t down) {
  if (mempool_get_core_id() != 0u)
    return;
  f2.down = down;
  f2.early = 0u;
  f2.rows = down ? f2.rows_dn : f2.rows_up;
  f2.tile_rows = down ? Q38_FFN_DN_TILE : Q38_FFN_GU_ROWS;
  f2.pv = Q38_VC_ENGINES / (f2.rows / f2.tile_rows);
  if (NUM_TILES_PER_GROUP % f2.pv)
    q38_fatal("[qwen38_fatal] ffn group panels=", f2.pv);
  const uint32_t columns = (down ? f2.n_dn : f2.n_up) / NT;
  f2.steps = (columns + f2.pv * Q38_VC_COUNT - 1u) / (f2.pv * Q38_VC_COUNT) *
             (down || !f2.pair ? 1u : 2u);
  uint32_t p = Q38_MIN((uint32_t)&stage, (uint32_t)&weights[0][0]);
#if Q38_KIND == 0 && Q38_ATTN_VC
  // Attention out follows the sweep's final join; FFN follows attention.
  // Their handshakes/constants are dead in either down pass and in FFN up
  // (pair=1). Profiles remain live outside this area. QKV up keeps stage.
  if (down || f2.pair)
    p = Q38_MIN(p, (uint32_t)&__l1_attn_reclaim);
#endif
  p = f2_align(p);
  f2.arena_base = p;
  f2.residual = (__fp16 *)p;
  p += down ? f2_align(NUM_GROUPS * NUM_TILES_PER_GROUP * f2.tile_rows * NT *
                       sizeof(__fp16))
            : f2_align(f2.rows * f2.k_up * sizeof(__fp16));
  if (down)
    f2.residual = vc_group_window((uint32_t)f2.residual);
  // Up passes read the residual only for the RMSNorm, which ends in
  // f2_sync_all before f2_agent loads a weight: the slots, Z and ship
  // overlay it, and the input follows whichever ends last.
  const uint32_t overlay = Q38_FFN_UP_OVERLAY && !down;
  const uint32_t input_bytes =
      f2_align(f2.rows * (down ? f2.k_dn : f2.k_up) * sizeof(__fp16));
  const uint32_t residual_end = p;
  if (overlay) {
    p = f2.arena_base;
  } else {
    f2.input = (__fp16 *)p;
    p += input_bytes;
  }
  const uint32_t piece = (down ? f2.k_dn : f2.k_up) * NT * sizeof(__fp16);
  f2.slot_bytes = (f2.pv * piece + Q38_VC_ROW - 1u) / Q38_VC_ROW * Q38_VC_ROW;
  f2.w_offset = VC_GLOBAL_ROW(p) * Q38_VC_ROW;
#if Q38_F2_C1
  f2c1.w_flat = p; // the same rows: all VCs' panels of a slot, VC-major
#endif
  f2.slots = Q38_FFN_LAG && (!down || f2.dn_lag) ? 3u : 2u;
  p += f2.slots * f2.slot_bytes / Q38_VC_ROW * VC_GLOBAL_ROW_BYTES;
#if Q38_F2_Z_PLACE
  f2.z = (__fp16 *)p; // the same bytes, engine-major over the chip
#else
  f2.z = vc_group_window(p);
#endif
  const uint32_t z_bytes =
      NUM_TILES_PER_GROUP *
      (down ? f2.steps * f2.tile_rows * NT : 4u * F2_Z_ELEMS) * 2u;
  p += (z_bytes + VC_ROW_BYTES - 1u) / VC_ROW_BYTES * VC_GLOBAL_ROW_BYTES;
  f2.x_buf[0] = f2.x_buf[1] = f2.input;
  if (Q38_FFN_X_PREFETCH && down) {
    f2.x_buf[1] = (__fp16 *)p;
    p += input_bytes;
  }
  f2.ship_columns = f2.pv * NT;
  f2.ship_rows = NUM_TILES_PER_GROUP / f2.pv * f2.tile_rows;
  f2.ship_elems = f2.ship_rows * f2.ship_columns;
  f2.ship = vc_group_window(p);
  if (!down)
    p += (2u * f2.ship_elems * 2u + VC_ROW_BYTES - 1u) / VC_ROW_BYTES *
         VC_GLOBAL_ROW_BYTES;
  if (overlay) {
    const uint32_t res_bytes = residual_end - f2.arena_base;
    const uint32_t slot0 = f2.slot_bytes / Q38_VC_ROW * VC_GLOBAL_ROW_BYTES;
    if (Q38_FFN_UP_EARLY_WEIGHT && p >= f2.arena_base + slot0 + res_bytes) {
      f2.residual = (__fp16 *)(p - res_bytes);
      f2.early = 1u;
    }
    p = Q38_MAX(p, residual_end);
    f2.input = (__fp16 *)p;
    p += input_bytes;
  }
#if Q38_F2_X_PLACE == 2
  // Each level-2 rectangle's own copy of X, rows of that window.
  f2c1.x_win = VC_WINDOW_BASE_2 + VC_GLOBAL_ROW(p) * vc_row_stride(2u);
  p += (input_bytes + vc_row_stride(2u) - 1u) / vc_row_stride(2u) *
       VC_GLOBAL_ROW_BYTES;
#endif
#if Q38_F2_NORM_HOME
  // The shares take the residual's rows: the same bytes, group by group.
  if (!down)
    f2h.res_row = VC_GLOBAL_ROW(f2.residual);
#endif
#if F2_AV_PACK
  // Exact shape guard: no other caller or experimental placement is packed.
  f2.packed_out = down && f2.dn_lag && f2.slots == 3u && f2.k_dn == 4096u &&
                  f2.k_last == 2048u && f2.slices == 2u && f2.rows == 256u &&
                  f2.tile_rows == 32u && NT == 64u && f2.pv == 2u &&
                  f2.steps == 10u && NUM_TILES_PER_GROUP == 4u &&
                  Q38_VC_COUNT == 4u && NUM_GROUPS == 16u;
  if (f2.packed_out) {
    const uint32_t slot = f2.slot_bytes / Q38_VC_ROW * VC_GLOBAL_ROW_BYTES;
    const uint32_t w = f2.arena_base + input_bytes;
    f2.input = f2.x_buf[0] = (__fp16 *)f2.arena_base;
    f2.w_offset = VC_GLOBAL_ROW(w) * Q38_VC_ROW;
    f2.z = vc_group_window(w + 3u * slot);
    // Slot 2: step 5 is its last large-K tenant. Steps 6..9 use slots 0/1.
    f2.z_tail = vc_group_window(w + 2u * slot + 3u * slot / 4u);
    // Slot 1: the small-K weights end at half the slot; X then residual.
    f2.x_buf[1] = (__fp16 *)(w + slot + slot / 2u);
    f2.residual = vc_group_window(w + slot + 3u * slot / 4u);
    p = w + 3u * slot +
        6u * NUM_TILES_PER_GROUP * f2.tile_rows * NT * 2u * NUM_GROUPS;
  }
#endif
#ifdef Q38_F2_LAYOUT_PRINT
  printf("[f2_layout] down=%u arena=%x residual=%x input=%x z=%x ship=%x "
         "slot=%x slots=%u p=%x\n",
         down, f2.arena_base, (uint32_t)f2.residual, (uint32_t)f2.input,
         (uint32_t)f2.z, (uint32_t)f2.ship, f2.slot_bytes, f2.slots, p);
#endif
  f2.peak = Q38_MAX(f2.peak, p);
#if Q38_G2_ITEM_KEEP && Q38_X8_BATCH
  if (p > g2_item_limit())
    q38_fatal("[qwen38_fatal] l2 ffn over GDN params=", p - g2_item_limit());
#else
  if (p > (uint32_t)&__heap_end)
    q38_fatal("[qwen38_fatal] l2 ffn over L1=", p - (uint32_t)&__heap_end);
#endif
#if Q38_SCHEDULE_PROBE
  if (down) {
    ++probe_ffn_blocks;
    probe_projection_groups += f2.steps * f2.slices;
  } else {
    probe_gateup_groups += f2_groups();
  }
#endif
}

// Every record is reset by its owning tile/group, before the phase barrier.
// Weight-cache ownership spans Down chunks and batches; reset it at FFN entry.
#if Q38_GDN_ENTRY_INIT
static void f2_zero(volatile uint32_t *p, uint32_t bytes) {
  for (uint32_t i = 0u; i < bytes / sizeof(uint32_t); ++i)
    p[i] = 0u;
}

// Every core, before the first edge barrier (which fences): nothing is in
// flight yet. Each record is zeroed by the core that owns it.
static void f2_init_records(void) {
  // f2 field by field: f2check follows named field stores, not a pointer.
#if Q38_F2_REP
  if (mempool_is_group_coordinator()) {
#else
  if (mempool_get_core_id() == 0u) {
#endif
    f2.residual = f2.input = f2.z = f2.ship = 0;
    f2.x_buf[0] = f2.x_buf[1] = 0;
    f2.down = f2.rows = f2.tile_rows = f2.pv = f2.steps = 0u;
    f2.slot_bytes = f2.w_offset = f2.slots = 0u;
    f2.ship_rows = f2.ship_columns = f2.ship_elems = f2.peak = 0u;
    f2.first = f2.slice = f2.arena_base = f2.weight_epoch = f2.early = 0u;
    f2.q_base = 0u;
    f2.pair = f2.k_up = f2.n_up = f2.rows_up = f2.nseg = 0u;
    f2.seg_panels[0] = f2.seg_panels[1] = f2.seg_panels[2] = 0u;
    f2.seg_w[0] = f2.seg_w[1] = f2.seg_w[2] = 0;
    f2.g_out = 0;
    f2.g_kc = f2.g_rows = 0u;
    f2.k_dn = f2.k_full = f2.n_dn = f2.rows_dn = f2.slices = 0u;
    f2.end = f2.dn_rows = f2.dn_lag = 0u;
    f2.w_dn = f2.dn_in = f2.res_src = 0;
    f2.dn_out = 0;
#if Q38_L2_TILED
    f2.k_last = f2.g_main = f2.g_kc_last = 0u;
#endif
#if G2_O_PADDED
    f2.dn_pitch = 0u;
#endif
  }
#if Q38_F2_C1
  if (mempool_get_core_id() == 0u)
    f2_zero((volatile uint32_t *)&f2c1, sizeof(f2c1));
#endif
  if (tnc_is_tile_agent())
    f2_zero((volatile uint32_t *)&f2_progress[mempool_get_redmule_id()],
            sizeof(f2_progress[0]));
  if (mempool_is_group_coordinator()) {
    const uint32_t g = mempool_get_group_id();
    f2_zero((volatile uint32_t *)&f2_ship[g], sizeof(f2_ship[0]));
    f2_zero((volatile uint32_t *)&f2_down[g], sizeof(f2_down[0]));
#if Q38_F2_NORM_HOME
    f2_zero((volatile uint32_t *)&f2_home[g], sizeof(f2_home[0]));
#endif
  }
}
#endif

static void f2_reset_progress(void) {
  if (tnc_is_tile_agent()) {
    const uint32_t e = mempool_get_redmule_id();
    f2_progress[e].pair_done = f2_progress[e].down_done = 0u;
    for (uint32_t w = 0u; w < F2_LANES; ++w)
      f2_progress[e].act[w] = f2_progress[e].res[w] = 0u;
  }
  if (mempool_is_group_coordinator()) {
    const uint32_t g = mempool_get_group_id();
    f2_ship[g].issued = f2_ship[g].released = f2_ship[g].pending = 0u;
    f2_down[g].res_issued = f2_down[g].res_ready = f2_down[g].out_issued =
        f2_down[g].res_pending = f2_down[g].out_pending = 0u;
  }
}

// A down pass's K slice restarts the step counts (before the slice barrier).
static void f2_slice_reset(void) {
  if (tnc_is_tile_agent()) {
    const uint32_t e = mempool_get_redmule_id();
    f2_progress[e].down_done = 0u;
  }
}

// A phase rendezvous of the whole machine (sync_all: the fractal_sync tree).
#if Q38_F2_REP
// Every f2 writer is core 0 (group 0), between two of these barriers.
static void f2_sync_all(void) {
  if (mempool_get_core_id() == 0u)
    for (uint32_t g = 1u; g < NUM_GROUPS; ++g)
      for (uint32_t i = 0u; i < sizeof(struct f2_state) / 4u; ++i)
        f2_rep[g].pad[i] = f2_rep[0].pad[i];
  sync_all();
}
#else
static void f2_sync_all(void) { sync_all(); }
#endif

static uint32_t f2_weight_slot(uint32_t t) {
#if F2_AV_PACK
  if (f2.packed_out)
    return f2.slice == 0u && t >= 6u ? t & 1u : t % 3u;
#endif
  return ((f2.down ? f2.weight_epoch * f2.steps : 0u) + t) % f2.slots;
}

#ifndef F2_ZN
#define F2_ZN NUM_TILES_PER_GROUP
#define F2_ZE(e) ((e) % NUM_TILES_PER_GROUP)
#endif
// The same step/engine tiles, with only the packed out pass's tail relocated.
static __fp16 *f2_down_z(uint32_t n, uint32_t engine) {
  __fp16 *z = f2.z;
#if F2_AV_PACK
  if (f2.packed_out && n >= 6u) {
    z = f2.z_tail;
    n -= 6u;
  }
#endif
  return z + (n * F2_ZN + F2_ZE(engine)) * f2.tile_rows * NT;
}

#if Q38_KIND != 0
static NOSTACK uint32_t f2_panels(uint32_t v, uint32_t t);
#endif
static uint32_t f2_panels(uint32_t v, uint32_t t) {
  const uint32_t first = f2_n(t) * f2.pv * Q38_VC_COUNT + v * f2.pv;
  const uint32_t columns = (f2.down ? f2.n_dn : f2.n_up) / NT;
  return first >= columns ? 0u : Q38_MIN(f2.pv, columns - first);
}

// Panel `first` of step t's weight matrix (pv panels, never across segments).
static const __fp16 *f2_panel(uint32_t first, uint32_t t, uint32_t slice) {
  if (f2.down)
    return f2.w_dn + (first * f2.k_full + slice * f2.k_dn) * NT;
  if (f2.pair)
    return f2.seg_w[t & 1u] + first * f2.k_up * NT;
  uint32_t s = 0u;
  while (s + 1u < f2.nseg && first >= f2.seg_panels[s])
    first -= f2.seg_panels[s++];
  return f2.seg_w[s] + first * f2.k_up * NT;
}

static NOSTACK dma_id_t f2_load_at(uint32_t v, uint32_t t, uint32_t slice,
                                   uint32_t slot) {
#if Q38_F2_C1
  (void)v, (void)t, (void)slice, (void)slot;
  return 0u; // resident
#endif
  const uint32_t panels = f2_panels(v, t);
  if (!panels)
    return 0u;
  const uint32_t group = mempool_get_group_id();
  const uint32_t k = f2.down ? F2_K(slice) : f2.k_up;
  const uint32_t first = f2_n(t) * f2.pv * Q38_VC_COUNT + v * f2.pv;
  const __fp16 *const source = f2_panel(first, t, slice);
  const uint32_t rows = k * NT * 2u / Q38_VC_ROW;
  const uint32_t offset = f2.w_offset + slot * f2.slot_bytes;
  void *dest = (void *)(offset / Q38_VC_ROW * VC_GLOBAL_ROW_BYTES +
                        group * VC_ROW_BYTES);
#if Q38_L2_TILED
  // One descriptor per panel: four VC rows (one 8 KiB L2 row) by rows / 4
  // planes, so every plane step is a whole L2 row (split-free domain fold).
  // A panel across the home wrap goes as two, cut at the wrap row.
  if (rows % 4u == 0u) {
    const uint32_t stride = (f2.down ? f2.k_full : f2.k_up) * NT * 2u;
    const char *src =
        (const char *)source + group % Q38_VC_GROUPS * VC_ROW_BYTES;
    char *dst = (char *)dest;
    dma_id_t id = 0u;
    for (uint32_t p = 0u; p < panels; ++p, src += stride) {
      uint32_t planes = rows / 4u, cut = planes;
#if DOMAIN_L2_HOME
      if (dma_domain_home((uint32_t)src)) {
        const uint32_t wrapo = DMA_DOMAIN_HOME_WRAP / DOMAINS;
        const uint32_t row = ((uint32_t)src - DMA_DOMAIN_HOME_BASE) /
                             VC_GLOBAL_ROW_BYTES * VC_GLOBAL_ROW_BYTES;
        const uint32_t left = (wrapo - row % wrapo) / VC_GLOBAL_ROW_BYTES;
        cut = Q38_MIN(planes, left);
      }
#endif
      for (uint32_t at = 0u; at < planes; at += cut, cut = planes - at)
        id = dma_local_memcpy_3d_nonblocking(
            0u, dst + at * 4u * VC_GLOBAL_ROW_BYTES,
            src + at * VC_GLOBAL_ROW_BYTES, VC_ROW_BYTES, VC_GLOBAL_ROW_BYTES,
            Q38_VC_ROW, 4u, VC_GLOBAL_ROW_BYTES, Q38_VC_ROW, cut);
      dst += rows * VC_GLOBAL_ROW_BYTES;
    }
    return id; // DONE_ID is monotone: the last covers them all
  }
#endif
  return dma_local_memcpy_3d_nonblocking(
      0u, dest, (const char *)source + group % Q38_VC_GROUPS * VC_ROW_BYTES,
      VC_ROW_BYTES, VC_GLOBAL_ROW_BYTES, Q38_VC_ROW, rows, VC_GLOBAL_ROW_BYTES,
      (f2.down ? f2.k_full : f2.k_up) * NT * 2u - (rows - 1u) * Q38_VC_ROW,
      panels);
}

static dma_id_t f2_load(uint32_t v, uint32_t t) {
  return f2_load_at(v, t, f2.slice, f2_weight_slot(t));
}

static NOSTACK dma_id_t f2_first_weight(uint32_t v) {
  const uint32_t g = mempool_get_group_id();
  if (f2_down[g].weight_valid) {
    if (f2.down && (f2_down[g].weight_epoch != f2.weight_epoch ||
                    f2_down[g].weight_slot != f2_weight_slot(0u)))
      q38_fatal("[qwen38_fatal] stale weight epoch=", f2.weight_epoch);
    f2_down[g].weight_valid = 0u;
    return f2_down[g].weight_pending;
  }
  return f2_load(v, 0u);
}

static NOSTACK void f2_prefetch_weight(uint32_t v) {
  uint32_t slice = f2.slice + 1u;
  if (slice == f2.slices) {
    if (f2.first + f2.rows >= f2.end)
      return;
    slice = 0u;
  }
  const uint32_t g = mempool_get_group_id();
  const uint32_t slot =
      F2_PACKED_OUT ? 0u : (f2_weight_slot(f2.steps - 1u) + 1u) % f2.slots;
  f2_down[g].weight_pending = f2_load_at(v, 0u, slice, slot);
  f2_down[g].weight_epoch = f2.weight_epoch + 1u;
  f2_down[g].weight_slot = slot;
  f2_down[g].weight_valid = 1u;
}

// The next epoch's X slice (this batch's next slice or the next batch's
#if G2_O_PADDED
// The down X of rows [first, first + f2.rows) of a slice, in L2.
static const __fp16 *f2_x_src(uint32_t slice, uint32_t first) {
  if (f2.dn_pitch)
    return f2.dn_in + (slice * f2.dn_rows + first) * f2.dn_pitch;
  return f2.dn_in + slice * f2.dn_rows * f2.k_dn + first * F2_K(slice);
}

// The group coordinator: rows of k_bytes, pitch_bytes apart in L2 (on 8 KiB
// L2 rows), packed at dst (on an 8 KiB L1 row); this group's 512-byte L1
// stripes only. Stripes and rows realign every lcm(k_bytes, 8 KiB) of dst:
// one 2-D descriptor per row-local piece of a stripe in that period, the
// periods as its repetitions.
static NOSTACK dma_id_t f2_x_gather(__fp16 *dst, const __fp16 *src,
                                    uint32_t rows, uint32_t k_bytes,
                                    uint32_t pitch_bytes) {
  const uint32_t stripe = VC_GLOBAL_ROW_BYTES / NUM_GROUPS;
  const uint32_t d = dma_l1_alias((uint32_t)dst);
  uint32_t a = k_bytes, b = VC_GLOBAL_ROW_BYTES;
  while (b) {
    const uint32_t t = a % b;
    a = b, b = t;
  }
  const uint32_t period = k_bytes / a * VC_GLOBAL_ROW_BYTES;
  const uint32_t n = period / k_bytes;
  if (d % VC_GLOBAL_ROW_BYTES || (uint32_t)src % VC_GLOBAL_ROW_BYTES ||
      pitch_bytes % VC_GLOBAL_ROW_BYTES || k_bytes > pitch_bytes || rows % n)
    q38_fatal("[qwen38_fatal] x gather shape=", rows);
  dma_id_t id = 0u;
  for (uint32_t s = mempool_get_group_id() * stripe; s < period;
       s += VC_GLOBAL_ROW_BYTES)
    for (uint32_t o = s; o < s + stripe;) {
      const uint32_t c = o % k_bytes,
                     len = Q38_MIN(s + stripe - o, k_bytes - c);
      id = dma_local_memcpy_2d_nonblocking(
          0u, (void *)(d + o),
          (const char *)src + o / k_bytes * pitch_bytes + c, len, period,
          n * pitch_bytes, rows / n);
      o += len;
    }
  return id;
}
#endif

// first) into the other buffer; this group's rows only, waited at the end of
// f2_agent, before the slice barrier.
static NOSTACK void f2_prefetch_x(void) {
#if Q38_F2_C1
  return; // resident
#endif
  uint32_t slice = f2.slice + 1u, first = f2.first;
  if (slice == f2.slices) {
    if (f2.first + f2.rows >= f2.end)
      return;
    slice = 0u, first += f2.rows;
  }
#if G2_O_PADDED
  if (f2.dn_pitch) {
    f2_down[mempool_get_group_id()].x_pending = f2_x_gather(
        f2.x_buf[(f2.weight_epoch + 1u) & 1u], f2_x_src(slice, first), f2.rows,
        F2_K(slice) * sizeof(__fp16), f2.dn_pitch * sizeof(__fp16));
    return;
  }
#endif
#if Q38_L2_TILED
  const dma_coordinator_op_t op = dma_coordinator_submit_linear(
      0u, f2.x_buf[(f2.weight_epoch + 1u) & 1u],
      f2.dn_in + slice * f2.dn_rows * f2.k_dn + first * F2_K(slice),
      f2.rows * F2_K(slice) * sizeof(__fp16));
#else
  const dma_coordinator_op_t op = dma_coordinator_submit_linear(
      0u, f2.x_buf[(f2.weight_epoch + 1u) & 1u],
      f2.dn_in + (slice * f2.dn_rows + first) * f2.k_dn,
      f2.rows * f2.k_dn * sizeof(__fp16));
#endif
  if (op.status)
    q38_fatal("[qwen38_fatal] x prefetch status=", op.status);
  f2_down[mempool_get_group_id()].x_pending = op.id;
}

// Only the group issuer calls this. Each completed DMA frees its source
// generation; activation readiness alone does not permit buffer reuse.
static NOSTACK void f2_ship_progress(void) {
  if (f2.down)
    return;
  const uint32_t g = mempool_get_group_id(), v = g / Q38_VC_GROUPS;
  if (f2_ship[g].pending) {
    if (!dma_local_done_id(0u, f2_ship[g].pending))
      return;
    f2_ship[g].pending = 0u;
    __atomic_store_n(&f2_ship[g].released, f2_ship[g].issued, __ATOMIC_RELEASE);
  }
  const uint32_t n = f2_ship[g].issued;
  if (n >= f2_groups())
    return;
  for (uint32_t e = g * NUM_TILES_PER_GROUP; e < (g + 1u) * NUM_TILES_PER_GROUP;
       ++e)
    if (f2_min_act(e) < n + 1u)
      return;
  ++f2_ship[g].issued;
  const uint32_t panels = f2_panels(v, f2.pair ? 2u * n : n);
  if (!panels) {
    __atomic_store_n(&f2_ship[g].released, n + 1u, __ATOMIC_RELEASE);
    return;
  }
  // Rows in one bank stripe are adjacent in the staging slot. Their L2
  // destinations have the full G row stride; plane stride is relative to
  // the last row, as required by the iDMA 3-D descriptor contract.
  const uint32_t row_bytes = f2.ship_columns * sizeof(__fp16);
  const uint32_t inner = VC_ROW_BYTES / row_bytes;
  const uint32_t row = g % Q38_VC_GROUPS * f2.ship_rows;
  const uint32_t col = (n * Q38_VC_COUNT + v) * f2.ship_columns;
  const __fp16 *source = f2.ship + (n & 1u) * f2.ship_elems;
  // A producer panel group can cross the consumer's K boundary.
  // Preserve the source row stride while exporting each destination piece.
  uint32_t exported = 0u, remaining = panels * NT;
#if Q38_L2_TILED
  // Slices of g_kc columns at an 8 KiB pitch up to g_main, then one slice at
  // pitch g_kc_last. A short pitch packs m rows per 8 KiB L2 row: each
  // descriptor takes at most m rows of a stripe, and its planes step whole
  // L2 rows, or it takes one stripe only.
  while (remaining) {
    const uint32_t column = col + exported;
    const uint32_t slice = column / f2.g_kc;
    const uint32_t pitch = column < f2.g_main ? f2.g_kc : f2.g_kc_last;
    const uint32_t width = Q38_MIN(remaining, pitch - column % f2.g_kc);
    const uint32_t p = pitch * sizeof(__fp16);
    const uint32_t row_fit =
        p % VC_GLOBAL_ROW_BYTES == 0u || VC_GLOBAL_ROW_BYTES % p == 0u;
    const uint32_t m =
        p < VC_GLOBAL_ROW_BYTES ? VC_GLOBAL_ROW_BYTES / p : inner;
    const uint32_t sub = row_fit ? Q38_MIN(inner, m) : inner;
    const uint32_t stripes = f2.ship_rows / inner;
    const uint32_t whole = !row_fit || inner * p % VC_GLOBAL_ROW_BYTES == 0u;
    __fp16 *const dst = f2.g_out + slice * f2.g_rows * f2.g_kc +
                        (f2.first + row) * pitch + column % f2.g_kc;
    const char *const src = (const char *)Q38_VC_GLOBAL(source + exported);
    if (!row_fit) {
      // Neither a divisor nor a multiple of the row: per stripe, runs of rows
      // whose bursts move forward inside their L2 rows (one 2-D descriptor
      // each), cut where a row's burst wraps to the next L2 row's start.
      const uint32_t len = width * sizeof(__fp16);
      for (uint32_t s = 0u; s < stripes; ++s)
        for (uint32_t h = 0u; h < inner;) {
          const uint32_t a = (uint32_t)(dst + (s * inner + h) * pitch);
          uint32_t n = 1u;
          while (h + n < inner &&
                 (a + n * p) % VC_GLOBAL_ROW_BYTES >
                     (a + (n - 1u) * p) % VC_GLOBAL_ROW_BYTES &&
                 (a + n * p) % VC_GLOBAL_ROW_BYTES + len <= VC_GLOBAL_ROW_BYTES)
            ++n;
          f2_ship[g].pending = dma_local_memcpy_2d_nonblocking(
              0u, (void *)a, src + s * VC_GLOBAL_ROW_BYTES + h * row_bytes, len,
              p, row_bytes, n);
          h += n;
        }
      remaining -= width;
      exported += width;
      continue;
    }
    for (uint32_t s = 0u; s < (whole ? 1u : stripes); ++s)
      for (uint32_t h = 0u; h < inner; h += sub)
        f2_ship[g].pending = dma_local_memcpy_3d_nonblocking(
            0u, dst + (s * inner + h) * pitch,
            src + s * VC_GLOBAL_ROW_BYTES + h * row_bytes,
            width * sizeof(__fp16), p, row_bytes, sub,
            inner * p - (sub - 1u) * p,
            VC_GLOBAL_ROW_BYTES - (sub - 1u) * row_bytes, whole ? stripes : 1u);
    remaining -= width;
    exported += width;
  }
#else
  while (remaining) {
    const uint32_t column = col + exported;
    const uint32_t slice = column / f2.g_kc;
    const uint32_t width = Q38_MIN(remaining, f2.g_kc - column % f2.g_kc);
    f2_ship[g].pending = dma_local_memcpy_3d_nonblocking(
        0u,
        f2.g_out + slice * f2.g_rows * f2.g_kc + (f2.first + row) * f2.g_kc +
            column % f2.g_kc,
        Q38_VC_GLOBAL(source + exported), width * sizeof(__fp16),
        f2.g_kc * sizeof(__fp16), row_bytes, inner, f2.g_kc * sizeof(__fp16),
        VC_GLOBAL_ROW_BYTES - (inner - 1u) * row_bytes, f2.ship_rows / inner);
    remaining -= width;
    exported += width;
  }
#endif
}

static uint8_t f2_configure(uint32_t t, uint32_t engine) {
  const uint32_t v = engine / Q38_VC_ENGINES, ev = engine % Q38_VC_ENGINES;
  const uint32_t r = ev / f2.pv, q = ev % f2.pv;
  const uint32_t n = f2_n(t);
  const uint32_t k = f2.down ? F2_K(f2.slice) : f2.k_up;
#if Q38_F2_W_PLACE
  const __fp16 *w = (const __fp16 *)(VC_WINDOW_BASE_1 + f2.w_offset +
                                     f2_weight_slot(t) * f2.slot_bytes) +
                    q * k * NT;
#else
  const __fp16 *w =
      (const __fp16 *)(f2c1.w_flat +
                       (f2_weight_slot(t) * Q38_VC_COUNT + v) * f2.slot_bytes) +
      q * k * NT;
#endif
  __fp16 *z =
      f2.z + F2_ZE(engine) * 4u * F2_Z_ELEMS +
      ((f2.down || !f2.pair ? 0u : 2u * (t & 1u)) + (n & 1u)) * F2_Z_ELEMS;
#if F2_AV_PACK
  if (f2.down)
    z = f2_down_z(n, engine);
#else
  if (f2.down)
    z = f2.z + (n * F2_ZN + F2_ZE(engine)) * f2.tile_rows * NT;
#endif
  const uint8_t target = (uint8_t)tnc_redmule_acquire_target();
#if Q38_F2_X_PLACE == 2
  const uint32_t x = f2c1.x_win + r * f2.tile_rows * k * sizeof(__fp16);
#else
  const uint32_t x = (uint32_t)(f2.input + r * f2.tile_rows * k);
#endif
  redmule_cfg(x, (uint32_t)w, (uint32_t)z, (uint16_t)f2.tile_rows, (uint16_t)k,
              (uint16_t)NT, 0u, f2.down && f2.slice ? GEMM : MATMUL, Float16);
  redmule_x_shift_set(0);
  (void)v;
  return target;
}

// Each tile has one small residual slot. Partial outputs live until the
// final slice, so output DMA never blocks the next RedMulE output address.
static NOSTACK dma_id_t f2_down_transfer(uint32_t n, uint32_t output) {
#if F2_AV_PACK
  const uint32_t g = mempool_get_group_id(), v = g / Q38_VC_GROUPS;
  const uint32_t inner = VC_ROW_BYTES / (NT * sizeof(__fp16));
#else
  const uint32_t g = mempool_get_group_id();
  const uint32_t inner = VC_ROW_BYTES / (NT * sizeof(__fp16));
  // Per-call bases, so that n and the VC index do not stay live in the loop
  // (the lock-step driver's 512-byte stack has no room for one more spill).
  const uint32_t panel0 = (n * Q38_VC_COUNT + g / Q38_VC_GROUPS) * f2.pv;
#if F2_AV_PACK
  __fp16 *const local0 = output ? f2_down_z(n, 0u) : f2.residual;
#else
  __fp16 *const local0 =
      output ? f2.z + n * NUM_TILES_PER_GROUP * f2.tile_rows * NT : f2.residual;
#endif
  __fp16 *const remote0 = output ? f2.dn_out : (__fp16 *)f2.res_src;
#endif
  dma_id_t pending = 0u;
  for (uint32_t t = 0u; t < NUM_TILES_PER_GROUP; ++t) {
    const uint32_t e = g * NUM_TILES_PER_GROUP + t;
    const uint32_t r = (e % Q38_VC_ENGINES) / f2.pv;
#if F2_AV_PACK
    const uint32_t q = e % f2.pv;
    const uint32_t panel = (n * Q38_VC_COUNT + v) * f2.pv + q;
#else
    const uint32_t panel = panel0 + e % f2.pv;
#endif
    if (panel * NT >= f2.n_dn)
      continue;
#if F2_AV_PACK
    __fp16 *local =
        output ? f2_down_z(n, e) : f2.residual + t * f2.tile_rows * NT;
    __fp16 *remote = (output ? f2.dn_out : (__fp16 *)f2.res_src) +
                     (f2.first + r * f2.tile_rows) * f2.n_dn + panel * NT;
#else
    __fp16 *local = local0 + t * f2.tile_rows * NT;
    __fp16 *remote =
        remote0 + (f2.first + r * f2.tile_rows) * f2.n_dn + panel * NT;
#endif
    const uint32_t packed = NT * sizeof(__fp16);
#if Q38_L2_TILED
#if !F2_AV_PACK
    // A stripe's rows as at most two runs that stay inside their L2 rows (cut
    // where a row wraps to the next L2 row's start), planes over the stripes
    // when a stripe's rows span whole L2 rows.
    const uint32_t row_pitch = f2.n_dn * 2u;
    if (inner * row_pitch % VC_GLOBAL_ROW_BYTES == 0u) {
      const uint32_t q = (uint32_t)remote % VC_GLOBAL_ROW_BYTES;
      const uint32_t b = row_pitch % VC_GLOBAL_ROW_BYTES;
      uint32_t k = 1u;
      while (k < inner && q + k * b + packed <= VC_GLOBAL_ROW_BYTES)
        ++k;
      const uint32_t q2 = (q + k * b) % VC_GLOBAL_ROW_BYTES;
      if (k == inner ||
          q2 + (inner - k - 1u) * b + packed <= VC_GLOBAL_ROW_BYTES) {
        char *const l1 = (char *)Q38_VC_GLOBAL(local);
        for (uint32_t j = 0u; j < inner; j += k, k = inner - k) {
          char *const a = (char *)remote + j * row_pitch;
          char *const c = l1 + j * packed;
          const uint32_t l2_plane = inner * row_pitch - (k - 1u) * row_pitch;
          const uint32_t l1_plane = VC_GLOBAL_ROW_BYTES - (k - 1u) * packed;
          pending = dma_local_memcpy_3d_nonblocking(
              0u, output ? a : c, output ? c : a, packed,
              output ? row_pitch : packed, output ? packed : row_pitch, k,
              output ? l2_plane : l1_plane, output ? l1_plane : l2_plane,
              f2.tile_rows / inner);
        }
        continue;
      }
    }
    // Otherwise: L2 rows repeat their offset in an 8 KiB L2 row every `step`
    // rows: one descriptor per class of a stripe's rows.
#else
    // L2 rows repeat their offset in an 8 KiB L2 row every `step` rows: one
    // descriptor per class of a stripe's rows, planes over the stripes.
#endif
    const uint32_t pitch = f2.n_dn * 2u, low = pitch & (0u - pitch);
    const uint32_t step =
        low < VC_GLOBAL_ROW_BYTES ? VC_GLOBAL_ROW_BYTES / low : 1u;
    if (inner % step == 0u) {
      const uint32_t per = inner / step;
      const uint32_t l2_plane = inner * pitch - (per - 1u) * step * pitch;
      const uint32_t l1_plane =
          VC_GLOBAL_ROW_BYTES - (per - 1u) * step * packed;
      char *const l1 = (char *)Q38_VC_GLOBAL(local);
      for (uint32_t j = 0u; j < step; ++j) {
        char *const a = (char *)remote + j * pitch, *const b = l1 + j * packed;
        pending = dma_local_memcpy_3d_nonblocking(
            0u, output ? a : b, output ? b : a, packed,
            output ? step * pitch : step * packed,
            output ? step * packed : step * pitch, per,
            output ? l2_plane : l1_plane, output ? l1_plane : l2_plane,
            f2.tile_rows / inner);
      }
      continue;
    }
#endif
    const uint32_t global = VC_GLOBAL_ROW_BYTES - (inner - 1u) * packed;
    pending = dma_local_memcpy_3d_nonblocking(
        0u, output ? (void *)remote : Q38_VC_GLOBAL(local),
        output ? Q38_VC_GLOBAL(local) : (void *)remote, packed,
        output ? f2.n_dn * 2u : packed, output ? packed : f2.n_dn * 2u, inner,
        output ? f2.n_dn * 2u : global, output ? global : f2.n_dn * 2u,
        f2.tile_rows / inner);
  }
  return pending;
}

static NOSTACK void f2_down_progress(void) {
  if (!f2.down || f2.slice + 1u != f2.slices)
    return;
  const uint32_t g = mempool_get_group_id();
  if (f2_down[g].res_pending && dma_local_done_id(0u, f2_down[g].res_pending)) {
    f2_down[g].res_pending = 0u;
    __atomic_store_n(&f2_down[g].res_ready, f2_down[g].res_issued,
                     __ATOMIC_RELEASE);
  }
  if (f2_down[g].out_pending && dma_local_done_id(0u, f2_down[g].out_pending))
    f2_down[g].out_pending = 0u;
  uint32_t done = f2.steps;
  for (uint32_t t = 0u; t < NUM_TILES_PER_GROUP; ++t)
    done = Q38_MIN(done, f2_min_res(g * NUM_TILES_PER_GROUP + t));
  if (!f2_down[g].out_pending && f2_down[g].out_issued < done) {
    f2_down[g].out_pending = f2_down_transfer(f2_down[g].out_issued, 1u);
    ++f2_down[g].out_issued;
  }
  if (!f2_down[g].res_pending && done >= f2_down[g].res_issued &&
      f2_down[g].res_issued < f2.steps) {
    f2_down[g].res_pending = f2_down_transfer(f2_down[g].res_issued, 0u);
    ++f2_down[g].res_issued;
    if (!f2_down[g].res_pending)
      __atomic_store_n(&f2_down[g].res_ready, f2_down[g].res_issued,
                       __ATOMIC_RELEASE);
  }
}

static void f2_progress_dma(void) {
  if (f2.down)
    f2_down_progress();
  else
    f2_ship_progress();
}

#if Q38_FFN_WATCHDOG
static NOSTACK void f2_watchdog(uint32_t *n, uint32_t where) {
  if (++*n != Q38_FFN_WATCHDOG)
    return;
  const uint32_t g = mempool_get_group_id(), core = mempool_get_core_id();
  const uint32_t v = g * NUM_TILES_PER_GROUP / Q38_VC_ENGINES;
  mempool_wait(1u + g * 20000u + core % NUM_CORES_PER_TILE * 1000u);
  printf("[f2_wd] t=%u g=%u core=%u where=%u down=%u slice=%u epoch=%u "
         "res=%u/%u out=%u pend=%u/%u x=%u wv=%u\n",
         mempool_get_timer(), g, core, where, f2.down, f2.slice,
         f2.weight_epoch, f2_down[g].res_issued, f2_down[g].res_ready,
         f2_down[g].out_issued, f2_down[g].res_pending, f2_down[g].out_pending,
         f2_down[g].x_pending, f2_down[g].weight_valid);
  for (uint32_t i = 0u; i < Q38_VC_ENGINES; ++i) {
    const uint32_t e = v * Q38_VC_ENGINES + i;
    printf("[f2_wd] g=%u e=%u step=%u down=%u res=%u\n", g, e,
           f2_progress[e].step_done, f2_progress[e].down_done, f2_min_res(e));
  }
  mempool_wait(1u + (NUM_GROUPS - g) * 20000u);
  q38_fatal("[qwen38_fatal] f2 watchdog where=", where);
}
#define F2_WATCHDOG(n, where) f2_watchdog(&(n), (where))
#else
#define F2_WATCHDOG(n, where) ((void)(n))
#endif

#if F2_AV_PACK
static uint32_t f2_vc_step_done(uint32_t v);
#endif
static NOSTACK void f2_agent_tail(uint32_t issuer);

static NOSTACK void f2_agent(void) {
  const uint32_t engine = mempool_get_redmule_id();
  const uint32_t v = engine / Q38_VC_ENGINES, q = engine % f2.pv;
  const uint32_t issuer = mempool_is_group_coordinator();
  vc_subscribe(FSYNC_EVT_FSYNC | FSYNC_EVT_ENGINE |
               (issuer ? FSYNC_EVT_DMA : 0u));
  dma_id_t pending = issuer ? f2_first_weight(v) : 0u;
  while (pending && !dma_local_done_id(0u, pending))
    vc_sleep(1u, 0u);
  vc_barrier();
  uint8_t next = q < f2_panels(v, 0u) ? f2_configure(0u, engine) : 0u;
  const uint32_t lag = f2.slots == 3u;
  uint32_t load = issuer && !lag && f2.steps > 1u ? 1u : ~0u;
  for (uint32_t t = 0u; t < f2.steps; ++t) {
    const uint32_t n = f2_n(t);
    const uint32_t mine = q < f2_panels(v, t);
    const uint8_t target = next;
    while (!f2.down && n >= 2u && f2_min_act(engine) < n - 1u) {
      if (issuer)
        f2_progress_dma();
      mempool_wait(32u);
    }
    if (lag && t)
      vc_barrier_wait(); // weights of step t landed; step t-2 done
    if (mine)
      tnc_redmule_trigger();
    if (lag && t + 1u < f2.steps) {
      // Step t+1's slot held step t-2. The issuer's load must land before
      // it arrives; the others arrive at once and wait before step t+1.
      if (issuer) {
        pending = f2_load(v, t + 1u);
        while (pending && !dma_local_done_id(0u, pending)) {
          f2_progress_dma();
          vc_sleep(4u, t);
        }
        pending = 0u;
      }
      vc_barrier_arrive();
    }
    if (load != ~0u) {
      pending = f2_load(v, load);
      load = ~0u;
    }
    if (Q38_FFN_X_PREFETCH && issuer && f2.down && t == 0u)
      f2_prefetch_x();
    if (t + 1u < f2.steps && q < f2_panels(v, t + 1u))
      next = f2_configure(t + 1u, engine);
    if (Q38_FFN_NEXT_WEIGHT_PREFETCH && issuer && f2.down && t + 1u == f2.steps)
      f2_prefetch_weight(v);
    while (mine && !tnc_redmule_done(target)) {
      if (issuer)
        f2_progress_dma();
      vc_sleep(3u, t);
    }
    if (f2.down) {
      __atomic_store_n(&f2_progress[engine].down_done, n + 1u,
                       __ATOMIC_RELEASE);
      f2_wake();
    } else if (!f2.pair || (t & 1u)) {
      __atomic_store_n(&f2_progress[engine].pair_done, n + 1u,
                       __ATOMIC_RELEASE);
      f2_wake();
    }
    if (issuer)
      f2_progress_dma();
    if (t + 1u == f2.steps || lag)
      continue;
    while (pending && !dma_local_done_id(0u, pending)) {
      if (issuer)
        f2_progress_dma();
      vc_sleep(4u, t);
    }
    vc_barrier();
    pending = 0u;
    if (issuer && t + 2u < f2.steps)
      load = t + 2u;
  }
  f2_agent_tail(issuer);
}

// The issuer drains the pass's ships, outputs and prefetches.
static NOSTACK void f2_agent_tail(uint32_t issuer) {
#if F2_AV_PACK
  if (f2.packed_out && f2.slice == 0u && issuer) {
    // X1 aliases the upper half of slot 1, including its step-9 large-K
    // weights. Every engine of this VC must finish before X may overwrite it.
    const uint32_t v = mempool_get_redmule_id() / Q38_VC_ENGINES;
    while (f2_vc_step_done(v) < f2.q_base + f2.steps)
      mempool_wait(32u);
    f2_prefetch_x();
  }
#endif
  if (issuer && !f2.down) {
    const uint32_t g = mempool_get_group_id();
    while (__atomic_load_n(&f2_ship[g].released, __ATOMIC_ACQUIRE) <
           f2_groups()) {
      f2_progress_dma();
      mempool_wait(32u);
    }
  }
  uint32_t wd = 0u;
  if (issuer && f2.down && f2.slice + 1u == f2.slices) {
    const uint32_t g = mempool_get_group_id();
    while (f2_down[g].out_issued < f2.steps || f2_down[g].out_pending) {
      f2_down_progress();
      mempool_wait(32u);
      F2_WATCHDOG(wd, 1u);
    }
  }
  if (issuer && f2.down && f2_down[mempool_get_group_id()].weight_valid) {
    const dma_id_t id = f2_down[mempool_get_group_id()].weight_pending;
    while (id && !dma_local_done_id(0u, id)) {
      f2_down_progress();
      vc_sleep(6u, f2.weight_epoch);
      F2_WATCHDOG(wd, 4u);
    }
  }
  if (Q38_FFN_X_PREFETCH && issuer && f2.down) {
    const uint32_t g = mempool_get_group_id();
    const dma_id_t id = f2_down[g].x_pending;
    while (id && !dma_local_done_id(0u, id)) {
      vc_sleep(6u, f2.weight_epoch);
      F2_WATCHDOG(wd, 8u);
    }
    f2_down[g].x_pending = 0u;
  }
  fsync_subscribe_self(0u);
}

#if Q38_FFN_QUEUE
// Steps every engine of VC v has finished.
static __attribute__((noinline)) uint32_t f2_vc_step_done(uint32_t v) {
  uint32_t m = ~0u;
#if Q38_KIND != 0
#pragma clang loop unroll(disable)
#endif
  for (uint32_t i = 0u; i < Q38_VC_ENGINES; ++i) {
    const uint32_t x = __atomic_load_n(
        &f2_progress[v * Q38_VC_ENGINES + i].step_done, __ATOMIC_ACQUIRE);
    m = Q38_MIN(m, x); // once: see f2_min_act
  }
  return m;
}

// An issuer's weight loads: step k goes to slot k % slots once every engine
// of the VC has finished step k - slots, the slot's previous tenant.
typedef struct {
  dma_id_t pend[3];
  uint32_t issued, landed, next_epoch, base;
} f2_q_t;

static void f2_q_retire(f2_q_t *s) {
  while (s->landed < s->issued &&
         (!s->pend[s->landed % f2.slots] ||
          dma_local_done_id(0u, s->pend[s->landed % f2.slots])))
    ++s->landed;
}

// Retire landed loads and issue the next ones while their slots are free.
// Returns 1 while a load waits on other engines (not an event: poll).
static NOSTACK uint32_t f2_q_service(f2_q_t *s, uint32_t v) {
  f2_progress_dma();
  f2_q_retire(s);
  for (;;) {
    const uint32_t k = s->issued;
    if (k == f2.steps &&
        (s->next_epoch || !f2.down || !Q38_FFN_NEXT_WEIGHT_PREFETCH))
      return 0u;
    if (k >= f2.slots) {
      uint32_t required = k - f2.slots + 1u;
#if F2_AV_PACK
      if (f2.packed_out) {
        // Each epoch starts in slot 0. The large-K tail reuses 0/1 at
        // distance two. Next epoch's slot 0 last held step 8 (large K) or
        // step 9 (small K); do not evict it until every reader has finished.
        if (k == f2.steps)
          required = f2.slice == 0u ? 9u : 10u;
        else if (f2.slice == 0u && k >= 8u)
          required = k - 1u;
      }
#endif
      const uint32_t done = f2_vc_step_done(v);
      if (done < s->base + required)
        return 1u;
    }
    if (k == f2.steps) { // the next K slice's first weights
      s->next_epoch = 1u;
      f2_prefetch_weight(v);
      return 0u;
    }
    s->pend[k % f2.slots] = f2_load(v, k);
    s->issued = k + 1u;
    if (Q38_FFN_X_PREFETCH && f2.down && s->issued == Q38_MIN(2u, f2.steps) &&
        !(F2_PACKED_OUT && f2.slice == 0u))
      f2_prefetch_x();
  }
}

#if Q38_FFN_W_CHECK
// Mismatches among 4 sampled halfwords of engine q's panel of step t.
static uint32_t f2_w_check(uint32_t v, uint32_t q, uint32_t t) {
  const uint32_t k = f2.down ? F2_K(f2.slice) : f2.k_up, bytes = k * NT * 2u;
  const uint32_t first = f2_n(t) * f2.pv * Q38_VC_COUNT + v * f2.pv + q;
  const char *const l1 = (const char *)(VC_WINDOW_BASE_1 + f2.w_offset +
                                        f2_weight_slot(t) * f2.slot_bytes) +
                         q * bytes;
  const char *const l2 = (const char *)f2_panel(first, t, f2.slice);
  // Step t + slots, the slot's next tenant: a bad sample that matches it
  // means its load landed early (bit 8 per such sample in the result).
  const uint32_t tn = t + f2.slots;
  const char *const l2n =
      tn < f2.steps
          ? (const char *)f2_panel(
                f2_n(tn) * f2.pv * Q38_VC_COUNT + v * f2.pv + q, tn, f2.slice)
          : 0;
  uint32_t bad = 0u;
  for (uint32_t i = 0u; i < 4u; ++i) {
    const uint32_t o = (i * 7919u * 2u + t * 64u) % bytes & ~1u;
    const uint16_t x = *(const volatile uint16_t *)(l1 + o);
    if (x != *(const volatile uint16_t *)(l2 + o))
      bad += l2n && x == *(const volatile uint16_t *)(l2n + o) ? 0x101u : 1u;
  }
  return bad;
}
// Per-engine check state, in L1: the agent's 512 B stack has no room for it.
static struct {
  uint32_t pre, post, n, first, vcmin;
} f2_wck[NUM_REDMULE_TILES] __attribute__((section(".l1")));
#define w_pre f2_wck[engine].pre
#define w_post f2_wck[engine].post
#define w_n f2_wck[engine].n
#define w_first f2_wck[engine].first
#define w_vcmin f2_wck[engine].vcmin
// Before a trigger; w_first keeps the first bad step (bit 16: after its job).
#define F2_W_PRE(k)                                                            \
  do {                                                                         \
    const uint32_t bad_ = f2_w_check(v, q, (k));                               \
    w_pre += bad_;                                                             \
    if (bad_ && w_first == ~0u)                                                \
      w_first = (k);                                                           \
  } while (0)
#endif

// DMA the workers drive (ships; a last K slice's residuals and outputs): their
// progress is no event for the issuer, which polls while any is left.
static uint32_t f2_q_busy(void) {
  const uint32_t g = mempool_get_group_id();
  if (!f2.down)
    return f2_ship[g].released < f2_groups();
  return f2.slice + 1u == f2.slices && f2_down[g].out_issued < f2.steps;
}

#if Q38_KIND != 0
static NOSTACK void f2_q_idle(f2_q_t *s, uint32_t v, uint32_t issuer,
                              uint32_t tag, uint32_t step);
#endif
static void f2_q_idle(f2_q_t *s, uint32_t v, uint32_t issuer, uint32_t tag,
                      uint32_t step) {
  if (issuer && (f2_q_service(s, v) | f2_q_busy()))
    mempool_wait(32u);
  else
    vc_sleep(tag, step);
}

// VC barrier phase k releases step k: issuers arrive once their part of its
// weights has landed, the other agents at once. No load is programmed between
// the arrival and the trigger: a submit stalls while the DMA request FIFO is
// full (dma_collective.h), and the VC, then this engine, would wait for it.
static NOSTACK void f2_q_phase(f2_q_t *s, uint32_t v, uint32_t issuer,
                               uint32_t k) {
  uint32_t wd = 0u;
  if (issuer)
    f2_q_retire(s);
  while (issuer && s->landed <= k) {
    f2_q_idle(s, v, issuer, 4u, k);
    F2_WATCHDOG(wd, 6u);
  }
#if F2_AV_PACK
  if (f2.packed_out && f2.slice == 0u && k == 6u && issuer)
    while (f2_vc_step_done(v) < s->base + 6u)
      mempool_wait(32u);
#endif
  vc_barrier_arrive();
  while (fsync_status() & FSYNC_STATUS_BUSY_MASK)
    vc_sleep(2u, k);
}

static NOSTACK void f2_agent_q(void) {
  const uint32_t engine = mempool_get_redmule_id();
  const uint32_t v = engine / Q38_VC_ENGINES, q = engine % f2.pv;
  const uint32_t issuer = mempool_is_group_coordinator();
  vc_subscribe(FSYNC_EVT_FSYNC | FSYNC_EVT_ENGINE |
               (issuer ? FSYNC_EVT_DMA : 0u));
  const uint32_t base = f2.q_base;
  f2_q_t s = {{0u, 0u, 0u}, 0u, 0u, 0u, base};
  if (issuer) {
    s.pend[0] = f2_first_weight(v);
    s.issued = 1u;
    if (Q38_FFN_X_PREFETCH && f2.down && f2.steps == 1u)
      f2_prefetch_x();
  }
  uint8_t target[2] = {0u, 0u}; // steps t and t + 1, by t & 1
#if Q38_FFN_W_CHECK
  w_pre = w_post = w_n = 0u;
  w_first = w_vcmin = ~0u;
#endif
  for (uint32_t k = 0u; k < Q38_MIN(2u, f2.steps); ++k) {
    f2_q_phase(&s, v, issuer, k);
    if (q < f2_panels(v, k)) {
#if Q38_FFN_W_CHECK
      F2_W_PRE(k);
#endif
      target[k & 1u] = f2_configure(k, engine);
      tnc_redmule_trigger();
    }
    if (k == 0u)
      F2_STAMP(5u);
  }
  for (uint32_t t = 0u; t < f2.steps; ++t) {
    const uint32_t n = f2_n(t);
    if (q < f2_panels(v, t)) {
      while (!tnc_redmule_done(target[t & 1u]))
        f2_q_idle(&s, v, issuer, 3u, t);
#if Q38_FFN_W_CHECK
      const uint32_t bad = f2_w_check(v, q, t);
      w_post += bad;
      if (bad && w_first == ~0u) {
        w_first = t | 1u << 16;
        w_vcmin = f2_vc_step_done(v) - base; // the VC's finished steps then
      }
      ++w_n;
#endif
    }
    __atomic_store_n(&f2_progress[engine].step_done, base + t + 1u,
                     __ATOMIC_RELEASE);
    if (f2.down) {
      __atomic_store_n(&f2_progress[engine].down_done, n + 1u,
                       __ATOMIC_RELEASE);
      f2_wake();
    } else if (!f2.pair || (t & 1u)) {
      __atomic_store_n(&f2_progress[engine].pair_done, n + 1u,
                       __ATOMIC_RELEASE);
      f2_wake();
    }
    const uint32_t k = t + 2u;
    if (k >= f2.steps)
      continue;
#if F2_AV_PACK
    if (f2.packed_out && f2.slice == 0u) {
      if (k == 6u)
        continue; // retire step 5 before the slot-2 reuse rendezvous
      if (k == 7u) {
        f2_q_phase(&s, v, issuer, 6u);
        if (q < f2_panels(v, 6u)) {
          target[0] = f2_configure(6u, engine);
          tnc_redmule_trigger();
        }
      }
    }
#endif
    // Step k's Z tile held step k - 2's (same pair slot): activated yet?
    const uint32_t nk = f2_n(k);
    while (!f2.down && nk >= 2u && f2_min_act(engine) < nk - 1u) {
      if (issuer)
        f2_q_service(&s, v);
      mempool_wait(32u);
    }
    f2_q_phase(&s, v, issuer, k);
    if (q < f2_panels(v, k)) {
#if Q38_FFN_W_CHECK
      F2_W_PRE(k);
#endif
      target[k & 1u] = f2_configure(k, engine);
      tnc_redmule_trigger();
    }
  }
#if Q38_FFN_W_CHECK
  // Engines print in turn: concurrent printf interleaves by character.
  mempool_wait(1u + engine * 3000u);
  if (engine == 0u || w_pre || w_post)
    printf("[f2_wcheck] engine=%u down=%u epoch=%u steps=%u pre=%x post=%x "
           "first=%x vcmin=%x\n",
           engine, f2.down, f2.weight_epoch, w_n, w_pre, w_post, w_first,
           w_vcmin);
#endif
#if Q38_FFN_W_CHECK
#undef w_pre
#undef w_post
#undef w_n
#undef w_first
#undef w_vcmin
#endif
  F2_STAMP(1u);
  uint32_t wd = 0u;
  while (issuer && f2_q_service(&s, v)) {
    mempool_wait(32u);
    F2_WATCHDOG(wd, 2u);
  }
  f2_agent_tail(issuer);
  F2_STAMP(2u);
}
#endif

static NOSTACK void f2_worker(void) {
  const uint32_t g = mempool_get_group_id();
  const uint32_t e =
      g * NUM_TILES_PER_GROUP + mempool_get_tile_id() % NUM_TILES_PER_GROUP;
  const uint32_t v = e / Q38_VC_ENGINES, ev = e % Q38_VC_ENGINES;
  const uint32_t q = ev % f2.pv;
  const __fp16 *z = f2.z + F2_ZE(e) * 4u * F2_Z_ELEMS;
  const uint32_t w = mempool_get_core_id() % NUM_CORES_PER_TILE - 1u;
  if (f2.down && f2.slice + 1u != f2.slices) {
    fsync_subscribe_self(0u);
    return;
  }
  vc_subscribe(Q38_FFN_EVT ? FSYNC_EVT_CORE : FSYNC_EVT_ENGINE);
  const uint32_t count = f2.down ? f2.steps : f2_groups();
  for (uint32_t n = 0u; n < count; ++n) {
    const uint32_t panel = (n * Q38_VC_COUNT + v) * f2.pv + q;
    const uint32_t live = panel * NT < (f2.down ? f2.n_dn : f2.n_up);
    if (live && f2.down) {
      f2_worker_wait(&f2_progress[e].down_done, n + 1u);
      uint32_t wd = 0u;
      while (__atomic_load_n(&f2_down[g].res_ready, __ATOMIC_ACQUIRE) <
             n + 1u) {
        mempool_wait(32u);
        F2_WATCHDOG(wd, 3u);
      }
#if F2_AV_PACK
      __fp16 *tile = f2_down_z(n, e);
#else
      __fp16 *tile =
          f2.z + (n * NUM_TILES_PER_GROUP + e % NUM_TILES_PER_GROUP) *
                     f2.tile_rows * NT;
#endif
      const __fp16 *res =
          f2.residual + e % NUM_TILES_PER_GROUP * f2.tile_rows * NT;
      for (uint32_t i = w; i < f2.tile_rows; i += Q38_FFN_WORKERS)
        llm_residual_add_f16(Q38_VC_GLOBAL(tile + i * NT),
                             Q38_VC_GLOBAL(res + i * NT), NT, 0u, 1u);
      __sync_synchronize();
      __atomic_store_n(&f2_progress[e].res[w], n + 1u, __ATOMIC_RELEASE);
    } else if (!f2.down) {
      if (live) {
        f2_worker_wait(&f2_progress[e].pair_done, n + 1u);
        while (n >= 2u &&
               __atomic_load_n(&f2_ship[g].released, __ATOMIC_ACQUIRE) < n - 1u)
          mempool_wait(32u);
        const __fp16 *gate = z + (n & 1u) * F2_Z_ELEMS;
        const __fp16 *up = z + (2u + (n & 1u)) * F2_Z_ELEMS;
        __fp16 *dest =
            f2.ship + (n & 1u) * f2.ship_elems +
            (e % NUM_TILES_PER_GROUP / f2.pv * f2.tile_rows * f2.ship_columns +
             q * NT);
        for (uint32_t i = w; i < f2.tile_rows; i += Q38_FFN_WORKERS)
          if (f2.pair)
            q38_activation16(F2_ZG(gate + i * NT), F2_ZG(up + i * NT),
                             Q38_VC_GLOBAL(dest + i * f2.ship_columns), NT, 1);
          else
            q38_copy16(Q38_VC_GLOBAL(dest + i * f2.ship_columns),
                       F2_ZG(gate + i * NT), NT);
        __sync_synchronize();
      }
      __atomic_store_n(&f2_progress[e].act[w], n + 1u, __ATOMIC_RELEASE);
    } else {
      __atomic_store_n(&f2_progress[e].res[w], n + 1u, __ATOMIC_RELEASE);
    }
  }
  fsync_subscribe_self(0u);
}

static NOSTACK void f2_execute(void) {
  if (tnc_is_tile_agent())
#if Q38_FFN_QUEUE
    f2_agent_q();
#else
    f2_agent();
#endif
  else if (mempool_get_core_id() % NUM_CORES_PER_TILE <= Q38_FFN_WORKERS)
    f2_worker();
  else
    fsync_subscribe_self(0u);
#if Q38_FFN_EVT
  // The epoch-checked barrier tolerates a stale VC response or core event.
  f2_sync_all();
  F2_STAMP(3u);
#else
  // Workers may finish before their agent's last VC barrier. Keep them
  // unsubscribed until this legacy join; the shim broadcasts to tile mates.
  sync_all();
  fsync_subscribe_self(FSYNC_EVT_FSYNC);
#endif
  // Read by the agents only after the next pass's f2_sync_all.
  if (mempool_get_core_id() == 0u)
    f2.q_base += f2.steps;
}

// Keep collective DMA's temporary descriptor out of the phase driver's
// frame; LLVM otherwise unrolls the small toy loops beyond a 512-byte stack.
static NOSTACK void f2_transfer(void *dest, const void *source,
                                uint32_t bytes) {
  transfer(dest, source, bytes);
}

// With f2.early (an up pass), the issuers load step 0's weights under the
// RMSNorm; norm_rows_sliced calls this after its gamma rows have landed.
static NOSTACK void f2_early_weight(void) {
  if (!f2.early || !mempool_is_group_coordinator())
    return;
  const uint32_t g = mempool_get_group_id();
  f2_down[g].weight_pending =
      f2_load(mempool_get_redmule_id() / Q38_VC_ENGINES, 0u);
  f2_down[g].weight_valid = 1u;
}

#if Q38_F2_NORM_HOME
// An up-pass batch loop that opts in, before its batch's first f2_sync_all:
// the gamma buffer for f2_home_row and, at the loop's first batch, cleared
// flags.
static void f2_home_setup(__fp16 *gamma_buf, uint32_t first, uint32_t w0) {
  if (mempool_get_core_id() == 0u)
    f2h.gamma = gamma_buf;
  if (first == w0 && mempool_is_group_coordinator())
    f2_home[mempool_get_group_id()].ready = 0u;
}

// After that f2_sync_all, in place of f2_transfer + norm_rows_sliced: the
// batch's RMSNorm input from the rows at `src` (L2), each group's share
// normed by the group's own workers.
static NOSTACK void f2_norm_batch(const __fp16 *src, const __fp16 *scale,
                                  __fp16 *gamma_buf, uint32_t first,
                                  uint32_t w0) {
  const uint32_t b = (first - w0) / Q38_FFN_ROWS;
  const uint32_t g = mempool_get_group_id();
  const uint32_t lane = mempool_get_core_id() % NUM_CORES_PER_TILE;
  transfer(gamma_buf, scale, Q38_D * 2);
  if (mempool_is_group_coordinator()) {
    const dma_id_t id = f2_home_issue(g, src + first * Q38_D);
    f2_early_weight();
    while (!dma_local_done_id(0u, id))
      mempool_wait(16u);
    __atomic_store_n(&f2_home[g].ready, b + 1u, __ATOMIC_RELEASE);
  } else if (lane >= 1u && lane <= Q38_FFN_WORKERS) {
    while (__atomic_load_n(&f2_home[g].ready, __ATOMIC_ACQUIRE) < b + 1u)
      mempool_wait(16u);
    for (uint32_t k =
             mempool_get_tile_id() % NUM_TILES_PER_GROUP * Q38_FFN_WORKERS +
             lane - 1u;
         k < F2_HOME_ROWS; k += F2_HOME_WORKERS)
      f2_home_row(g, k, f2.input);
    __sync_synchronize();
  }
  f2_sync_all();
}
#endif

#if Q38_FFN_X_CHECK
static uint16_t f2_x_tag(const void *a) {
  return (uint16_t)(((uint32_t)a >> 1) * 40503u >> 7);
}

// Core 0, before the first down pass: tag every slice of rows [start, end).
static NOSTACK void f2_x_seed(uint32_t start) {
  if (mempool_get_core_id() != 0u)
    return;
  for (uint32_t first = start; first < f2.end; first += f2.rows_dn)
    for (uint32_t slice = 0u; slice < f2.slices; ++slice) {
      volatile uint16_t *const src =
          (volatile uint16_t *)(f2.dn_in +
                                (slice * f2.dn_rows + first) * f2.k_dn);
      for (uint32_t i = 0u; i < f2.rows_dn * f2.k_dn; i += F2_X_CHECK_STRIDE)
        src[i] = f2_x_tag((const void *)&src[i]);
    }
}
#endif

// A down slice's X: in place already when the previous slice prefetched it.
static NOSTACK void f2_down_input(void) {
#if Q38_L2_TILED
  const __fp16 *const src =
      f2.dn_in + f2.slice * f2.dn_rows * f2.k_dn + f2.first * F2_K(f2.slice);
#else
  const __fp16 *const src =
      f2.dn_in + (f2.slice * f2.dn_rows + f2.first) * f2.k_dn;
#endif
  F2_STAMP(4u);
#if G2_O_PADDED
  _Static_assert(!Q38_FFN_X_CHECK, "The X check assumes packed rows");
  if (f2.dn_pitch) {
    if (!Q38_FFN_X_PREFETCH || !f2.weight_epoch) {
      if (mempool_is_group_coordinator()) {
        const dma_id_t id = f2_x_gather(
            f2.input, f2_x_src(f2.slice, f2.first), f2.rows,
            F2_K(f2.slice) * sizeof(__fp16), f2.dn_pitch * sizeof(__fp16));
        while (id && !dma_local_done_id(0u, id))
          ;
      }
      f2_sync_all(); // every group's stripes have landed
    }
    return;
  }
#endif
  if (!Q38_FFN_X_PREFETCH || !f2.weight_epoch)
    f2_transfer(f2.input, src, f2.rows * F2_K(f2.slice) * sizeof(__fp16));
#if Q38_FFN_X_CHECK
  if (mempool_get_core_id() == 0u) {
    uint32_t bad = 0u, n = 0u;
    for (uint32_t i = 0u; i < f2.rows * f2.k_dn; i += F2_X_CHECK_STRIDE, ++n)
      bad += ((const volatile uint16_t *)f2.input)[i] != f2_x_tag(&src[i]);
    printf("[f2_xcheck] epoch=%u buf=%u checked=%u bad=%u\n", f2.weight_epoch,
           f2.input == f2.x_buf[1], n, bad);
  }
#endif
}

#if Q38_X8_BATCH
// Rows [w0, w1) only: one batch of the batch-major layer (layer_resident.c).
static NOSTACK void ffn_rows(uint32_t w0, uint32_t w1) {
#else
static NOSTACK void ffn(void) {
  const uint32_t w0 = Q38_EXEC_START, w1 = Q38_EXEC_END;
#endif
  const uint32_t core = mempool_get_core_id();
  if (core == 0u) {
    f2.peak = 0u;
    f2.pair = 1u, f2.k_up = Q38_D, f2.n_up = Q38_I, f2.rows_up = Q38_FFN_ROWS;
    f2.seg_w[0] = w_gate, f2.seg_w[1] = w_up, f2.nseg = 2u;
    f2.g_out = F2_G_L2, f2.g_kc = F2_KC, f2.g_rows = Q38_R;
    f2.k_dn = F2_KC, f2.k_full = Q38_I, f2.n_dn = Q38_D;
    f2.rows_dn = Q38_FFN_DN_ROWS, f2.slices = F2_DN_SLICES;
#if Q38_L2_TILED
    f2.k_last = 0u, f2.g_main = ~0u; // even slices, one pitch
#endif
    f2.dn_lag = 0u;
    f2.end = w1, f2.w_dn = w_down, f2.dn_in = F2_G_L2;
#if G2_O_PADDED
    f2.dn_pitch = 0u;
#endif
    f2.dn_rows = Q38_R, f2.res_src = X8_MIX, f2.dn_out = X8_OUT;
    f2.q_base = 0u;
    // f2 sits in remote L1 and f2_layout reloads it at once on this core:
    // a fence first (GVSoC L1 reload-after-store).
    __sync_synchronize();
  }
  if (tnc_is_tile_agent())
    f2_progress[mempool_get_redmule_id()].step_done = 0u;
  if (mempool_is_group_coordinator()) {
    const uint32_t g = mempool_get_group_id();
    f2_down[g].weight_valid = f2_down[g].weight_pending = 0u;
    f2_down[g].x_pending = 0u;
  }
#if Q38_FFN_X_CHECK
  f2_x_seed(w0);
#endif
#pragma clang loop unroll(disable)
  for (uint32_t first = w0; !Q38_FFN_DEBUG_DOWN_ONLY && first < w1;
       first += Q38_FFN_ROWS) {
    f2_layout(0u);
    f2_reset_progress();
#if Q38_F2_NORM_HOME
    f2_home_setup(gamma, first, w0);
#endif
    if (core == 0u)
      f2.first = first;
    f2_sync_all();
#if Q38_F2_NORM_HOME
    f2_norm_batch(X8_MIX, w_ffn_norm, gamma, first, w0);
#else
    f2_transfer(f2.residual, X8_MIX + first * Q38_D,
                Q38_FFN_ROWS * Q38_D * sizeof(__fp16));
    norm_rows_sliced(f2.residual, w_ffn_norm, f2.input, Q38_FFN_ROWS, gamma);
#endif
    f2_execute();
  }
#pragma clang loop unroll(disable)
  for (uint32_t first = w0; !Q38_X8_FFN_UP_ONLY && first < w1;
       first += Q38_FFN_DN_ROWS) {
    f2_layout(1u);
    f2_reset_progress();
    if (core == 0u)
      f2.first = first;
    f2_sync_all();
#pragma clang loop unroll(disable)
    for (uint32_t slice = 0u; slice < F2_DN_SLICES; ++slice) {
      if (core == 0u) {
        f2.slice = slice;
        f2.weight_epoch = (first - w0) / Q38_FFN_DN_ROWS * F2_DN_SLICES + slice;
        f2.input = f2.x_buf[f2.weight_epoch & 1u];
      }
      f2_slice_reset();
      f2_sync_all();
      f2_down_input();
      f2_execute();
    }
  }
}

#if Q38_F2_C1
// One timed pass of kind Q38_F2_C1 over the first rows, weights resident.
static NOSTACK void f2_c1(void) {
  const uint32_t core = mempool_get_core_id();
  const uint32_t down = Q38_F2_C1 >= 4;
  if (core == 0u) {
    const uint32_t rows = down ? Q38_FFN_DN_ROWS : Q38_FFN_ROWS;
    const uint32_t tile = down ? Q38_FFN_DN_TILE : Q38_FFN_GU_ROWS;
    const uint32_t pv = Q38_VC_ENGINES / (rows / tile);
    const uint32_t groups = Q38_F2_C1 == 2   ? 1u
                            : Q38_F2_C1 == 3 ? Q38_F2_C1_STEPS / 2u
                                             : Q38_F2_C1_STEPS;
    const uint32_t cols = groups * pv * Q38_VC_COUNT * NT;
    f2.peak = 0u, f2.q_base = 0u, f2.first = 0u;
    f2.pair = Q38_F2_C1 == 3;
    f2.k_up = Q38_D;
    f2.n_up = cols, f2.rows_up = Q38_FFN_ROWS;
    f2.seg_w[0] = w_gate, f2.seg_w[1] = w_up, f2.nseg = f2.pair ? 2u : 1u;
    f2.seg_panels[0] = cols / NT;
    f2.g_out = F2_G_L2, f2.g_kc = cols, f2.g_rows = Q38_FFN_ROWS;
    // A middle K slice: GEMM accumulation, no residual or output DMA.
    f2.k_dn = Q38_F2_C1 == 4 ? Q38_F2_C1_KO / 2u : Q38_KC;
    f2.k_full = Q38_F2_C1 == 4 ? Q38_F2_C1_KO : Q38_I, f2.n_dn = cols;
    f2.rows_dn = Q38_FFN_DN_ROWS, f2.slices = Q38_FFN_DOWN_SLICES;
    f2.slice = 1u, f2.weight_epoch = 1u, f2.dn_lag = 0u;
    f2.end = Q38_FFN_DN_ROWS, f2.w_dn = w_down, f2.dn_in = F2_G_L2;
#if G2_O_PADDED
    f2.dn_pitch = 0u;
#endif
    f2.dn_rows = Q38_R, f2.res_src = result_mixer, f2.dn_out = result_output;
  }
  if (tnc_is_tile_agent())
    f2_progress[mempool_get_redmule_id()].step_done = 0u;
  if (mempool_is_group_coordinator()) {
    const uint32_t g = mempool_get_group_id();
    f2_down[g].weight_valid = f2_down[g].weight_pending = 0u;
    f2_down[g].x_pending = 0u;
  }
  f2_layout(down);
  f2_reset_progress();
  f2_sync_all();
  const uint32_t start = mempool_get_timer();
  f2_execute();
  if (core == 0u)
    printf("[f2_c1] kind=%u steps=%u k=%u n=%u rows=%u pv=%u w=%u z=%u x=%u "
           "peak=%x start=%u end=%u cycles=%u\n",
           Q38_F2_C1, f2.steps, down ? f2.k_dn : f2.k_up,
           down ? f2.n_dn : f2.n_up, f2.rows, f2.pv, Q38_F2_W_PLACE,
           Q38_F2_Z_PLACE, Q38_F2_X_PLACE, f2.peak, start, mempool_get_timer(),
           mempool_get_timer() - start);
}
#endif
