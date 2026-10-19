// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// GDN v2 (Q38_GDN_V2): the gated-delta-net mixer with whole-sequence
// projections on the L2 GEMM engine (q38_ffn_l2.c) and a chunked gated delta
// rule on RedMulE. Included by layer_resident.c; not a translation unit.
//
// Chunk core (64 tokens, WY form). With Lambda_i the decay accumulated over
// the chunk and G_ij = Lambda_i / Lambda_j:
//   A  = strict_lower(diag(beta) K K^T * G),  T = (I + A)^-1  (Spatz, FP32)
//   U~ = (T diag beta) V,  W = -(T diag(beta Lambda)) K         (RedMulE)
//   U  = U~ + W S,  O = (Lambda Q) S + (Q K^T * G)_lower U     (RedMulE)
//   S' = Lambda_last S + K^T (diag(Lambda_last / Lambda) U)     (RedMulE)
// The decays sit in T and U rather than in V and K, so K^T is the only
// transpose (gdn_chunk_check.py "rm16b": as accurate as scaling the operands).
// Group g owns key head g and its three value heads; tile q of the group owns
// state columns 32q..32q+31 of each of them. The state stays FP16.

_Static_assert(Q38_NK == NUM_GROUPS && Q38_NV == 3u * Q38_NK &&
                   Q38_KD == 128u && Q38_VD == 128u &&
                   NUM_TILES_PER_GROUP == 4u && (NT == 32u || NT == 64u) &&
                   NUM_CORES_PER_TILE == 4u &&
                   NUM_SCALAR_CORES_PER_TILE == 1u && Q38_R % 64u == 0u,
               "GDN v2 maps one key head per group and a 1 + 3 tile");

// Debug: project one row batch, run the chunk core over the first N tokens
// and dump their contexts of heads 0, 1 and NV-1 (reference.py comparison).
#ifndef Q38_GDN2_DEBUG_TOKENS
#define Q38_GDN2_DEBUG_TOKENS 0u
#endif
// Debug: skip the in-projection (P holds whatever L2 holds; control only).
#ifndef Q38_GDN2_DEBUG_SKIP_PROJ
#define Q38_GDN2_DEBUG_SKIP_PROJ 0
#endif
// Debug (timing): skip the chunk core and the FFN and run the
// out-projection over the first N rows only.
#ifndef Q38_GDN2_DEBUG_OUT_ROWS
#define Q38_GDN2_DEBUG_OUT_ROWS 0u
#endif
// Debug (timing): stop after the GDN mixer (no FFN).
#ifndef Q38_GDN2_DEBUG_NO_FFN
#define Q38_GDN2_DEBUG_NO_FFN 0
#endif
#define G2_TOKENS (Q38_GDN2_DEBUG_TOKENS ? Q38_GDN2_DEBUG_TOKENS : Q38_R)
// Out-projection K slices: O is stored as [slices][R][GDIM / slices]. RedMulE
// reads X at a row stride of its K, and a 12 KiB stride (one slice) puts the
// 32 rows of a job on only two offsets of the 8 KiB global L1 row.
#ifndef Q38_GDN2_OUT_SLICES
#define Q38_GDN2_OUT_SLICES 2u
#endif
#define G2_OK (Q38_GDIM / Q38_GDN2_OUT_SLICES)
// Out-projection: three weight slots (engines meet one step behind).
#ifndef Q38_GDN2_OUT_LAG
#define Q38_GDN2_OUT_LAG 1u
#endif
#define G2_OUT_ROWS                                                            \
  (Q38_GDN2_DEBUG_OUT_ROWS ? Q38_GDN2_DEBUG_OUT_ROWS : G2_TOKENS)
// Debug: export every context (the debug check reads heads 0, 1, NV-1);
// 0 keeps a full run's sampled export (timing controls).
#ifndef Q38_GDN2_DEBUG_CTX
#define Q38_GDN2_DEBUG_CTX (Q38_GDN2_DEBUG_TOKENS != 0u)
#endif
#define G2_PROJECT_ROWS (Q38_GDN2_DEBUG_TOKENS ? Q38_FFN_ROWS : Q38_R)
// The stages take their row window [w0, w1) as arguments in the batch-major
// layer (Q38_X8_BATCH, layer_resident.c) and are whole-sequence otherwise.
#if Q38_X8_BATCH
_Static_assert(!Q38_GDN2_DEBUG_TOKENS && !Q38_GDN2_DEBUG_OUT_ROWS &&
                   !Q38_GDN2_DEBUG_NO_FFN,
               "Batch-major GDN runs the full layer");
#define G2_WIN_VOID uint32_t w0, uint32_t w1
#define G2_WIN_PARAMS , uint32_t w0, uint32_t w1
#define G2_WIN_ARGS , w0, w1
#define G2_WIN(a, b)
#else
#define G2_WIN_VOID void
#define G2_WIN_PARAMS
#define G2_WIN_ARGS
#define G2_WIN(a, b) const uint32_t w0 = (a), w1 = (b)
#endif
#define G2_C 64u                            // tokens per chunk
#define G2_H 3u                             // value heads per key head
#define G2_QW 32u                           // state columns per tile
#define G2_CH (2u * Q38_KD + G2_H * Q38_VD) // convolved channels per group
#define G2_PCOLS (Q38_QKVDIM + Q38_GDIM + 2u * Q38_NV)
// P row stride: whole 512 B L2 granules. The L2 interleaver scrambles only
// a burst's start address, so no DMA row may cross a granule.
#define G2_PSTRIDE ((G2_PCOLS + 255u) / 256u * 256u)
#define G2_COL_Q(g) ((g)*Q38_KD)
#define G2_COL_K(g) (Q38_NK * Q38_KD + (g)*Q38_KD)
#define G2_COL_V(g) (2u * Q38_NK * Q38_KD + (g)*G2_H * Q38_VD)
_Static_assert(G2_OK *Q38_GDN2_OUT_SLICES == Q38_GDIM &&
                   G2_OK % (G2_H * Q38_VD) == 0,
               "A group's gated columns lie in one out-projection slice");
#define G2_COL_Z(g) (Q38_QKVDIM + (g)*G2_H * Q38_VD)
#define G2_COL_A(g) (Q38_QKVDIM + Q38_GDIM + (g)*G2_H)
#define G2_COL_B(g) (Q38_QKVDIM + Q38_GDIM + Q38_NV + (g)*G2_H)
// L2 scratch at the top of L2: projections P, gated rows O and contexts. The
// FFN's G later overlays P and O (dead by then) but not the contexts, which
// the end-of-run check reads.
// Debug: bytes left free at the top of L2 above the GDN buffers.
#ifndef Q38_GDN2_L2_GAP
#define Q38_GDN2_L2_GAP 0u
#endif
#if Q38_X8_BATCH
// Batch-major: P must outlive the FFN of the previous batch (the chunk core
// reads its three history rows), so the GDN buffers sit below G.
#define G2_L2_TOP ((uint32_t)F2_G_L2 - Q38_GDN2_L2_GAP)
#else
#define G2_L2_TOP (Q38_L2_TOP - Q38_GDN2_L2_GAP)
#endif
#define G2_P_L2 ((__fp16 *)(G2_L2_TOP - Q38_R * G2_PSTRIDE * 2u))
#define G2_O_L2 ((__fp16 *)((uint32_t)G2_P_L2 - Q38_R * Q38_GDIM * 2u))
#define G2_CTX_L2 ((__fp16 *)((uint32_t)G2_O_L2 - Q38_NV * Q38_R * Q38_VD * 2u))
// Debug (numeric, chunk core only): gdn-design/mkpload.py writes P's first
// rows over w_out, which a Q38_GDN2_DEBUG_TOKENS run never reads.
#ifndef Q38_GDN2_DEBUG_PLOAD
#define Q38_GDN2_DEBUG_PLOAD 0
#endif
// Debug (numeric, with DEBUG_TOKENS and PLOAD): also run the out-projection
// over the debug rows and dump four whole result_mixer rows. P then lies
// over w_gate (mkpload.py ... w_gate), as the out-projection reads w_out.
#ifndef Q38_GDN2_DEBUG_MIXER
#define Q38_GDN2_DEBUG_MIXER 0
#endif
#if Q38_GDN2_DEBUG_PLOAD
#define G2_PSRC ((const __fp16 *)(Q38_GDN2_DEBUG_MIXER ? w_gate : w_out))
#else
#define G2_PSRC ((const __fp16 *)G2_P_L2)
#endif
_Static_assert((G2_PSTRIDE + Q38_GDIM) >= Q38_I,
               "Contexts must lie below the FFN's G");
#if Q38_L2_TILED
// P slice-major: q|k|v|z in slices of 4096 columns (8 KiB L2 rows), then the
// a|b columns and padding at a pitch of G2_PSTRIDE - G2_PMAIN.
#define G2_PSLICE 4096u
#define G2_PMAIN (Q38_QKVDIM + Q38_GDIM)
#define G2_PTAIL (G2_PSTRIDE - G2_PMAIN)
_Static_assert(G2_PMAIN % G2_PSLICE == 0u && 8192u % (G2_PTAIL * 2u) == 0u &&
                   (G2_PSLICE % Q38_KD == 0u && G2_PSLICE % Q38_VD == 0u),
               "P slices are whole L2 rows and the tail divides one");
#define G2_PAT(row, col)                                                       \
  (G2_PSRC + ((col) < G2_PMAIN                                                 \
                  ? (col) / G2_PSLICE * Q38_R * G2_PSLICE + (row)*G2_PSLICE +  \
                        (col) % G2_PSLICE                                      \
                  : G2_PMAIN * Q38_R + (row)*G2_PTAIL + (col)-G2_PMAIN))
#define G2_PPITCH(col) ((col) < G2_PMAIN ? G2_PSLICE : G2_PTAIL)
// O rows (pitch G2_OK) whose starts repeat their offset in an 8 KiB L2 row.
#define G2_OSTEP (8192u / (G2_OK * 2u & (0u - G2_OK * 2u)))
_Static_assert((G2_OK * 2u & (0u - G2_OK * 2u)) < 8192u &&
                   G2_C / 4u % G2_OSTEP == 0u,
               "O row classes divide a tile's rows");
#endif

// Group-window arena: one instance per group (vc_group_window). RedMulE uses
// these pointers; Spatz, scalar code and the DMA use the global alias, row by
// row. Every member starts on a 512-byte window row and no matrix row
// straddles one.
#define G2_ROW __attribute__((aligned(512)))
// One chunk's raw projected rows. Two of them: the next chunk's land while
// this one computes.
typedef struct {
  __fp16 qk[G2_C + 3u][2u * Q38_KD] G2_ROW; // q|k, 3 history rows first
  __fp16 v[G2_H][G2_C + 4u][Q38_VD] G2_ROW; // 3 history rows, one pad row
  __fp16 ab[G2_C][8] G2_ROW;                // a0 a1 a2 b0 b1 b2
} g2_raw_t;
// One chunk's state-independent operands. Two of them: the next chunk's
// convolution and gates run while RedMulE works on this one.
typedef struct {
  __fp16 q[G2_C][Q38_KD] G2_ROW; // convolved, then normalized
  __fp16 k[G2_C][Q38_KD] G2_ROW;
  __fp16 vq[G2_H][4][G2_C][G2_QW] G2_ROW; // convolved V, by state quarter
  float lam[G2_H][G2_C] G2_ROW;           // Lambda_i
  float ilam[G2_H][G2_C] G2_ROW;          // 1 / Lambda_i
  float beta[G2_H][G2_C] G2_ROW;
  float nbl[G2_H][G2_C] G2_ROW; // -beta_i Lambda_i
  // K^T: chunk c's state update still reads it while chunk c+1's is built.
  __fp16 kt[Q38_KD][G2_C] G2_ROW;
} g2_pre_t;
// Each tile's own data, in its tile window (vc.h): its quarter of the state,
// of U and of O, which only its RedMulE and its Spatz cores write, the gated
// rows its cores produce and the gated-norm scale. Spatz and RedMulE address
// it through the tile window; other tiles and the DMA through the global
// alias of that tile's stripe (g2_tglob).
typedef struct {
  __fp16 s[G2_H][Q38_KD][G2_QW]; // state columns 32q .. 32q+31
  __fp16 u[G2_H][G2_C][G2_QW];   // U~, U, then diag(L / Lambda) U
  __fp16 o[G2_H][G2_C][G2_QW];   // context quarter
  // z rows 16q .. 16q+15 of the three heads, gated in place; by chunk parity
  // (the next chunk's land while this one's are gated and exported).
  __fp16 gz[2][G2_C / 4u][G2_H][Q38_VD]; // [b][row][head]: L2's order
  float gamma32[Q38_VD];                 // gated-norm scale, widened
  union {
    struct {
      float a32[G2_C][G2_C]; // tiles 0 .. 2: A of head q (FP32)
      float t32[G2_C][G2_C]; // tiles 0 .. 2: T of head q (FP32)
    };
    __fp16 dl[G2_H][G2_C][G2_C]; // tile 3: diag(Lambda_h), for Lambda Q
  };
  __fp16 tn[G2_C][G2_C]; // tiles 0 .. 2: -T diag(beta Lambda), for W_q
  // [0], [1]: compute-core barrier count and epoch; [2]: blocks handed to the
  // tile's engine (tile 3: diagonals; tiles 0 .. 2: level-3 operands);
  // [3]: the engine's level-3 result epoch.
  uint32_t bar[VC_TILE_ROW_BYTES / 4u];
} g2_tile_t;
#define G2_TILE_ROWS (sizeof(g2_tile_t) / VC_TILE_ROW_BYTES)
_Static_assert(sizeof(g2_tile_t) % VC_TILE_ROW_BYTES == 0u,
               "Tile data takes whole tile-window rows");

typedef struct {
  g2_raw_t raw[2];
  g2_pre_t pre[2];
  __fp16 gkk[G2_C][G2_C] G2_ROW;               // K K^T
  __fp16 gqk[G2_C][G2_C] G2_ROW;               // Q K^T
  __fp16 tb[G2_H][G2_C][G2_C] G2_ROW;          // T diag(beta)
  __fp16 p[G2_H][G2_C][G2_C] G2_ROW;           // (Q K^T * G)_lower
  __fp16 lq[G2_H][G2_C][Q38_KD] G2_ROW;        // Lambda Q
  __fp16 w[G2_H][G2_C][Q38_KD] G2_ROW;         // W
  float tmp[12][G2_C] G2_ROW;                  // per compute core
  uint32_t ntask[2][VC_ROW_BYTES / 8u] G2_ROW; // [b][0]: next task of pre[b]
#if Q38_SAMPLED_CHECKS && !Q38_GDN2_DEBUG_CTX
  uint32_t runs[Q38_RUNS_CONTEXT] G2_ROW; // run_start_context, in L1
#endif
  float lt[3][4][32] G2_ROW; // per compute core, in its own tile: [core][tile]
  __fp16 conv[4][G2_CH] G2_ROW; // conv taps of this group
  __fp16 gamma[Q38_VD] G2_ROW;  // gated-norm scale
  // Window rows whose 128-byte tile stripes hold each tile's g2_tile_t.
  uint8_t tile_rows[G2_TILE_ROWS][VC_ROW_BYTES] G2_ROW;
} g2_group_t;
_Static_assert(G2_CH * 2u % 128u == 0u,
               "64-channel conv blocks never straddle a window row");
// 128-element pieces of the state and of V by quarter.
#define G2_S_PIECES (G2_H * 4u * Q38_KD * G2_QW / 128u)
#define G2_VQ_PIECES (G2_H * 4u * G2_C * G2_QW / 128u)

static struct { uint32_t in_proj, chunks, out_proj; } g2_run Q38_VC_SMALL;

// The arena's group-window address, from link-time constants: every function
// keeps it in a register (g2_self). A shared pointer variable would be
// reloaded after each vector asm statement by every core of the cluster.
static g2_group_t *g2_arena(void) {
  return (g2_group_t *)vc_group_window(
      f2_align(Q38_MIN((uint32_t)&stage, (uint32_t)&weights[0][0])));
}
#define G2 g2_self
#define G2_SELF g2_group_t *const g2_self = g2_arena()

// Global alias of a group-window element (valid within one window row). The
// hart id is read by a non-volatile asm, so the compiler keeps the group
// offset in a register instead of re-deriving it for every access.
_Static_assert((VC_ROW_BYTES & (VC_ROW_BYTES - 1u)) == 0u &&
                   VC_GLOBAL_ROW_BYTES == VC_ROW_BYTES * NUM_GROUPS &&
                   VC_WINDOW_BASE_0 % VC_ROW_BYTES == 0u,
               "Window rows are power-of-two stripes of a global row");
static inline uint32_t g2_hart(void) {
  uint32_t r;
  asm("csrr %0, mhartid" : "=r"(r));
  return DOMAINS > 1 ? r % NUM_CORES : r; // within the domain (addrmap.h)
}
static inline void *g2_glob(const void *window) {
  const uint32_t o = (uint32_t)window - VC_WINDOW_BASE_0;
  return (void *)(VC_GLOBAL_BASE + (o & ~(VC_ROW_BYTES - 1u)) * NUM_GROUPS +
                  (o & (VC_ROW_BYTES - 1u)) +
                  g2_hart() / NUM_CORES_PER_GROUP * VC_ROW_BYTES);
}
#define G2G(ptr) ((__typeof__(ptr))g2_glob(ptr))
// The tile-window alias of whole group-window rows: the requester's own
// 128-byte stripe of each row, contiguous in the tile window (vc.h).
static inline void *g2_tw(const void *window) {
  return (void *)(VC_TILE_WINDOW_BASE + ((uint32_t)window - VC_WINDOW_BASE_0) /
                                            VC_ROW_BYTES * VC_TILE_ROW_BYTES);
}
#define G2T(ptr) ((__typeof__(ptr))g2_tw(ptr))
// This tile's g2_tile_t (tile window) and the global alias of the same
// element in tile q of this group.
#define G2L ((g2_tile_t *)g2_tw(&G2->tile_rows[0][0]))
static inline void *g2_tglob(const void *tile_window, uint32_t q) {
  const uint32_t o = (uint32_t)tile_window - VC_TILE_WINDOW_BASE;
  return (void *)(VC_GLOBAL_BASE + o / VC_TILE_ROW_BYTES * VC_GLOBAL_ROW_BYTES +
                  (g2_hart() / NUM_CORES_PER_GROUP * NUM_TILES_PER_GROUP + q) *
                      VC_TILE_ROW_BYTES +
                  o % VC_TILE_ROW_BYTES);
}
#define G2TG(ptr, q) ((__typeof__(ptr))g2_tglob(ptr, q))
// Per-row helpers are inlined into per-task loops (G2_LOOP functions), so
// constants and bases are set up once per task rather than once per row.
#define G2_INL static inline __attribute__((always_inline))

static uint32_t g2_tile(void) {
  return mempool_get_tile_id() % NUM_TILES_PER_GROUP;
}

static uint32_t g2_is_sched(void) {
  return mempool_get_core_id() % NUM_CORES_PER_TILE == 0u;
}

// Compute core index within the group: 0..11 (cores 1..3 of the 4 tiles).
static uint32_t g2_cid(void) {
  return g2_tile() * 3u + mempool_get_core_id() % NUM_CORES_PER_TILE - 1u;
}

// Debug: core 0 stamps every group barrier's exit (printed by gdn2()).
#ifndef Q38_GDN2_STAMPS
#define Q38_GDN2_STAMPS 0
#endif
#if Q38_GDN2_STAMPS
static uint32_t g2_stamp[64] Q38_VC_SMALL;
static uint32_t g2_nstamp Q38_VC_SMALL;
#endif

// Every core of the group.
static NOSTACK void g2_sync(void) {
  __sync_synchronize();
  q38_barrier(FSYNC_ID_H, fsync_aggr_of_vc_level(0u));
  __sync_synchronize();
#if Q38_GDN2_STAMPS
  if (mempool_get_core_id() == 0u && g2_nstamp < 64u)
    g2_stamp[g2_nstamp++] = mempool_get_timer();
#endif
}

// The group's four tile schedulers only, on the other barrier id: a compute
// core already waiting in g2_sync must not take this response for its own.
static NOSTACK void g2_sched_sync(void) {
  __sync_synchronize();
  q38_barrier_n(FSYNC_ID_V, fsync_aggr_of_vc_level(0u), 1u);
  __sync_synchronize();
}

// Rows of row_bytes, win_stride apart in the group window (win_stride divides
// 512) from or to a strided L2 matrix. Successive window rows lie 8 KiB apart
// globally, so a block goes as a leading partial window row, whole window
// rows and a trailing partial one. Plane strides are relative to the last
// row of the previous plane.
static dma_id_t g2_dma_rows(char *global, uint32_t win_stride, char *l2,
                            uint32_t row_bytes, uint32_t l2_stride,
                            uint32_t per, uint32_t planes, uint32_t to_l2) {
  const uint32_t win_plane = VC_GLOBAL_ROW_BYTES - (per - 1u) * win_stride;
  return to_l2 ? dma_local_memcpy_3d_nonblocking(0u, l2, global, row_bytes,
                                                 l2_stride, win_stride, per,
                                                 l2_stride, win_plane, planes)
               : dma_local_memcpy_3d_nonblocking(0u, global, l2, row_bytes,
                                                 win_stride, l2_stride, per,
                                                 win_plane, l2_stride, planes);
}

static dma_id_t g2_dma(const void *window, uint32_t win_stride, const void *l2,
                       uint32_t row_bytes, uint32_t l2_stride, uint32_t rows,
                       uint32_t to_l2) {
  const uint32_t inner = VC_ROW_BYTES / win_stride;
  // Offset of the block's first row in its window row (minus the column).
  const uint32_t off = ((uint32_t)window - VC_WINDOW_BASE_0) % VC_ROW_BYTES /
                       win_stride * win_stride;
  uint32_t lead = (VC_ROW_BYTES - off) / win_stride;
  char *win = (char *)window, *mem = (char *)l2;
  dma_id_t id = 0u;
  if (off) {
    lead = lead < rows ? lead : rows;
    id = g2_dma_rows(G2G(win), win_stride, mem, row_bytes, l2_stride, lead, 1u,
                     to_l2);
    win += lead * win_stride;
    mem += lead * l2_stride;
    rows -= lead;
  }
  const uint32_t planes = rows / inner;
  if (planes) {
    id = g2_dma_rows(G2G(win), win_stride, mem, row_bytes, l2_stride, inner,
                     planes, to_l2);
    win += planes * VC_ROW_BYTES;
    mem += planes * inner * l2_stride;
    rows -= planes * inner;
  }
  if (rows)
    id = g2_dma_rows(G2G(win), win_stride, mem, row_bytes, l2_stride, rows, 1u,
                     to_l2);
  return id;
}

static void g2_dma_wait(dma_id_t id) {
  while (id && !dma_local_done_id(0u, id))
    mempool_wfi();
}

// RedMulE job on this tile's engine; returns its completion target.
static uint8_t g2_job(const void *x, const void *w, void *z, uint32_t m,
                      uint32_t n, uint32_t k, uint32_t accumulate) {
  const uint8_t target = (uint8_t)tnc_redmule_acquire_target();
  redmule_cfg((uint32_t)x, (uint32_t)w, (uint32_t)z, (uint16_t)m, (uint16_t)n,
              (uint16_t)k, 0u, accumulate ? GEMM : MATMUL, Float16);
  redmule_x_shift_set(0);
  tnc_redmule_trigger();
  return target;
}

static void g2_job_wait(uint8_t target) {
  while (!tnc_redmule_done(target))
    mempool_wfi();
}

// --- group coordinator: DMA -------------------------------------------------

// One-time: this group's conv taps and the gated-norm scale.
static NOSTACK void g2_load_params(void) {
  G2_SELF;
  const uint32_t g = mempool_get_group_id(), stride = Q38_QKVDIM * 2u;
  dma_id_t id = 0u;
  for (uint32_t tap = 0; tap < 4u; ++tap) {
    id = g2_dma(&G2->conv[tap][0], Q38_KD * 2u,
                w_conv + tap * Q38_QKVDIM + G2_COL_Q(g), Q38_KD * 2u, stride,
                1u, 0u);
    id = g2_dma(&G2->conv[tap][Q38_KD], Q38_KD * 2u,
                w_conv + tap * Q38_QKVDIM + G2_COL_K(g), Q38_KD * 2u, stride,
                1u, 0u);
    for (uint32_t h = 0u; h < G2_H; ++h)
      id = g2_dma(&G2->conv[tap][2u * Q38_KD + h * Q38_VD], Q38_VD * 2u,
                  w_conv + tap * Q38_QKVDIM + G2_COL_V(g) + h * Q38_VD,
                  Q38_VD * 2u, stride, 1u, 0u);
  }
  id = g2_dma(&G2->gamma[0], Q38_VD * 2u, w_gated_norm, Q38_VD * 2u,
              Q38_VD * 2u, 1u, 0u);
#if Q38_SAMPLED_CHECKS && !Q38_GDN2_DEBUG_CTX
  id = dma_local_memcpy_nonblocking(0u, G2G(&G2->runs[0]), run_start_context,
                                    sizeof(G2->runs));
#endif
  g2_dma_wait(id);
}

// The raw projected rows of the chunk at t0 into raw, with 3 history rows
// (the first chunk's are zeroed by the compute cores).
#if Q38_L2_TILED
// a or b (column col): blocks of G2_AB_ROWS rows, one L2 row at the tail
// pitch and within one window row each.
#define G2_AB_ROWS (8192u / (G2_PTAIL * 2u))
_Static_assert(G2_C % G2_AB_ROWS == 0u &&
                   VC_ROW_BYTES %
                           (G2_AB_ROWS * sizeof(((g2_raw_t *)0)->ab[0])) ==
                       0u,
               "a|b blocks are whole L2 rows inside window rows");
static dma_id_t g2_load_ab(__fp16 *win, uint32_t t0, uint32_t col) {
  dma_id_t id = 0u;
  for (uint32_t r = 0u; r < G2_C; r += G2_AB_ROWS)
    id = dma_local_memcpy_2d_nonblocking(
        0u, G2G(win + r * 8u), G2_PAT(t0 + r, col), 6u, 8u * sizeof(__fp16),
        G2_PTAIL * 2u, G2_AB_ROWS);
  return id;
}

static NOSTACK dma_id_t g2_load(uint32_t t0, g2_raw_t *raw) {
  const uint32_t g = mempool_get_group_id();
  const uint32_t first = t0 ? t0 - 3u : 0u, rows = t0 ? G2_C + 3u : G2_C;
  const uint32_t dst = t0 ? 0u : 3u;
  dma_id_t id =
      g2_dma(&raw->qk[dst][0], sizeof(raw->qk[0]), G2_PAT(first, G2_COL_Q(g)),
             Q38_KD * 2u, G2_PPITCH(G2_COL_Q(g)) * 2u, rows, 0u);
  id = g2_dma(&raw->qk[dst][Q38_KD], sizeof(raw->qk[0]),
              G2_PAT(first, G2_COL_K(g)), Q38_KD * 2u,
              G2_PPITCH(G2_COL_K(g)) * 2u, rows, 0u);
  for (uint32_t h = 0u; h < G2_H; ++h) {
    const uint32_t col = G2_COL_V(g) + h * Q38_VD;
    id = g2_dma(&raw->v[h][dst][0], Q38_VD * 2u, G2_PAT(first, col),
                Q38_VD * 2u, G2_PPITCH(col) * 2u, rows, 0u);
  }
  id = g2_load_ab(&raw->ab[0][0], t0, G2_COL_A(g));
  id = g2_load_ab(&raw->ab[0][3], t0, G2_COL_B(g));
  return id; // DONE_ID is monotone: the last covers them all
}
#else
static NOSTACK dma_id_t g2_load(uint32_t t0, g2_raw_t *raw) {
  const uint32_t g = mempool_get_group_id(), stride = G2_PSTRIDE * 2u;
  const uint32_t first = t0 ? t0 - 3u : 0u, rows = t0 ? G2_C + 3u : G2_C;
  const uint32_t dst = t0 ? 0u : 3u;
  const __fp16 *const src = G2_PSRC + first * G2_PSTRIDE;
  const __fp16 *const now = G2_PSRC + t0 * G2_PSTRIDE;
  dma_id_t id = g2_dma(&raw->qk[dst][0], sizeof(raw->qk[0]), src + G2_COL_Q(g),
                       Q38_KD * 2u, stride, rows, 0u);
  id = g2_dma(&raw->qk[dst][Q38_KD], sizeof(raw->qk[0]), src + G2_COL_K(g),
              Q38_KD * 2u, stride, rows, 0u);
  for (uint32_t h = 0u; h < G2_H; ++h) {
    id = g2_dma(&raw->v[h][dst][0], Q38_VD * 2u, src + G2_COL_V(g) + h * Q38_VD,
                Q38_VD * 2u, stride, rows, 0u);
  }
  // a and b of the three heads: six raw FP16 words per row.
  id = g2_dma(&raw->ab[0][0], sizeof(raw->ab[0]), now + G2_COL_A(g), 6u, stride,
              G2_C, 0u);
  id = g2_dma(&raw->ab[0][3], sizeof(raw->ab[0]), now + G2_COL_B(g), 6u, stride,
              G2_C, 0u);
  return id; // DONE_ID is monotone: the last covers them all
}
#endif

// A tile's gated-row block <-> 16 L2 rows of the group's three heads (768
// contiguous bytes each): six tile-window rows per L2 row, one descriptor.
#define G2_ZPIECES (G2_H * Q38_VD * 2u / VC_TILE_ROW_BYTES)
static dma_id_t g2_dma_gz(__fp16 *l2, uint32_t l2_stride, uint32_t b,
                          uint32_t q, uint32_t to_l2) {
  G2_SELF;
  char *const tile = (char *)G2TG(&G2L->gz[b][0][0][0], q);
  const uint32_t l2_plane = l2_stride - (G2_ZPIECES - 1u) * VC_TILE_ROW_BYTES;
  return to_l2 ? dma_local_memcpy_3d_nonblocking(
                     0u, l2, tile, VC_TILE_ROW_BYTES, VC_TILE_ROW_BYTES,
                     VC_GLOBAL_ROW_BYTES, G2_ZPIECES, l2_plane,
                     VC_GLOBAL_ROW_BYTES, G2_C / 4u)
               : dma_local_memcpy_3d_nonblocking(
                     0u, tile, l2, VC_TILE_ROW_BYTES, VC_GLOBAL_ROW_BYTES,
                     VC_TILE_ROW_BYTES, G2_ZPIECES, VC_GLOBAL_ROW_BYTES,
                     l2_plane, G2_C / 4u);
}

#if Q38_L2_TILED
// Pieces p0 .. p0 + np - 1 of rows r0, r0 + step, ... (nr of them) of a
// tile's block; l2 is piece p0 of row r0, pitch the L2 row stride. The planes
// step whole L2 rows when step * pitch does.
static dma_id_t g2_dma_gzp(char *l2, uint32_t pitch, uint32_t step, uint32_t b,
                           uint32_t q, uint32_t to_l2, uint32_t r0, uint32_t nr,
                           uint32_t p0, uint32_t np) {
  G2_SELF;
  char *const tile = (char *)G2TG(&G2L->gz[b][0][0][0], q) +
                     (r0 * G2_ZPIECES + p0) * VC_GLOBAL_ROW_BYTES;
  const uint32_t l2_plane = step * pitch - (np - 1u) * VC_TILE_ROW_BYTES;
  const uint32_t tile_plane =
      (step * G2_ZPIECES - np + 1u) * VC_GLOBAL_ROW_BYTES;
  return to_l2 ? dma_local_memcpy_3d_nonblocking(
                     0u, l2, tile, VC_TILE_ROW_BYTES, VC_TILE_ROW_BYTES,
                     VC_GLOBAL_ROW_BYTES, np, l2_plane, tile_plane, nr)
               : dma_local_memcpy_3d_nonblocking(
                     0u, tile, l2, VC_TILE_ROW_BYTES, VC_GLOBAL_ROW_BYTES,
                     VC_TILE_ROW_BYTES, np, tile_plane, l2_plane, nr);
}

// Rows r0, r0 + step, ... of a tile's block from l2 (row r0, piece 0), cut
// where a row's 768 bytes cross an 8 KiB L2 row (all rows of the class cross
// at the same piece when step * pitch is whole L2 rows).
static dma_id_t g2_dma_gzc(char *l2, uint32_t pitch, uint32_t step, uint32_t b,
                           uint32_t q, uint32_t to_l2, uint32_t r0,
                           uint32_t nr) {
  const uint32_t room = 8192u - (uint32_t)l2 % 8192u;
  const uint32_t n1 = Q38_MIN(G2_ZPIECES, room / VC_TILE_ROW_BYTES);
  dma_id_t id = g2_dma_gzp(l2, pitch, step, b, q, to_l2, r0, nr, 0u, n1);
  if (n1 < G2_ZPIECES)
    id = g2_dma_gzp(l2 + n1 * VC_TILE_ROW_BYTES, pitch, step, b, q, to_l2, r0,
                    nr, n1, G2_ZPIECES - n1);
  return id;
}

// The z rows of the chunk at t0 from P's slices: a group whose z columns
// cross a slice takes two descriptors.
static NOSTACK dma_id_t g2_load_z(uint32_t t0) {
  const uint32_t g = mempool_get_group_id(), b = t0 / G2_C & 1u;
  const uint32_t col = G2_COL_Z(g);
  const uint32_t n1 = Q38_MIN(G2_ZPIECES, (G2_PSLICE - col % G2_PSLICE) * 2u /
                                              VC_TILE_ROW_BYTES);
  dma_id_t id = 0u;
  for (uint32_t q = 0u; q < NUM_TILES_PER_GROUP; ++q) {
    const uint32_t row = t0 + q * (G2_C / 4u);
    id = g2_dma_gzp((char *)G2_PAT(row, col), G2_PSLICE * 2u, 1u, b, q, 0u, 0u,
                    G2_C / 4u, 0u, n1);
    if (n1 < G2_ZPIECES)
      id = g2_dma_gzp((char *)G2_PAT(row, col + n1 * VC_TILE_ROW_BYTES / 2u),
                      G2_PSLICE * 2u, 1u, b, q, 0u, 0u, G2_C / 4u, n1,
                      G2_ZPIECES - n1);
  }
  return id;
}
#else
// The z rows of the chunk at t0, rows 16q .. 16q+15 into tile q, over the
// gated rows of chunk t0 - 2C.
static NOSTACK dma_id_t g2_load_z(uint32_t t0) {
  const uint32_t g = mempool_get_group_id(), b = t0 / G2_C & 1u;
  dma_id_t id = 0u;
  for (uint32_t q = 0u; q < NUM_TILES_PER_GROUP; ++q)
    id = g2_dma_gz((__fp16 *)G2_PSRC + (t0 + q * (G2_C / 4u)) * G2_PSTRIDE +
                       G2_COL_Z(g),
                   G2_PSTRIDE * 2u, b, q, 0u);
  return id;
}
#endif

// Gated rows (in the tiles) and contexts of the chunk at t0 to L2. The
// contexts are a checked diagnostic only: a debug run exports them all, a
// full run only the sampled runs that fall in this chunk, straight into
// result_sample_context. O is rewritten by the next chunk's U, O jobs only
// after the coordinator has waited for this transfer.
static NOSTACK dma_id_t g2_store(uint32_t t0) {
  G2_SELF;
  const uint32_t g = mempool_get_group_id(), b = t0 / G2_C & 1u;
  dma_id_t id = 0u;
#if Q38_L2_TILED
  // O rows repeat their place in an 8 KiB L2 row every G2_OSTEP rows: one
  // class of rows per descriptor (two where its rows cross an L2 row).
  for (uint32_t q = 0u; q < NUM_TILES_PER_GROUP; ++q) {
    char *const l2 =
        (char *)(G2_O_L2 +
                 (g * G2_H * Q38_VD / G2_OK * Q38_R + t0 + q * (G2_C / 4u)) *
                     G2_OK +
                 g * G2_H * Q38_VD % G2_OK);
    for (uint32_t r = 0u; r < G2_OSTEP; ++r)
      id = g2_dma_gzc(l2 + r * G2_OK * 2u, G2_OK * 2u, G2_OSTEP, b, q, 1u, r,
                      G2_C / 4u / G2_OSTEP);
  }
#else
  for (uint32_t q = 0u; q < NUM_TILES_PER_GROUP; ++q)
    id = g2_dma_gz(
        G2_O_L2 +
            (g * G2_H * Q38_VD / G2_OK * Q38_R + t0 + q * (G2_C / 4u)) * G2_OK +
            g * G2_H * Q38_VD % G2_OK,
        G2_OK * 2u, b, q, 1u);
#endif
#if Q38_GDN2_DEBUG_CTX
  for (uint32_t h = 0u; h < G2_H; ++h) {
    const uint32_t head = g * G2_H + h;
    for (uint32_t q = 0u; q < NUM_TILES_PER_GROUP; ++q) // 64 B rows, 2 a row
      id = g2_dma_rows(
          (char *)G2TG(&G2L->o[h][0][0], q), G2_QW * 2u,
          (char *)(G2_CTX_L2 + (head * Q38_R + t0) * Q38_VD + q * G2_QW),
          G2_QW * 2u, Q38_VD * 2u, VC_TILE_ROW_BYTES / (G2_QW * 2u),
          G2_C * G2_QW * 2u / VC_TILE_ROW_BYTES, 1u);
  }
#elif Q38_SAMPLED_CHECKS
  // Runs over [head][row][VD], split at quarter (tile) and row boundaries.
  const uint32_t *const runs = G2G(&G2->runs[0]);
  for (uint32_t i = 0u; i < Q38_RUNS_CONTEXT; ++i)
    for (uint32_t p = runs[i]; p < runs[i] + Q38_RUNLEN_CONTEXT;) {
      const uint32_t head = p / (Q38_R * Q38_VD), row = p / Q38_VD % Q38_R;
      const uint32_t col = p % Q38_VD;
      const uint32_t n =
          Q38_MIN(runs[i] + Q38_RUNLEN_CONTEXT - p, G2_QW - col % G2_QW);
      if (head / G2_H == g && row - t0 < G2_C)
        id = dma_local_memcpy_nonblocking(
            0u, Q38_RES_SAMPLE_CONTEXT + i * Q38_RUNLEN_CONTEXT + (p - runs[i]),
            G2TG(&G2L->o[head % G2_H][row - t0][col % G2_QW], col / G2_QW),
            n * 2u);
      p += n;
    }
#endif
  return id;
}

// The convolution state: the last four raw rows (tokens R-4 .. R-1).
static NOSTACK void g2_store_conv(g2_raw_t *raw) {
  const uint32_t g = mempool_get_group_id(), stride = Q38_RES_CONV_PITCH;
  dma_id_t id = g2_dma(&raw->qk[G2_C - 1u][0], sizeof(raw->qk[0]),
                       Q38_RES_CONV + G2_COL_Q(g), Q38_KD * 2u, stride, 4u, 1u);
  id = g2_dma(&raw->qk[G2_C - 1u][Q38_KD], sizeof(raw->qk[0]),
              Q38_RES_CONV + G2_COL_K(g), Q38_KD * 2u, stride, 4u, 1u);
  for (uint32_t h = 0u; h < G2_H; ++h)
    id = g2_dma(&raw->v[h][G2_C - 1u][0], Q38_VD * 2u,
                Q38_RES_CONV + G2_COL_V(g) + h * Q38_VD, Q38_VD * 2u, stride,
                4u, 1u);
  g2_dma_wait(id);
}

#if Q38_X8_BATCH
// The FP16 state between row batches: each tile's g2_tile_t.s, contiguous in
// L2 below the contexts, [group][tile].
#define G2_SC_BYTES sizeof(((g2_tile_t *)0)->s)
#define G2_SC_L2                                                               \
  ((char *)G2_CTX_L2 - NUM_GROUPS * NUM_TILES_PER_GROUP * G2_SC_BYTES)
_Static_assert(G2_SC_BYTES % VC_TILE_ROW_BYTES == 0u &&
                   G2_SC_BYTES % 512u == 0u && offsetof(g2_tile_t, s) == 0u,
               "The state takes whole tile-window rows and L2 granules");

#if Q38_X8_PIPE
// Layer pipeline: below domain dom's state slot, its mixer rows and its
// layer's output rows (the next layer's input). F2_G_L2 and the GDN buffers
// above follow the same per-domain span.
_Static_assert(Q38_GDN2_L2_GAP == 0u, "The pipeline spans start at G");
#define X8_DOM_TOP(dom)                                                        \
  (Q38_L2_TOP - (dom)*Q38_X8_SPAN - Q38_R * Q38_I * 2u -                       \
   Q38_R * (G2_PSTRIDE + Q38_GDIM + Q38_NV * Q38_VD) * 2u -                    \
   NUM_GROUPS * NUM_TILES_PER_GROUP * G2_SC_BYTES)
#define X8_MIX_L2(dom) ((__fp16 *)(X8_DOM_TOP(dom) - Q38_R * Q38_D * 2u))
#define X8_ACT_L2(dom) ((__fp16 *)(X8_DOM_TOP(dom) - 2u * Q38_R * Q38_D * 2u))
_Static_assert(!Q38_X8_SPAN ||
                   Q38_L2_TOP - X8_DOM_TOP(0u) + 2u * Q38_R * Q38_D * 2u <=
                       Q38_X8_SPAN,
               "A domain's scratch fits its span");
#if Q38_L2_HOME_SLOTS
_Static_assert(Q38_RES_HOME_END <= X8_DOM_TOP(0u) - 2u * Q38_R * Q38_D * 2u &&
                   Q38_RES_CONV_PITCH * 4u <= 0x20000u &&
                   Q38_NV * Q38_KD * Q38_VD * 4u <= 0x300000u,
               "The home result slots fit below the scratch");
#endif
#endif

// The state of the group's four tiles to (to_l2) or from its L2 slot: one
// descriptor per tile, whole tile-window rows.
static NOSTACK dma_id_t g2_state_dma(uint32_t to_l2) {
  G2_SELF;
  const uint32_t g = mempool_get_group_id();
  const uint32_t rows = G2_SC_BYTES / VC_TILE_ROW_BYTES;
  dma_id_t id = 0u;
  for (uint32_t q = 0u; q < NUM_TILES_PER_GROUP; ++q) {
    char *const l2 = G2_SC_L2 + (g * NUM_TILES_PER_GROUP + q) * G2_SC_BYTES;
    char *const tile = (char *)G2TG(&G2L->s[0][0][0], q);
#if Q38_L2_TILED
    // Planes of one 8 KiB L2 row each: whole-row plane steps on both sides.
    const uint32_t per = VC_GLOBAL_ROW_BYTES / VC_TILE_ROW_BYTES;
    _Static_assert(G2_SC_BYTES % VC_GLOBAL_ROW_BYTES == 0u,
                   "The state takes whole 8 KiB L2 rows");
#else
    const uint32_t per = rows;
#endif
    id = to_l2 ? dma_local_memcpy_3d_nonblocking(
                     0u, l2, tile, VC_TILE_ROW_BYTES, VC_TILE_ROW_BYTES,
                     VC_GLOBAL_ROW_BYTES, per, VC_TILE_ROW_BYTES,
                     VC_GLOBAL_ROW_BYTES, rows / per)
               : dma_local_memcpy_3d_nonblocking(
                     0u, tile, l2, VC_TILE_ROW_BYTES, VC_GLOBAL_ROW_BYTES,
                     VC_TILE_ROW_BYTES, per, VC_GLOBAL_ROW_BYTES,
                     VC_TILE_ROW_BYTES, rows / per);
  }
  return id;
}
#endif

// --- compute cores: Spatz
// -----------------------------------------------------

// Debug (timing only): a hardware vfexp stand-in, one vector op per exp.
#ifndef Q38_GDN2_VFEXP
#define Q38_GDN2_VFEXP 0
#endif
#if Q38_GDN2_VFEXP
#define G2_EXP() asm volatile("vfmul.vv v0, v0, v0")
#else
#define G2_EXP() llm_rvv_expf_m8()
#endif

// v24 op= sum_{j<n} scal[j] row(j) for vl FP32 elements (LMUL <= 2, vtype
// set by the caller, v24 initialized). Rows rotate through v16 .. v22 and
// the products through two accumulators (v24, v28), so loads run ahead of
// the FMAs; the scalars load four at a time so their latencies overlap.
#define G2_AXPY1(op, acc, vs, s, addr)                                         \
  do {                                                                         \
    asm volatile("vle32.v " vs ", (%0)" : : "r"(addr) : "memory");             \
    asm volatile(op " " acc ", %0, " vs : : "f"(s));                           \
  } while (0)
#define G2_AXPYS(op, n, scal, row)                                             \
  do {                                                                         \
    uint32_t j_ = 0u;                                                          \
    asm volatile("vmv.v.i v28, 0");                                            \
    for (; j_ + 4u <= (n); j_ += 4u) {                                         \
      const float s0_ = (scal)[j_], s1_ = (scal)[j_ + 1u];                     \
      const float s2_ = (scal)[j_ + 2u], s3_ = (scal)[j_ + 3u];                \
      G2_AXPY1(op, "v24", "v16", s0_, row(j_));                                \
      G2_AXPY1(op, "v28", "v18", s1_, row(j_ + 1u));                           \
      G2_AXPY1(op, "v24", "v20", s2_, row(j_ + 2u));                           \
      G2_AXPY1(op, "v28", "v22", s3_, row(j_ + 3u));                           \
    }                                                                          \
    for (; j_ < (n); ++j_) {                                                   \
      const float s_ = (scal)[j_];                                             \
      G2_AXPY1(op, "v24", "v16", s_, row(j_));                                 \
    }                                                                          \
    asm volatile("vfadd.vv v24, v24, v28");                                    \
  } while (0)

// This compute core's scratch in its own tile (32 floats).
static float *g2_lt(void) {
  G2_SELF;
  return G2G(
      &G2->lt[mempool_get_core_id() % NUM_CORES_PER_TILE - 1u][g2_tile()][0]);
}

// v8 = SiLU(v24) for vl FP32 elements (e32, m8 selected); v0, v16 scratch.
static inline __attribute__((always_inline)) void g2_silu_v24(void) {
  const float one = 1.0f;
  asm volatile("vfsgnjn.vv v0, v24, v24");
  G2_EXP();
  asm volatile("vfadd.vf v0, v0, %0" : : "f"(one));
  llm_rvv_recip_pos_m8();
  asm volatile("vfmul.vv v8, v8, v24");
}

// Activations (the convolution's and z's SiLU) in FP16 arithmetic: twice the
// FP32 rate, no widening, ~11% of a chunk; contexts' RMS error vs the
// reference 1.16e-6 instead of 1.0e-6 (ref RMS 5e-4; 128-token check).
// 0: FP32 rounded to FP16 as the reference does.
#ifndef Q38_GDN2_F16ACT
#define Q38_GDN2_F16ACT 1
#endif

// v0 = exp(v0) in FP16 (e16, m4 selected), the argument clamped to +-9.5 so
// that 2^n and 1 / (1 + exp) stay normal; v4, v8 scratch.
static inline __attribute__((always_inline)) void g2_exp16(void) {
#if Q38_GDN2_VFEXP
  asm volatile("vfmul.vv v0, v0, v0");
#else
  const float hi = llm_f16box(9.5f), lo = llm_f16box(-9.5f);
  const float l2e = llm_f16box(1.4426950408f), ln2 = llm_f16box(0.6931471806f);
  const float a4 = llm_f16box(1.0f / 24.0f), a3 = llm_f16box(1.0f / 6.0f);
  const float a2 = llm_f16box(0.5f), a1 = llm_f16box(1.0f);
  asm volatile("vfmin.vf v0, v0, %0" : : "f"(hi));
  asm volatile("vfmax.vf v0, v0, %0" : : "f"(lo));
  asm volatile("vfmul.vf v4, v0, %0" : : "f"(l2e));
  asm volatile("vfcvt.x.f.v v8, v4");
  asm volatile("vfcvt.f.x.v v4, v8");
  asm volatile("vfnmsac.vf v0, %0, v4" : : "f"(ln2));
  asm volatile("vfmv.v.f v4, %0" : : "f"(a4));
  asm volatile("vfmul.vv v4, v4, v0");
  asm volatile("vfadd.vf v4, v4, %0" : : "f"(a3));
  asm volatile("vfmul.vv v4, v4, v0");
  asm volatile("vfadd.vf v4, v4, %0" : : "f"(a2));
  asm volatile("vfmul.vv v4, v4, v0");
  asm volatile("vfadd.vf v4, v4, %0" : : "f"(a1));
  asm volatile("vfmul.vv v4, v4, v0");
  asm volatile("vfadd.vf v4, v4, %0" : : "f"(a1));
  asm volatile("vadd.vx v8, v8, %0" : : "r"(15));
  asm volatile("vsll.vi v8, v8, 10");
  asm volatile("vfmul.vv v0, v4, v8");
#endif
}

// Load n <= 128 FP16 elements (one window row at most) and leave SiLU(x),
// FP16, in v0 (e16, m4 selected on return).
G2_INL void g2_silu_row16(const __fp16 *x, uint32_t n) {
#if Q38_GDN2_F16ACT
  const float one = llm_f16box(1.0f), two = llm_f16box(2.0f);
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(n));
  asm volatile("vle16.v v16, (%0)" : : "r"(x) : "memory");
  asm volatile("vfsgnjn.vv v0, v16, v16");
  g2_exp16();
  asm volatile("vfadd.vf v0, v0, %0" : : "f"(one));
  // 1 / (1 + exp(-x)): seed from the bits, two Newton steps.
  asm volatile("vrsub.vx v8, v0, %0" : : "r"(0x7798));
  asm volatile("vfmul.vv v12, v0, v8");
  asm volatile("vfrsub.vf v12, v12, %0" : : "f"(two));
  asm volatile("vfmul.vv v8, v8, v12");
  asm volatile("vfmul.vv v12, v0, v8");
  asm volatile("vfrsub.vf v12, v12, %0" : : "f"(two));
  asm volatile("vfmul.vv v8, v8, v12");
  asm volatile("vfmul.vv v0, v8, v16");
#else
  const float zh = llm_f16box(0.0f);
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(n));
  asm volatile("vle16.v v0, (%0)" : : "r"(x) : "memory");
  asm volatile("vfwadd.vf v24, v0, %0" : : "f"(zh));
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(n));
  g2_silu_v24();
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(n));
  asm volatile("vfncvt.f.f.w v0, v8");
#endif
}

// x[0..n) = 0 (one window row at most).
static void g2_zero16(__fp16 *x, uint32_t n) {
  while (n) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(n));
    asm volatile("vmv.v.i v0, 0");
    asm volatile("vse16.v v0, (%0)" : : "r"(x) : "memory");
    x += vl;
    n -= vl;
  }
}

// x[0..n) *= scale (FP16 in place, FP32 product).
G2_INL void g2_scale16(__fp16 *x, uint32_t n, float scale) {
  const float zh = llm_f16box(0.0f);
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(n));
  asm volatile("vle16.v v0, (%0)" : : "r"(x) : "memory");
  asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(n));
  asm volatile("vfmul.vf v8, v8, %0" : : "f"(scale));
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(n));
  asm volatile("vfncvt.f.f.w v0, v8");
  asm volatile("vse16.v v0, (%0)" : : "r"(x) : "memory");
}

// Raw row r of the group's convolution channel c (0 .. G2_CH).
static const __fp16 *g2_raw(const g2_raw_t *raw, uint32_t c, uint32_t r) {
  if (c < 2u * Q38_KD)
    return &raw->qk[r][c];
  c -= 2u * Q38_KD;
  return &raw->v[c / Q38_VD][r][c % Q38_VD];
}

// exp(X) in place for the selected vtype (FP32), T and N scratch groups of the
// same size: llm_rvv_expf_m8 with the registers as arguments.
#if Q38_GDN2_VFEXP
#define G2_EXPF(X, T, N) asm volatile("vfmul.vv " X ", " X ", " X)
#else
#define G2_EXPF(X, T, N)                                                       \
  do {                                                                         \
    asm volatile("vfmin.vf " X ", " X ", %0" : : "f"(80.0f));                  \
    asm volatile("vfmax.vf " X ", " X ", %0" : : "f"(-80.0f));                 \
    asm volatile("vfmul.vf " T ", " X ", %0" : : "f"(1.4426950408f));          \
    asm volatile("vfcvt.x.f.v " N ", " T);                                     \
    asm volatile("vfcvt.f.x.v " T ", " N);                                     \
    asm volatile("vfnmsac.vf " X ", %0, " T : : "f"(0.6931471806f));           \
    asm volatile("vfmv.v.f " T ", %0" : : "f"(0.0083333338f));                 \
    asm volatile("vfmul.vv " T ", " T ", " X);                                 \
    asm volatile("vfadd.vf " T ", " T ", %0" : : "f"(0.041666668f));           \
    asm volatile("vfmul.vv " T ", " T ", " X);                                 \
    asm volatile("vfadd.vf " T ", " T ", %0" : : "f"(0.16666667f));            \
    asm volatile("vfmul.vv " T ", " T ", " X);                                 \
    asm volatile("vfadd.vf " T ", " T ", %0" : : "f"(0.5f));                   \
    asm volatile("vfmul.vv " T ", " T ", " X);                                 \
    asm volatile("vfadd.vf " T ", " T ", %0" : : "f"(1.0f));                   \
    asm volatile("vfmul.vv " T ", " T ", " X);                                 \
    asm volatile("vfadd.vf " T ", " T ", %0" : : "f"(1.0f));                   \
    asm volatile("vadd.vx " N ", " N ", %0" : : "r"(127));                     \
    asm volatile("vsll.vi " N ", " N ", 23");                                  \
    asm volatile("vfmul.vv " X ", " T ", " N);                                 \
  } while (0)
#endif
// R = 1 / X for positive X (llm_rvv_recip_pos_m8), T scratch.
#define G2_RECIP(X, R, T)                                                      \
  do {                                                                         \
    asm volatile("vrsub.vx " R ", " X ", %0" : : "r"(0x7ef311c3u));            \
    asm volatile("vfmul.vv " T ", " X ", " R);                                 \
    asm volatile("vfrsub.vf " T ", " T ", %0" : : "f"(2.0f));                  \
    asm volatile("vfmul.vv " R ", " R ", " T);                                 \
    asm volatile("vfmul.vv " T ", " X ", " R);                                 \
    asm volatile("vfrsub.vf " T ", " T ", %0" : : "f"(2.0f));                  \
    asm volatile("vfmul.vv " R ", " R ", " T);                                 \
  } while (0)

// FP16 forms for e16 vtypes (Q38_GDN2_F16ACT): exp(X) in place, the argument
// clamped to +-9.5 so that 2^n and 1 / (1 + exp) stay normal (g2_exp16), and
// R = 1 / X from the bits plus two Newton steps.
#if Q38_GDN2_VFEXP
#define G2_EXPH(X, T, N) asm volatile("vfmul.vv " X ", " X ", " X)
#else
#define G2_EXPH(X, T, N)                                                       \
  do {                                                                         \
    asm volatile("vfmin.vf " X ", " X ", %0" : : "f"(llm_f16box(9.5f)));       \
    asm volatile("vfmax.vf " X ", " X ", %0" : : "f"(llm_f16box(-9.5f)));      \
    asm volatile("vfmul.vf " T ", " X ", %0"                                   \
                 :                                                             \
                 : "f"(llm_f16box(1.4426950408f)));                            \
    asm volatile("vfcvt.x.f.v " N ", " T);                                     \
    asm volatile("vfcvt.f.x.v " T ", " N);                                     \
    asm volatile("vfnmsac.vf " X ", %0, " T                                    \
                 :                                                             \
                 : "f"(llm_f16box(0.6931471806f)));                            \
    asm volatile("vfmv.v.f " T ", %0" : : "f"(llm_f16box(1.0f / 24.0f)));      \
    asm volatile("vfmul.vv " T ", " T ", " X);                                 \
    asm volatile("vfadd.vf " T ", " T ", %0"                                   \
                 :                                                             \
                 : "f"(llm_f16box(1.0f / 6.0f)));                              \
    asm volatile("vfmul.vv " T ", " T ", " X);                                 \
    asm volatile("vfadd.vf " T ", " T ", %0" : : "f"(llm_f16box(0.5f)));       \
    asm volatile("vfmul.vv " T ", " T ", " X);                                 \
    asm volatile("vfadd.vf " T ", " T ", %0" : : "f"(llm_f16box(1.0f)));       \
    asm volatile("vfmul.vv " T ", " T ", " X);                                 \
    asm volatile("vfadd.vf " T ", " T ", %0" : : "f"(llm_f16box(1.0f)));       \
    asm volatile("vadd.vx " N ", " N ", %0" : : "r"(15));                      \
    asm volatile("vsll.vi " N ", " N ", 10");                                  \
    asm volatile("vfmul.vv " X ", " T ", " N);                                 \
  } while (0)
#endif
#define G2_RECIPH(X, R, T)                                                     \
  do {                                                                         \
    asm volatile("vrsub.vx " R ", " X ", %0" : : "r"(0x7798));                 \
    asm volatile("vfmul.vv " T ", " X ", " R);                                 \
    asm volatile("vfrsub.vf " T ", " T ", %0" : : "f"(two16));                 \
    asm volatile("vfmul.vv " R ", " R ", " T);                                 \
    asm volatile("vfmul.vv " T ", " X ", " R);                                 \
    asm volatile("vfrsub.vf " T ", " T ", %0" : : "f"(two16));                 \
    asm volatile("vfmul.vv " R ", " R ", " T);                                 \
  } while (0)
// Y = SiLU(X) in FP16 arithmetic, A, R, T scratch groups (R also holds Y's
// reciprocal factor); needs one16 and two16 (boxed FP16) in scope.
#define G2_SILUH(X, Y, A, R, T)                                                \
  do {                                                                         \
    asm volatile("vfsgnjn.vv " A ", " X ", " X);                               \
    G2_EXPH(A, R, T);                                                          \
    asm volatile("vfadd.vf " A ", " A ", %0" : : "f"(one16));                  \
    G2_RECIPH(A, R, T);                                                        \
    asm volatile("vfmul.vv " Y ", " R ", " X);                                 \
  } while (0)

// One output row of the convolution, then SiLU: rows x_r .. x_r+3 sit in A,
// B, C, D (e16 m2), the taps in v24 .. v31. A is free after the first FMA,
// so x_r+4 (if next) loads into it while the other three run. The sum is
// rounded to FP16 (the convolution output), widened back, and its SiLU
// rounded to FP16 is left in v12 (e16 m2 selected on return).
#define G2_CONV_ROW(A, B, C, D, next)                                          \
  do {                                                                         \
    asm volatile("vfwmul.vv v8, " A ", v24");                                  \
    if (next)                                                                  \
      asm volatile("vle16.v " A ", (%0)" : : "r"(next) : "memory");            \
    asm volatile("vfwmacc.vv v8, " B ", v26");                                 \
    asm volatile("vfwmacc.vv v8, " C ", v28");                                 \
    asm volatile("vfwmacc.vv v8, " D ", v30");                                 \
    asm volatile("vfncvt.f.f.w v12, v8");                                      \
    G2_CONV_SILU();                                                            \
  } while (0)
#if Q38_GDN2_F16ACT
// SiLU of the rounded convolution output in v12, FP16 arithmetic.
#define G2_CONV_SILU() G2_SILUH("v12", "v12", "v14", "v16", "v18")
#else
// Widened back, SiLU in FP32, rounded to FP16 (q38_activation16 mode 1).
#define G2_CONV_SILU()                                                         \
  do {                                                                         \
    asm volatile("vfwadd.vf v8, v12, %0" : : "f"(zh));                         \
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(64));             \
    asm volatile("vfsgnjn.vv v12, v8, v8");                                    \
    G2_EXPF("v12", "v16", "v20");                                              \
    asm volatile("vfadd.vf v12, v12, %0" : : "f"(one));                        \
    G2_RECIP("v12", "v16", "v20");                                             \
    asm volatile("vfmul.vv v8, v16, v8");                                      \
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(64));             \
    asm volatile("vfncvt.f.f.w v12, v8");                                      \
  } while (0)
#endif

// Causal 4-tap convolution (reference.py:345) of the 64 channels of block
// blk for output rows r0 .. r0+n-1 (n a multiple of 4), then SiLU
// (reference.py:353; q38_activation16 mode 1: FP32, rounded to FP16): FP32
// sums of exact FP16 products, rounded to FP16. Each raw row is loaded once.
G2_INL void g2_conv_block(g2_pre_t *pre, const g2_raw_t *raw, uint32_t blk,
                          uint32_t r0, uint32_t n, __fp16 *local) {
  G2_SELF;
  const float zh = llm_f16box(0.0f), one = 1.0f;
  const float one16 = llm_f16box(1.0f), two16 = llm_f16box(2.0f);
  (void)one;
  (void)one16;
  (void)two16;
  const uint32_t c = 64u * blk;
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(64));
  asm volatile("vle16.v v24, (%0)" : : "r"(G2G(&G2->conv[0][c])) : "memory");
  asm volatile("vle16.v v26, (%0)" : : "r"(G2G(&G2->conv[1][c])) : "memory");
  asm volatile("vle16.v v28, (%0)" : : "r"(G2G(&G2->conv[2][c])) : "memory");
  asm volatile("vle16.v v30, (%0)" : : "r"(G2G(&G2->conv[3][c])) : "memory");
  asm volatile("vle16.v v0, (%0)" : : "r"(G2G(g2_raw(raw, c, r0))) : "memory");
  asm volatile("vle16.v v2, (%0)"
               :
               : "r"(G2G(g2_raw(raw, c, r0 + 1u)))
               : "memory");
  asm volatile("vle16.v v4, (%0)"
               :
               : "r"(G2G(g2_raw(raw, c, r0 + 2u)))
               : "memory");
  asm volatile("vle16.v v6, (%0)"
               :
               : "r"(G2G(g2_raw(raw, c, r0 + 3u)))
               : "memory");
  const uint32_t h = (c - 2u * Q38_KD) / Q38_VD;
  const uint32_t qq = (c - 2u * Q38_KD) % Q38_VD / G2_QW;
  for (uint32_t r = r0; r < r0 + n; ++r) {
    // x_r+4 (the next output row's newest input), none after the last row.
    const __fp16 *const next =
        r + 1u < r0 + n ? G2G(g2_raw(raw, c, r + 4u)) : (const __fp16 *)0;
    switch ((r - r0) % 4u) {
    case 0u:
      G2_CONV_ROW("v0", "v2", "v4", "v6", next);
      break;
    case 1u:
      G2_CONV_ROW("v2", "v4", "v6", "v0", next);
      break;
    case 2u:
      G2_CONV_ROW("v4", "v6", "v0", "v2", next);
      break;
    default:
      G2_CONV_ROW("v6", "v0", "v2", "v4", next);
      break;
    }
    if (local) { // q, k halves: row r - r0 of a 128-wide local block
      asm volatile("vse16.v v12, (%0)"
                   :
                   : "r"(local + (r - r0) * Q38_KD)
                   : "memory");
    } else if (c < Q38_KD) {
      asm volatile("vse16.v v12, (%0)" : : "r"(G2G(&pre->q[r][c])) : "memory");
    } else if (c < 2u * Q38_KD) {
      asm volatile("vse16.v v12, (%0)"
                   :
                   : "r"(G2G(&pre->k[r][c - Q38_KD]))
                   : "memory");
    } else {
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(G2_QW));
      asm volatile("vse16.v v12, (%0)"
                   :
                   : "r"(G2G(&pre->vq[h][qq][r][0]))
                   : "memory");
      asm volatile("vse16.v v13, (%0)"
                   :
                   : "r"(G2G(&pre->vq[h][qq + 1u][r][0]))
                   : "memory");
      asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(64));
    }
  }
}

// 64 raw FP16 gate inputs of head word w (rab rows are 16 B, 32 to a window
// row), widened to FP32 in v24 (e32, m8, vl 64 selected on return).
static void g2_gate_inputs(const g2_raw_t *raw, uint32_t w) {
  const float zh = llm_f16box(0.0f);
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(32));
  asm volatile("vlse16.v v0, (%0), %1"
               :
               : "r"(G2G(&raw->ab[0][w])), "r"(sizeof(raw->ab[0]))
               : "memory");
  asm volatile("vlse16.v v1, (%0), %1"
               :
               : "r"(G2G(&raw->ab[32][w])), "r"(sizeof(raw->ab[0]))
               : "memory");
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(G2_C));
  asm volatile("vfwadd.vf v24, v0, %0" : : "f"(zh));
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(G2_C));
}

// Gates of head h for the 64 rows: lambda = exp(-exp(A_log) softplus(a +
// dt_bias)), beta = sigmoid(b) (reference.py:331); Lambda_i = prod_{j<=i}
// lambda_j through the log, accumulated in order as the reference does.
static NOSTACK void g2_gates(g2_pre_t *pre, const g2_raw_t *raw, uint32_t h,
                             float *tmp) {
  const uint32_t head = mempool_get_group_id() * G2_H + h;
  const float rate = llm_expf(w_a_log[head]), dt = w_dt_bias[head];
  const float zero = 0.0f, one = 1.0f, two = 2.0f;
  static const float series[6] = {1.0f / 11.0f, 1.0f / 9.0f, 1.0f / 7.0f,
                                  1.0f / 5.0f,  1.0f / 3.0f, 1.0f};
  // softplus(x) = max(x, 0) + log1p(exp(-|x|)) as q38_softplus.
  g2_gate_inputs(raw, h);
  asm volatile("vfadd.vf v24, v24, %0" : : "f"(dt));
  asm volatile("vfsgnjx.vv v0, v24, v24");
  asm volatile("vfsgnjn.vv v0, v0, v0");
  G2_EXP();
  asm volatile("vse32.v v0, (%0)" : : "r"(tmp) : "memory"); // y
  asm volatile("vfadd.vf v0, v0, %0" : : "f"(two));
  llm_rvv_recip_pos_m8();
  asm volatile("vle32.v v16, (%0)" : : "r"(tmp) : "memory");
  asm volatile("vfmul.vv v0, v8, v16"); // z = y / (2 + y)
  asm volatile("vfmul.vv v8, v0, v0");  // z^2
  asm volatile("vfmv.v.f v16, %0" : : "f"(1.0f / 13.0f));
  for (uint32_t i = 0u; i < 6u; ++i) {
    asm volatile("vfmul.vv v16, v16, v8");
    asm volatile("vfadd.vf v16, v16, %0" : : "f"(series[i]));
  }
  asm volatile("vfmul.vv v16, v16, v0");
  asm volatile("vfadd.vv v16, v16, v16");
  asm volatile("vfmax.vf v0, v24, %0" : : "f"(zero));
  asm volatile("vfadd.vv v0, v0, v16");
  asm volatile("vfmul.vf v0, v0, %0" : : "f"(rate));
  asm volatile("vse32.v v0, (%0)" : : "r"(tmp) : "memory");
  asm volatile("fence" ::: "memory");
  float log_cum = 0.0f;
  for (uint32_t r = 0u; r < G2_C; ++r) {
    log_cum -= tmp[r];
    tmp[r] = log_cum;
  }
  asm volatile("fence" ::: "memory");
  float *const lam = G2G(&pre->lam[h][0]), *const ilam = G2G(&pre->ilam[h][0]);
  asm volatile("vle32.v v0, (%0)" : : "r"(tmp) : "memory");
  G2_EXP();
  asm volatile("vse32.v v0, (%0)" : : "r"(lam) : "memory");
  asm volatile("vle32.v v0, (%0)" : : "r"(tmp) : "memory");
  asm volatile("vfsgnjn.vv v0, v0, v0");
  G2_EXP();
  asm volatile("vse32.v v0, (%0)" : : "r"(ilam) : "memory");
  // beta, and -beta Lambda for T's narrowing.
  g2_gate_inputs(raw, 3u + h);
  asm volatile("vfsgnjn.vv v0, v24, v24");
  G2_EXP();
  asm volatile("vfadd.vf v0, v0, %0" : : "f"(one));
  llm_rvv_recip_pos_m8();
  asm volatile("vse32.v v8, (%0)" : : "r"(G2G(&pre->beta[h][0])) : "memory");
  asm volatile("vle32.v v16, (%0)" : : "r"(lam) : "memory");
  asm volatile("vfmul.vv v8, v8, v16");
  asm volatile("vfsgnjn.vv v8, v8, v8");
  asm volatile("vse32.v v8, (%0)" : : "r"(G2G(&pre->nbl[h][0])) : "memory");
}

// An FP32 factor as two boxed FP16 scalars, hi + lo: x hi + x lo in FP16
// arithmetic is x f within about one FP16 ulp, at the FP16 rate and with no
// widening or narrowing (the state may lose precision for speed).
G2_INL void g2_split(float f, float *hi, float *lo) {
  float h, back, l;
  asm volatile("fcvt.h.s %0, %1" : "=f"(h) : "f"(f));
  asm volatile("fcvt.s.h %0, %1" : "=f"(back) : "f"(h));
  asm volatile("fcvt.h.s %0, %1" : "=f"(l) : "f"(f - back));
  *hi = h;
  *lo = l;
}

// Row r of q (sel 0) or k (sel 1), SiLU applied by the convolution:
// L2-normalized (reference.py:353; q also by 1/sqrt(KD)); k also gives its
// column of K^T (elements j, j+4, ... lie 8 KiB apart globally).
G2_INL void g2_qk_row(g2_pre_t *pre, uint32_t sel, uint32_t r, __fp16 *row) {
  G2_SELF;
  __fp16 *const x = sel ? G2G(&pre->k[r][0]) : G2G(&pre->q[r][0]);
  const float zero = 0.0f;
  // The convolution (with SiLU) left the row in this tile's banks.
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(Q38_KD));
  asm volatile("vle16.v v0, (%0)" : : "r"(row) : "memory");
  asm volatile("vfwmul.vv v8, v0, v0"); // FP32 squares
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(1));
  asm volatile("vfmv.s.f v16, %0" : : "f"(zero));
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(Q38_KD));
  asm volatile("vfredusum.vs v16, v8, v16");
  float sum;
  asm volatile("vfmv.f.s %0, v16" : "=f"(sum));
  float scale = llm_rsqrt(sum + 1e-6f);
  if (!sel)
    scale *= 1.0f / 11.313708499f; // 1 / sqrt(KD)
  // The scaled row in FP16 arithmetic, scale as hi + lo (g2_split).
  float hi, lo;
  g2_split(scale, &hi, &lo);
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(Q38_KD));
  asm volatile("vfmul.vf v8, v0, %0" : : "f"(hi));
  asm volatile("vfmacc.vf v8, %0, v0" : : "f"(lo));
  asm volatile("vse16.v v8, (%0)" : : "r"(x) : "memory");
  if (sel) { // Lambda q is built by tile 3's engine during the solve
    asm volatile("vse16.v v8, (%0)" : : "r"(row) : "memory");
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(Q38_KD / 4u));
    for (uint32_t j = 0u; j < 4u; ++j) {
      asm volatile("vlse16.v v0, (%0), %1" : : "r"(row + j), "r"(8) : "memory");
      asm volatile("vsse16.v v0, (%0), %1"
                   :
                   : "r"(G2G(&pre->kt[j][r])), "r"(VC_GLOBAL_ROW_BYTES)
                   : "memory");
    }
  }
}

// T's 32-block (1, 0) on the tile's engine, FP16 (gdn_chunk_check.py v3l3:
// as accurate as FP32 here): 32 x 32 blocks packed over A's rows 0 .. 31,
// which the levels before have finished with.
#define G2_PK(i) ((__fp16(*)[32]) & G2L->a32[8u * (i)][0])
enum { G2_PK_T22, G2_PK_T11, G2_PK_NA21, G2_PK_X };

// --- the solve: tile q < 3 solves head q in its own banks, tile 3 builds P
// and Lambda Q for the three heads.

// The three compute cores of this tile: a counter and an epoch in the tile's
// own banks, released by a core-raised event (fsync_raise), not the tree.
static NOSTACK void g2_tile_sync(void) {
  G2_SELF;
  volatile uint32_t *const b = G2TG(&G2L->bar[0], g2_tile());
  __sync_synchronize(); // this core's vector stores are visible
  const uint32_t epoch = b[1];
  if (__atomic_fetch_add(&b[0], 1u, __ATOMIC_ACQ_REL) == 2u) {
    b[0] = 0u;
    __sync_synchronize();
    b[1] = epoch + 1u;
    fsync_raise(0xeu); // cores 1 .. 3 of the tile
  } else {
    while (b[1] == epoch)
      mempool_wfi();
  }
}

// v8 (e32 m4) = scale * fp32(row16[0..64)) * v24 (1 / Lambda, held), then
// zero for j >= n, left in v16. Whole rows are loaded and the mask is a
// tail-undisturbed move: the model mishandles e16 loads with an odd vl > 1.
G2_INL void g2_ratio(const __fp16 *row16, float scale, uint32_t n) {
  const float zh = llm_f16box(0.0f);
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(G2_C));
  asm volatile("vle16.v v0, (%0)" : : "r"(row16) : "memory");
  asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(G2_C));
  asm volatile("vfmul.vv v8, v8, v24");
  asm volatile("vfmul.vf v8, v8, %0" : : "f"(scale));
  asm volatile("vmv.v.i v16, 0");
  if (n) {
    asm volatile("vsetvli zero, %0, e32, m4, tu, ma" : : "r"(n));
    asm volatile("vmv.v.v v16, v8");
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(G2_C));
  }
}

// Diagonal block blk of T (rows 16 blk .. +15) by forward substitution; the
// rows were zeroed with A.
G2_INL void g2_solve_block(uint32_t blk) {
  G2_SELF;
  const float one = 1.0f;
  const uint32_t q = g2_tile(), i0 = 16u * blk;
  for (uint32_t i = i0; i < i0 + 16u; ++i) {
    const float *const a = G2TG(&G2L->a32[i][i0], q);
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(16));
    asm volatile("vmv.v.i v24, 0");
#define G2_ROW_T(j) (&G2L->t32[i0 + (j)][i0])
    G2_AXPYS("vfnmsac.vf", i - i0, a, G2_ROW_T);
#undef G2_ROW_T
    asm volatile("vse32.v v24, (%0)" : : "r"(&G2L->t32[i][i0]) : "memory");
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(1));
    asm volatile("vfmv.s.f v24, %0" : : "f"(one));
    asm volatile("vse32.v v24, (%0)" : : "r"(&G2L->t32[i][i]) : "memory");
  }
}

// Row 16 (2p + 1) + rr, columns 32p .. 32p + 15: -(T_RR A_RC) T_CC.
G2_INL void g2_solve_l2(uint32_t p, uint32_t rr) {
  G2_SELF;
  const uint32_t q = g2_tile();
  const uint32_t r0 = 16u * (2u * p + 1u), c0 = 32u * p, r = r0 + rr;
  const float *const tr = G2TG(&G2L->t32[r][r0], q);
  float *const x = g2_lt();
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(16));
  asm volatile("vmv.v.i v24, 0");
#define G2_ROW_A(m) (&G2L->a32[r0 + (m)][c0])
  G2_AXPYS("vfmacc.vf", rr + 1u, tr, G2_ROW_A); // X = (T_RR A_RC)[rr]
#undef G2_ROW_A
  asm volatile("vse32.v v24, (%0)" : : "r"(x) : "memory");
  asm volatile("fence" ::: "memory");
  asm volatile("vmv.v.i v24, 0");
#define G2_ROW_T(l) (&G2L->t32[c0 + (l)][c0])
  G2_AXPYS("vfnmsac.vf", 16u, x, G2_ROW_T); // -X T_CC
#undef G2_ROW_T
  asm volatile("vse32.v v24, (%0)" : : "r"(&G2L->t32[r][c0]) : "memory");
}

// Row r (32 .. 63), columns 0 .. 31: -(T_22 A_21) T_11.
G2_INL void g2_solve_l3(uint32_t r) {
  G2_SELF;
  const uint32_t q = g2_tile();
  const float *const tr = G2TG(&G2L->t32[r][32], q);
  float *const x = g2_lt();
  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(32));
  asm volatile("vmv.v.i v24, 0");
#define G2_ROW_A(m) (&G2L->a32[32u + (m)][0])
  G2_AXPYS("vfmacc.vf", r - 31u, tr, G2_ROW_A); // X = (T_22 A_21)[r]
#undef G2_ROW_A
  asm volatile("vse32.v v24, (%0)" : : "r"(x) : "memory");
  asm volatile("fence" ::: "memory");
  asm volatile("vmv.v.i v24, 0");
#define G2_ROW_T(l) (&G2L->t32[(l)][0])
  G2_AXPYS("vfnmsac.vf", 32u, x, G2_ROW_T); // -X T_11
#undef G2_ROW_T
  asm volatile("vse32.v v24, (%0)" : : "r"(&G2L->t32[r][0]) : "memory");
}

// Row r of T to FP16 as T diag(beta) and -T diag(beta Lambda); beta and
// -beta Lambda held in v16 and v20.
G2_INL void g2_narrow_t(uint32_t h, uint32_t r) {
  G2_SELF;
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(G2_C));
  asm volatile("vle32.v v8, (%0)" : : "r"(&G2L->t32[r][0]) : "memory");
  asm volatile("vfmul.vv v0, v8, v16");
  asm volatile("vfmul.vv v8, v8, v20");
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(G2_C));
  asm volatile("vfncvt.f.f.w v24, v0");
  asm volatile("vse16.v v24, (%0)" : : "r"(G2G(&G2->tb[h][r][0])) : "memory");
  asm volatile("vfncvt.f.f.w v28, v8");
  asm volatile("vse16.v v28, (%0)" : : "r"(&G2L->tn[r][0]) : "memory");
}

// A rows i0 .. i1-1 of head h (A_ij = beta_i Lambda_i / Lambda_j (K K^T)_ij,
// j < i), T's rows zeroed; 1 / Lambda held in v24.
G2_INL void g2_a_rows(g2_pre_t *pre, uint32_t h, uint32_t i0, uint32_t i1) {
  G2_SELF;
  const float *const lam = G2G(&pre->lam[h][0]);
  const float *const beta = G2G(&pre->beta[h][0]);
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(G2_C));
  asm volatile("vle32.v v24, (%0)" : : "r"(G2G(&pre->ilam[h][0])) : "memory");
  for (uint32_t i = i0; i < i1; ++i) {
    g2_ratio(G2G(&G2->gkk[i][0]), beta[i] * lam[i], i);
    asm volatile("vse32.v v16, (%0)" : : "r"(&G2L->a32[i][0]) : "memory");
    asm volatile("vmv.v.i v8, 0");
    asm volatile("vse32.v v8, (%0)" : : "r"(&G2L->t32[i][0]) : "memory");
  }
}

// Rows of T to FP16 (g2_narrow_t), beta and -beta Lambda loaded first.
G2_INL void g2_narrow_rows(g2_pre_t *pre, uint32_t h, uint32_t r0, uint32_t r1,
                           uint32_t step) {
  G2_SELF;
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(G2_C));
  asm volatile("vle32.v v16, (%0)" : : "r"(G2G(&pre->beta[h][0])) : "memory");
  asm volatile("vle32.v v20, (%0)" : : "r"(G2G(&pre->nbl[h][0])) : "memory");
  for (uint32_t r = r0; r < r1; r += step)
    g2_narrow_t(h, r);
}

// 32 FP32 elements to FP16 (negated if neg), into a packed 64-byte row.
G2_INL void g2_pack_row(__fp16 *dst, const float *src, uint32_t neg) {
  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(32));
  asm volatile("vle32.v v0, (%0)" : : "r"(src) : "memory");
  if (neg)
    asm volatile("vfsgnjn.vv v0, v0, v0");
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(32));
  asm volatile("vfncvt.f.f.w v4, v0");
  asm volatile("vse16.v v4, (%0)" : : "r"(dst) : "memory");
}

// A packed FP16 row of 32 back into FP32 (T_21 into T's row).
G2_INL void g2_unpack_row(float *dst, const __fp16 *src) {
  const float zh = llm_f16box(0.0f);
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(32));
  asm volatile("vle16.v v0, (%0)" : : "r"(src) : "memory");
  asm volatile("vfwadd.vf v4, v0, %0" : : "f"(zh));
  asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(32));
  asm volatile("vse32.v v4, (%0)" : : "r"(dst) : "memory");
}

// T = (I + A)^-1 of head q in FP32 by the three compute cores c of tile q:
// A rows; the 16 x 16 diagonal blocks by forward substitution; T_RC =
// -T_RR A_RC T_CC for the 16-blocks (1, 0), (3, 2) and the 32-block (1, 0);
// T diag(beta), -T diag(beta Lambda) to FP16. Each core builds the A rows of
// the diagonal block it solves first, so the first level needs no barrier;
// tile barriers separate the levels after it.
//   1: core c: A rows and diagonal block c; cores 1, 2: A rows of block 3
//   2: core 0: diagonal block 3; cores 1, 2: rows of 16-block (1, 0)
//   3: rows of 16-block (3, 2); T_11, -A_21 packed to FP16
//   4: T_22 packed, the engine runs the 32-block (1, 0) while the cores narrow
//      rows 0 .. 31; then rows 32 .. 63 get T_21 and are narrowed
static NOSTACK void g2_solve_head(g2_pre_t *pre, uint32_t c) {
  G2_SELF;
  const uint32_t h = g2_tile();
  g2_a_rows(pre, h, 16u * c, 16u * c + 16u);
  g2_solve_block(c);
  if (c)
    g2_a_rows(pre, h, 40u + 8u * c, 48u + 8u * c);
  g2_tile_sync();
  if (c == 0u)
    g2_solve_block(3u);
  else
    for (uint32_t rr = c - 1u; rr < 16u; rr += 2u)
      g2_solve_l2(0u, rr);
  g2_tile_sync();
  for (uint32_t rr = c; rr < 16u; rr += 3u)
    g2_solve_l2(1u, rr);
  // FP16 T_11 and -A_21 (complete) for the engine.
  for (uint32_t r = c; r < 32u; r += 3u) {
    g2_pack_row(G2_PK(G2_PK_T11)[r], &G2L->t32[r][0], 0);
    g2_pack_row(G2_PK(G2_PK_NA21)[r], &G2L->a32[32u + r][0], 1);
  }
  volatile uint32_t *const b = G2TG(&G2L->bar[0], h);
  const uint32_t done = b[3];
  g2_tile_sync();
  for (uint32_t r = c; r < 32u; r += 3u) // T_22, complete after (3, 2)
    g2_pack_row(G2_PK(G2_PK_T22)[r], &G2L->t32[32u + r][32], 0);
  __sync_synchronize();
  __atomic_fetch_add(&b[2], 1u, __ATOMIC_RELEASE);
  fsync_raise(1u); // the tile's scheduler: X = T_22 (-A_21), T_21 = X T_11
  g2_narrow_rows(pre, h, c, 32u, 3u); // rows 0 .. 31 meanwhile
  while (b[3] == done)
    mempool_wfi();
  for (uint32_t r = 32u + c; r < G2_C; r += 3u) {
    g2_unpack_row(&G2L->t32[r][0], G2_PK(G2_PK_T22)[r - 32u]);
    g2_narrow_rows(pre, h, r, r + 1u, 1u);
  }
}

// Tile 3, core c: the diagonal of diag(Lambda_c) for the engine's Lambda Q
// (FP16 Lambda), then head c's P = (Q K^T * Lambda_i / Lambda_j)_lower.
static NOSTACK void g2_p_lq(g2_pre_t *pre, uint32_t h) {
  G2_SELF;
  const float *const lam = G2G(&pre->lam[h][0]);
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(G2_C));
  asm volatile("vle32.v v8, (%0)" : : "r"(lam) : "memory");
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(G2_C));
  asm volatile("vfncvt.f.f.w v0, v8");
  asm volatile("vsse16.v v0, (%0), %1"
               :
               : "r"(&G2L->dl[h][0][0]), "r"((G2_C + 1u) * 2u)
               : "memory");
  __sync_synchronize();
  volatile uint32_t *const b = G2TG(&G2L->bar[0], g2_tile());
  __atomic_fetch_add(&b[2], 1u, __ATOMIC_RELEASE);
  fsync_raise(1u); // the tile's scheduler
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(G2_C));
  asm volatile("vle32.v v24, (%0)" : : "r"(G2G(&pre->ilam[h][0])) : "memory");
  for (uint32_t i = 0u; i < G2_C; ++i) {
    g2_ratio(G2G(&G2->gqk[i][0]), lam[i], i + 1u);
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(G2_C));
    asm volatile("vfncvt.f.f.w v0, v16");
    asm volatile("vse16.v v0, (%0)" : : "r"(G2G(&G2->p[h][i][0])) : "memory");
  }
}

// Gated RMSNorm of context row r of head h (reference.py:380), assembled from
// the four state quarters, into this tile's gated rows. FP32 throughout:
// out = x * inv * (gamma * SiLU(z)), rounded once (the reference rounds after
// the norm, the scale and the gate: within an FP16 ulp or two).
G2_INL void g2_gate_row(uint32_t b, uint32_t h, uint32_t r, uint32_t q) {
  G2_SELF;
  const float zero = 0.0f, one = 1.0f, zh = llm_f16box(0.0f);
  __fp16 *const z = &G2L->gz[b][r % (G2_C / 4u)][h][0];
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(Q38_VD));
  asm volatile("vle16.v v0, (%0)" : : "r"(z) : "memory");
#if Q38_GDN2_F16ACT
  // SiLU(z) in FP16 arithmetic, widened into v24.
  const float one16 = llm_f16box(1.0f), two16 = llm_f16box(2.0f);
  G2_SILUH("v0", "v8", "v4", "v8", "v12");
  asm volatile("vfwadd.vf v24, v8, %0" : : "f"(zh));
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(Q38_VD));
  (void)one;
#else
  asm volatile("vfwadd.vf v24, v0, %0" : : "f"(zh));
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(Q38_VD));
  asm volatile("vfsgnjn.vv v8, v24, v24");
  G2_EXPF("v8", "v16", "v0");
  asm volatile("vfadd.vf v8, v8, %0" : : "f"(one));
  G2_RECIP("v8", "v0", "v16");
#endif
  // v8 and v16 are free: the gamma row and the context quarters load while
  // the SiLU completes.
  asm volatile("vle32.v v16, (%0)" : : "r"(&G2L->gamma32[0]) : "memory");
  // The four quarters of the context row: this tile's from its own banks.
  const __fp16 *const o = &G2L->o[h][r][0];
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(G2_QW));
  asm volatile("vle16.v v8, (%0)"
               :
               : "r"(q == 0u ? o : G2TG(o, 0u))
               : "memory");
  asm volatile("vle16.v v9, (%0)"
               :
               : "r"(q == 1u ? o : G2TG(o, 1u))
               : "memory");
  asm volatile("vle16.v v10, (%0)"
               :
               : "r"(q == 2u ? o : G2TG(o, 2u))
               : "memory");
  asm volatile("vle16.v v11, (%0)"
               :
               : "r"(q == 3u ? o : G2TG(o, 3u))
               : "memory");
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(Q38_VD));
#if Q38_GDN2_F16ACT
  asm volatile("vfmul.vv v0, v24, v16"); // gamma SiLU(z)
#else
  asm volatile("vfmul.vv v0, v0, v24"); // SiLU(z)
  asm volatile("vfmul.vv v0, v0, v16"); // gamma SiLU(z)
#endif
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(Q38_VD));
  asm volatile("vfwadd.vf v24, v8, %0" : : "f"(zh)); // x
  asm volatile("vfwmul.vv v16, v8, v8");             // x^2
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(1));
  asm volatile("vfmv.s.f v8, %0" : : "f"(zero));
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(Q38_VD));
  asm volatile("vfredusum.vs v8, v16, v8");
  asm volatile("vfmul.vv v24, v24, v0"); // x gamma SiLU(z), during the sum
  float sum;
  asm volatile("vfmv.f.s %0, v8" : "=f"(sum));
  const float inverse = llm_rsqrt(sum * (1.0f / (float)Q38_VD) + 1e-6f);
  asm volatile("vfmul.vf v24, v24, %0" : : "f"(inverse));
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(Q38_VD));
  asm volatile("vfncvt.f.f.w v0, v24");
  asm volatile("vse16.v v0, (%0)" : : "r"(z) : "memory");
}

// This tile's quarter of U of head h times Lambda_last / Lambda_r, for the
// state update: per-row hi / lo factors in v16..v19 / v20..v23 (32 per row,
// four rows at a time), FP16 arithmetic, local banks.
G2_INL void g2_scale_u(g2_pre_t *pre, uint32_t h) {
  G2_SELF;
  const float last = G2G(&pre->lam[h][0])[G2_C - 1u];
  const float *const ilam = G2G(&pre->ilam[h][0]);
  for (uint32_t r = 0u; r < G2_C; r += 4u) {
    float hi[4], lo[4];
    for (uint32_t i = 0u; i < 4u; ++i)
      g2_split(last * ilam[r + i], &hi[i], &lo[i]);
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(G2_QW));
    asm volatile("vfmv.v.f v16, %0" : : "f"(hi[0]));
    asm volatile("vfmv.v.f v17, %0" : : "f"(hi[1]));
    asm volatile("vfmv.v.f v18, %0" : : "f"(hi[2]));
    asm volatile("vfmv.v.f v19, %0" : : "f"(hi[3]));
    asm volatile("vfmv.v.f v20, %0" : : "f"(lo[0]));
    asm volatile("vfmv.v.f v21, %0" : : "f"(lo[1]));
    asm volatile("vfmv.v.f v22, %0" : : "f"(lo[2]));
    asm volatile("vfmv.v.f v23, %0" : : "f"(lo[3]));
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(4u * G2_QW));
    __fp16 *const u = &G2L->u[h][r][0];
    asm volatile("vle16.v v0, (%0)" : : "r"(u) : "memory");
    asm volatile("vfmul.vv v8, v0, v16");
    asm volatile("vfmacc.vv v8, v0, v20");
    asm volatile("vse16.v v8, (%0)" : : "r"(u) : "memory");
  }
}

// x[0..n) *= f (FP16 in place, f as hi + lo; contiguous, n % 256 == 0).
G2_INL void g2_scale16_split(__fp16 *x, uint32_t n, float hi, float lo) {
  asm volatile("vsetvli zero, %0, e16, m8, ta, ma" : : "r"(256));
  for (uint32_t i = 0u; i < n; i += 256u) {
    asm volatile("vle16.v v0, (%0)" : : "r"(x + i) : "memory");
    asm volatile("vfmul.vf v8, v0, %0" : : "f"(hi));
    asm volatile("vfmacc.vf v8, %0, v0" : : "f"(lo));
    asm volatile("vse16.v v8, (%0)" : : "r"(x + i) : "memory");
  }
}

// --- tile scheduler: RedMulE
// --------------------------------------------------

// K K^T (tile 0) and Q K^T (tile 1).
// K K^T on tiles 0, 1 and Q K^T on tiles 2, 3, 32 rows each (a full array
// height); not waited.
static NOSTACK uint8_t g2_rm_gram(g2_pre_t *pre) {
  G2_SELF;
  const uint32_t q = g2_tile(), r = q % 2u * (G2_C / 2u);
  return q < 2u ? g2_job(&pre->k[r][0], pre->kt, &G2->gkk[r][0], G2_C / 2u,
                         Q38_KD, G2_C, 0u)
                : g2_job(&pre->q[r][0], pre->kt, &G2->gqk[r][0], G2_C / 2u,
                         Q38_KD, G2_C, 0u);
}

// Tiles 0 .. 2: T's 32-block (1, 0) once the compute cores have packed the
// operands: X = T_22 (-A_21), then T_21 = X T_11 over T_22's block, and the
// result epoch is raised to the compute cores.
static NOSTACK void g2_rm_l3(void) {
  G2_SELF;
  volatile uint32_t *const b = G2TG(&G2L->bar[0], g2_tile());
  while (b[2] < 3u)
    mempool_wfi();
  b[2] = 0u;
  g2_job(G2_PK(G2_PK_T22), G2_PK(G2_PK_NA21), G2_PK(G2_PK_X), 32u, 32u, 32u,
         0u);
  g2_job_wait(g2_job(G2_PK(G2_PK_X), G2_PK(G2_PK_T11), G2_PK(G2_PK_T22), 32u,
                     32u, 32u, 0u));
  __sync_synchronize();
  b[3] = b[3] + 1u;
  fsync_raise(0xeu);
}

// Tile 3: Lambda_h Q = diag(Lambda_h) Q on the engine, once the compute cores
// have written the diagonals (a counter in the tile's banks, raised to this
// core). Waited before barrier 3.
static NOSTACK uint8_t g2_rm_lq(g2_pre_t *pre) {
  G2_SELF;
  volatile uint32_t *const b = G2TG(&G2L->bar[0], g2_tile());
  while (b[2] < G2_H)
    mempool_wfi();
  b[2] = 0u;
  uint8_t last = 0u;
  for (uint32_t h = 0u; h < G2_H; ++h)
    last = g2_job(G2L->dl[h], pre->q, G2->lq[h], G2_C, G2_C, Q38_KD, 0u);
  return last;
}

// U~ = (T diag beta) V (each tile: its quarter of the three heads); W_h =
// -(T diag(beta Lambda)) K (tile h).
static NOSTACK void g2_rm_wu(g2_pre_t *pre) {
  G2_SELF;
  const uint32_t q = g2_tile();
  uint8_t last = 0u;
  for (uint32_t h = 0u; h < G2_H; ++h)
    last = g2_job(G2->tb[h], pre->vq[h][q], G2L->u[h], G2_C, G2_C, G2_QW, 0u);
  if (q < G2_H)
    last = g2_job(G2L->tn, pre->k, G2->w[q], G2_C, G2_C, Q38_KD, 0u);
  g2_job_wait(last);
}

// U = U~ + W S, O = Lambda Q S + P U (state-dependent; the state is old).
static NOSTACK void g2_rm_uo(void) {
  G2_SELF;
  uint8_t last = 0u;
  for (uint32_t h = 0u; h < G2_H; ++h) {
    g2_job(G2->w[h], G2L->s[h], G2L->u[h], G2_C, Q38_KD, G2_QW, 1u);
    g2_job(G2->lq[h], G2L->s[h], G2L->o[h], G2_C, Q38_KD, G2_QW, 0u);
    last = g2_job(G2->p[h], G2L->u[h], G2L->o[h], G2_C, G2_C, G2_QW, 1u);
  }
  g2_job_wait(last);
}

// S += K^T (diag(Lambda_last / Lambda) U), after Spatz scaled S by
// Lambda_last and U's rows.
static NOSTACK uint8_t g2_rm_state(g2_pre_t *pre) {
  G2_SELF;
  uint8_t last = 0u;
  for (uint32_t h = 0u; h < G2_H; ++h)
    last = g2_job(pre->kt, G2L->u[h], G2L->s[h], Q38_KD, G2_C, G2_QW, 1u);
  return last;
}

// --- phases
// -------------------------------------------------------------------

static NOSTACK void gdn2_layout(void) {
  if (mempool_get_core_id() != 0u)
    return;
  // The L2 GEMM engine's arena (f2_layout), dead between the projections.
  const uint32_t p =
      f2_align(Q38_MIN((uint32_t)&stage, (uint32_t)&weights[0][0]));
  const uint32_t rows = (sizeof(g2_group_t) + VC_ROW_BYTES - 1u) / VC_ROW_BYTES;
  const uint32_t end = p + rows * VC_GLOBAL_ROW_BYTES;
  if (end > (uint32_t)&__heap_end)
    q38_fatal("[qwen38_fatal] gdn2 over L1=", end - (uint32_t)&__heap_end);
#if Q38_X8_PIPE && Q38_SAMPLED_CHECKS && !DOMAIN_L2_HOME
  // The lowest scratch row must clear the fixture (gold_sample_recurrent is
  // its last tensor). Not with DOMAIN_L2_HOME: its folded DMA aliases the
  // fixture anyway (timing-only, values garbage by design).
  const uint32_t low = (uint32_t)X8_ACT_L2(mempool_get_domain_id());
  if (low < (uint32_t)gold_sample_recurrent + 0x10000u)
    q38_fatal("[qwen38_fatal] x8 scratch over the fixture=", low);
#endif
}

// In-projection of all rows: [qkv | z | a b] = RMSNorm(x) W_in, to P in L2.
static NOSTACK void gdn2_project(G2_WIN_VOID) {
  G2_WIN(0u, G2_PROJECT_ROWS);
  const uint32_t core = mempool_get_core_id();
  if (core == 0u) {
    f2.peak = 0u;
    // Whole panels: with NT = 64 the 96 a|b columns take two, and P's row
    // padding (G2_PSTRIDE) holds the extra 32 (unchanged for NT = 32).
    f2.pair = 0u, f2.k_up = Q38_D, f2.n_up = (G2_PCOLS + NT - 1u) / NT * NT;
    f2.rows_up = Q38_FFN_ROWS, f2.nseg = 3u;
    f2.seg_w[0] = w_qkv, f2.seg_panels[0] = Q38_QKVDIM / NT;
    f2.seg_w[1] = w_z, f2.seg_panels[1] = Q38_GDIM / NT;
    f2.seg_w[2] = w_ab, f2.seg_panels[2] = (2u * Q38_NV + NT - 1u) / NT;
#if Q38_L2_TILED
    f2.g_out = G2_P_L2, f2.g_kc = G2_PSLICE, f2.g_rows = Q38_R;
    f2.g_main = G2_PMAIN, f2.g_kc_last = G2_PSTRIDE - G2_PMAIN;
#else
    f2.g_out = G2_P_L2, f2.g_kc = G2_PSTRIDE, f2.g_rows = Q38_R;
#endif
    // f2 sits in remote L1 and f2_layout reloads it at once on this core:
    // a fence first (GVSoC L1 reload-after-store).
    __sync_synchronize();
  }
  if (mempool_is_group_coordinator())
    f2_down[mempool_get_group_id()].weight_valid = 0u;
#pragma clang loop unroll(disable)
  for (uint32_t first = w0; first < w1; first += Q38_FFN_ROWS) {
    f2_layout(0u);
    f2_reset_progress();
#if Q38_F2_NORM_HOME
    f2_home_setup(gamma, first, w0);
#endif
    if (core == 0u)
      f2.first = first;
    f2_sync_all();
#if Q38_F2_NORM_HOME
    f2_norm_batch(X8_IN, w_input_norm, gamma, first, w0);
#else
    f2_transfer(f2.residual, X8_IN + first * Q38_D,
                Q38_FFN_ROWS * Q38_D * sizeof(__fp16));
    norm_rows_sliced(f2.residual, w_input_norm, f2.input, Q38_FFN_ROWS, gamma);
#endif
    f2_execute();
  }
}

// Out-projection with the residual: mixer = x + O W_out, to result_mixer.
static NOSTACK void gdn2_output(G2_WIN_VOID) {
  G2_WIN(0u, G2_OUT_ROWS);
  const uint32_t core = mempool_get_core_id();
  _Static_assert(G2_OUT_ROWS % Q38_FFN_DN_ROWS == 0,
                 "Out-projection rows are whole down batches");
  if (core == 0u) {
    f2.k_dn = G2_OK, f2.k_full = Q38_GDIM, f2.n_dn = Q38_D;
    f2.rows_dn = Q38_FFN_DN_ROWS, f2.slices = Q38_GDN2_OUT_SLICES;
#if Q38_L2_TILED
    f2.k_last = 0u;
#endif
    f2.dn_lag = Q38_GDN2_OUT_LAG;
    f2.end = w1;
    f2.w_dn = w_out, f2.dn_in = G2_O_L2, f2.dn_rows = Q38_R;
    f2.res_src = X8_IN, f2.dn_out = X8_MIX;
#if Q38_FFN_STAMPS
    f2_nst = 0u; // .l1 powers up undefined
#endif
    // f2 sits in remote L1 and f2_layout reloads it at once on this core:
    // a fence first (GVSoC L1 reload-after-store).
    __sync_synchronize();
  }
  if (mempool_is_group_coordinator()) {
    const uint32_t g = mempool_get_group_id();
    f2_down[g].weight_valid = f2_down[g].weight_pending = 0u;
    f2_down[g].x_pending = 0u;
  }
#pragma clang loop unroll(disable)
  for (uint32_t first = w0; first < w1; first += Q38_FFN_DN_ROWS) {
    f2_layout(1u);
    f2_reset_progress();
    if (core == 0u)
      f2.first = first;
    f2_sync_all();
#pragma clang loop unroll(disable)
    for (uint32_t slice = 0u; slice < Q38_GDN2_OUT_SLICES; ++slice) {
      if (core == 0u) {
        f2.slice = slice;
        f2.weight_epoch =
            (first - w0) / Q38_FFN_DN_ROWS * Q38_GDN2_OUT_SLICES + slice;
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

// The scheduler and compute halves of the chunk loop run the same sequence of
// group barriers. They are separate, non-inlined functions so that no FP or
// vector instruction of the compute half (hoisted constants included) can
// reach a scheduler, which is rv32ima; ctrlcheck.py checks g2_sched_loop.
#define G2_CTRL __attribute__((noinline, section(".text.ctrl")))
#define G2_CHUNKS ((w1 - w0) / G2_C) // of the window

// Tile schedulers: RedMulE jobs; the group coordinator also moves the data,
// one chunk ahead. Chunk c's operands sit in pre[c & 1], its raw rows in
// raw[c & 1]; the compute cores convolve chunk c + 1 during U~, W, U and O.
static G2_CTRL void g2_sched_loop(uint32_t coord G2_WIN_PARAMS) {
  G2_SELF;
  G2_WIN(0u, G2_TOKENS);
  dma_id_t load = 0u, zload = 0u, store = 0u;
  if (coord) {
    g2_load_params();
#if Q38_X8_BATCH
    if (w0) // the state the previous batch left (no zeroing then)
      g2_state_dma(0u);
#endif
    g2_load(w0, &G2->raw[0]);
    g2_dma_wait(g2_load_z(w0));
  }
  g2_sync(); // 0 raw rows of chunk 0
  g2_sync(); // convolution and gates of chunk 0
  // The engine queue orders the jobs: chunk c's Gram (issued once its q, k,
  // K^T exist at barrier 4 of c-1, done under the gate rows, waited before
  // barrier 5), then chunk c-1's state update, which runs under the solve and
  // ends before chunk c's U~, W, U, O read the state.
  // Raw rows go one chunk ahead of their convolution: chunk c+2's are
  // requested under chunk c's gate rows (raw[c & 1] is free from then on).
  if (coord && G2_CHUNKS > 1u)
    load = g2_load(w0 + G2_C, &G2->raw[1]);
  g2_job_wait(g2_rm_gram(&G2->pre[0]));
  g2_sync(); // K K^T, Q K^T of chunk 0
  uint8_t state_job = 0u;
  for (uint32_t c = 0u; c < G2_CHUNKS; ++c) {
    g2_pre_t *const pre = &G2->pre[c & 1u];
    const dma_id_t zcur = zload; // chunk c's z rows (issued in chunk c-1)
    // The coordinator programs chunk c-1's exports during the solve, where
    // it has nothing else to do until the level-3 jobs.
    if (coord && c)
      store = g2_store(w0 + (c - 1u) * G2_C);
    uint8_t lq_job = 0u;
    if (g2_tile() == 3u)
      lq_job = g2_rm_lq(pre);
    else
      g2_rm_l3();
    if (coord) {
      // Chunk c-1's contexts and gated rows have left O and its gated-row
      // buffer: the next chunk's z rows may land there, U, O may follow.
      g2_dma_wait(store);
      if (c + 1u < G2_CHUNKS)
        zload = g2_load_z(w0 + (c + 1u) * G2_C);
      g2_dma_wait(load); // chunk c+1's raw rows, for its convolution
    }
    if (lq_job)
      g2_job_wait(lq_job);
    g2_sync(); // 3 T diag(beta), -T diag(beta Lambda) (tiles 0 .. 2), P,
               //   Lambda Q (tile 3)
    g2_rm_wu(pre);
    g2_sched_sync(); // U~, W (schedulers only)
    g2_rm_uo();
    if (coord)
      g2_dma_wait(zcur); // chunk c's z rows, for its gate rows
    g2_sync();           // 4 U, O; next chunk's convolution and gates
    const uint8_t gram_job =
        c + 1u < G2_CHUNKS ? g2_rm_gram(&G2->pre[(c + 1u) & 1u]) : 0u;
    if (coord && c + 2u < G2_CHUNKS)
      load = g2_load(w0 + (c + 2u) * G2_C, &G2->raw[c & 1u]);
    if (gram_job)
      g2_job_wait(gram_job);
    g2_sync(); // 5 gated rows, scaled state and U; next chunk's K K^T, Q K^T
    state_job = g2_rm_state(pre);
  }
  if (state_job)
    g2_job_wait(state_job);
  g2_sync();
  if (coord) {
    g2_dma_wait(g2_store(w0 + (G2_CHUNKS - 1u) * G2_C));
    if (!Q38_GDN2_DEBUG_TOKENS && w1 == Q38_R)
      g2_store_conv(&G2->raw[(G2_CHUNKS - 1u) & 1u]);
#if Q38_X8_BATCH
    if (w1 != Q38_R) // the next batch's initial state
      g2_dma_wait(g2_state_dma(1u));
#endif
  }
  g2_sync(); // the raw rows are free
  g2_sync(); // the FP32 state, staged over them
  if (coord && w1 == G2_TOKENS) {
    const float(*const fs)[Q38_VD] = (const float(*)[Q38_VD]) & G2->raw[0];
    dma_id_t id = 0u;
    for (uint32_t h = 0u; h < G2_H; ++h) {
      const uint32_t head = mempool_get_group_id() * G2_H + h;
#if Q38_L2_TILED
      // One window row per 512 B state row: 16 rows fill an 8 KiB L2 row,
      // so planes of 16 step whole rows on both sides.
      _Static_assert(sizeof(fs[0]) == VC_ROW_BYTES && Q38_KD % 16u == 0u,
                     "State rows are window rows, 16 to an L2 row");
      id = dma_local_memcpy_3d_nonblocking(
          0u, Q38_RES_RECURRENT + head * Q38_KD * Q38_VD,
          G2G(&fs[h * Q38_KD][0]), VC_ROW_BYTES, VC_ROW_BYTES,
          VC_GLOBAL_ROW_BYTES, 16u, VC_ROW_BYTES, VC_GLOBAL_ROW_BYTES,
          Q38_KD / 16u);
#else
      id = g2_dma(fs[h * Q38_KD], sizeof(fs[0]),
                  Q38_RES_RECURRENT + head * Q38_KD * Q38_VD, sizeof(fs[0]),
                  sizeof(fs[0]), Q38_KD, 1u);
#endif
    }
    g2_dma_wait(id);
  }
}

// Gates and convolution of a chunk: three gate tasks, then 10 channel
// blocks x 8 row segments.
// One chunk's operands, as tasks taken from a group counter (the cost per
// task differs): 16 q / k tasks of 8 full rows (both 64-channel halves into
// this core's block in its tile, then SiLU'd rows L2-normalized, K^T), the
// three gate tasks, then 48 V tasks of 64 channels x 8 rows.
#define G2_TASKS (16u + G2_H + 48u)
static NOSTACK void g2_pre_chunk(g2_pre_t *pre, const g2_raw_t *raw,
                                 uint32_t cid, float *tmp, uint32_t b) {
  G2_SELF;
  // A 1 KiB block per core over T's dead rows 32 .. 63 (tile 3: past the
  // diagonals of diag(Lambda)), free between solves.
  __fp16 *const blk = (__fp16 *)&G2L->t32[32][0] + cid % 3u * 8u * Q38_KD;
  uint32_t *const next = G2G(&G2->ntask[b][0]);
  for (uint32_t t;
       (t = __atomic_fetch_add(next, 1u, __ATOMIC_RELAXED)) < G2_TASKS;) {
    if (t < 16u) {
      const uint32_t sel = t / 8u, r0 = t % 8u * 8u;
      g2_conv_block(pre, raw, 2u * sel, r0, 8u, blk);
      g2_conv_block(pre, raw, 2u * sel + 1u, r0, 8u, blk + 64u);
      for (uint32_t i = 0u; i < 8u; ++i)
        g2_qk_row(pre, sel, r0 + i, blk + i * Q38_KD);
    } else if (t < 16u + G2_H) {
      g2_gates(pre, raw, t - 16u, tmp);
    } else {
      const uint32_t v = t - 16u - G2_H;
      g2_conv_block(pre, raw, 4u + v / 8u, v % 8u * 8u, 8u, (__fp16 *)0);
    }
  }
}

// Per-task loops of the compute cores (12 per group, cid 0 .. 11).
// Core c of tile q: head c's gated rows 16q .. 16q+15, then head c's
// quarter q of U and of the state, all in this tile's banks.
static NOSTACK void g2_task_gate(g2_pre_t *pre, uint32_t b, uint32_t cid) {
  G2_SELF;
  const uint32_t q = cid / 3u, h = cid % 3u;
  for (uint32_t r = q * (G2_C / 4u); r < (q + 1u) * (G2_C / 4u); ++r)
    g2_gate_row(b, h, r, q);
  g2_scale_u(pre, h);
  float hi, lo;
  g2_split(G2G(&pre->lam[h][0])[G2_C - 1u], &hi, &lo);
  g2_scale16_split(&G2L->s[h][0][0], Q38_KD * G2_QW, hi, lo);
}

#ifndef Q38_GDN2_MEMPROBE
#define Q38_GDN2_MEMPROBE 0
#endif
#if Q38_GDN2_MEMPROBE
// Debug: Spatz load/store cost, tile window vs group window (over W, which
// is dead until the first U~, W jobs). Core 1 of tile 0, group 0 only.
static NOSTACK void g2_memprobe(void) {
  G2_SELF;
  const uint32_t gw = (uint32_t)&G2->w[0][0][0];
  float *const tw =
      (float *)(VC_TILE_WINDOW_BASE +
                (gw - VC_WINDOW_BASE_0) / VC_ROW_BYTES * VC_TILE_ROW_BYTES);
  // bytes 256 .. 511 of the window row: tiles 2 and 3, both remote
  float *const gg = (float *)g2_glob((const void *)gw) + 64u;
  for (uint32_t mode = 0u; mode < 12u; ++mode) {
    float *const b = mode & 1u ? gg : tw;
    float x;
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(64));
    asm volatile("vmv.v.i v8, 0");
    asm volatile("fence" ::: "memory");
    const uint32_t t0 = mempool_get_timer();
    for (uint32_t i = 0u; i < 16u; ++i) {
      // group window: rows 512 B apart in the window, 8 KiB apart globally
      float *const a =
          mode & 1u ? b + i * (VC_GLOBAL_ROW_BYTES / 4u) : b + i * 64u;
      if (mode < 2u) { // 16 independent loads
        asm volatile("vle32.v v0, (%0)" : : "r"(a) : "memory");
      } else if (mode < 4u) { // 16 dependent loads (latency)
        asm volatile("vle32.v v0, (%0)" : : "r"(a) : "memory");
        asm volatile("vfmv.f.s %0, v0" : "=f"(x));
      } else if (mode < 6u) { // 16 stores
        asm volatile("vse32.v v8, (%0)" : : "r"(a) : "memory");
      } else if (mode < 8u) { // load + store of the same rows
        asm volatile("vle32.v v0, (%0)" : : "r"(a) : "memory");
        asm volatile("vse32.v v0, (%0)" : : "r"(a) : "memory");
      } else if (mode < 10u) { // load + independent FMA (overlap?)
        asm volatile("vle32.v v0, (%0)" : : "r"(a) : "memory");
        asm volatile("vfmul.vv v16, v8, v8");
      } else if (mode == 10u) { // FMA only
        asm volatile("vfmul.vv v16, v8, v8");
      } else { // store + independent FMA
        asm volatile("vse32.v v8, (%0)" : : "r"(b + i * 64u) : "memory");
        asm volatile("vfmul.vv v16, v4, v4");
      }
    }
    asm volatile("vfmv.f.s %0, v0" : "=f"(x));
    asm volatile("fence" ::: "memory");
    const uint32_t t1 = mempool_get_timer();
    printf("[g2_memprobe] mode=%u %s cycles=%u\n", mode,
           mode & 1u ? "group" : "tile", t1 - t0);
  }
}
#endif

// Compute cores (Spatz): convolution, gates, operands, solve, gated norm.
static NOSTACK void g2_compute_loop(uint32_t cid G2_WIN_PARAMS) {
  G2_SELF;
  G2_WIN(0u, G2_TOKENS);
#if Q38_GDN2_MEMPROBE
  if (cid == 0u && mempool_get_group_id() == 0u)
    g2_memprobe();
#endif
  float *const tmp = G2G(&G2->tmp[cid][0]);
  // Zero the first chunk's history rows and the state. A later window's
  // come from P (g2_load) and from the previous window (g2_state_dma).
  if (!w0) {
    for (uint32_t r = cid; r < 3u * (1u + G2_H); r += 12u)
      if (r < 3u)
        g2_zero16(G2G(&G2->raw[0].qk[r][0]), 2u * Q38_KD);
      else
        g2_zero16(G2G(&G2->raw[0].v[(r - 3u) / 3u][(r - 3u) % 3u][0]), Q38_VD);
    g2_zero16(&G2L->s[cid % 3u][0][0], Q38_KD * G2_QW);
  }
  if (cid == 0u)
    *G2G(&G2->ntask[0][0]) = 0u;
  if (cid % 3u == 0u) { // .l1 powers up undefined
    volatile uint32_t *const b = G2TG(&G2L->bar[0], g2_tile());
    b[0] = b[1] = b[2] = b[3] = 0u;
  }
  if (cid >= 9u) // tile 3: the zero background of diag(Lambda_h)
    g2_zero16(&G2L->dl[cid % 3u][0][0], G2_C * G2_C);
  g2_sync();
  if (cid % 3u == 0u) { // one core per tile fills its tile's copy
    const float zh = llm_f16box(0.0f);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(Q38_VD));
    asm volatile("vle16.v v0, (%0)" : : "r"(G2G(&G2->gamma[0])) : "memory");
    asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(Q38_VD));
    asm volatile("vse32.v v8, (%0)" : : "r"(&G2L->gamma32[0]) : "memory");
  }
  g2_pre_chunk(&G2->pre[0], &G2->raw[0], cid, tmp, 0u);
  g2_sync();
  g2_sync(); // K K^T, Q K^T of chunk 0 (later chunks': before barrier 5)
  for (uint32_t c = 0u; c < G2_CHUNKS; ++c) {
    g2_pre_t *const pre = &G2->pre[c & 1u];
    if (cid == 0u) // next used after barrier 3, last used a chunk ago
      *G2G(&G2->ntask[(c + 1u) & 1u][0]) = 0u;
    if (cid < 9u)
      g2_solve_head(pre, cid % 3u);
    else
      g2_p_lq(pre, cid % 3u);
    g2_sync();
    if (c + 1u < G2_CHUNKS)
      g2_pre_chunk(&G2->pre[(c + 1u) & 1u], &G2->raw[(c + 1u) & 1u], cid, tmp,
                   (c + 1u) & 1u);
    g2_sync(); // U, O
    g2_task_gate(pre, c & 1u, cid);
    g2_sync();
  }
  g2_sync();
  g2_sync(); // the conv state has left the raw rows
  // The final state, FP32, as result_recurrent[head][KD][VD] rows over the
  // raw rows and the (dead) operands; the coordinator ships them.
  float(*const fs)[Q38_VD] = (float(*)[Q38_VD]) & G2->raw[0];
  _Static_assert(sizeof(float) * G2_H * Q38_KD * Q38_VD <=
                         sizeof(G2->raw) + sizeof(G2->pre) &&
                     offsetof(g2_group_t, pre) == sizeof(G2->raw),
                 "The FP32 state fits over the raw rows and the operands");
  // Core c of tile q: quarter q of head c's rows. Last window only.
  const float zh = llm_f16box(0.0f);
  const uint32_t q = cid / 3u, h = cid % 3u;
  for (uint32_t d = 0u; w1 == G2_TOKENS && d < Q38_KD; ++d) {
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(G2_QW));
    asm volatile("vle16.v v0, (%0)" : : "r"(&G2L->s[h][d][0]) : "memory");
    asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(G2_QW));
    asm volatile("vse32.v v8, (%0)"
                 :
                 : "r"(G2G(&fs[h * Q38_KD + d][q * G2_QW]))
                 : "memory");
  }
  g2_sync();
}

// The chunked recurrence over all tokens.
static NOSTACK void gdn2_chunks(G2_WIN_VOID) {
  const uint32_t sched = g2_is_sched();
#if Q38_GDN2_STAMPS
  if (mempool_get_core_id() == 0u)
    g2_nstamp = 0u;
#endif
  gdn2_layout();
  f2_sync_all(); // G2 is published
  // After the barrier: f2_sync_all resets every core to barrier events only,
  // and the schedulers sleep on their engine and the group DMA.
  fsync_subscribe_self(0u);
  fsync_subscribe_self(FSYNC_EVT_FSYNC | FSYNC_EVT_CORE |
                       (sched ? FSYNC_EVT_ENGINE | FSYNC_EVT_DMA : 0u));
  if (sched)
    g2_sched_loop(mempool_is_group_coordinator() G2_WIN_ARGS);
  else
    g2_compute_loop(g2_cid() G2_WIN_ARGS);
  fsync_subscribe_self(0u);
  fsync_subscribe_self(FSYNC_EVT_FSYNC);
  f2_sync_all();
}

#if !Q38_X8_BATCH
static NOSTACK void gdn2(void) {
#if Q38_GDN2_DEBUG_PLOAD
  // The loader is still writing the sidecar: wait for the preloaded rows,
  // which mkpload.py makes the last segment, to carry its sentinel.
  if (mempool_get_core_id() == 0u)
    while (*(volatile const uint32_t *)(G2_PSRC + G2_PCOLS) != 0x600DF00Du)
      mempool_wait(1000u);
#endif
  const uint32_t t0 = mempool_get_timer();
  if (!Q38_GDN2_DEBUG_SKIP_PROJ)
    gdn2_project();
  const uint32_t t1 = mempool_get_timer();
  if (!Q38_GDN2_DEBUG_OUT_ROWS)
    gdn2_chunks();
  const uint32_t t2 = mempool_get_timer();
#if Q38_GDN2_DEBUG_TOKENS
  if (mempool_get_core_id() == 0u)
    printf("[gdn2] in_proj=%u chunks=%u tokens=%u\n", t1 - t0, t2 - t1,
           Q38_GDN2_DEBUG_TOKENS);
#if Q38_GDN2_STAMPS
  if (mempool_get_core_id() == 0u)
    for (uint32_t i = 0u; i < g2_nstamp; ++i)
      printf("[gdn2_stamp] %u %u\n", i, g2_stamp[i]);
#endif
#if Q38_GDN2_DEBUG_MIXER
  gdn2_output();
  if (mempool_get_core_id() == 0u) {
    printf("[gdn2] out_proj=%u rows=%u\n", mempool_get_timer() - t2,
           G2_OUT_ROWS);
    const uint32_t rows[4] = {0u, 77u, 170u, G2_OUT_ROWS - 1u};
    for (uint32_t i = 0u; i < 4u; ++i)
      q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, result_mixer + rows[i] * Q38_D,
              result_mixer + rows[i] * Q38_D, Q38_D, 0x00000000u /* 0.0f */, 1);
  }
#endif
  if (mempool_get_core_id() == 0u) { // core 0 records DPI checks
#if Q38_GDN2_DEBUG_MIXER
#elif Q38_GDN2_DEBUG_CTX
    const uint32_t heads[3] = {0u, 1u, Q38_NV - 1u};
    for (uint32_t i = 0u; i < 3u; ++i)
      q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, G2_CTX_L2 + heads[i] * Q38_R * Q38_VD,
              G2_CTX_L2 + heads[i] * Q38_R * Q38_VD,
              Q38_GDN2_DEBUG_TOKENS * Q38_VD, 0x00000000u /* 0.0f */, 1);
#elif Q38_SAMPLED_CHECKS
    // The sampled context runs the chunk loop copied (those in the tokens run).
    q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, result_sample_context,
            result_sample_context, Q38_RUNS_CONTEXT * Q38_RUNLEN_CONTEXT,
            0x00000000u /* 0.0f */, 1);
#endif
  }
  return;
#endif
  gdn2_output();
  const uint32_t t3 = mempool_get_timer();
  if (mempool_get_core_id() == 0u) {
    g2_run.in_proj = t1 - t0;
    g2_run.chunks = t2 - t1;
    g2_run.out_proj = t3 - t2;
    printf("[gdn2] in_proj=%u chunks=%u out_proj=%u total=%u\n", t1 - t0,
           t2 - t1, t3 - t2, t3 - t0);
  }
  // The context samples were copied chunk by chunk (g2_store).
}
#endif
