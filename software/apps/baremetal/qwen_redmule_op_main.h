// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __QWEN_REDMULE_OP_MAIN_H__
#define __QWEN_REDMULE_OP_MAIN_H__

#include <stdint.h>

#include "printf.h"
#include "runtime.h"

#include "baremetal/mempool_llm_tiling.h"
#include "baremetal/mempool_qwen_redmule_op.h"

#ifndef QWEN_BENCHMARK_NAME
#error "QWEN_BENCHMARK_NAME must be defined"
#endif
#ifndef QWEN_LOGICAL_M
#error "QWEN_LOGICAL_M must be defined"
#endif
#ifndef QWEN_LOGICAL_N
#error "QWEN_LOGICAL_N must be defined"
#endif
#ifndef QWEN_LOGICAL_P
#error "QWEN_LOGICAL_P must be defined"
#endif
#ifndef QWEN_TILE_N
#error "QWEN_TILE_N must be defined"
#endif
#ifndef QWEN_DECODE
#define QWEN_DECODE 0
#endif
#ifndef QWEN_OUTPUT_CLASS
#define QWEN_OUTPUT_CLASS 0
#endif

#define QWEN_OUTPUT_DEFAULT 0
#define QWEN_OUTPUT_AB 1
#define QWEN_OUTPUT_PV 2
#define QWEN_PORT_WIDTH (REDMULE_H * (REDMULE_P + 1))

// ---------------------------------------------------------------- tile policy
//
// 0 -- the stage-1 reference shape: an 8x8 engine grid for prefill, 1x16 for
//      decode, and the two attention output classes pinned to their head
//      geometry. Every 4x4 DSE number is measured against this, so it must not
//      move; policy 0 is byte-for-byte what this header always emitted.
//
// 1 -- fill: split whatever engines the chip has as squarely as the operand
//      shape allows, then shrink until the tile fits L1. This is what a
//      256-engine part needs. At 64 engines it agrees with policy 0 on prefill
//      and takes more columns on decode, so a policy-1 pack must be compared
//      against a policy-1 pack, not against the stage-1 numbers.
//
// Select with APP_DEFINES=-DQWEN_TILE_POLICY=1.
#ifndef QWEN_TILE_POLICY
#define QWEN_TILE_POLICY 0
#endif

// Largest power of two <= x, and the largest whose square is <= x. Both are
// used inside #if, so no casts and no sizeof.
#define QWEN_P2_LE(x)                                                          \
  ((x) >= 256   ? 256                                                          \
   : (x) >= 128 ? 128                                                          \
   : (x) >= 64  ? 64                                                           \
   : (x) >= 32  ? 32                                                           \
   : (x) >= 16  ? 16                                                           \
   : (x) >= 8   ? 8                                                            \
   : (x) >= 4   ? 4                                                            \
   : (x) >= 2   ? 2                                                            \
                : 1)
#define QWEN_P2_SQRT(x)                                                        \
  ((x) >= 65536   ? 256                                                        \
   : (x) >= 16384 ? 128                                                        \
   : (x) >= 4096  ? 64                                                         \
   : (x) >= 1024  ? 32                                                         \
   : (x) >= 256   ? 16                                                         \
   : (x) >= 64    ? 8                                                          \
   : (x) >= 16    ? 4                                                          \
   : (x) >= 4     ? 2                                                          \
                  : 1)
#define QWEN_MIN(a, b) ((a) < (b) ? (a) : (b))

// How far each direction may be split. One engine shard is PORT_WIDTH rows by
// PORT_WIDTH columns, so a direction can hold at most operand/PORT_WIDTH
// groups. In decode M is the batch and is smaller than one shard, which floors
// to a single row group that pads it -- the behaviour policy 0 hard-codes.
#define QWEN_NE_R_CAP QWEN_P2_LE(QWEN_LOGICAL_M / QWEN_PORT_WIDTH)
#if QWEN_OUTPUT_CLASS == QWEN_OUTPUT_AB
#define QWEN_NE_C_CAP (64 / QWEN_PORT_WIDTH)
#elif QWEN_OUTPUT_CLASS == QWEN_OUTPUT_PV
#define QWEN_NE_C_CAP (256 / QWEN_PORT_WIDTH)
#else
#define QWEN_NE_C_CAP QWEN_P2_LE(QWEN_LOGICAL_P / QWEN_PORT_WIDTH)
#endif

// Square split of T engines, clamped to the shape, with the slack handed back
// to whichever direction can still use it. Square is not arbitrary: the operand
// traffic is 2*NE_R*NE_C*PORT_WIDTH*TILE_N whatever the split, but the resident
// tile is (NE_R + NE_C)*PORT_WIDTH*TILE_N + ..., so a square grid is the one
// that fits in the least L1.
#define QWEN_SPLIT_C0(T) QWEN_MIN(QWEN_P2_SQRT(T), QWEN_NE_C_CAP)
#define QWEN_SPLIT_R(T) QWEN_MIN((T) / QWEN_SPLIT_C0(T), QWEN_NE_R_CAP)
#define QWEN_SPLIT_C(T) QWEN_MIN((T) / QWEN_SPLIT_R(T), QWEN_NE_C_CAP)

// L1_TILE_BUDGET_BYTES carries (long) casts, which a preprocessor conditional
// may not contain. Restate it cast-free -- the preprocessor already evaluates
// in intmax_t -- and assert below that the two agree.
#define QWEN_L1_BUDGET                                                         \
  ((NUM_BANKS * L1_BANK_SIZE - NUM_CORES * SEQ_MEM_SIZE) * 9 / 10)
#define QWEN_TILE_BYTES(r, c)                                                  \
  (((r)*QWEN_PORT_WIDTH * QWEN_TILE_N + QWEN_TILE_N * (c)*QWEN_PORT_WIDTH +    \
    (r)*QWEN_PORT_WIDTH * (c)*QWEN_PORT_WIDTH) *                               \
   2)
#define QWEN_FITS(T)                                                           \
  (QWEN_TILE_BYTES(QWEN_SPLIT_R(T), QWEN_SPLIT_C(T)) <= QWEN_L1_BUDGET)

#if QWEN_TILE_POLICY == 1

// Take the whole chip, then halve until the tile is resident. Four steps is
// three more than any current operation needs.
#if QWEN_FITS(NUM_REDMULE_TILES)
#define QWEN_ENGINE_BUDGET NUM_REDMULE_TILES
#elif QWEN_FITS(NUM_REDMULE_TILES / 2)
#define QWEN_ENGINE_BUDGET (NUM_REDMULE_TILES / 2)
#elif QWEN_FITS(NUM_REDMULE_TILES / 4)
#define QWEN_ENGINE_BUDGET (NUM_REDMULE_TILES / 4)
#elif QWEN_FITS(NUM_REDMULE_TILES / 8)
#define QWEN_ENGINE_BUDGET (NUM_REDMULE_TILES / 8)
#else
#define QWEN_ENGINE_BUDGET (NUM_REDMULE_TILES / 16)
#endif

#define QWEN_NE_R QWEN_SPLIT_R(QWEN_ENGINE_BUDGET)
#define QWEN_NE_C QWEN_SPLIT_C(QWEN_ENGINE_BUDGET)

#else // stage-1 reference shape

#if QWEN_DECODE
#define QWEN_NE_R 1
#if QWEN_OUTPUT_CLASS == QWEN_OUTPUT_AB
#define QWEN_NE_C (64 / QWEN_PORT_WIDTH)
#elif QWEN_OUTPUT_CLASS == QWEN_OUTPUT_PV
#define QWEN_NE_C (256 / QWEN_PORT_WIDTH)
#else
#define QWEN_NE_C 16
#endif
#else
#define QWEN_NE_R 8
#if QWEN_OUTPUT_CLASS == QWEN_OUTPUT_AB
#define QWEN_NE_C (64 / QWEN_PORT_WIDTH)
#elif QWEN_OUTPUT_CLASS == QWEN_OUTPUT_PV
#define QWEN_NE_C (256 / QWEN_PORT_WIDTH)
#else
#define QWEN_NE_C 8
#endif
#endif

#endif // QWEN_TILE_POLICY

#define QWEN_TILE_M (QWEN_NE_R * QWEN_PORT_WIDTH)
#define QWEN_TILE_P (QWEN_NE_C * QWEN_PORT_WIDTH)

#define QWEN_ACTIVE_ENGINES (QWEN_NE_R * QWEN_NE_C)
#define QWEN_ROWS_PER_ENGINE (QWEN_TILE_M / QWEN_NE_R)
#define QWEN_COLS_PER_ENGINE (QWEN_TILE_P / QWEN_NE_C)

_Static_assert(QWEN_L1_BUDGET == L1_TILE_BUDGET_BYTES,
               "the cast-free L1 budget has drifted from L1_TILE_BUDGET_BYTES");
_Static_assert(QWEN_NE_R *QWEN_NE_C <= NUM_REDMULE_TILES,
               "Qwen tile asks for more engines than the chip has");
_Static_assert(QWEN_DECODE == 0 || QWEN_DECODE == 1,
               "QWEN_DECODE must be Boolean");
_Static_assert(QWEN_OUTPUT_CLASS >= QWEN_OUTPUT_DEFAULT &&
                   QWEN_OUTPUT_CLASS <= QWEN_OUTPUT_PV,
               "invalid Qwen output class");
_Static_assert(QWEN_PORT_WIDTH == 32 || QWEN_PORT_WIDTH == 64,
               "the first Qwen tranche supports Stage-1 S1/S2 only");
_Static_assert(QWEN_ACTIVE_ENGINES > 0 &&
                   QWEN_ACTIVE_ENGINES <= NUM_REDMULE_TILES,
               "Qwen tile requests an invalid RedMulE count");
_Static_assert(QWEN_TILE_M % QWEN_NE_R == 0 && QWEN_TILE_P % QWEN_NE_C == 0,
               "Qwen row/column groups must divide the tile");
_Static_assert(QWEN_ROWS_PER_ENGINE % QWEN_PORT_WIDTH == 0 &&
                   QWEN_COLS_PER_ENGINE % QWEN_PORT_WIDTH == 0,
               "each Qwen engine shard must tile the RedMulE port");
_Static_assert(QWEN_TILE_N % QWEN_PORT_WIDTH == 0,
               "Qwen contraction must tile the RedMulE port");
_Static_assert(QWEN_ROWS_PER_ENGINE <= UINT16_MAX &&
                   QWEN_TILE_N <= UINT16_MAX &&
                   QWEN_COLS_PER_ENGINE <= UINT16_MAX,
               "Qwen RedMulE dimensions must fit uint16_t");
_Static_assert((((long)QWEN_TILE_M * QWEN_TILE_N +
                 (long)QWEN_TILE_N * QWEN_TILE_P +
                 (long)QWEN_TILE_M * QWEN_TILE_P) *
                (long)sizeof(__fp16)) <= L1_TILE_BUDGET_BYTES,
               "Qwen one-operation tile exceeds the L1 budget");

// ---------------------------------------------------------------------------
// Virtual-cluster operand placement (WP3). Undefined => every operand stays in
// the flat global interleave and this file is byte-identical to the pre-VC
// build; that identity is checked by an ELF hash, not by inspection.
//
// QWEN_VC_W_LEVEL places W in that fractal level's window. Only W, and only
// for a single-engine-row map: with NE_R > 1 a W column block is read by every
// engine row, so confining it to one rectangle turns uniform bank load into a
// hotspot rather than buying locality. X and Z are the row-aligned pair and
// come later (they share a rectangle with each other, never with W -- engine
// rows and columns are orthogonal partitions, so at most one side can be
// confined).
// ---------------------------------------------------------------------------
#ifdef QWEN_VC_W_LEVEL
#include "vc.h"

_Static_assert(QWEN_NE_R == 1,
               "QWEN_VC_W_LEVEL assumes a single engine row: with NE_R > 1 "
               "each W block has NE_R readers spread across rectangles.");
_Static_assert(QWEN_VC_W_LEVEL < VC_MAX_LEVELS, "VC level above the mesh");

// The tile budget must be re-derived against the LEVEL's aperture, not full
// L1: confinement trades capacity for locality, and a rectangle holds
// 4^k / NUM_GROUPS of the array.
#define QWEN_VC_WSIZE_(k) VC_WINDOW_SIZE_##k
#define QWEN_VC_WSIZE(k) QWEN_VC_WSIZE_(k)

// Every RedMulE tile is one engine, so a level-k rectangle owns 4^k groups
// worth of them.
#define QWEN_VC_ENGINES_PER_RECT                                               \
  ((uint32_t)(1u << (2 * QWEN_VC_W_LEVEL)) * (NUM_REDMULE_TILES / NUM_GROUPS))
#define QWEN_VC_W_BYTES_PER_ENGINE                                             \
  ((uint32_t)(sizeof(__fp16) * QWEN_TILE_N * QWEN_COLS_PER_ENGINE))
// Only ACTIVE engines have a W block, and a rectangle never holds more than
// the whole active set, so size the reservation by the smaller of the two.
// Rectangles that hold no active engine over-reserve harmlessly; what matters
// is that the figure is the same for every core, since the reservation is
// collective within a rectangle.
#define QWEN_VC_W_ENGINES                                                      \
  ((QWEN_VC_ENGINES_PER_RECT < (uint32_t)QWEN_ACTIVE_ENGINES)                  \
       ? QWEN_VC_ENGINES_PER_RECT                                              \
       : (uint32_t)QWEN_ACTIVE_ENGINES)
#define QWEN_VC_W_BYTES_PER_RECT                                               \
  (QWEN_VC_W_ENGINES * QWEN_VC_W_BYTES_PER_ENGINE)

_Static_assert(QWEN_VC_W_BYTES_PER_RECT <= QWEN_VC_WSIZE(QWEN_VC_W_LEVEL),
               "W does not fit this level's rectangle: shrink TILE_N or the "
               "engine grid, or place W at a larger level");

// This core's engine-slot within its own rectangle, and the rectangle's first
// engine in global numbering. Groups are row-major and a rectangle is a
// contiguous (gy, gx) block, so the slot is the raster position of the group
// inside the rectangle times the tiles per group.
static inline uint32_t qwen_vc_rect_first_engine(void) {
  const vc_rect_t r = vc_rect(QWEN_VC_W_LEVEL);
  // Under Morton the rectangle's groups are a contiguous id range starting at
  // base_gid, so this is a multiply and nothing else. The raster form needed a
  // per-axis recomposition that was identity only on the diagonal -- it
  // survived rectangle (0,0) and was wrong everywhere else.
  return r.base_gid * (uint32_t)NUM_TILES_PER_GROUP;
}
#endif

#ifdef QWEN_VC_W_LEVEL
// MUST be initialised explicitly before use. `.l1` is NOBITS and the bss range
// is EMPTY (__bss_start == __bss_end), so nothing zeroes it and it powers up X
// in RTL simulation. An X here is not a cosmetic wrong count: it feeds the
// ternary below, so the X reaches a BRANCH, the next PC becomes X, and that
// core wedges -- taking all the others down with it at the following barrier.
uint32_t qwen_vc_fill_bad __attribute__((section(".l1")));
#endif

__fp16 qwen_l1_x[QWEN_TILE_M * QWEN_TILE_N]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
#ifndef QWEN_VC_W_LEVEL
__fp16 qwen_l1_w[QWEN_TILE_N * QWEN_TILE_P]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
#endif
__fp16 qwen_l1_z[QWEN_TILE_M * QWEN_TILE_P]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));

__fp16 qwen_l2_x[QWEN_TILE_M * QWEN_TILE_N]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
__fp16 qwen_l2_w[QWEN_TILE_N * QWEN_TILE_P]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
__fp16 qwen_l2_z[QWEN_TILE_M * QWEN_TILE_P]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  const uint32_t redmule_id = mempool_get_redmule_id();
  mempool_barrier_init(core_id);

  if (core_id == 0)
    printf("\n[%s] logical=%dx%dx%d tile=%dx%dx%d map=%dx%d "
           "engines=%d/%d port=%d xshift=%d\n",
           QWEN_BENCHMARK_NAME, QWEN_LOGICAL_M, QWEN_LOGICAL_N, QWEN_LOGICAL_P,
           QWEN_TILE_M, QWEN_TILE_N, QWEN_TILE_P, QWEN_NE_R, QWEN_NE_C,
           QWEN_ACTIVE_ENGINES, NUM_REDMULE_TILES, QWEN_PORT_WIDTH,
           LLM_RM_X_SHIFT);
#ifndef QWEN_VC_W_LEVEL
  dma_collective_memcpy_nonblocking(qwen_l1_x, qwen_l2_x,
                                    sizeof(__fp16) * QWEN_TILE_M * QWEN_TILE_N);
  dma_collective_memcpy_blocking(qwen_l1_w, qwen_l2_w,
                                 sizeof(__fp16) * QWEN_TILE_N * QWEN_TILE_P);
#else
  dma_collective_memcpy_blocking(qwen_l1_x, qwen_l2_x,
                                 sizeof(__fp16) * QWEN_TILE_M * QWEN_TILE_N);
#endif
  mempool_barrier(num_cores);

  if (redmule_id < QWEN_ACTIVE_ENGINES) {
    llm_zero_f16(qwen_l1_z +
                     redmule_id * QWEN_ROWS_PER_ENGINE * QWEN_COLS_PER_ENGINE,
                 QWEN_ROWS_PER_ENGINE * QWEN_COLS_PER_ENGINE);
  }
  mempool_barrier(num_cores);

#ifdef QWEN_VC_W_LEVEL
  // Reserve before allocating: a level's base depends on the lower levels'
  // reservations, so a late reservation moves live arenas (contract
  // [allocator]). One reservation, then one allocation, both collective.
  vc_init();
  mempool_barrier(num_cores);
  if (vc_reserve(QWEN_VC_W_LEVEL, QWEN_VC_W_BYTES_PER_RECT) < 0) {
    if (core_id == 0)
      printf("[%s] VC reservation of %u bytes at level %d failed\n",
             QWEN_BENCHMARK_NAME, (unsigned)QWEN_VC_W_BYTES_PER_RECT,
             QWEN_VC_W_LEVEL);
    return 1;
  }
  __fp16 *const qwen_vc_w =
      (__fp16 *)vc_malloc(QWEN_VC_W_LEVEL, QWEN_VC_W_BYTES_PER_RECT);
  const uint32_t qwen_vc_off =
      (uint32_t)qwen_vc_w - vc_window_base(QWEN_VC_W_LEVEL);
  const uint32_t qwen_vc_first = qwen_vc_rect_first_engine();
#ifdef QWEN_VC_TRACE
  if (core_id == 0)
    printf("[%s] VC cfg: rect_bytes=%u wsize=%u engines=%u active=%u first=%u "
           "off=%u\n",
           QWEN_BENCHMARK_NAME, (unsigned)QWEN_VC_W_BYTES_PER_RECT,
           (unsigned)QWEN_VC_WSIZE(QWEN_VC_W_LEVEL),
           (unsigned)QWEN_VC_W_ENGINES, (unsigned)QWEN_ACTIVE_ENGINES,
           (unsigned)qwen_vc_first, (unsigned)qwen_vc_off);
#define QWEN_MARK(m)                                                           \
  do {                                                                         \
    if (core_id == 0)                                                          \
      printf("[%s] MARK %s\n", QWEN_BENCHMARK_NAME, (m));                      \
  } while (0)
#else
#define QWEN_MARK(m)                                                           \
  do {                                                                         \
  } while (0)
#endif
  if (core_id == 0)
    qwen_vc_fill_bad = 0;
  QWEN_MARK("vc-alloc-done");
  mempool_barrier(num_cores);
  QWEN_MARK("bar-after-alloc");

  // Every core of a rectangle pulls that rectangle's W blocks out of L2, and
  // each group's coordinator issues its own stripe. The descriptor is a GLOBAL
  // alias -- the DMA never sees a VC address, which is what keeps one
  // implementation of the map (contract [dma]).
  // Seed the L2 source at the positions the fill check samples. qwen_l2_w is
  // .l2_bss and therefore zero, so comparing it after the DMA would compare
  // zeros to zeros and pass wherever the data landed -- the same vacuity the
  // first alias probe had.
#define QWEN_VC_FILL_N 8
  if (core_id == 0)
    for (uint32_t i = 0; i < QWEN_VC_FILL_N; i++)
      *(volatile uint16_t *)(qwen_l2_w +
                             i * (QWEN_TILE_N * QWEN_COLS_PER_ENGINE /
                                  QWEN_VC_FILL_N)) = (uint16_t)(0x5C00u + i);
  mempool_barrier(num_cores);
  QWEN_MARK("seed-done");
  {
    const vc_rect_t rect = vc_rect(QWEN_VC_W_LEVEL);
    for (uint32_t e = 0; e < QWEN_VC_W_ENGINES; e++) {
      if (qwen_vc_first + e >= QWEN_ACTIVE_ENGINES)
        break;
      const vc_desc_t dst =
          vc_descriptor(&rect, qwen_vc_off + e * QWEN_VC_W_BYTES_PER_ENGINE,
                        QWEN_VC_W_BYTES_PER_ENGINE);
      const vc_desc_t src = {
          .base =
              (uint32_t)(qwen_l2_w + (uint32_t)(qwen_vc_first + e) *
                                         QWEN_TILE_N * QWEN_COLS_PER_ENGINE),
          .len = QWEN_VC_W_BYTES_PER_ENGINE,
          .reps = 1u,
          .stride = 0u};
      vc_dma_memcpy_blocking(&dst, &src);
    }
  }
  mempool_barrier(num_cores);
  QWEN_MARK("dma-done");
  // Every core of a tile shares a redmule_id, so gating the PRINT on it lets
  // several cores print at once and the UART interleaves into soup. Check on
  // the engine cores, record, print from one.
  // ONE writer, and only for a rectangle that was actually filled. This used to
  // be gated on redmule_id == qwen_vc_first, which is a PER-RECTANGLE value, so
  // all four rectangle leaders (cores 0/32/64/96 at level 1 on dse_s1_winner)
  // ran the check and all stored into the single shared flag below. Only
  // rectangle 0 owns active engines, so the other three read VC memory the DMA
  // never wrote -- i.e. X -- and raced their X into the flag. Whether core 0
  // then read its own 0 or someone else's X depended purely on store timing,
  // which is why inserting printfs made the hang disappear.
  if (core_id == qwen_vc_first * NUM_CORES_PER_TILE &&
      qwen_vc_first < QWEN_ACTIVE_ENGINES) {
    uint32_t bad = 0;
    for (uint32_t i = 0; i < QWEN_VC_FILL_N; i++)
      if (*(volatile uint16_t *)(qwen_vc_w +
                                 i * (QWEN_TILE_N * QWEN_COLS_PER_ENGINE /
                                      QWEN_VC_FILL_N)) !=
          (uint16_t)(0x5C00u + i))
        bad++;
    qwen_vc_fill_bad = bad;
    // The store is POSTED. Without this the value can still be in flight when
    // the barrier releases, and the reader gets the pre-store contents.
    __sync_synchronize();
  }
  mempool_barrier(num_cores);
  if (core_id == 0)
    printf("[%s] VC W fill %s (%u/%u samples)\n", QWEN_BENCHMARK_NAME,
           qwen_vc_fill_bad ? "MISMATCH" : "OK",
           (unsigned)(QWEN_VC_FILL_N - qwen_vc_fill_bad), QWEN_VC_FILL_N);
    // Alias equivalence on this allocation, which is what [EOC] retval=0 cannot
    // show: these apps are perf-only, and qwen_l2_w is untouched .l2_bss, so a
    // read-back against L2 would compare zeros to zeros and pass wherever the
    // pointer resolved. Instead write a pattern through the DMA's view
    // (vc_global_addr, plain arithmetic) and read it through the compute view
    // (vc_ptr, the hardware permuter). Two independent address computations, so
    // agreement cannot be accidental. INTEGER pattern on purpose: the probe
    // tests ADDRESSES, and routing it through __fp16 dragged in two unrelated
    // datapath bugs that presented as an address failure -- this toolchain
    // compiles
    // (__fp16)<integer> as a BITCAST, and Spatz's widen_fp16_to_fp32 mishandles
    // the subnormals that produced. 0xA5xx also cannot alias zero-filled
    // memory. Sixteen samples, all L1, before the fill overwrites the region --
    // an L2-crossing sweep costs hours of simulation.
#define QWEN_VC_PROBE_N 16
  {
    const vc_rect_t prect = vc_rect(QWEN_VC_W_LEVEL);
    const uint32_t step =
        (QWEN_VC_W_BYTES_PER_RECT / QWEN_VC_PROBE_N) & ~(uint32_t)1;
    if (core_id == qwen_vc_first * NUM_CORES_PER_TILE) {
      for (uint32_t i = 0; i < QWEN_VC_PROBE_N; i++)
        *(volatile uint16_t *)vc_global_addr(&prect, qwen_vc_off + i * step) =
            (uint16_t)(0xA500u + i);
      __sync_synchronize();
    }
    mempool_barrier(num_cores);
    QWEN_MARK("probe-written");
    if (redmule_id < QWEN_ACTIVE_ENGINES) {
      uint32_t bad = 0;
      for (uint32_t i = 0; i < QWEN_VC_PROBE_N; i++)
        if (*(volatile uint16_t *)vc_ptr(QWEN_VC_W_LEVEL,
                                         qwen_vc_off + i * step) !=
            (uint16_t)(0xA500u + i))
          bad++;
      if (bad)
        printf("[%s] VC alias probe FAILED on engine %u: %u/%u\n",
               QWEN_BENCHMARK_NAME, (unsigned)redmule_id, (unsigned)bad,
               QWEN_VC_PROBE_N);
      else if (redmule_id == 0)
        printf("[%s] VC alias probe OK (level %d, %u samples, %u B/rect)\n",
               QWEN_BENCHMARK_NAME, QWEN_VC_W_LEVEL, QWEN_VC_PROBE_N,
               (unsigned)QWEN_VC_W_BYTES_PER_RECT);
    }
  }
  mempool_barrier(num_cores);

  // End-to-end proof that the fill landed where the ENGINES will read it: the
  // pattern went into L2, the DMA moved it through the global alias, and this
  // reads it back through the VC pointer. Engine 0 only, 8 samples, outside
  // the timed region.  mempool_barrier(num_cores);
#endif

  const uint32_t time_init = mempool_get_timer();
  mempool_start_benchmark();
#ifdef QWEN_VC_W_LEVEL
  // NE_R == 1 is asserted above, so there is exactly one X row block and the
  // engine's W block is its slot within its own rectangle.
  if (redmule_id < QWEN_ACTIVE_ENGINES)
    qwen_redmule_op_engine_f16(
        qwen_l1_x,
        qwen_vc_w + (uint32_t)(redmule_id - qwen_vc_first) * QWEN_TILE_N *
                        QWEN_COLS_PER_ENGINE,
        qwen_l1_z + redmule_id * QWEN_ROWS_PER_ENGINE * QWEN_COLS_PER_ENGINE,
        QWEN_ROWS_PER_ENGINE, QWEN_TILE_N, QWEN_COLS_PER_ENGINE);
#else
  qwen_redmule_op_tile_f16(qwen_l1_x, qwen_l1_w, qwen_l1_z, QWEN_TILE_N,
                           QWEN_ROWS_PER_ENGINE, QWEN_COLS_PER_ENGINE,
                           QWEN_NE_C, QWEN_ACTIVE_ENGINES, redmule_id);
#endif
  mempool_log_barrier(2, core_id);
  mempool_stop_benchmark();
  const uint32_t time_end = mempool_get_timer();

  dma_collective_memcpy_blocking(qwen_l2_z, qwen_l1_z,
                                 sizeof(__fp16) * QWEN_TILE_M * QWEN_TILE_P);
  if (core_id == 0) {
    printf("[%s] stage=gemm cycles=%u\n", QWEN_BENCHMARK_NAME,
           time_end - time_init);
    printf("[%s] kernel takes %u clock cycles\n", QWEN_BENCHMARK_NAME,
           time_end - time_init);
  }
  mempool_barrier(num_cores);
  return 0;
}

#endif // __QWEN_REDMULE_OP_MAIN_H__
