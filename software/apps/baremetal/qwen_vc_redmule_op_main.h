// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Virtual-cluster placement variant of the Qwen RedMulE GEMM driver.
//
// WHY THIS FILE EXISTS. qwen_redmule_op_main.h is the calibrated baseline and
// every 4x4 stage-1 number is measured against it, so it must not move. This
// is a sibling: the arithmetic, the engine kernel and the shape of the timed
// region are identical, and only the PLACEMENT of the operands changes.
//
// THE PROBLEM. At 8x8 the flat global interleave spreads every operand over
// all 64 groups, so 4x the engines returns ~2.4x the throughput and engine
// utilisation falls from ~90% to ~60%
// (docs/report/2026-08-20-monsterpool-scaleup.html). Per-group resources are
// unchanged, so what got worse is distance and path overlap.
//
// THE MODEL. Z[M*P] = X[M*N] . W[N*P] on an NE_R x NE_C engine grid. Engine
// (r,c) reads X row-band r and W column-band c and owns Z block (r,c), so:
//
//   * X row-band r is read by every engine of engine-row r;
//   * W col-band c is read by every engine of engine-column c;
//   * Z is private and is never shared by anyone.
//
// TWO FACTS SET THE WHOLE DESIGN.
//
// 1. The L1 cost of confining an operand is EXACTLY its replication factor,
//    and is independent of the level. A level-k allocation of n rows occupies
//    n rows of every one of its 4^k groups, so the per-group rows it consumes
//    are total_bytes * replication / (VC_ROW_BYTES * NUM_GROUPS) -- the 4^k
//    cancels. Level is therefore FREE and is purely a locality knob; only
//    replication costs memory. That is why the levels are per-operand here
//    rather than one level for the whole tile.
//
// 2. Z is private, so it can always sit at the FINEST level at zero cost. X
//    and W cannot both be replication-free unless one of them stays global:
//    making X free needs a VC that spans a whole engine-row, making W free
//    needs one that spans a whole engine-column, and only the root spans both.
//
// THE MAP. The engine grid is cut into contiguous runs of SUB_R x SUB_C
// engines, laid out row-major over an RR x RC grid of runs. Morton group ids
// make a contiguous run of redmule_id exactly one rectangle, so the map is
// divides and modulos and nothing else. SUB_C == NE_C reproduces the
// baseline's raster order exactly.

#ifndef __QWEN_VC_REDMULE_OP_MAIN_H__
#define __QWEN_VC_REDMULE_OP_MAIN_H__

#include <stdint.h>

#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "baremetal/mempool_llm_tiling.h"
#include "baremetal/mempool_qwen_redmule_op.h"

// vc.h must follow runtime.h: runtime.h defines mempool_wfi static while
// synchronization.h declares it non-static, and clang-format sorts in-block.
#include "vc.h"

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

// ------------------------------------------------------------- engine budget
// Policy 1 of the baseline header, verbatim: the 8x8 scale-up numbers this DSE
// is measured against were taken with it, so the grid must agree exactly.
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
#define QWEN_MAX(a, b) ((a) > (b) ? (a) : (b))

#define QWEN_NE_R_CAP QWEN_P2_LE(QWEN_LOGICAL_M / QWEN_PORT_WIDTH)
#if QWEN_OUTPUT_CLASS == QWEN_OUTPUT_AB
#define QWEN_NE_C_CAP QWEN_MAX(64 / QWEN_PORT_WIDTH, 1)
#elif QWEN_OUTPUT_CLASS == QWEN_OUTPUT_PV
#define QWEN_NE_C_CAP QWEN_MAX(256 / QWEN_PORT_WIDTH, 1)
#else
#define QWEN_NE_C_CAP QWEN_P2_LE(QWEN_LOGICAL_P / QWEN_PORT_WIDTH)
#endif

#define QWEN_SPLIT_C0(T) QWEN_MIN(QWEN_P2_SQRT(T), QWEN_NE_C_CAP)
#define QWEN_SPLIT_R(T) QWEN_MIN((T) / QWEN_SPLIT_C0(T), QWEN_NE_R_CAP)
#define QWEN_SPLIT_C(T) QWEN_MIN((T) / QWEN_SPLIT_R(T), QWEN_NE_C_CAP)

#define QWEN_L1_BUDGET                                                         \
  ((NUM_BANKS * L1_BANK_SIZE - NUM_CORES * SEQ_MEM_SIZE) * 9 / 10)
#define QWEN_TILE_BYTES(r, c)                                                  \
  (((r)*QWEN_PORT_WIDTH * QWEN_TILE_N + QWEN_TILE_N * (c)*QWEN_PORT_WIDTH +    \
    (r)*QWEN_PORT_WIDTH * (c)*QWEN_PORT_WIDTH) *                               \
   2)
#define QWEN_FITS(T)                                                           \
  (QWEN_TILE_BYTES(QWEN_SPLIT_R(T), QWEN_SPLIT_C(T)) <= QWEN_L1_BUDGET)

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
#define QWEN_TILE_M (QWEN_NE_R * QWEN_PORT_WIDTH)
#define QWEN_TILE_P (QWEN_NE_C * QWEN_PORT_WIDTH)
#define QWEN_ACTIVE_ENGINES (QWEN_NE_R * QWEN_NE_C)
#define QWEN_ROWS_PER_ENGINE (QWEN_TILE_M / QWEN_NE_R)
#define QWEN_COLS_PER_ENGINE (QWEN_TILE_P / QWEN_NE_C)

// ------------------------------------------------------------ N chunking
// The contraction actually executed and measured. Splitting N shrinks the
// resident X and W proportionally, which is what buys replication headroom:
// with NCHUNK chunks the same L1 budget affords NCHUNK times the replication.
// One chunk is measured; the whole-tile compute cost is cycles * NCHUNK,
// and utilisation is scale-invariant so it compares directly. Reloading
// between chunks is a DMA cost that double buffering hides, and is excluded
// here exactly as the baseline excludes its own operand load.
#ifndef QWEN_VC_NCHUNK
#define QWEN_VC_NCHUNK 1
#endif
#define QWEN_KN (QWEN_TILE_N / QWEN_VC_NCHUNK)

// -------------------------------------------------------------- the map
// SUB_R x SUB_C engines per contiguous run of redmule_id. SUB_C == NE_C is the
// baseline's raster order.
#ifndef QWEN_VC_SUB_R
#define QWEN_VC_SUB_R 1
#endif
#ifndef QWEN_VC_SUB_C
#define QWEN_VC_SUB_C QWEN_NE_C
#endif

// Per-operand level, or -1 to leave the operand in the flat global interleave.
#ifndef QWEN_VC_X_LEVEL
#define QWEN_VC_X_LEVEL (-1)
#endif
#ifndef QWEN_VC_W_LEVEL
#define QWEN_VC_W_LEVEL (-1)
#endif
#ifndef QWEN_VC_Z_LEVEL
#define QWEN_VC_Z_LEVEL (-1)
#endif

#define QWEN_X_VC (QWEN_VC_X_LEVEL >= 0)
#define QWEN_W_VC (QWEN_VC_W_LEVEL >= 0)
#define QWEN_Z_VC (QWEN_VC_Z_LEVEL >= 0)
#define QWEN_ANY_VC (QWEN_X_VC || QWEN_W_VC || QWEN_Z_VC)

#define QWEN_ENG_PER_GROUP (NUM_REDMULE_TILES / NUM_GROUPS)
#define QWEN_RUN (QWEN_VC_SUB_R * QWEN_VC_SUB_C)
#define QWEN_RR (QWEN_NE_R / QWEN_VC_SUB_R)
#define QWEN_RC (QWEN_NE_C / QWEN_VC_SUB_C)

// Geometry of one level-k rectangle expressed in the engine grid.
#define QWEN_ENG_PER_VC(k) ((1u << (2 * (k))) * (unsigned)QWEN_ENG_PER_GROUP)
#define QWEN_RUNS_PER_VC(k) (QWEN_ENG_PER_VC(k) / (unsigned)QWEN_RUN)
#define QWEN_RECT_COLS(k) QWEN_MIN(QWEN_RUNS_PER_VC(k), (unsigned)QWEN_RC)
#define QWEN_RECT_ROWS(k) QWEN_MAX(QWEN_RUNS_PER_VC(k) / (unsigned)QWEN_RC, 1u)

// Engine rows / columns a level-k rectangle owns, and hence the replication
// each operand pays there.
#define QWEN_X_ENGROWS(k) ((unsigned)QWEN_VC_SUB_R * QWEN_RECT_ROWS(k))
#define QWEN_W_ENGCOLS(k) ((unsigned)QWEN_VC_SUB_C * QWEN_RECT_COLS(k))
#define QWEN_X_REPL(k) ((unsigned)QWEN_RC / QWEN_RECT_COLS(k))
#define QWEN_W_REPL(k) ((unsigned)QWEN_RR / QWEN_RECT_ROWS(k))

// Per-rectangle operand bytes.
#define QWEN_XB                                                                \
  (QWEN_X_ENGROWS(QWEN_VC_X_LEVEL) * (unsigned)QWEN_ROWS_PER_ENGINE *          \
   (unsigned)QWEN_KN * 2u)
#define QWEN_WB                                                                \
  ((unsigned)QWEN_KN * QWEN_W_ENGCOLS(QWEN_VC_W_LEVEL) *                       \
   (unsigned)QWEN_COLS_PER_ENGINE * 2u)
#define QWEN_ZB                                                                \
  (QWEN_ENG_PER_VC(QWEN_VC_Z_LEVEL) * (unsigned)QWEN_ROWS_PER_ENGINE *         \
   (unsigned)QWEN_COLS_PER_ENGINE * 2u)

_Static_assert(QWEN_NE_R % QWEN_VC_SUB_R == 0 && QWEN_NE_C % QWEN_VC_SUB_C == 0,
               "the engine sub-block must divide the engine grid");
_Static_assert(QWEN_ACTIVE_ENGINES % QWEN_RUN == 0,
               "the engine sub-block must tile the active engine set");
_Static_assert(QWEN_TILE_N % QWEN_VC_NCHUNK == 0,
               "NCHUNK must divide the contraction");
_Static_assert(QWEN_KN % QWEN_PORT_WIDTH == 0,
               "the measured contraction must tile the RedMulE port");
_Static_assert(QWEN_ROWS_PER_ENGINE <= UINT16_MAX && QWEN_KN <= UINT16_MAX &&
                   QWEN_COLS_PER_ENGINE <= UINT16_MAX,
               "RedMulE dimensions must fit uint16_t");
#if QWEN_X_VC
_Static_assert(QWEN_VC_X_LEVEL < VC_LEVELS, "X level above the mesh");
_Static_assert(QWEN_ENG_PER_VC(QWEN_VC_X_LEVEL) >= (unsigned)QWEN_RUN,
               "X level is finer than one engine run");
#endif
#if QWEN_W_VC
_Static_assert(QWEN_VC_W_LEVEL < VC_LEVELS, "W level above the mesh");
_Static_assert(QWEN_ENG_PER_VC(QWEN_VC_W_LEVEL) >= (unsigned)QWEN_RUN,
               "W level is finer than one engine run");
#endif
#if QWEN_Z_VC
_Static_assert(QWEN_VC_Z_LEVEL < VC_LEVELS, "Z level above the mesh");
_Static_assert(QWEN_ENG_PER_VC(QWEN_VC_Z_LEVEL) >= (unsigned)QWEN_RUN,
               "Z level is finer than one engine run");
#endif

// --------------------------------------------------------------- buffers
#if !QWEN_X_VC
__fp16 qwen_l1_x[QWEN_TILE_M * QWEN_KN]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
#endif
#if !QWEN_W_VC
__fp16 qwen_l1_w[QWEN_KN * QWEN_TILE_P]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
#endif
#if !QWEN_Z_VC
__fp16 qwen_l1_z[QWEN_TILE_M * QWEN_TILE_P]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
#endif

__fp16 qwen_l2_x[QWEN_TILE_M * QWEN_KN]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
__fp16 qwen_l2_w[QWEN_KN * QWEN_TILE_P]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));

// `.l1` is NOBITS and the bss range is empty, so nothing zeroes this and it
// powers up X in RTL simulation. An X reaching a branch wedges the core.
uint32_t qwen_vc_alias_bad __attribute__((section(".l1")));

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  const uint32_t redmule_id = mempool_get_redmule_id();
  mempool_barrier_init(core_id);

  const uint32_t is_eng = redmule_id < (uint32_t)QWEN_ACTIVE_ENGINES;
  // Engine coordinates under the blocked map.
  const uint32_t run = is_eng ? redmule_id / QWEN_RUN : 0;
  const uint32_t in_run = is_eng ? redmule_id % QWEN_RUN : 0;
  const uint32_t r = (run / QWEN_RC) * QWEN_VC_SUB_R + in_run / QWEN_VC_SUB_C;
  const uint32_t c = (run % QWEN_RC) * QWEN_VC_SUB_C + in_run % QWEN_VC_SUB_C;

  if (core_id == 0) {
    printf("\n[%s] tile=%dx%dx%d map=%dx%d sub=%dx%d engines=%d/%d nchunk=%d "
           "| levels X%d W%d Z%d | repl X%u W%u\n",
           QWEN_BENCHMARK_NAME, QWEN_TILE_M, QWEN_KN, QWEN_TILE_P, QWEN_NE_R,
           QWEN_NE_C, QWEN_VC_SUB_R, QWEN_VC_SUB_C, QWEN_ACTIVE_ENGINES,
           NUM_REDMULE_TILES, QWEN_VC_NCHUNK, QWEN_VC_X_LEVEL, QWEN_VC_W_LEVEL,
           QWEN_VC_Z_LEVEL,
           (unsigned)(QWEN_X_VC ? QWEN_X_REPL(QWEN_VC_X_LEVEL) : 1u),
           (unsigned)(QWEN_W_VC ? QWEN_W_REPL(QWEN_VC_W_LEVEL) : 1u));
    qwen_vc_alias_bad = 0;
  }
  mempool_barrier(num_cores);

#if QWEN_ANY_VC
  vc_init();
  mempool_barrier(num_cores);

  // Reserve every level before allocating from any of them: a level's arena
  // base depends on the higher levels' reservations, so a late reservation
  // moves live arenas (vc.h [allocator]).
  uint32_t want[VC_MAX_LEVELS] = {0};
  const uint32_t rs[VC_MAX_LEVELS] = {vc_row_stride(0), vc_row_stride(1),
                                      vc_row_stride(2), vc_row_stride(3)};
#define QWEN_RND(b, k) ((((b) + rs[k] - 1) / rs[k]) * rs[k])
#if QWEN_X_VC
  want[QWEN_VC_X_LEVEL] += QWEN_RND(QWEN_XB, QWEN_VC_X_LEVEL);
#endif
#if QWEN_W_VC
  want[QWEN_VC_W_LEVEL] += QWEN_RND(QWEN_WB, QWEN_VC_W_LEVEL);
#endif
#if QWEN_Z_VC
  want[QWEN_VC_Z_LEVEL] += QWEN_RND(QWEN_ZB, QWEN_VC_Z_LEVEL);
#endif
  for (uint32_t k = 0; k < (uint32_t)VC_LEVELS; k++)
    if (want[k] && vc_reserve(k, want[k]) < 0) {
      if (core_id == 0)
        printf("[%s] VC reserve of %u B at level %u FAILED\n",
               QWEN_BENCHMARK_NAME, (unsigned)want[k], (unsigned)k);
      return 1;
    }
    // Allocation order is fixed, so every member of a rectangle agrees.
#if QWEN_X_VC
  __fp16 *const vx =
      (__fp16 *)vc_malloc(QWEN_VC_X_LEVEL, QWEN_RND(QWEN_XB, QWEN_VC_X_LEVEL));
#endif
#if QWEN_W_VC
  __fp16 *const vw =
      (__fp16 *)vc_malloc(QWEN_VC_W_LEVEL, QWEN_RND(QWEN_WB, QWEN_VC_W_LEVEL));
#endif
#if QWEN_Z_VC
  __fp16 *const vz =
      (__fp16 *)vc_malloc(QWEN_VC_Z_LEVEL, QWEN_RND(QWEN_ZB, QWEN_VC_Z_LEVEL));
#endif
  if (
#if QWEN_X_VC
      vx == NULL ||
#endif
#if QWEN_W_VC
      vw == NULL ||
#endif
#if QWEN_Z_VC
      vz == NULL ||
#endif
      0) {
    if (core_id == 0)
      printf("[%s] VC malloc FAILED\n", QWEN_BENCHMARK_NAME);
    return 1;
  }
  mempool_barrier(num_cores);

  // Alias equivalence, on the operand confined at the finest level. [EOC]
  // retval=0 cannot show this: the apps are perf-only and the L2 sources are
  // untouched .l2_bss, so a read-back against L2 compares zeros to zeros.
  // Write through the DMA's view (vc_global_addr, plain arithmetic) and read
  // through the compute view (vc_ptr, the hardware permuter): two independent
  // address computations, so agreement cannot be accidental.
  {
#if QWEN_Z_VC
    const uint32_t plevel = QWEN_VC_Z_LEVEL;
    const uint32_t pbytes = QWEN_RND(QWEN_ZB, QWEN_VC_Z_LEVEL);
    const uint32_t poff = (uint32_t)vz - vc_window_base(plevel);
#elif QWEN_X_VC
    const uint32_t plevel = QWEN_VC_X_LEVEL;
    const uint32_t pbytes = QWEN_RND(QWEN_XB, QWEN_VC_X_LEVEL);
    const uint32_t poff = (uint32_t)vx - vc_window_base(plevel);
#else
    const uint32_t plevel = QWEN_VC_W_LEVEL;
    const uint32_t pbytes = QWEN_RND(QWEN_WB, QWEN_VC_W_LEVEL);
    const uint32_t poff = (uint32_t)vw - vc_window_base(plevel);
#endif
#define QWEN_VC_PROBE_N 16
    const vc_rect_t prect = vc_rect(plevel);
    const uint32_t step = (pbytes / QWEN_VC_PROBE_N) & ~(uint32_t)1;
    const uint32_t lead =
        (prect.base_gid * QWEN_ENG_PER_GROUP) * (NUM_CORES / NUM_REDMULE_TILES);
    if (core_id == lead) {
      for (uint32_t i = 0; i < QWEN_VC_PROBE_N; i++)
        *(volatile uint16_t *)vc_global_addr(&prect, poff + i * step) =
            (uint16_t)(0xA500u + i);
      // Read the pattern back through the same view before releasing the
      // readers. The stores are posted and the fence orders them against each
      // other, not against their arrival in the bank; without this the barrier
      // can release while the last ones are still in the NoC.
      volatile uint16_t sink = 0;
      for (uint32_t i = 0; i < QWEN_VC_PROBE_N; i++)
        sink = *(volatile uint16_t *)vc_global_addr(&prect, poff + i * step);
      (void)sink;
      __sync_synchronize();
    }
    mempool_barrier(num_cores);
    if (is_eng) {
      uint32_t bad = 0;
      for (uint32_t i = 0; i < QWEN_VC_PROBE_N; i++)
        if (*(volatile uint16_t *)vc_ptr(plevel, poff + i * step) !=
            (uint16_t)(0xA500u + i))
          bad++;
      if (bad) {
        qwen_vc_alias_bad = bad;
        __sync_synchronize();
      }
    }
    mempool_barrier(num_cores);
    if (core_id == 0)
      printf("[%s] VC alias probe %s (level %u)\n", QWEN_BENCHMARK_NAME,
             qwen_vc_alias_bad ? "FAILED" : "OK", (unsigned)plevel);
  }
  mempool_barrier(num_cores);
#endif // QWEN_ANY_VC

  // Operand bases. A confined operand is indexed by this engine's band WITHIN
  // its own rectangle; the VCs partition the r (and c) axis into contiguous
  // blocks, so the local index is a modulo.
#if QWEN_X_VC
  const __fp16 *const xe = vx + (r % QWEN_X_ENGROWS(QWEN_VC_X_LEVEL)) *
                                    QWEN_ROWS_PER_ENGINE * QWEN_KN;
#else
  const __fp16 *const xe = qwen_l1_x + r * QWEN_ROWS_PER_ENGINE * QWEN_KN;
#endif
#if QWEN_W_VC
  const __fp16 *const we = vw + (c % QWEN_W_ENGCOLS(QWEN_VC_W_LEVEL)) *
                                    QWEN_KN * QWEN_COLS_PER_ENGINE;
#else
  const __fp16 *const we = qwen_l1_w + c * QWEN_KN * QWEN_COLS_PER_ENGINE;
#endif
#if QWEN_Z_VC
  __fp16 *const ze = vz + (redmule_id % QWEN_ENG_PER_VC(QWEN_VC_Z_LEVEL)) *
                              QWEN_ROWS_PER_ENGINE * QWEN_COLS_PER_ENGINE;
#else
  __fp16 *const ze =
      qwen_l1_z + redmule_id * QWEN_ROWS_PER_ENGINE * QWEN_COLS_PER_ENGINE;
#endif

  if (is_eng)
    llm_zero_f16(ze, QWEN_ROWS_PER_ENGINE * QWEN_COLS_PER_ENGINE);
  mempool_barrier(num_cores);

  const uint32_t time_init = mempool_get_timer();
  mempool_start_benchmark();
  if (is_eng)
    qwen_redmule_op_engine_f16(xe, we, ze, QWEN_ROWS_PER_ENGINE, QWEN_KN,
                               QWEN_COLS_PER_ENGINE);
  // Software log barrier: the hardware fractal_sync barrier is not stable
  // enough to sit inside a timed region yet.
  mempool_log_barrier(2, core_id);
  mempool_stop_benchmark();
  const uint32_t time_end = mempool_get_timer();

  if (core_id == 0)
    printf("[%s] stage=gemm cycles=%u\n", QWEN_BENCHMARK_NAME,
           time_end - time_init);
  mempool_barrier(num_cores);
  return qwen_vc_alias_bad ? 1 : 0;
}

#endif // __QWEN_VC_REDMULE_OP_MAIN_H__
