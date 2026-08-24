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
  ((x) >= 256  ? 256                                                           \
   : (x) >= 128 ? 128                                                          \
   : (x) >= 64  ? 64                                                           \
   : (x) >= 32  ? 32                                                           \
   : (x) >= 16  ? 16                                                           \
   : (x) >= 8   ? 8                                                            \
   : (x) >= 4   ? 4                                                            \
   : (x) >= 2   ? 2                                                            \
                : 1)
#define QWEN_P2_SQRT(x)                                                        \
  ((x) >= 65536  ? 256                                                         \
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
// PORT_WIDTH columns, so a direction can hold at most operand/PORT_WIDTH groups.
// In decode M is the batch and is smaller than one shard, which floors to a
// single row group that pads it -- the behaviour policy 0 hard-codes.
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
  (((r) * QWEN_PORT_WIDTH * QWEN_TILE_N +                                      \
    QWEN_TILE_N * (c) * QWEN_PORT_WIDTH +                                      \
    (r) * QWEN_PORT_WIDTH * (c) * QWEN_PORT_WIDTH) *                           \
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
_Static_assert(QWEN_NE_R * QWEN_NE_C <= NUM_REDMULE_TILES,
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

__fp16 qwen_l1_x[QWEN_TILE_M * QWEN_TILE_N]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
__fp16 qwen_l1_w[QWEN_TILE_N * QWEN_TILE_P]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
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

  if (core_id == 0) {
    printf("\n[%s] logical=%dx%dx%d tile=%dx%dx%d map=%dx%d "
           "engines=%d/%d port=%d xshift=%d\n",
           QWEN_BENCHMARK_NAME, QWEN_LOGICAL_M, QWEN_LOGICAL_N, QWEN_LOGICAL_P,
           QWEN_TILE_M, QWEN_TILE_N, QWEN_TILE_P, QWEN_NE_R, QWEN_NE_C,
           QWEN_ACTIVE_ENGINES, NUM_REDMULE_TILES, QWEN_PORT_WIDTH,
           LLM_RM_X_SHIFT);
    dma_memcpy_blocking(qwen_l1_x, qwen_l2_x,
                        sizeof(__fp16) * QWEN_TILE_M * QWEN_TILE_N);
    dma_memcpy_blocking(qwen_l1_w, qwen_l2_w,
                        sizeof(__fp16) * QWEN_TILE_N * QWEN_TILE_P);
  }
  mempool_barrier(num_cores);

  if (redmule_id < QWEN_ACTIVE_ENGINES) {
    llm_zero_f16(qwen_l1_z +
                     redmule_id * QWEN_ROWS_PER_ENGINE * QWEN_COLS_PER_ENGINE,
                 QWEN_ROWS_PER_ENGINE * QWEN_COLS_PER_ENGINE);
  }
  mempool_barrier(num_cores);

  const uint32_t time_init = mempool_get_timer();
  mempool_start_benchmark();
  qwen_redmule_op_tile_f16(qwen_l1_x, qwen_l1_w, qwen_l1_z, QWEN_TILE_N,
                           QWEN_ROWS_PER_ENGINE, QWEN_COLS_PER_ENGINE,
                           QWEN_NE_C, QWEN_ACTIVE_ENGINES, redmule_id);
  mempool_log_barrier(2, core_id);
  mempool_stop_benchmark();
  const uint32_t time_end = mempool_get_timer();

  if (core_id == 0) {
    dma_memcpy_blocking(qwen_l2_z, qwen_l1_z,
                        sizeof(__fp16) * QWEN_TILE_M * QWEN_TILE_P);
    printf("[%s] stage=gemm cycles=%u\n", QWEN_BENCHMARK_NAME,
           time_end - time_init);
    printf("[%s] kernel takes %u clock cycles\n", QWEN_BENCHMARK_NAME,
           time_end - time_init);
  }
  mempool_barrier(num_cores);
  return 0;
}

#endif // __QWEN_REDMULE_OP_MAIN_H__
