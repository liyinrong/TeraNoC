// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Virtual-cluster placement variant of the Qwen RedMulE GEMM driver.
//
// WHY THIS FILE EXISTS. qwen_redmule_op_main.h is the calibrated baseline and
// every 4x4 stage-1 number is measured against it, so it must not move. This
// is a sibling, not an edit: the baseline keeps its ELF-hash identity and this
// file is free to change the operand PLACEMENT while leaving the arithmetic,
// the engine kernel and the timed region byte-identical in shape.
//
// THE PROBLEM. At 8x8 the flat global interleave spreads every operand over
// all 64 groups, so 4x the engines returns only ~2.4x the throughput and
// engine utilisation falls from ~90% to ~60% with a 27-46% straggler tail
// (docs/report/2026-08-20-monsterpool-scaleup.html). Per-group resources are
// unchanged, so what got worse is distance and path overlap, not bandwidth.
//
// THE LEVER. Z[M*P] = X[M*N] . W[N*P] on an NE_R x NE_C engine grid, cut into
// an RR x RC grid of virtual-cluster rectangles:
//
//   * X row-band r is read by every engine in engine-row r, so cutting the
//     columns RC ways replicates X RC times;
//   * W col-band c is read by every engine in engine-column c, so cutting the
//     rows RR ways replicates W RR times;
//   * Z is private to its engine and is never replicated.
//
// RC == 1 therefore confines X and Z for free, RR == 1 confines W for free,
// and a square cut pays the least total replication for full confinement.
// Which operands are actually placed in the window is a SEPARATE choice from
// the cut -- QWEN_VC_X / _W / _Z -- because leaving one operand global is a
// real design point, not a degenerate one.
//
// Morton group ids (mempool_pkg.sv:688-718) are what make the mapping free: a
// contiguous run of redmule_id is exactly one level-k rectangle, so the engine
// to rectangle map is a divide and nothing else.

#ifndef __QWEN_VC_REDMULE_OP_MAIN_H__
#define __QWEN_VC_REDMULE_OP_MAIN_H__

#include <stdint.h>

#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "baremetal/mempool_llm_tiling.h"
#include "baremetal/mempool_qwen_redmule_op.h"

// vc.h must follow runtime.h: runtime.h defines mempool_wfi static while
// synchronization.h declares it non-static, and clang-format sorts within a
// block. Separate block, deliberately.
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
// Policy 1 of the baseline header, verbatim: split the whole chip as squarely
// as the operand shape allows, then halve until the tile is resident. The 8x8
// scale-up numbers this DSE is measured against were taken with policy 1, so
// the grid must agree with it exactly or the comparison is meaningless.
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

#define QWEN_NE_R_CAP QWEN_P2_LE(QWEN_LOGICAL_M / QWEN_PORT_WIDTH)
#if QWEN_OUTPUT_CLASS == QWEN_OUTPUT_AB
#define QWEN_NE_C_CAP (64 / QWEN_PORT_WIDTH)
#elif QWEN_OUTPUT_CLASS == QWEN_OUTPUT_PV
#define QWEN_NE_C_CAP (256 / QWEN_PORT_WIDTH)
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

// ------------------------------------------------------- VC placement policy
//
// QWEN_VC_LEVEL   fractal level k: rectangle is 2^k x 2^k groups.
// QWEN_VC_RR/_RC  the rectangle grid laid over the engine grid. Their product
//                 must equal the number of rectangles the level defines, so
//                 the whole chip is covered exactly once.
// QWEN_VC_X/_W/_Z which operands move into the window. An operand left at 0
//                 stays in the flat global interleave, exactly as the baseline
//                 places it.
#ifndef QWEN_VC_LEVEL
#error "QWEN_VC_LEVEL must be defined (fractal level of the rectangle)"
#endif
#ifndef QWEN_VC_RR
#error "QWEN_VC_RR must be defined (rectangle rows over the engine grid)"
#endif
#ifndef QWEN_VC_RC
#error "QWEN_VC_RC must be defined (rectangle cols over the engine grid)"
#endif
#ifndef QWEN_VC_X
#define QWEN_VC_X 1
#endif
#ifndef QWEN_VC_W
#define QWEN_VC_W 0
#endif
#ifndef QWEN_VC_Z
#define QWEN_VC_Z 1
#endif

// Bulk DMA prefill of the confined operands, default OFF.
//
// These are perf-only kernels: qwen_l2_* is untouched .l2_bss, so the prefill
// moves zeros and the measured GEMM is a pure timing measurement either way.
// It is also entirely OUTSIDE the timed region, so its presence cannot change
// a reported cycle count. What it does do is drive the pre-WP4 1-D frontend
// with count[0]*count[1] strided transfers per rectangle, and the model's DMA
// completion path retires a slice while that slice's read beats are still in
// flight (valgrind: IDmaBeAxi::resp_meth reading an IdmaTransfer freed by
// IDmaMeDist::ack_transfer). That corrupts the heap and later aborts an
// unrelated wide transaction.
//
// Addressability of the confined buffers is proven by the alias probe below,
// which writes through vc_global_addr and reads through vc_ptr -- two
// independent address computations -- so nothing is lost by leaving the bulk
// move off. Turn it on with -DQWEN_VC_DMA_FILL=1 once the DMA path is fixed.
#ifndef QWEN_VC_DMA_FILL
#define QWEN_VC_DMA_FILL 0
#endif

// RedMulE engines per group, and per level-k rectangle.
#define QWEN_ENG_PER_GROUP (NUM_REDMULE_TILES / NUM_GROUPS)
#define QWEN_RECT_GROUPS (1u << (2 * QWEN_VC_LEVEL))
#define QWEN_ENG_PER_RECT (QWEN_RECT_GROUPS * QWEN_ENG_PER_GROUP)
// Rectangles the MAP covers, not the rectangles the mesh has. An operation
// whose shape caps the engine grid below the chip (attention PV at S2 tops out
// at 128 of 256 because P = 256 leaves only 4 column groups at PORT_WIDTH 64)
// lands on a contiguous Morton PREFIX of the rectangles; the rest hold no
// engine and allocate nothing. Sizing this from NUM_GROUPS instead would
// demand a rectangle grid the engine grid cannot supply.
#define QWEN_NUM_RECTS (QWEN_ACTIVE_ENGINES / QWEN_ENG_PER_RECT)
#define QWEN_MESH_RECTS (NUM_GROUPS / QWEN_RECT_GROUPS)

// Engine sub-grid owned by one rectangle.
#define QWEN_SUB_R (QWEN_NE_R / QWEN_VC_RR)
#define QWEN_SUB_C (QWEN_NE_C / QWEN_VC_RC)

// Per-rectangle operand slices, in f16 elements.
#define QWEN_X_ELEMS_PER_RECT ((QWEN_TILE_M / QWEN_VC_RR) * QWEN_TILE_N)
#define QWEN_W_ELEMS_PER_RECT (QWEN_TILE_N * (QWEN_TILE_P / QWEN_VC_RC))
#define QWEN_Z_ELEMS_PER_RECT                                                  \
  ((QWEN_TILE_M / QWEN_VC_RR) * (QWEN_TILE_P / QWEN_VC_RC))

#define QWEN_VC_WSIZE_(k) VC_WINDOW_SIZE_##k
#define QWEN_VC_WSIZE(k) QWEN_VC_WSIZE_(k)

_Static_assert(QWEN_VC_LEVEL < VC_MAX_LEVELS, "VC level above the mesh");
_Static_assert(QWEN_VC_RR *QWEN_VC_RC == QWEN_NUM_RECTS,
               "the rectangle grid must tile the mesh exactly once");
_Static_assert(QWEN_NE_R % QWEN_VC_RR == 0 && QWEN_NE_C % QWEN_VC_RC == 0,
               "the rectangle grid must divide the engine grid");
_Static_assert(QWEN_SUB_R *QWEN_SUB_C == QWEN_ENG_PER_RECT,
               "engine sub-grid must fill the rectangle's engines; the engine "
               "budget disagrees with the level");
_Static_assert(QWEN_ACTIVE_ENGINES % QWEN_ENG_PER_RECT == 0,
               "the active engine set must be a whole number of rectangles");
_Static_assert(QWEN_NUM_RECTS <= QWEN_MESH_RECTS,
               "the map asks for more rectangles than the mesh has");
// Confinement trades capacity for locality: size against the LEVEL's aperture,
// never against full L1.
_Static_assert((((long)QWEN_VC_X * QWEN_X_ELEMS_PER_RECT +
                 (long)QWEN_VC_W * QWEN_W_ELEMS_PER_RECT +
                 (long)QWEN_VC_Z * QWEN_Z_ELEMS_PER_RECT) *
                2) <= QWEN_VC_WSIZE(QWEN_VC_LEVEL),
               "confined operands do not fit the level's rectangle: use a "
               "larger level, a squarer cut, or chunk the contraction");
_Static_assert(QWEN_TILE_N % QWEN_PORT_WIDTH == 0,
               "Qwen contraction must tile the RedMulE port");
_Static_assert(QWEN_ROWS_PER_ENGINE <= UINT16_MAX &&
                   QWEN_TILE_N <= UINT16_MAX &&
                   QWEN_COLS_PER_ENGINE <= UINT16_MAX,
               "Qwen RedMulE dimensions must fit uint16_t");

// --------------------------------------------------------------- buffers
//
// An operand placed in a VC window has no static array: it is bump-allocated
// out of the level's arena, identically by every member of the rectangle. An
// operand left global keeps the baseline's `.l1_prio` array.
#if !QWEN_VC_X
__fp16 qwen_l1_x[QWEN_TILE_M * QWEN_TILE_N]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
#endif
#if !QWEN_VC_W
__fp16 qwen_l1_w[QWEN_TILE_N * QWEN_TILE_P]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
#endif
#if !QWEN_VC_Z
__fp16 qwen_l1_z[QWEN_TILE_M * QWEN_TILE_P]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));
#endif

__fp16 qwen_l2_x[QWEN_TILE_M * QWEN_TILE_N]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
__fp16 qwen_l2_w[QWEN_TILE_N * QWEN_TILE_P]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));
__fp16 qwen_l2_z[QWEN_TILE_M * QWEN_TILE_P]
    __attribute__((aligned(4 * NUM_BANKS), section(".l2_bss")));

// `.l1` is NOBITS and the bss range is EMPTY, so nothing zeroes these and they
// power up X in RTL simulation. An X that reaches a branch wedges the core and
// takes the whole cluster down at the next barrier. Initialise explicitly.
uint32_t qwen_vc_alias_bad __attribute__((section(".l1")));

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  const uint32_t redmule_id = mempool_get_redmule_id();
  mempool_barrier_init(core_id);

  // Engine coordinates under the BLOCKED map. Morton makes a contiguous
  // redmule_id run one rectangle, so the rectangle index is a divide; the
  // engine's global (row, col) is then its rectangle's origin plus its slot
  // inside the rectangle's sub-grid. The baseline's flat raster map
  // (r = id / NE_C) is the RR = NUM_RECTS, RC = 1 special case of this.
  // redmule_id is (uint32_t)-1 on cores that drive no engine, so derive the
  // rectangle from the core's own group instead: base_gid / RECT_GROUPS is the
  // same index redmule_id / ENG_PER_RECT gives, because Morton makes a
  // rectangle a contiguous id range and every engine of a group shares it.
  const vc_rect_t myrect = vc_rect(QWEN_VC_LEVEL);
  const uint32_t rect = myrect.base_gid / QWEN_RECT_GROUPS;
  const uint32_t rect_active = rect < QWEN_NUM_RECTS;
  const uint32_t slot = rect_active ? (redmule_id % QWEN_ENG_PER_RECT) : 0;
  const uint32_t rr = rect / QWEN_VC_RC;
  const uint32_t rc = rect % QWEN_VC_RC;
  const uint32_t sr = slot / QWEN_SUB_C;
  const uint32_t sc = slot % QWEN_SUB_C;

  if (core_id == 0) {
    printf("\n[%s] logical=%dx%dx%d tile=%dx%dx%d map=%dx%d engines=%d/%d "
           "port=%d | VC L%d rects=%dx%d sub=%dx%d place=X%dW%dZ%d\n",
           QWEN_BENCHMARK_NAME, QWEN_LOGICAL_M, QWEN_LOGICAL_N, QWEN_LOGICAL_P,
           QWEN_TILE_M, QWEN_TILE_N, QWEN_TILE_P, QWEN_NE_R, QWEN_NE_C,
           QWEN_ACTIVE_ENGINES, NUM_REDMULE_TILES, QWEN_PORT_WIDTH,
           QWEN_VC_LEVEL, QWEN_VC_RR, QWEN_VC_RC, QWEN_SUB_R, QWEN_SUB_C,
           QWEN_VC_X, QWEN_VC_W, QWEN_VC_Z);
    qwen_vc_alias_bad = 0;
  }
  mempool_barrier(num_cores);

  // Reserve the whole level before allocating any of it: a level's base
  // depends on the higher levels' reservations, so a late reservation moves
  // live arenas (vc.h [allocator]).
  vc_init();
  mempool_barrier(num_cores);

  const uint32_t rowsz = vc_row_stride(QWEN_VC_LEVEL);
#define QWEN_VC_ROUND(b) ((((b) + rowsz - 1) / rowsz) * rowsz)
  const uint32_t xb = QWEN_VC_X ? QWEN_VC_ROUND(2u * QWEN_X_ELEMS_PER_RECT) : 0;
  const uint32_t wb = QWEN_VC_W ? QWEN_VC_ROUND(2u * QWEN_W_ELEMS_PER_RECT) : 0;
  const uint32_t zb = QWEN_VC_Z ? QWEN_VC_ROUND(2u * QWEN_Z_ELEMS_PER_RECT) : 0;

  if (vc_reserve(QWEN_VC_LEVEL, xb + wb + zb) < 0) {
    if (core_id == 0)
      printf("[%s] VC reserve of %u B at level %d FAILED\n",
             QWEN_BENCHMARK_NAME, (unsigned)(xb + wb + zb), QWEN_VC_LEVEL);
    return 1;
  }
  // Collective and deterministic: every member of every rectangle walks the
  // same call sequence, so all rectangles agree on these offsets and differ
  // only in the physical groups they resolve to.
  __fp16 *const vx = xb ? (__fp16 *)vc_malloc(QWEN_VC_LEVEL, xb) : NULL;
  __fp16 *const vw = wb ? (__fp16 *)vc_malloc(QWEN_VC_LEVEL, wb) : NULL;
  __fp16 *const vz = zb ? (__fp16 *)vc_malloc(QWEN_VC_LEVEL, zb) : NULL;
  if ((xb && !vx) || (wb && !vw) || (zb && !vz)) {
    if (core_id == 0)
      printf("[%s] VC malloc FAILED at level %d\n", QWEN_BENCHMARK_NAME,
             QWEN_VC_LEVEL);
    return 1;
  }
  const uint32_t wbase = vc_window_base(QWEN_VC_LEVEL);
  mempool_barrier(num_cores);

  // ------------------------------------------------------------------ fill
  // One core per rectangle pulls its own slices out of L2. The engine-blocked
  // L2 layout makes every slice CONTIGUOUS at the source -- X is packed by
  // engine row, W by engine column, Z by flat engine id, and a rectangle owns
  // a contiguous run of each -- so only the VC destination is strided, which
  // is what vc_descriptor walks. Untimed, and deliberately many small 1-D
  // transfers: there is no double buffering here and DMA cost is not part of
  // the result.
  const uint32_t rect_lead_core =
      (myrect.base_gid * QWEN_ENG_PER_GROUP) * (NUM_CORES / NUM_REDMULE_TILES);
#if QWEN_VC_DMA_FILL
  if (rect_active && core_id == rect_lead_core) {
    if (xb) {
      const vc_desc_t dst = vc_descriptor(&myrect, (uint32_t)vx - wbase,
                                          2u * QWEN_X_ELEMS_PER_RECT);
      const vc_desc_t src = {
          .base = (uint32_t)(qwen_l2_x + rr * QWEN_X_ELEMS_PER_RECT),
          .len = 2u * QWEN_X_ELEMS_PER_RECT,
          .count = {1, 1},
          .stride = {0, 0}};
      vc_dma_memcpy_blocking(&dst, &src);
    }
    if (wb) {
      const vc_desc_t dst = vc_descriptor(&myrect, (uint32_t)vw - wbase,
                                          2u * QWEN_W_ELEMS_PER_RECT);
      const vc_desc_t src = {
          .base = (uint32_t)(qwen_l2_w + rc * QWEN_W_ELEMS_PER_RECT),
          .len = 2u * QWEN_W_ELEMS_PER_RECT,
          .count = {1, 1},
          .stride = {0, 0}};
      vc_dma_memcpy_blocking(&dst, &src);
    }
  }
#endif // QWEN_VC_DMA_FILL
  // Operands left global keep the baseline's single-core DMA.
  if (core_id == 0) {
#if !QWEN_VC_X
    dma_memcpy_blocking(qwen_l1_x, qwen_l2_x,
                        sizeof(__fp16) * QWEN_TILE_M * QWEN_TILE_N);
#endif
#if !QWEN_VC_W
    dma_memcpy_blocking(qwen_l1_w, qwen_l2_w,
                        sizeof(__fp16) * QWEN_TILE_N * QWEN_TILE_P);
#endif
  }
  mempool_barrier(num_cores);

  // ------------------------------------------------------- alias equivalence
  // [EOC] retval=0 cannot show this: the apps are perf-only and qwen_l2_* is
  // untouched .l2_bss, so a read-back against L2 compares zeros to zeros and
  // passes wherever the pointer resolved. Write through the DMA's view
  // (vc_global_addr, plain arithmetic) and read through the compute view
  // (vc_ptr, the hardware permuter): two independent address computations, so
  // agreement cannot be accidental. Integer pattern on purpose -- the probe
  // tests ADDRESSES, and routing it through __fp16 drags in unrelated
  // datapath behaviour.
#define QWEN_VC_PROBE_N 16
  if (zb) {
    const uint32_t zoff = (uint32_t)vz - wbase;
    const uint32_t step = (zb / QWEN_VC_PROBE_N) & ~(uint32_t)1;
    if (rect_active && core_id == rect_lead_core) {
      for (uint32_t i = 0; i < QWEN_VC_PROBE_N; i++)
        *(volatile uint16_t *)vc_global_addr(&myrect, zoff + i * step) =
            (uint16_t)(0xA500u + i);
      // Read the pattern back through the SAME (global) view before releasing
      // the readers. The stores are posted and the fence orders them against
      // each other, not against their arrival in the bank: without this the
      // barrier can release while the last stores are still in the NoC, and an
      // engine reading through the VC view sees the old contents. That showed
      // up as a probe failure only at level 0 -- the shortest path, so the
      // smallest window -- and disappeared the moment a printf was added to
      // the reader, which is the signature of a visibility race rather than a
      // wrong address.
      volatile uint16_t sink = 0;
      for (uint32_t i = 0; i < QWEN_VC_PROBE_N; i++)
        sink = *(volatile uint16_t *)vc_global_addr(&myrect, zoff + i * step);
      (void)sink;
      __sync_synchronize();
    }
    mempool_barrier(num_cores);
    if (rect_active && redmule_id < QWEN_ACTIVE_ENGINES) {
      uint32_t bad = 0;
      for (uint32_t i = 0; i < QWEN_VC_PROBE_N; i++)
        if (*(volatile uint16_t *)vc_ptr(QWEN_VC_LEVEL, zoff + i * step) !=
            (uint16_t)(0xA500u + i)) {
          bad++;
#ifdef QWEN_VC_PROBE_TRACE
          if (bad == 1)
            printf(
                "[probe] eng=%u rect=%u gid=%u base_gid=%u off=%u vcptr=0x%x "
                "glob=0x%x want=0x%x got=0x%x\n",
                (unsigned)redmule_id, (unsigned)rect,
                (unsigned)(mempool_get_core_id() / NUM_CORES_PER_GROUP),
                (unsigned)myrect.base_gid, (unsigned)(zoff + i * step),
                (unsigned)(uint32_t)vc_ptr(QWEN_VC_LEVEL, zoff + i * step),
                (unsigned)vc_global_addr(&myrect, zoff + i * step),
                (unsigned)(0xA500u + i),
                (unsigned)*(volatile uint16_t *)vc_ptr(QWEN_VC_LEVEL,
                                                       zoff + i * step));
#endif
        }
      if (bad) {
        qwen_vc_alias_bad = bad;
        __sync_synchronize();
      }
    }
    mempool_barrier(num_cores);
    if (core_id == 0)
      printf("[%s] VC alias probe %s (level %d, %u samples)\n",
             QWEN_BENCHMARK_NAME, qwen_vc_alias_bad ? "FAILED" : "OK",
             QWEN_VC_LEVEL, QWEN_VC_PROBE_N);
  }
  mempool_barrier(num_cores);

  // Zero this engine's Z block. Must happen after the probe, which writes a
  // pattern into the same rows.
  __fp16 *const ze =
      zb ? vz + slot * QWEN_ROWS_PER_ENGINE * QWEN_COLS_PER_ENGINE : NULL;
#if QWEN_VC_Z
  if (rect_active && redmule_id < QWEN_ACTIVE_ENGINES)
    llm_zero_f16(ze, QWEN_ROWS_PER_ENGINE * QWEN_COLS_PER_ENGINE);
#else
  if (rect_active && redmule_id < QWEN_ACTIVE_ENGINES)
    llm_zero_f16(qwen_l1_z +
                     redmule_id * QWEN_ROWS_PER_ENGINE * QWEN_COLS_PER_ENGINE,
                 QWEN_ROWS_PER_ENGINE * QWEN_COLS_PER_ENGINE);
#endif
  mempool_barrier(num_cores);

  // ------------------------------------------------------------------ timed
  // Operand bases. A confined operand is indexed by the engine's slot WITHIN
  // its rectangle; a global one keeps the baseline's whole-grid index.
#if QWEN_VC_X
  const __fp16 *const xe = vx + sr * QWEN_ROWS_PER_ENGINE * QWEN_TILE_N;
#else
  const __fp16 *const xe =
      qwen_l1_x + (rr * QWEN_SUB_R + sr) * QWEN_ROWS_PER_ENGINE * QWEN_TILE_N;
#endif
#if QWEN_VC_W
  const __fp16 *const we = vw + sc * QWEN_TILE_N * QWEN_COLS_PER_ENGINE;
#else
  const __fp16 *const we =
      qwen_l1_w + (rc * QWEN_SUB_C + sc) * QWEN_TILE_N * QWEN_COLS_PER_ENGINE;
#endif
#if QWEN_VC_Z
  __fp16 *const zeng = ze;
#else
  __fp16 *const zeng =
      qwen_l1_z + redmule_id * QWEN_ROWS_PER_ENGINE * QWEN_COLS_PER_ENGINE;
#endif

  const uint32_t time_init = mempool_get_timer();
  mempool_start_benchmark();
  if (rect_active && redmule_id < QWEN_ACTIVE_ENGINES)
    qwen_redmule_op_engine_f16(xe, we, zeng, QWEN_ROWS_PER_ENGINE, QWEN_TILE_N,
                               QWEN_COLS_PER_ENGINE);
  // Software log barrier: the hardware fractal_sync barrier is not stable
  // enough to sit inside a timed region yet.
  mempool_log_barrier(2, core_id);
  mempool_stop_benchmark();
  const uint32_t time_end = mempool_get_timer();

  if (core_id == 0) {
    printf("[%s] stage=gemm cycles=%u\n", QWEN_BENCHMARK_NAME,
           time_end - time_init);
    printf("[%s] kernel takes %u clock cycles\n", QWEN_BENCHMARK_NAME,
           time_end - time_init);
  }
  mempool_barrier(num_cores);
  return qwen_vc_alias_bad ? 1 : 0;
}

#endif // __QWEN_VC_REDMULE_OP_MAIN_H__
