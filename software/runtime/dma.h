// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Samuel Riedel, ETH Zurich

#ifndef _DMA_H_
#define _DMA_H_

#include <stddef.h>
#include <stdint.h>

#include "addrmap.h"
#include "fsync.h"
#include "runtime.h"

// iDMA v0.6.5 reg32_3d register map. The descriptor registers are one context
// per group; status and transfer ids are per stream.
//
// The engine has no interrupt and no error register: software polls STATUS
// (busy) and DONE_ID (completion), and both are exact. The one event the
// hardware emits is the completion pulse behind FSYNC_EVT_DMA, which
// dma_event_gate uses; the runtime polls.
// Second 64 KiB of the peripheral window (mempool_pkg::DMABaseAddr).
#define DMA_BASE (CONTROL_REGISTER_OFFSET + 0x10000u)
#define DMA_CONF_OFFSET 0x000u
#define DMA_STATUS_OFFSET(stream) (0x004u + 4u * (stream))
#define DMA_NEXT_ID_OFFSET(stream) (0x044u + 4u * (stream))
#define DMA_DONE_ID_OFFSET(stream) (0x084u + 4u * (stream))
#define DMA_DST_ADDR_OFFSET 0x0d0u
#define DMA_SRC_ADDR_OFFSET 0x0d8u
#define DMA_LENGTH_OFFSET 0x0e0u
#define DMA_DST_STRIDE_2_OFFSET 0x0e8u
#define DMA_SRC_STRIDE_2_OFFSET 0x0f0u
#define DMA_REPS_2_OFFSET 0x0f8u
#define DMA_DST_STRIDE_3_OFFSET 0x100u
#define DMA_SRC_STRIDE_3_OFFSET 0x108u
#define DMA_REPS_3_OFFSET 0x110u

#define DMA_CONF_ENABLE_ND_BIT 10u
#define DMA_CONF_ENABLE_ND_MASK (0x3u << DMA_CONF_ENABLE_ND_BIT)
#define DMA_STATUS_BUSY_MASK 0x3ffu

typedef uint32_t dma_id_t;

enum {
  DMA_LOCAL_OK = 0,
  DMA_LOCAL_ERROR_ARGUMENT = 0x80000001u,
};

// THE ENGINE'S DESCRIPTOR REGISTERS ARE ONE CONTEXT PER GROUP.
//
// DMA_BASE carries no group index: the context is the issuing core's own
// group's. Programming a transfer takes SEVERAL writes -- CONF, SRC, DST,
// LENGTH, the strides, then the NEXT_ID read that launches it -- so the context
// has exactly one issuer, and the hardware enforces which: only core 0 of tile
// 0, i.e. mempool_is_group_coordinator(), is wired to it (mempool_tile.sv,
// mempool_group.sv). Core 0 of the other tiles reads 0 with an error response
// and its writes are dropped; the other cores have no DMA window at all, so an
// access goes out of the SoC port like any unmapped address. Every DMA access,
// status polls included, must therefore come from the coordinator; others
// learn completion from it (dma_collective.c) or from the DMA done event.
//
// KEEP EACH CONTIGUOUS CHUNK WITHIN THE APPLICABLE L1 AND L2 STRIPES.
//
// An L2 AXI burst must stay within one bank stripe:
// AXI_DATA_WIDTH/8 * AXI_WIDTH_INTERLEAVED bytes (64 B on minpool).
// axi_L2_interleaver permutes only the burst's start address; it does not
// split bursts at bank boundaries. Crossing one can silently access the
// wrong data. AXI beat size is configuration-dependent: AXI_DATA_WIDTH/8.
//
// Every L1 address must belong to the issuing group's stripe of the global
// interleaved map, NUM_BANKS/NUM_GROUPS*4 bytes (64 B on minpool).
// mempool_group selects a local tile and drops the address's group field
// when constructing its bank/word address. A foreign-group address therefore
// aliases the issuing group's banks at the same word, with a displacement
// determined by the two group ids; it is not always one stripe low.
// Splitting L2 bursts alone does not correct a foreign-group L1 address.
//
// With the default burst configuration, keep each contiguous descriptor
// chunk within both applicable stripes, including its starting offset.
// For 2-D/3-D descriptors this applies to each repeated chunk: one descriptor
// may cover multiple rows if its strides preserve L1 ownership and L2 burst
// boundaries. dma_collective.c handles these constraints for linear copies;
// raw dma_local_* callers must enforce them themselves.

static inline volatile uint32_t *dma_local_reg(uint32_t offset) {
  return (volatile uint32_t *)(DMA_BASE + offset);
}

static inline uint32_t dma_local_status(uint32_t stream) {
  return *dma_local_reg(DMA_STATUS_OFFSET(stream));
}

static inline uint32_t dma_local_idle(uint32_t stream) {
  return (dma_local_status(stream) & DMA_STATUS_BUSY_MASK) == 0;
}

static inline dma_id_t dma_local_completed_id(uint32_t stream) {
  return *dma_local_reg(DMA_DONE_ID_OFFSET(stream));
}

static inline uint32_t dma_local_id_done(dma_id_t done, dma_id_t target) {
  return (int32_t)(done - target) >= 0;
}

static inline uint32_t dma_local_done_id(uint32_t stream, dma_id_t id) {
  return dma_local_id_done(dma_local_completed_id(stream), id);
}

// Both waits POLL. The engine's completion pulse (FSYNC_EVT_DMA) is exact, but
// a poll is as cheap here -- the register file is ~4 cycles a read, L1 speed --
// and it needs no subscription discipline: no unsubscribe to restore, no stale
// pending bit to keep out of the next barrier. Measured on dse_s1_winner at
// 512 KiB, sleeping instead of polling moved nothing: bw in 1559/1559.
//
// Wait until one request has landed; waiting an id waits every earlier one.
static inline uint32_t dma_local_wait_id(uint32_t stream, dma_id_t id) {
  if (stream >= (uint32_t)NUM_DMAS_PER_GROUP || id == 0)
    return DMA_LOCAL_ERROR_ARGUMENT;
  while (!dma_local_done_id(stream, id))
    mempool_wait(4u);
  __sync_synchronize();
  return DMA_LOCAL_OK;
}

// Wait until every request accepted by one stream has landed. The status
// includes the frontend FIFO, ND midend and backend, so a wait right after an
// asynchronous launch cannot observe a false idle gap.
static inline uint32_t dma_local_wait(uint32_t stream) {
  if (stream >= (uint32_t)NUM_DMAS_PER_GROUP)
    return DMA_LOCAL_ERROR_ARGUMENT;
  while (!dma_local_idle(stream))
    mempool_wait(4u);
  __sync_synchronize();
  return DMA_LOCAL_OK;
}

#if DOMAINS > 1
#include "dma_fold.h"
#endif

// The global-map alias of an L1 pointer: identity, except that a domain build
// (addrmap.h) sees its global map through its top-level window. Code that
// classifies L1 against L2 by address works on aliases.
static inline uint32_t dma_l1_alias(uint32_t a) {
#if DOMAINS > 1
  if (a >= DOMAIN_L1_DATA_BASE && a - DOMAIN_L1_DATA_BASE < DMA_DOMAIN_L1_BYTES)
    return a - DOMAIN_L1_DATA_BASE;
#endif
  return a;
}

#if DOMAINS > 1 && defined(DMA_DOMAIN_CHECK) && DMA_DOMAIN_CHECK
// DMA_DOMAIN_CHECK also flags any read whose chip source reaches the group
// RO-cache window [L2 base, _erodata): cached and bypass DMA reads there are
// modelled unfaithfully (gvsoc-da 10-19). Counted per domain, first 8 printed.
extern uint32_t volatile dma_check_ro_hits;
static void dma_check_word(char tag, uint32_t v);
static __attribute__((noinline)) void dma_check_ro(uint32_t c, uint32_t a2,
                                                   uint32_t a3, uint32_t len) {
  extern char _erodata[];
  if (!dma_l2_low_overlap(c, a2, a3, len, (uint32_t)_erodata))
    return;
  if (__atomic_fetch_add(&dma_check_ro_hits, 1u, __ATOMIC_RELAXED) < 8u) {
    dma_check_word('O', c);
    dma_check_word('M', mempool_get_domain_id());
    dma_check_word('a', a2);
    dma_check_word('z', len);
  }
}
// ... and any write whose chip destination reaches L2 below the home base or
// the program image, whichever ends higher: a home walk that wraps onto the
// image overwrites code (window C hang, fold carry errors; 10-19).
extern uint32_t volatile dma_check_wr_hits;
static __attribute__((noinline)) void dma_check_wr(uint32_t c, uint32_t a2,
                                                   uint32_t a3, uint32_t len) {
  extern char __l2_bss_end[];
  const uint32_t img = (uint32_t)__l2_bss_end;
  const uint32_t end =
      img > DMA_DOMAIN_HOME_BASE ? img : (uint32_t)DMA_DOMAIN_HOME_BASE;
  if (!dma_l2_low_overlap(c, a2, a3, len, end))
    return;
  if (__atomic_fetch_add(&dma_check_wr_hits, 1u, __ATOMIC_RELAXED) < 8u) {
    dma_check_word('W', c);
    dma_check_word('M', mempool_get_domain_id());
    dma_check_word('a', a2);
    dma_check_word('z', len);
  }
}
#endif

// Always inlined where a caller must stay a leaf (dma_domain_home_walk); the
// plain dma_local_issue_nd below leaves the choice to the compiler.
static inline __attribute__((always_inline)) dma_id_t
dma_local_issue_nd_inl(uint32_t stream, void *dest, const void *src, size_t len,
                       uint32_t dst_stride_2, uint32_t src_stride_2,
                       uint32_t reps_2, uint32_t dst_stride_3,
                       uint32_t src_stride_3, uint32_t reps_3,
                       uint32_t dimensions) {
#if DOMAINS > 1
  {
    // The midend steps stride_2 after each burst and stride_3 from a plane's
    // LAST burst (dma_collective.c), so translate each where it is taken.
    const uint32_t d = (uint32_t)dest, s = (uint32_t)src;
    const uint32_t r2 = dimensions >= 2 ? reps_2 : 1u;
    const uint32_t d2 = dimensions >= 2 ? dst_stride_2 : 0u;
    const uint32_t s2 = dimensions >= 2 ? src_stride_2 : 0u;
    if (dimensions >= 3) {
      dst_stride_3 = dma_domain_stride(d, d + (r2 - 1u) * d2, dst_stride_3,
                                       dst_stride_3 + (r2 - 1u) * d2, reps_3);
      src_stride_3 = dma_domain_stride(s, s + (r2 - 1u) * s2, src_stride_3,
                                       src_stride_3 + (r2 - 1u) * s2, reps_3);
    }
    if (dimensions >= 2) {
      dst_stride_2 = dma_domain_stride(d, d, d2, d2, r2);
      src_stride_2 = dma_domain_stride(s, s, s2, s2, r2);
    }
    dest = (void *)dma_domain_addr(d);
    src = (const void *)dma_domain_addr(s);
#if defined(DMA_DOMAIN_CHECK) && DMA_DOMAIN_CHECK
    {
      const uint32_t a2 = dimensions >= 2 ? (r2 - 1u) * src_stride_2 : 0u;
      dma_check_ro((uint32_t)src, a2,
                   dimensions >= 3 ? (reps_3 - 1u) * (a2 + src_stride_3) : 0u,
                   (uint32_t)len);
      const uint32_t w2 = dimensions >= 2 ? (r2 - 1u) * dst_stride_2 : 0u;
      dma_check_wr((uint32_t)dest, w2,
                   dimensions >= 3 ? (reps_3 - 1u) * (w2 + dst_stride_3) : 0u,
                   (uint32_t)len);
    }
#endif
  }
#endif

  // Program only the dimensions this descriptor uses: CONF's ENABLE_ND field
  // tells the ND midend how many to walk, so the registers above it are dead.
  uint32_t conf = *dma_local_reg(DMA_CONF_OFFSET);
  conf &= ~DMA_CONF_ENABLE_ND_MASK;
  if (dimensions >= 2)
    conf |= (dimensions - 1u) << DMA_CONF_ENABLE_ND_BIT;
  *dma_local_reg(DMA_CONF_OFFSET) = conf;
  *dma_local_reg(DMA_SRC_ADDR_OFFSET) = (uint32_t)src;
  *dma_local_reg(DMA_DST_ADDR_OFFSET) = (uint32_t)dest;
  *dma_local_reg(DMA_LENGTH_OFFSET) = (uint32_t)len;
  if (dimensions >= 2) {
    *dma_local_reg(DMA_SRC_STRIDE_2_OFFSET) = src_stride_2;
    *dma_local_reg(DMA_DST_STRIDE_2_OFFSET) = dst_stride_2;
    *dma_local_reg(DMA_REPS_2_OFFSET) = reps_2;
  }
  if (dimensions >= 3) {
    *dma_local_reg(DMA_SRC_STRIDE_3_OFFSET) = src_stride_3;
    *dma_local_reg(DMA_DST_STRIDE_3_OFFSET) = dst_stride_3;
    *dma_local_reg(DMA_REPS_3_OFFSET) = reps_3;
  }
  __sync_synchronize();
  const dma_id_t id = *dma_local_reg(DMA_NEXT_ID_OFFSET(stream));
  __sync_synchronize();
  return id;
}

static inline dma_id_t
dma_local_issue_nd(uint32_t stream, void *dest, const void *src, size_t len,
                   uint32_t dst_stride_2, uint32_t src_stride_2,
                   uint32_t reps_2, uint32_t dst_stride_3,
                   uint32_t src_stride_3, uint32_t reps_3,
                   uint32_t dimensions) {
  return dma_local_issue_nd_inl(stream, dest, src, len, dst_stride_2,
                                src_stride_2, reps_2, dst_stride_3,
                                src_stride_3, reps_3, dimensions);
}

#if DOMAINS > 1 && !defined(DOMAIN_L2_SPREAD)
#define DOMAIN_L2_SPREAD                                                       \
  0 // measured slower for RedMulE (rows arrive out of need order)
#endif
#if DOMAINS > 1 && DOMAIN_L2_SPREAD
// The chip's L2 period (its L1 row) is DOMAINS domain rows, and the domains run
// the same walks over the same L2 offsets in lockstep, so they would all hit
// the same banks. Rotating a walk so that domain d starts d domain rows further
// into the period puts the domains on disjoint banks; data is unchanged.
#define DMA_DOMAIN_L2_PERIOD (DMA_DOMAIN_ROW_BYTES * DOMAINS)
_Static_assert((DMA_DOMAIN_L2_PERIOD & (DMA_DOMAIN_L2_PERIOD - 1u)) == 0,
               "L2 spread needs a power-of-two period");
// k with k * step == target (mod period), or ~0 if no k < reps does it.
static inline uint32_t dma_domain_rotation(uint32_t step, uint32_t target,
                                           uint32_t reps) {
  step &= DMA_DOMAIN_L2_PERIOD - 1u;
  if (!step)
    return ~0u;
  const uint32_t t = (uint32_t)__builtin_ctz(step);
  if (target & ((1u << t) - 1u))
    return ~0u;
  const uint32_t a = step >> t;
  uint32_t inv = a; // Newton: odd a^-1 mod 2^32, 3 -> 48 bits
  for (int i = 0; i < 4; ++i)
    inv *= 2u - a * inv;
  const uint32_t k = ((target >> t) * inv) & ((DMA_DOMAIN_L2_PERIOD >> t) - 1u);
  return k < reps ? k : ~0u;
}

// A read burst to another L2 bank waits for the stream's reads in flight (the
// FlooNoC NoRoB order), and the chip's interleave moves a domain-row walk to
// another bank at every row, so each row pays a whole round trip. Rows r and
// r + p of a walk whose L2 step has period p share a bank: walking them back to
// back pays it once per p-row plane instead.
#ifndef DOMAIN_L2_BANKS
#define DOMAIN_L2_BANKS 32u // the chip's L2 channels: 4N on an NxN group mesh
#endif
#ifndef DOMAIN_L2_MAX_PLANES
#define DOMAIN_L2_MAX_PLANES 4u // residues a 3-D walk may split into
#endif
#ifndef DOMAIN_L2_MAX_DESC
#define DOMAIN_L2_MAX_DESC                                                     \
  8u // descriptors a walk split per outer plane may use
#endif
#define DMA_DOMAIN_L2_GRANULE (DMA_DOMAIN_L2_PERIOD / DOMAIN_L2_BANKS)
static inline uint32_t dma_domain_planes(uint32_t step) {
  const uint32_t m = step & (DMA_DOMAIN_L2_PERIOD - 1u);
  if (!m || step % DMA_DOMAIN_L2_GRANULE)
    return 1u;
  return DMA_DOMAIN_L2_PERIOD >> __builtin_ctz(m);
}
// Planes [j0, j1) of a 2-D walk, all of n rows, as one 3-D walk.
static inline dma_id_t dma_domain_planes_2d(uint32_t stream, char *d,
                                            const char *s, size_t len,
                                            uint32_t d2, uint32_t s2,
                                            uint32_t n, uint32_t p, uint32_t j0,
                                            uint32_t j1) {
  return dma_local_issue_nd(stream, d + j0 * d2, s + j0 * s2, len, p * d2,
                            p * s2, n, d2 - (n - 1u) * p * d2,
                            s2 - (n - 1u) * p * s2, j1 - j0, 3);
}
// Planes [j0, j1) of a 2-D walk of `reps` rows: the first reps % p planes are
// one row longer, so split there.
static inline dma_id_t dma_domain_planes_run(uint32_t stream, char *d,
                                             const char *s, size_t len,
                                             uint32_t d2, uint32_t s2,
                                             uint32_t reps, uint32_t p,
                                             uint32_t j0, uint32_t j1) {
  const uint32_t hi = reps % p, n = reps / p;
  dma_id_t id = 0;
  if (j0 < hi)
    id = dma_domain_planes_2d(stream, d, s, len, d2, s2, n + 1u, p, j0,
                              j1 < hi ? j1 : hi);
  if (j1 > hi && j1 > j0)
    id = dma_domain_planes_2d(stream, d, s, len, d2, s2, n, p,
                              j0 > hi ? j0 : hi, j1);
  return id;
}
#endif

#if DOMAINS > 1 && DOMAIN_L2_HOME
#ifndef DOMAIN_L2_MAX_DESC
#define DOMAIN_L2_MAX_DESC 8u
#endif
// One outer plane of a home-folded walk whose L2 step divides a row: q steps
// per row become the inner dimension, rows the outer one, so each row crossing
// is the same carry. Rows that do not fill a row run as a 2-D tail.
static inline dma_id_t dma_domain_home_plane(uint32_t stream, char *d,
                                             const char *s, size_t len,
                                             uint32_t d2, uint32_t s2,
                                             uint32_t reps, uint32_t q) {
  const uint32_t rows = reps / q, tail = reps % q;
  dma_id_t id = 0;
  if (rows)
    id = dma_local_issue_nd_inl(stream, d, s, len, d2, s2, q, d2, s2, rows,
                                rows > 1u ? 3u : 2u);
  if (tail)
    id = dma_local_issue_nd_inl(stream, d + rows * q * d2, s + rows * q * s2,
                                len, d2, s2, tail, 0u, 0u, 1u,
                                tail > 1u ? 2u : 1u);
  return id;
}
// Every outer plane of a home-folded walk, out of line with its loop state in
// a per-core L1 slot instead of on the stack: the per-core stack is 512 B, and
// the attention agents' DMA paths overflowed it with the walk inlined.
typedef struct {
  char *d;
  const char *s;
  size_t len;
  uint32_t d2, s2, reps, q, dq, sq;
} dma_home_walk_t;
static dma_home_walk_t dma_home_walks[NUM_CORES]
    __attribute__((section(".l1")));
static __attribute__((noinline)) dma_id_t
dma_domain_home_walk(uint32_t stream, dma_home_walk_t *w, uint32_t r3) {
  dma_id_t id = 0;
  for (uint32_t i = 0; i < r3; ++i)
    id = dma_domain_home_plane(stream, w->d + i * w->dq, w->s + i * w->sq,
                               w->len, w->d2, w->s2, w->reps, w->q);
  return id;
}
#endif

// Exact home fold (default off): every byte lands at dma_domain_addr(byte); a
// descriptor the fold cannot map as a whole is split (dma_fold_exact.h).
// DMA_DOMAIN_CHECK (debug, needs it): a descriptor whose raw domain addresses
// leave their region is printed straight to the console (_putchar prints
// domain 0 only), up to 8 per run, and dropped.
#ifndef DOMAIN_L2_EXACT
#define DOMAIN_L2_EXACT 0
#endif
#ifndef DMA_DOMAIN_CHECK
#define DMA_DOMAIN_CHECK 0
#endif
#if DMA_DOMAIN_CHECK && !DOMAIN_L2_EXACT
#error "DMA_DOMAIN_CHECK runs inside the exact fold: set DOMAIN_L2_EXACT"
#endif
// DMA_RO_PROBE (CHECK builds): each domain's core 0 reads 64 B from L2_BASE +
// 0x100 at start (synchronization.c), a planted positive control for the
// RO-window check: every domain must then report at least one hit.
#ifndef DMA_RO_PROBE
#define DMA_RO_PROBE 0
#endif
#if DMA_RO_PROBE && !DMA_DOMAIN_CHECK
#error "DMA_RO_PROBE is a control for DMA_DOMAIN_CHECK: set it"
#endif
// DMA_FOLD_PROFILE (debug, needs it): per call site (the hook's pc), calls,
// pieces (sum, min, max) and fold cycles, per domain; domain 0 prints every
// domain's table at exit (synchronization.c).
#ifndef DMA_FOLD_PROFILE
#define DMA_FOLD_PROFILE 0
#endif
#if DMA_FOLD_PROFILE && !DOMAIN_L2_EXACT
#error "DMA_FOLD_PROFILE runs inside the exact fold: set DOMAIN_L2_EXACT"
#endif
#if DOMAIN_L2_EXACT
#if !(DOMAINS > 1 && DOMAIN_L2_HOME) || DOMAIN_L2_SPREAD
#error "DOMAIN_L2_EXACT needs a domain build with DOMAIN_L2_HOME, no SPREAD"
#endif
#define DMA_FOLD_EMIT(stream, d, s, len, d2, s2, r2, d3, s3, r3, dims)         \
  dma_local_issue_nd_inl((stream), (void *)(d), (const void *)(s), (len),      \
                         (d2), (s2), (r2), (d3), (s3), (r3), (dims))
#include "dma_fold_exact.h"

#if DMA_DOMAIN_CHECK
extern char fake_uart;
extern uint32_t volatile dma_check_hits;
static __attribute__((noinline)) void dma_check_word(char tag, uint32_t v) {
  *(volatile char *)&fake_uart = tag;
  *(volatile char *)&fake_uart = '=';
  for (int32_t b = 28; b >= 0; b -= 4)
    *(volatile char *)&fake_uart = "0123456789abcdef"[(v >> b) & 15u];
  *(volatile char *)&fake_uart = tag == 'z' ? '\n' : ' ';
}
// Whether every burst of one side lies in its region: home L2, domain L1 (or
// its window), or, untranslated, the program image's L2.
static inline uint32_t dma_check_side(uint32_t a, uint32_t len, uint32_t t2,
                                      uint32_t r2, uint32_t t3, uint32_t r3) {
  const uint32_t k = dma_fold_kind(a);
  const uint32_t lo = k == 2u ? DMA_DOMAIN_HOME_BASE
                      : k == 1u
                          ? (a < DMA_DOMAIN_L1_BYTES ? 0u : DOMAIN_L1_DATA_BASE)
                          : (uint32_t)L2_BASE;
  const uint32_t n =
      k == 1u   ? DMA_DOMAIN_L1_BYTES
      : k == 2u ? (uint32_t)L2_BASE + (uint32_t)L2_SIZE - DMA_DOMAIN_HOME_BASE
                : DMA_DOMAIN_HOME_BASE - (uint32_t)L2_BASE;
  const uint32_t a2 = (r2 - 1u) * t2, a3 = (r3 - 1u) * ((r2 - 1u) * t2 + t3);
  const uint32_t x[4] = {a - lo, a + a2 - lo, a + a3 - lo, a + a2 + a3 - lo};
  for (uint32_t i = 0u; i < 4u; ++i)
    if (x[i] >= n || len > n - x[i])
      return 0u;
  return 1u;
}
#endif

// The domain path, on its own stack: the agents' DMA call chain already uses
// the whole 512 B core stack (stackdepth.py), so dma_fold_call runs this on a
// per-core L1 slot instead. It must hold the 64 B call frame plus
// dma_fold_entry and dma_fold_walk, counted as nested although the walk is a
// tail call (48 + 224 B with LLVM). DMA_DOMAIN_CHECK's RO-window check grows
// the walk (64 + 256 + 16 B), so CHECK builds get a larger slot.
#if DMA_DOMAIN_CHECK
#define DMA_FOLD_STACK_BYTES 448u
#else
#define DMA_FOLD_STACK_BYTES 384u
#endif
typedef struct {
  dma_fold_t w;
  uint32_t pad[(16u - sizeof(dma_fold_t) % 16u) % 16u / 4u];
  uint32_t stack[DMA_FOLD_STACK_BYTES / 4u];
} dma_fold_slot_t;
// One array for the program (synchronization.c), not one per translation unit.
extern dma_fold_slot_t dma_fold_slots[NUM_CORES];
// A core reaches its slot through the VC tile window: there the array's rows
// are its own tile's bank slices, NUM_CORES_PER_TILE contiguous slots. In the
// interleaved map a slot spans remote banks, and GVSoC can serve a reload
// that follows a store to the same word with the stale value: a spilled
// register or descriptor field of the fold went wrong (2026-10-19).
#ifndef VC_TILE_WINDOW_BASE
#error "DOMAIN_L2_EXACT keeps the fold slots in the VC tile window"
#endif
_Static_assert(NUM_CORES * sizeof(dma_fold_slot_t) % (NUM_BANKS * 4u) == 0u,
               "The fold slots fill whole global L1 rows");
// Slots and their stack tops 16-B aligned (psABI; the trampoline assumes it).
_Static_assert(sizeof(dma_fold_slot_t) % 16u == 0u &&
                   DMA_FOLD_STACK_BYTES % 16u == 0u,
               "Fold slots keep 16-B stack alignment");
// Tiles per global row; NUM_BANKS_PER_TILE is an unparenthesized product.
#define DMA_FOLD_ROW_TILES (NUM_BANKS / (NUM_BANKS_PER_TILE))
_Static_assert(DMA_FOLD_ROW_TILES *(NUM_BANKS_PER_TILE) == NUM_BANKS,
               "Whole tiles per global row");
static inline dma_fold_slot_t *dma_fold_slot(void) {
  // The array starts on a row: its tile-window offset is its global offset
  // over the tiles of a row.
  const uint32_t off =
      ((uint32_t)dma_fold_slots - DOMAIN_L1_DATA_BASE) / DMA_FOLD_ROW_TILES;
  return (dma_fold_slot_t *)(VC_TILE_WINDOW_BASE + off) +
         mempool_get_core_id() % NUM_CORES_PER_TILE;
}
#if DMA_FOLD_PROFILE
// Direct-mapped on the call site; a site whose entry another site holds counts
// in dma_fold_prof_lost instead.
#define DMA_FOLD_PROF_SITES 128u
typedef struct {
  uint32_t pc, calls, pieces, minp, maxp, cycles;
} dma_fold_prof_t;
extern dma_fold_prof_t dma_fold_prof[DMA_FOLD_PROF_SITES];
extern uint32_t volatile dma_fold_prof_lost;
static __attribute__((noinline)) void
dma_fold_prof_add(uint32_t pc, uint32_t pieces, uint32_t cycles) {
  dma_fold_prof_t *const e = &dma_fold_prof[(pc >> 2) % DMA_FOLD_PROF_SITES];
  if (__atomic_load_n(&e->pc, __ATOMIC_RELAXED) == 0u)
    __atomic_store_n(&e->pc, pc, __ATOMIC_RELAXED);
  if (__atomic_load_n(&e->pc, __ATOMIC_RELAXED) != pc) {
    __atomic_fetch_add(&dma_fold_prof_lost, 1u, __ATOMIC_RELAXED);
    return;
  }
  __atomic_fetch_add(&e->calls, 1u, __ATOMIC_RELAXED);
  __atomic_fetch_add(&e->pieces, pieces, __ATOMIC_RELAXED);
  __atomic_fetch_add(&e->cycles, cycles, __ATOMIC_RELAXED);
  __asm__ volatile("amominu.w zero, %1, (%0)"
                   :
                   : "r"(&e->minp), "r"(pieces)
                   : "memory");
  __asm__ volatile("amomaxu.w zero, %1, (%0)"
                   :
                   : "r"(&e->maxp), "r"(pieces)
                   : "memory");
}
#endif
static __attribute__((noinline, used)) dma_id_t dma_fold_entry(dma_fold_t *w) {
#if DMA_FOLD_PROFILE
  const uint32_t t0 = (uint32_t)mempool_get_timer();
#endif
#if DMA_DOMAIN_CHECK
  if (!dma_check_side(w->d, w->len, w->d2, w->r2, w->d3, w->r3) ||
      !dma_check_side(w->s, w->len, w->s2, w->r2, w->s3, w->r3)) {
    if (__atomic_fetch_add(&dma_check_hits, 1u, __ATOMIC_RELAXED) < 8u) {
      dma_check_word('R', w->ra);
      dma_check_word('C', mempool_get_core_id());
      dma_check_word('M', mempool_get_domain_id());
      dma_check_word('D', w->d);
      dma_check_word('S', w->s);
      dma_check_word('L', w->len);
      dma_check_word('d', w->d2);
      dma_check_word('s', w->s2);
      dma_check_word('r', w->r2);
      dma_check_word('e', w->d3);
      dma_check_word('f', w->s3);
      dma_check_word('q', w->r3);
      dma_check_word('z', w->dims);
    }
    return 0;
  }
#endif
#if DMA_FOLD_PROFILE
  dma_id_t id;
  uint32_t pieces = 1u;
  if (dma_fold_side_exact(w->d, w->len, w->d2, w->r2, w->d3, w->r3) &&
      dma_fold_side_exact(w->s, w->len, w->s2, w->r2, w->s3, w->r3))
    id = dma_local_issue_nd_inl(w->stream, (void *)w->d, (const void *)w->s,
                                w->len, w->d2, w->s2, w->r2, w->d3, w->s3,
                                w->r3, w->dims);
  else
    id = dma_fold_walk(w), pieces = w->n;
  dma_fold_prof_add(w->ra, pieces, (uint32_t)mempool_get_timer() - t0);
  return id;
#else
  if (dma_fold_side_exact(w->d, w->len, w->d2, w->r2, w->d3, w->r3) &&
      dma_fold_side_exact(w->s, w->len, w->s2, w->r2, w->s3, w->r3))
    return dma_local_issue_nd_inl(w->stream, (void *)w->d, (const void *)w->s,
                                  w->len, w->d2, w->s2, w->r2, w->d3, w->s3,
                                  w->r3, w->dims);
  return dma_fold_walk(w);
#endif
}
// Calls dma_fold_entry(&slot->w) with sp at the top of the slot's stack. The
// call saves every caller-saved register it may clobber on that stack, not on
// the caller's: the caller compiles as if no call were made, so its frame does
// not grow (a clobber list would make it keep its values in saved registers).
static inline __attribute__((always_inline)) dma_id_t
dma_fold_call(dma_fold_slot_t *slot) {
  register uint32_t a0 __asm__("a0") = (uint32_t)&slot->w;
  register uint32_t t1 __asm__("t1") =
      (uint32_t)&slot->stack[DMA_FOLD_STACK_BYTES / 4u];
  __asm__ volatile("addi t1, t1, -64\n\t"
                   "sw sp, 0(t1)\n\t"
                   "mv sp, t1\n\t"
                   "sw ra, 4(sp)\n\t"
                   "sw t0, 8(sp)\n\t"
                   "sw t2, 12(sp)\n\t"
                   "sw t3, 16(sp)\n\t"
                   "sw t4, 20(sp)\n\t"
                   "sw t5, 24(sp)\n\t"
                   "sw t6, 28(sp)\n\t"
                   "sw a1, 32(sp)\n\t"
                   "sw a2, 36(sp)\n\t"
                   "sw a3, 40(sp)\n\t"
                   "sw a4, 44(sp)\n\t"
                   "sw a5, 48(sp)\n\t"
                   "sw a6, 52(sp)\n\t"
                   "sw a7, 56(sp)\n\t"
                   "call dma_fold_entry\n\t"
                   "lw ra, 4(sp)\n\t"
                   "lw t0, 8(sp)\n\t"
                   "lw t2, 12(sp)\n\t"
                   "lw t3, 16(sp)\n\t"
                   "lw t4, 20(sp)\n\t"
                   "lw t5, 24(sp)\n\t"
                   "lw t6, 28(sp)\n\t"
                   "lw a1, 32(sp)\n\t"
                   "lw a2, 36(sp)\n\t"
                   "lw a3, 40(sp)\n\t"
                   "lw a4, 44(sp)\n\t"
                   "lw a5, 48(sp)\n\t"
                   "lw a6, 52(sp)\n\t"
                   "lw a7, 56(sp)\n\t"
                   "lw sp, 0(sp)"
                   : "+r"(a0), "+r"(t1)
                   :
                   : "memory");
  return a0;
}
#endif
static inline dma_id_t
dma_local_memcpy_nd_nonblocking(uint32_t stream, void *dest, const void *src,
                                size_t len, uint32_t dst_stride_2,
                                uint32_t src_stride_2, uint32_t reps_2,
                                uint32_t dst_stride_3, uint32_t src_stride_3,
                                uint32_t reps_3, uint32_t dimensions) {
  if (stream >= (uint32_t)NUM_DMAS_PER_GROUP || len == 0 || dimensions < 1 ||
      dimensions > 3)
    return 0;
#if DOMAIN_L2_EXACT
  {
    dma_fold_slot_t *const slot = dma_fold_slot();
    dma_fold_t *const w = &slot->w;
    w->stream = stream;
    w->d = (uint32_t)dest;
    w->s = (uint32_t)src;
    w->len = len;
    w->dims = dimensions;
    w->d2 = dimensions >= 2 ? dst_stride_2 : 0u;
    w->s2 = dimensions >= 2 ? src_stride_2 : 0u;
    w->r2 = dimensions >= 2 ? reps_2 : 1u;
    w->d3 = dimensions >= 3 ? dst_stride_3 : 0u;
    w->s3 = dimensions >= 3 ? src_stride_3 : 0u;
    w->r3 = dimensions >= 3 ? reps_3 : 1u;
#if DMA_FOLD_PROFILE
    {
      uint32_t pc;
      __asm__ volatile("auipc %0, 0" : "=r"(pc));
      w->ra = pc;
    }
#elif DMA_DOMAIN_CHECK
    w->ra = (uint32_t)__builtin_return_address(0);
#endif
    if (!w->r2 || !w->r3)
      return 0;
    return dma_fold_call(slot);
  }
#endif
#if DOMAINS > 1 && DOMAIN_L2_HOME
  {
    const uint32_t hs = dma_domain_home((uint32_t)src);
    const uint32_t step = hs ? src_stride_2 : dst_stride_2;
    const uint32_t at = dma_domain_off(hs ? (uint32_t)src : (uint32_t)dest);
    const uint32_t r3 = dimensions == 3 ? reps_3 : 1u;
    if (dimensions >= 2 && (hs || dma_domain_home((uint32_t)dest)) && step &&
        step < DMA_DOMAIN_ROW_BYTES && DMA_DOMAIN_ROW_BYTES % step == 0u &&
        at % DMA_DOMAIN_ROW_BYTES < step &&
        reps_2 > DMA_DOMAIN_ROW_BYTES / step && r3 <= DOMAIN_L2_MAX_DESC) {
      dma_home_walk_t *const w = &dma_home_walks[mempool_get_core_id()];
      w->d = (char *)dest;
      w->s = (const char *)src;
      w->len = len;
      w->d2 = dst_stride_2;
      w->s2 = src_stride_2;
      w->reps = reps_2;
      w->q = DMA_DOMAIN_ROW_BYTES / step;
      w->dq = (reps_2 - 1u) * dst_stride_2 + dst_stride_3;
      w->sq = (reps_2 - 1u) * src_stride_2 + src_stride_3;
      return dma_domain_home_walk(stream, w, r3);
    }
  }
#endif
#if DOMAINS > 1 && DOMAIN_L2_SPREAD
  const uint32_t target = mempool_get_domain_id() * DMA_DOMAIN_ROW_BYTES;
  const uint32_t l1s = dma_domain_l1((uint32_t)src);
  if (dimensions >= 2 && l1s != dma_domain_l1((uint32_t)dest)) {
    char *const d = (char *)dest;
    const char *const s = (const char *)src;
    const uint32_t r3 = dimensions == 3 ? reps_3 : 1u;
    const uint32_t step = l1s ? dst_stride_2 : src_stride_2;
    const uint32_t p = dma_domain_planes(step);
    // Bank-major, starting at the plane that puts domain d on its own banks.
    uint32_t rot = p > 1u ? dma_domain_rotation(step, target, p) : ~0u;
    rot = rot == ~0u ? 0u : rot;
    const uint32_t per = (rot ? 2u : 1u) * (reps_2 % p ? 2u : 1u);
    if (p > 1u && reps_2 >= 2u * p &&
        (dimensions == 2 || p > DOMAIN_L2_MAX_PLANES) &&
        r3 * per <= DOMAIN_L2_MAX_DESC) {
      // Each outer plane on its own: rows as a 3-D walk over their residues.
      const uint32_t dq = (reps_2 - 1u) * dst_stride_2 + dst_stride_3;
      const uint32_t sq = (reps_2 - 1u) * src_stride_2 + src_stride_3;
      dma_id_t id = 0;
      for (uint32_t i = 0; i < r3; ++i) {
        id = dma_domain_planes_run(stream, d + i * dq, s + i * sq, len,
                                   dst_stride_2, src_stride_2, reps_2, p, rot,
                                   p);
        if (rot)
          id = dma_domain_planes_run(stream, d + i * dq, s + i * sq, len,
                                     dst_stride_2, src_stride_2, reps_2, p, 0u,
                                     rot);
      }
      return id;
    }
    if (p > 1u && reps_2 >= 2u * p && dimensions == 3 &&
        p <= DOMAIN_L2_MAX_PLANES) {
      dma_id_t id = 0;
      for (uint32_t i = 0; i < p; ++i) {
        const uint32_t j = (rot + i) % p, n = (reps_2 - j + p - 1u) / p;
        // stride_3 leaves the plane's last row for row j of the next plane.
        const uint32_t back = (reps_2 - 1u - (n - 1u) * p);
        id = dma_local_issue_nd(stream, d + j * dst_stride_2,
                                s + j * src_stride_2, len, p * dst_stride_2,
                                p * src_stride_2, n,
                                dst_stride_3 + back * dst_stride_2,
                                src_stride_3 + back * src_stride_2, r3, 3);
      }
      return id;
    }
    uint32_t k = target ? dma_domain_rotation(step, target, reps_2) : ~0u;
    if (k != ~0u) {
      // Rows [k, reps_2) then [0, k) of every plane; stride_3 is taken from a
      // plane's last row, so each half re-aims it at its own first row.
      dma_local_issue_nd(stream, d + k * dst_stride_2, s + k * src_stride_2,
                         len, dst_stride_2, src_stride_2, reps_2 - k,
                         dst_stride_3 + k * dst_stride_2,
                         src_stride_3 + k * src_stride_2, r3, dimensions);
      return dma_local_issue_nd(stream, d, s, len, dst_stride_2, src_stride_2,
                                k, dst_stride_3 + (reps_2 - k) * dst_stride_2,
                                src_stride_3 + (reps_2 - k) * src_stride_2, r3,
                                dimensions);
    }
    const uint32_t dp = (reps_2 - 1u) * dst_stride_2 + dst_stride_3;
    const uint32_t sp = (reps_2 - 1u) * src_stride_2 + src_stride_3;
    k = target && dimensions == 3
            ? dma_domain_rotation(l1s ? dp : sp, target, r3)
            : ~0u;
    if (k != ~0u) {
      dma_local_issue_nd(stream, d + k * dp, s + k * sp, len, dst_stride_2,
                         src_stride_2, reps_2, dst_stride_3, src_stride_3,
                         r3 - k, 3);
      return dma_local_issue_nd(stream, d, s, len, dst_stride_2, src_stride_2,
                                reps_2, dst_stride_3, src_stride_3, k, 3);
    }
  }
#endif
  return dma_local_issue_nd(stream, dest, src, len, dst_stride_2, src_stride_2,
                            reps_2, dst_stride_3, src_stride_3, reps_3,
                            dimensions);
}

static inline dma_id_t dma_local_memcpy_nonblocking(uint32_t stream, void *dest,
                                                    const void *src,
                                                    size_t len) {
  return dma_local_memcpy_nd_nonblocking(stream, dest, src, len, 0, 0, 0, 0, 0,
                                         1, 1);
}

static inline dma_id_t
dma_local_memcpy_2d_nonblocking(uint32_t stream, void *dest, const void *src,
                                size_t len, uint32_t dst_stride,
                                uint32_t src_stride, uint32_t reps) {
  if (reps == 0)
    return 0;
  return dma_local_memcpy_nd_nonblocking(stream, dest, src, len, dst_stride,
                                         src_stride, reps, 0, 0, 1, 2);
}

static inline dma_id_t dma_local_memcpy_3d_nonblocking(
    uint32_t stream, void *dest, const void *src, size_t len,
    uint32_t dst_stride_2, uint32_t src_stride_2, uint32_t reps_2,
    uint32_t dst_stride_3, uint32_t src_stride_3, uint32_t reps_3) {
  if (reps_2 == 0 || reps_3 == 0)
    return 0;
  return dma_local_memcpy_nd_nonblocking(stream, dest, src, len, dst_stride_2,
                                         src_stride_2, reps_2, dst_stride_3,
                                         src_stride_3, reps_3, 3);
}

// The collective API, for every includer of dma.h. Last, because it builds on
// everything above.
#include "dma_collective.h"

#endif // _DMA_H_
