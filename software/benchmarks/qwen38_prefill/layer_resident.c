// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Complete prefill layers with bounded weight panels, bounded mixer activation
// windows, bounded FFN intermediates and an L2-resident residual stream, so no
// L1 buffer scales with the sequence. Reduced fixtures qualify arithmetic;
// real S=2048/4096 runs qualify the full workload. Blocking factors are
// explicit mapping choices, and small blocks trade occupancy for bounded
// storage. Q38_MIXER_ROWS=0 restores the whole-sequence resident control.
#include "async_task.h"
#include "baremetal/mempool_qwen38_kv.h"
#include "baremetal/mempool_qwen38_online_softmax.h"
#include "baremetal/mempool_qwen38_vector.h"
#include "dma.h"
#include "dma_stage_queue.h"
#include "hal_redmule.h"
#include "layer_data.h"
#include "mempool_checks.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define LOCAL __attribute__((section(".l1"), aligned(4 * NUM_BANKS)))
// Run both phases (0), mixer only (1), or FFN only (2). Numerical FFN-only
// checks use the golden mixer residual; timing probes use their seeded rows.
#ifndef Q38_PROBE_PHASE
#define Q38_PROBE_PHASE 0
#endif
#if Q38_PROBE_PHASE && !Q38_SCHEDULE_PROBE &&                                  \
    (Q38_KIND != 0 || Q38_SAMPLED_CHECKS)
#error "Numerical phase selection needs full-tensor attention fixtures"
#endif
#define MT (REDMULE_H * (REDMULE_P + 1))
#ifndef Q38_PROJECTION_COLUMNS
#define Q38_PROJECTION_COLUMNS 64u
#endif
#define NT Q38_PROJECTION_COLUMNS
// Attention key block: keys per online-softmax pass and per K/V tile.
#ifndef Q38_BK
#define Q38_BK 64u
#endif
#define BK ((uint32_t)Q38_BK)
#define Q38_MAX(lhs, rhs) ((lhs) > (rhs) ? (lhs) : (rhs))
#define Q38_MIN(lhs, rhs) ((lhs) < (rhs) ? (lhs) : (rhs))
// Keep stage frames out of main: see report_fatal on the 512-byte stack.
#define NOSTACK __attribute__((noinline))
#ifndef Q38_KV_REPLICATED
#define Q38_KV_REPLICATED 0
#endif
#ifndef Q38_KV_LEVEL
#define Q38_KV_LEVEL 0
#endif
#if Q38_KV_REPLICATED && !VIRTUAL_CLUSTER
#error "KV replicas require virtual_cluster=1 in both ELF and model"
#endif
#ifndef Q38_ASYNC
#define Q38_ASYNC 0
#endif
#ifndef Q38_VERBOSE_CHECKS
#define Q38_VERBOSE_CHECKS 0
#endif
#ifndef Q38_SAMPLED_CHECKS
#define Q38_SAMPLED_CHECKS 0
#endif
#ifndef Q38_DELAY_SEED
#define Q38_DELAY_SEED 0
#endif
#ifndef Q38_FFN_ROWS
#define Q38_FFN_ROWS 64u
#endif
#ifndef Q38_FFN_L2
#define Q38_FFN_L2 0
#endif
// GDN v2: whole-sequence projections on the L2 GEMM engine and the chunked
// recurrence on RedMulE (q38_gdn2.c); needs the L2 engine's L1 layout.
#ifndef Q38_GDN_V2
#define Q38_GDN_V2 0
#endif
#if Q38_GDN_V2 && !(Q38_FFN_L2 && Q38_KIND != 0)
#error "GDN v2 needs a GDN fixture and the L2 GEMM engine (Q38_FFN_L2)"
#endif
// Batch-major GDN layer (8x8 layer pipeline): each Q38_FFN_ROWS row batch
// runs in-projection, chunk core, out-projection and FFN in turn; the chunk
// core carries its FP16 state from batch to batch through L2 (q38_gdn2.c).
#ifndef Q38_X8_BATCH
#define Q38_X8_BATCH 0
#endif
// Batch-major GDN: keep layer parameters outside the shared phase arena,
// overlap the state import with chunk-0 preparation, and queue the tail DMAs.
// Whole-sequence (S1) builds already load parameters once and are unchanged.
#ifndef Q38_G2_ITEM_KEEP
#define Q38_G2_ITEM_KEEP 0
#endif
#if Q38_X8_BATCH &&                                                            \
    !(Q38_GDN_V2 && !Q38_SCHEDULE_PROBE && Q38_PROBE_PHASE == 0 &&             \
      Q38_FFN_ROWS % 128u == 0u && Q38_R % Q38_FFN_ROWS == 0u)
#error "Q38_X8_BATCH needs the GDN v2 full layer and 128-row-aligned batches"
#endif
// Layer pipeline (8x8 P2, q38_x8_pipe.c): layer l runs batch-major on domain
// l % DOMAINS; each batch's output rows reach layer l + 1 through L2, behind a
// ready flag.
#ifndef Q38_X8_PIPE
#define Q38_X8_PIPE 0
#endif
#ifndef Q38_X8_LAYERS
#define Q38_X8_LAYERS 2u
#endif
// Debug (numeric): every layer after the first reads a copy of the layer
// input, published like a real output, so all layers match the one-layer
// golden.
#ifndef Q38_X8_PIPE_RELAY
#define Q38_X8_PIPE_RELAY 0
#endif
// Debug (timing): P2 steady state without the fill. Layer l starts at batch
// R / Q38_FFN_ROWS - 1 - l, its input (l - 1, that batch) taken as published.
#ifndef Q38_X8_PIPE_SNAP
#define Q38_X8_PIPE_SNAP 0
#endif
// Debug (timing): stop after phase k of each domain's first item (1 in-proj,
// 2 chunk core, 3 out-proj, 4 FFN).
#ifndef Q38_X8_STOP_PHASE
#define Q38_X8_STOP_PHASE 0
#endif
// Debug (timing, with SNAP and STOP_PHASE 3): also time the next batch's
// chunk core (q38_x8_pipe.c), a warm item of the same layer.
#ifndef Q38_X8_WARM_CHUNKS
#define Q38_X8_WARM_CHUNKS 0
#endif
// Debug (timing window): stop the attention layer after QKV (1) or after the
// sweep (2); out-proj and FFN are skipped, values are incomplete.
#ifndef Q38_AV_STOP_PHASE
#define Q38_AV_STOP_PHASE 0
#endif
#if Q38_AV_STOP_PHASE && !(Q38_KIND == 0 && Q38_ATTN_VC)
#error "Q38_AV_STOP_PHASE stops the VC attention layer"
#endif
#if Q38_AV_STOP_PHASE
// Read at run time: out-proj and FFN stay compiled and referenced, so the
// code up to the stop is generated as in the full layer (no re-inlining).
static volatile uint32_t q38_av_stop = Q38_AV_STOP_PHASE;
#endif
// Debug (timing, with SNAP): the first item starts at its FFN (in-proj,
// chunk core and out-projection skipped; their inputs are garbage anyway).
#ifndef Q38_X8_START_FFN
#define Q38_X8_START_FFN 0
#endif
// Debug (timing): the FFN runs its up passes only (no down projection); in a
// layer pipeline only with a debug stop, also in FFN-only schedule probes.
#ifndef Q38_X8_FFN_UP_ONLY
#define Q38_X8_FFN_UP_ONLY 0
#endif
#if (Q38_X8_START_FFN && !Q38_X8_PIPE_SNAP) ||                                 \
    (Q38_X8_FFN_UP_ONLY && Q38_X8_PIPE && !Q38_X8_STOP_PHASE) ||               \
    (Q38_X8_FFN_UP_ONLY && !Q38_X8_PIPE && !Q38_SCHEDULE_PROBE)
#error "START_FFN is a snapshot knob; FFN_UP_ONLY needs a debug stop or probe"
#endif
// Debug (numeric): no layer; a ring of handoffs from domain to domain that
// checks the flags and the cross-domain DMA (q38_x8_pipe.c).
#ifndef Q38_X8_PIPE_TEST
#define Q38_X8_PIPE_TEST 0
#endif
// L2 bytes per domain below the top for its scratch (G, GDN buffers, mixer,
// output rows); 0 = the same addresses in every domain (DOMAIN_L2_HOME
// separates them, timing-only).
#ifndef Q38_X8_SPAN
#define Q38_X8_SPAN 0u
#endif
// Deployment L2 layouts (S1 and 8x8 alike): every L2-side burst stays in one
// 8 KiB L2 row with whole-row plane steps, so the domain fold never splits.
#ifndef Q38_L2_TILED
#define Q38_L2_TILED 0
#endif
// Tiled layouts: GDN O rows padded to one 8 KiB L2 row each, so a tile's
// gated rows go out as one descriptor and the out-projection gathers its X
// rows back (f2_x_gather). P2 home builds drop the context buffer, which
// only Q38_GDN2_DEBUG_CTX writes, to make room.
#ifndef Q38_G2_O_PAD
#define Q38_G2_O_PAD 0
#endif
// Batch-major path only (P2 and its S1 twins): S1's whole-sequence layer
// keeps the unpadded O.
#define G2_O_PADDED (Q38_G2_O_PAD && Q38_X8_BATCH)
// P2 home builds: the written L2 scratch sits in home segment 0 (below the
// wrap), and the GDN end state and last layer's output stay in the domain's
// home instead of the shared result_* buffers below the home base.
#if Q38_L2_TILED && DOMAINS > 1 && DOMAIN_L2_HOME && Q38_X8_PIPE
#define Q38_L2_HOME_SLOTS 1
#define Q38_L2_TOP (DMA_DOMAIN_HOME_BASE + DMA_DOMAIN_HOME_WRAP / DOMAINS)
#define Q38_RES_SAMPLE_CONTEXT ((__fp16 *)(DMA_DOMAIN_HOME_BASE + 0x210000u))
#define Q38_RES_CONV ((__fp16 *)(DMA_DOMAIN_HOME_BASE + 0x220000u))
#define Q38_RES_RECURRENT ((float *)(DMA_DOMAIN_HOME_BASE + 0x240000u))
#define Q38_RES_HOME_END (DMA_DOMAIN_HOME_BASE + 0x540000u)
// Conv history rows three L2 rows apart (whole-row plane steps).
#define Q38_RES_CONV_PITCH 0x6000u
#else
#define Q38_L2_HOME_SLOTS 0
#define Q38_L2_TOP (L2_BASE + L2_SIZE)
#define Q38_RES_SAMPLE_CONTEXT result_sample_context
#define Q38_RES_CONV result_conv
#define Q38_RES_RECURRENT result_recurrent
#define Q38_RES_CONV_PITCH (Q38_QKVDIM * 2u)
#endif
// Pipeline synchronization: 0 = per-(layer, batch) ready flags, core 0 polls;
// 1 = lock-step, one chip-wide fractal_sync barrier per step (quadrant q runs
// its k-th item in step k + q; a quadrant without an item that step arrives
// at once). Default lock-step; the snapshot and debug-stop knobs need flags.
#ifndef Q38_X8_SYNC
#define Q38_X8_SYNC (!Q38_X8_PIPE_SNAP && !Q38_X8_STOP_PHASE)
#endif
#if Q38_X8_SYNC && (Q38_X8_PIPE_SNAP || Q38_X8_STOP_PHASE)
#error "The lock-step pipeline has no snapshot or debug stop"
#endif
#if Q38_X8_PIPE && !Q38_X8_BATCH
#error "Q38_X8_PIPE runs the batch-major layer (Q38_X8_BATCH)"
#endif
#if Q38_X8_PIPE
// The current layer's rows: input, mixer (out-projection result) and output.
static struct {
  const __fp16 *in;
  __fp16 *mix, *out;
} x8_io __attribute__((section(".l1"), aligned(8)));
#define X8_IN x8_io.in
#define X8_MIX x8_io.mix
#define X8_OUT x8_io.out
#else
#define X8_IN input_init
#if Q38_KIND == 0 && DOMAINS > 1 && DOMAIN_L2_HOME && Q38_L2_TILED
// The FFN residual is the mixer that out-proj wrote to its home scratch
// (AV_MIXER, q38_attn_vc.c), which is defined after the FFN is included.
static inline __fp16 *av_mixer_home(void);
#define X8_MIX av_mixer_home()
#else
#define X8_MIX result_mixer
#endif
#define X8_OUT result_output
#endif
#ifndef Q38_VC_PROFILE
#define Q38_VC_PROFILE Q38_SCHEDULE_PROBE
#endif
#ifndef Q38_PROBE_SKIP_BURST
#define Q38_PROBE_SKIP_BURST (Q38_FFN_L2 || Q38_AV_PANEL_AWARE)
#endif
#ifndef Q38_FFN_GU_ROWS
#define Q38_FFN_GU_ROWS MT
#endif
#ifndef Q38_FFN_DN_ROWS
#define Q38_FFN_DN_ROWS (REDMULE_H == 8u ? 256u : 512u)
#endif
#ifndef Q38_FFN_DN_TILE
#define Q38_FFN_DN_TILE 32u
#endif
#if Q38_FFN_ROWS == 0 || Q38_FFN_ROWS % 64 != 0
#error "FFN row block must be a positive multiple of 64"
#endif
#ifndef Q38_WEIGHT_PANELS
#define Q38_WEIGHT_PANELS 2
#endif
#ifndef Q38_ADAPTIVE_PANELS
#define Q38_ADAPTIVE_PANELS 0
#endif
#ifndef Q38_GATEUP_PIPELINE
#define Q38_GATEUP_PIPELINE 0
#endif
#ifndef Q38_COORDINATOR_DMA
#define Q38_COORDINATOR_DMA 0
#endif
#ifndef Q38_KV_PREFETCH
#define Q38_KV_PREFETCH 0
#endif
// Attention key sweep: QK and PV on the owner engines, but each (head, row)'s
// masking, online-softmax update and accumulation on any core, so all cores
// share work that the 24 head agents otherwise did row by row. Every row's
// arithmetic and its key-block order are unchanged.
#ifndef Q38_ATTN_SPREAD
#define Q38_ATTN_SPREAD 0
#endif
// Interleaved K/V then query passes normalize the same rows; reuse them.
#ifndef Q38_ATTN_REUSE_NORM
#define Q38_ATTN_REUSE_NORM 0
#endif
// Attention supertiles: the K/V, Q/gate and output projections of
// Q38_ATTN_ST_ROWS rows each run as full-K groups of one rectangle per
// RedMulE, around the unchanged per-window causal key sweeps.
#ifndef Q38_ATTN_SUPERTILE
#define Q38_ATTN_SUPERTILE 0
#endif
#ifndef Q38_ATTN_ST_ROWS
#define Q38_ATTN_ST_ROWS 256u
#endif
#if Q38_ATTN_SUPERTILE &&                                                      \
    (Q38_KIND != 0 || !Q38_INTERLEAVE_KV_QUERY ||                              \
     !(Q38_GATEUP_PIPELINE && Q38_ASYNC) || Q38_WEIGHT_PANELS == 0 ||          \
     Q38_KV_REPLICATED || Q38_BACKGROUND_FFN_NORM || Q38_FFN_NORM_STREAM)
#error "Attention supertiles need interleaved async global attention"
#endif
// Group-local key sweep: each window's (head, 16-row tile) owners are split
// evenly over the groups; an owner's queries, scores, probabilities, PV tile,
// FP32 accumulator and softmax state live in its group's level-0 VC arena, its
// QK/PV jobs run on that group's engines, and its softmax and accumulation on
// that group's cores, with group-only barriers inside a key block.
#ifndef Q38_ATTN_LOCAL
#define Q38_ATTN_LOCAL 0
#endif
// Phased attention on the VC scheme (q38_attn_vc.c), and its rows per
// Q/gate/K/V and output projection block.
#ifndef Q38_ATTN_VC
#define Q38_ATTN_VC 0
#endif
#ifndef Q38_ATTN_VC_ROWS
#define Q38_ATTN_VC_ROWS 256u
#endif
#ifndef Q38_ATTN_VC_OROWS
#define Q38_ATTN_VC_OROWS Q38_ATTN_VC_ROWS
#endif
// KV-affine sweep batches; separate from projection and FFN row geometry.
#ifndef Q38_ATTN_VC_SWEEP
#define Q38_ATTN_VC_SWEEP 0
#endif
// A full attention layer: phased VC attention, then the L2 FFN.
#define Q38_ATTN_L2_LAYER (Q38_KIND == 0 && Q38_ATTN_VC && Q38_FFN_L2)
#ifndef Q38_AV_PROFILE
#define Q38_AV_PROFILE (Q38_SCHEDULE_PROBE || Q38_ATTN_VC_SWEEP)
#endif
#ifndef Q38_AV_PANEL_AWARE
#define Q38_AV_PANEL_AWARE 0
#endif
#ifndef Q38_AV_DUAL
#define Q38_AV_DUAL 0
#endif
#ifndef Q38_AV_DIRECT_DMA
#define Q38_AV_DIRECT_DMA 0
#endif
#ifndef Q38_AV_STATIC_CORES
#define Q38_AV_STATIC_CORES (Q38_AV_DUAL ? NUM_CORES_PER_TILE : 0)
#endif
#ifndef Q38_AV_ROW_UNROLL
#define Q38_AV_ROW_UNROLL 1
#endif
#ifndef Q38_AV_CACHE_GAMMA
#define Q38_AV_CACHE_GAMMA (Q38_AV_STATIC_CORES && VLEN == 1024)
#endif
_Static_assert(!Q38_AV_CACHE_GAMMA || (Q38_AV_STATIC_CORES && VLEN == 1024),
               "Retained gamma needs static VLEN1024 row blocks");
_Static_assert(!Q38_AV_STATIC_CORES ||
                   (Q38_AV_DUAL && Q38_AV_STATIC_CORES >= 2 &&
                    Q38_AV_STATIC_CORES <= NUM_CORES_PER_TILE),
               "Static projection needs two or more available tile cores");
_Static_assert(!Q38_AV_DIRECT_DMA || Q38_AV_PANEL_AWARE,
               "Direct panel DMA requires panel-aware projection");
_Static_assert(Q38_AV_ROW_UNROLL == 1 || Q38_AV_ROW_UNROLL == 2 ||
                   Q38_AV_ROW_UNROLL == 4,
               "Projection row batch must be one, two or four");
_Static_assert(!Q38_AV_DUAL || Q38_AV_PANEL_AWARE,
               "Two-core projection requires panel-aware epilogues");
// Sweep-only alternatives: 0 group joins, 1 tile handoffs, 2 early rescale.
#ifndef Q38_AV_SWEEP_MODE
#define Q38_AV_SWEEP_MODE 0
#endif
#ifndef Q38_AV_SWEEP_REUSE_QUERY
#define Q38_AV_SWEEP_REUSE_QUERY 0
#endif
#ifndef Q38_AV_SWEEP_ROW_DMA
#define Q38_AV_SWEEP_ROW_DMA 0
#endif
#ifndef Q38_AV_SWEEP_TREE_MAX
#define Q38_AV_SWEEP_TREE_MAX 0
#endif
#ifndef Q38_AV_SWEEP_INTERLEAVE
#define Q38_AV_SWEEP_INTERLEAVE 0
#endif
#ifndef Q38_AV_SWEEP_REGISTER
#define Q38_AV_SWEEP_REGISTER 0
#endif
// Lean softmax row; QSCALE moves 1/sqrt(HD) into the Q epilogue, EXP2 swaps
// in the degree-4 exp2 polynomial.
#ifndef Q38_AV_SWEEP_LEAN
#define Q38_AV_SWEEP_LEAN 0
#endif
#ifndef Q38_AV_SWEEP_QSCALE
#define Q38_AV_SWEEP_QSCALE 0
#endif
#ifndef Q38_AV_SWEEP_EXP2
#define Q38_AV_SWEEP_EXP2 0
#endif
// FP16 base-2 softmax row with an integer running maximum; needs QSCALE.
#ifndef Q38_AV_SWEEP_FAST
#define Q38_AV_SWEEP_FAST 0
#endif
// Scores and Z in the tile window (GVSoC model extension; tile-owned rows).
#ifndef Q38_AV_SWEEP_TILE_WIN
#define Q38_AV_SWEEP_TILE_WIN 0
#endif
// Dependency-only pipeline: no tile joins inside a key block, each core waits
// on the RedMulE jobs it consumes, and a split-phase KV barrier.
#ifndef Q38_AV_SWEEP_ASYNC
#define Q38_AV_SWEEP_ASYNC 0
#endif
// Store drains once after the rescale (1), or after softmax and rescale (2),
// instead of after every row.
#ifndef Q38_AV_SWEEP_LATE_DRAIN
#define Q38_AV_SWEEP_LATE_DRAIN 0
#endif
// Finish with the output gate fused into the normalisation and batched
// gated exports; the full context is not exported.
#ifndef Q38_AV_SWEEP_FINISH
#define Q38_AV_SWEEP_FINISH 0
#endif
#if Q38_AV_SWEEP_FINISH && !(Q38_ATTN_VC_SWEEP && Q38_AV_SWEEP_FAST)
#error "The fused finish needs the FP16 softmax sweep"
#endif
// All query windows as one key-block stream: a window's finish runs between
// blocks of the next one, and nothing joins at a window boundary.
#ifndef Q38_AV_SWEEP_OVERLAP
#define Q38_AV_SWEEP_OVERLAP 0
#endif
#if Q38_AV_SWEEP_OVERLAP &&                                                    \
    !(Q38_AV_SWEEP_ASYNC && Q38_AV_SWEEP_FINISH && Q38_AV_SWEEP_TILE_WIN)
#error "Window overlap needs the async pipeline, fused finish and tile window"
#endif
#if Q38_AV_SWEEP_MODE > 2 ||                                                   \
    ((Q38_AV_SWEEP_MODE || Q38_AV_SWEEP_REUSE_QUERY ||                         \
      Q38_AV_SWEEP_INTERLEAVE || Q38_AV_SWEEP_REGISTER || Q38_AV_SWEEP_LEAN || \
      Q38_AV_SWEEP_FAST || Q38_AV_SWEEP_TILE_WIN || Q38_AV_SWEEP_ASYNC) &&     \
     !Q38_ATTN_VC_SWEEP)
#error "Sweep alternatives require the KV-affine attention path"
#endif
#if Q38_AV_SWEEP_ASYNC && Q38_AV_SWEEP_MODE != 2
#error "The dependency-only pipeline needs early mode (tile-owned rows)"
#endif
#if Q38_AV_SWEEP_LATE_DRAIN > 2 ||                                             \
    (Q38_AV_SWEEP_LATE_DRAIN && !Q38_AV_SWEEP_ASYNC)
#error "Late drains are an option of the dependency-only pipeline"
#endif
#if Q38_AV_SWEEP_EXP2 && !Q38_AV_SWEEP_LEAN
#error "exp2 is an option of the lean softmax row"
#endif
#if Q38_AV_SWEEP_QSCALE && !(Q38_AV_SWEEP_LEAN || Q38_AV_SWEEP_FAST)
#error "The folded QK scale needs the lean or FP16 softmax row"
#endif
#if Q38_AV_SWEEP_FAST &&                                                       \
    (!Q38_AV_SWEEP_QSCALE || Q38_AV_SWEEP_LEAN || Q38_AV_SWEEP_REGISTER)
#error "The FP16 softmax row needs QSCALE and replaces lean/register rows"
#endif
#if Q38_AV_SWEEP_LEAN && Q38_AV_SWEEP_REGISTER
#error "Lean and register softmax rows are alternatives"
#endif
#if Q38_AV_SWEEP_QSCALE && !Q38_AV_PANEL_AWARE
#error "The folded QK scale is applied by the panel-aware Q epilogue"
#endif
#if Q38_AV_SWEEP_QSCALE
// 1/sqrt(HD) is a power of two, so applying it to Q is exact.
#define Q38_QK_SCALE (Q38_HD == 64u ? 0.125f : Q38_HD == 256u ? 0.0625f : 0.0f)
_Static_assert(Q38_HD == 64u || Q38_HD == 256u,
               "Folded QK scale needs HD 64 or 256");
#endif
#ifndef Q38_AV_SWEEP_TILE
#define Q38_AV_SWEEP_TILE Q38_MIN(MT, Q38_BK / 8u)
#endif
#ifndef Q38_AV_SWEEP_ROWS
#define Q38_AV_SWEEP_ROWS (8u * Q38_AV_SWEEP_TILE)
#endif
#if Q38_ATTN_VC_SWEEP &&                                                       \
    (!Q38_ATTN_VC ||                                                           \
     !(Q38_PROBE_PHASE == 1 || (Q38_ATTN_L2_LAYER && Q38_PROBE_PHASE == 0)))
#error                                                                         \
    "The KV-affine sweep needs phased VC attention, mixer-only or with the L2 FFN"
#endif
_Static_assert(Q38_KIND != 0 || Q38_BK <= 128u || Q38_ATTN_VC_SWEEP,
               "Wide softmax blocks need the KV-affine sweep");
#if Q38_ATTN_LOCAL && !((Q38_ATTN_SUPERTILE || Q38_ATTN_VC) && VIRTUAL_CLUSTER)
#error                                                                         \
    "The group-local key sweep needs attention supertiles or phases and VC windows"
#endif
// RedMulE accumulates the unnormalized context in FP16 (Z = P V + Z) instead
// of Spatz adding each PV tile into an FP32 accumulator. Spatz rescales only
// the rows whose running maximum moved. A precision change, not a rewrite of
// the result: the online-softmax algebra is unchanged.
// One-pass softmax row: scale, round, max, exp, sum and store in registers,
// instead of zeroing the probabilities, writing the scaled scores back, a
// scalar mask loop and two score reloads. Same vector arithmetic; the toy
// differs from the unfused path by up to 1 ulp on rows whose running maximum
// moves (scalar correction/denominator code compiles differently), with the
// same error against the ideal reference.
#ifndef Q38_ATTN_FUSED_SOFTMAX
#define Q38_ATTN_FUSED_SOFTMAX 0
#endif
#ifndef Q38_ATTN_RM_ACCUM
#define Q38_ATTN_RM_ACCUM 0
#endif
// Software-pipelined group-local key sweep; see query_sweep_pipe.
#ifndef Q38_ATTN_LOCAL_PIPE
#define Q38_ATTN_LOCAL_PIPE 0
#endif
#if Q38_ATTN_LOCAL_PIPE && !(Q38_ATTN_RM_ACCUM && Q38_KV_PREFETCH)
#error "The pipelined sweep needs RedMulE accumulation and K/V prefetch"
#endif
// Two-level FP16 accumulation on RedMulE: every PV starts fresh (MATMUL)
// into a group tile Zg, and a 16x16 identity GEMM folds it into the running
// context (Z = I Zg + Z). Short FMA chains keep the host-estimated error near
// FP32 accumulation's at no Spatz cost; RedMulE has slack beside the softmax.
// Tile-local softmax: each core handles only rows whose scores and
// probabilities lie in its own tile, and the per-row softmax state and
// correction are laid out per tile next to them.
#ifndef Q38_ATTN_TILE_LOCAL
#define Q38_ATTN_TILE_LOCAL 0
#endif
// Probe-only isolation of the pipelined key sweep's Spatz work: bit 0 drops
// its RedMulE QK/PV jobs, bit 1 its K/V DMA. Numerics are meaningless.
#ifndef Q38_ATTN_ISOLATE
#define Q38_ATTN_ISOLATE 0
#endif
#if Q38_ATTN_ISOLATE && !(Q38_SCHEDULE_PROBE && Q38_ATTN_LOCAL_PIPE)
#error "Key-sweep isolation is a scheduling probe of the pipelined sweep"
#endif
// Snitch's fence waits for its scalar LSU only (snitch.sv fence_stall), not
// for Spatz vector stores, so a barrier does not make them visible to RedMulE.
// With this set, a Spatz row that RedMulE reads next (rescaled Z, softmax
// probabilities) is read back and consumed before the barrier: Spatz orders a
// load after its own store to the same address.
#ifndef Q38_ATTN_DRAIN
#define Q38_ATTN_DRAIN 0
#endif
#ifndef Q38_ATTN_RM_FOLD
#define Q38_ATTN_RM_FOLD 0
#endif
#if Q38_ATTN_RM_FOLD && !Q38_ATTN_LOCAL_PIPE
#error "The RedMulE fold needs the pipelined sweep"
#endif
#if Q38_ATTN_RM_ACCUM && !(Q38_ATTN_LOCAL && Q38_REDMULE_MATMUL)
#error "RedMulE accumulation needs the group-local sweep and MATMUL mode"
#endif
#if Q38_ATTN_REUSE_NORM && !Q38_INTERLEAVE_KV_QUERY
#error "Reusing the K/V normalization needs interleaved K/V and query windows"
#endif
#if Q38_ATTN_SPREAD && !(Q38_GATEUP_PIPELINE && Q38_ASYNC)
#error "Spread attention needs the per-head attention stage buffers"
#endif
#ifndef Q38_INTERLEAVE_KV_QUERY
#define Q38_INTERLEAVE_KV_QUERY 0
#endif
#ifndef Q38_GDN_CHUNK_ROWS
#define Q38_GDN_CHUNK_ROWS 64u
#endif
#if Q38_SCHEDULE_PROBE
#ifndef Q38_PROBE_RECURRENT_K
#define Q38_PROBE_RECURRENT_K 32u
#endif
_Static_assert(Q38_PROBE_RECURRENT_K > 0 && Q38_PROBE_RECURRENT_K <= Q38_KD &&
                   Q38_PROBE_RECURRENT_K % 16u == 0,
               "Probe recurrence K must cover whole vector chunks");
#define Q38_RECUR_K Q38_PROBE_RECURRENT_K
#else
#define Q38_RECUR_K Q38_KD
#endif
#ifndef Q38_GDN_DOUBLE_CHUNK
#define Q38_GDN_DOUBLE_CHUNK 0
#endif
#ifndef Q38_GDN_RECUR_PROJECTION_OVERLAP
#define Q38_GDN_RECUR_PROJECTION_OVERLAP 0
#endif
#ifndef Q38_GDN_TRIPLE_PIPELINE
#define Q38_GDN_TRIPLE_PIPELINE 0
#endif
#ifndef Q38_STAGE_SEPARATE
#define Q38_STAGE_SEPARATE 0
#endif
#ifndef Q38_FFN_NORM_STREAM
#define Q38_FFN_NORM_STREAM 0
#endif
#ifndef Q38_BACKGROUND_FFN_NORM
#define Q38_BACKGROUND_FFN_NORM 0
#endif
#if Q38_FFN_NORM_STREAM && !Q38_STAGE_SEPARATE
#error "FFN norm stream needs independent mixer and FFN buffers"
#endif
#if Q38_BACKGROUND_FFN_NORM &&                                                 \
    (Q38_KIND != 0 || !Q38_FFN_NORM_STREAM || !Q38_INTERLEAVE_KV_QUERY ||      \
     !Q38_KV_PREFETCH || Q38_KV_REPLICATED)
#error "Background FFN norm needs interleaved global attention and L2 stream"
#endif
#if Q38_STAGE_SEPARATE && !(Q38_GATEUP_PIPELINE && Q38_ASYNC)
#error "Separate stage arenas require async gate/up panel buffers"
#endif
#if Q38_SCHEDULE_PROBE &&                                                      \
    (!Q38_GATEUP_PIPELINE || !Q38_ASYNC || Q38_WEIGHT_PANELS < 2)
#error "Scheduling probe needs async gate/up with two weight panels"
#endif
#if Q38_GDN_CHUNK_ROWS == 0 || Q38_GDN_CHUNK_ROWS % 16 != 0
#error "GDN recurrent chunks must contain a positive multiple of 16 rows"
#endif
#if Q38_GDN_RECUR_PROJECTION_OVERLAP &&                                        \
    (!Q38_GDN_DOUBLE_CHUNK || !Q38_ASYNC || Q38_WEIGHT_PANELS == 0 ||          \
     Q38_KIND == 0)
#error                                                                         \
    "GDN recurrence/projection overlap needs async double chunks and staged weights"
#endif
#if Q38_GDN_TRIPLE_PIPELINE &&                                                 \
    (!Q38_GDN_RECUR_PROJECTION_OVERLAP || !Q38_COORDINATOR_DMA)
#error "GDN triple pipeline needs QKV overlap and coordinator DMA"
#endif
#if Q38_KV_PREFETCH && Q38_KV_REPLICATED
#error "KV prefetch currently needs global KV placement"
#endif
// GDN ready queue: QKV/Z/control projections for the next Q38_GDN_PROJECT_ROWS
// supertile run as independent full-K rectangles while Spatz advances the
// ordered recurrence of the current one. Finished panel groups spill to a
// two-generation L2 ring; each recurrent chunk stages its rows back.
#ifndef Q38_GDN_READY_QUEUE
#define Q38_GDN_READY_QUEUE 0
#endif
#ifndef Q38_GDN_PROJECT_ROWS
#define Q38_GDN_PROJECT_ROWS 256u
#endif
#ifndef Q38_GDN_QUEUE_PANELS
#define Q38_GDN_QUEUE_PANELS 4u
#endif
// Diagnostic: start a group only after the previous group's spill landed, so
// iDMA never reads L1 while the engines stream.
#ifndef Q38_GDN_SPILL_SERIAL
#define Q38_GDN_SPILL_SERIAL 0
#endif
// Start a group only once every engine retired the previous one, so each
// group starts from idle engines as project_panel_jobs does after its barrier.
#ifndef Q38_GDN_GROUP_SYNC
#define Q38_GDN_GROUP_SYNC 1
#endif
// Retire copy of a finished Z tile into its landing slot: 1 one vector copy,
// 2 one copy per row as project_panel_jobs does, 0 none (diagnostic only:
// the spilled projections are then stale).
#ifndef Q38_GDN_RETIRE_COPY
#define Q38_GDN_RETIRE_COPY 1
#endif
// Diagnostic: a draining tile agent sleeps on its engine's EOC event instead
// of polling RUNNING_JOB while the engine streams.
#ifndef Q38_GDN_DRAIN_SLEEP
#define Q38_GDN_DRAIN_SLEEP 0
#endif
// 1: group g+2's weights stream in while group g+1 computes. 0: a group's
// weights are fetched only after the previous group retires, so weight DMA
// never overlaps the engines.
#ifndef Q38_GDN_WEIGHT_PREFETCH
#define Q38_GDN_WEIGHT_PREFETCH 1
#endif
// Diagnostic: every queued job writes Z slot 0 and is copied out before the
// successor starts, as the capacity burst's single job per engine does.
#ifndef Q38_GDN_Z_FIXED
#define Q38_GDN_Z_FIXED 0
#endif
#ifndef Q38_GDN_QUEUE_DEPTH
#define Q38_GDN_QUEUE_DEPTH 1u
#endif
#if Q38_GDN_QUEUE_DEPTH < 1 || Q38_GDN_QUEUE_DEPTH > 2
#error "GDN queue depth is one or two jobs per engine"
#endif
// Wide FFN: gate and up rectangles of a panel group share one job list, and
// the down projection stages Q38_FFN_DOWN_PANELS full-K panels at once in the
// whole weight area, single-buffered, so each group offers one job per engine.
#ifndef Q38_FFN_WIDE
#define Q38_FFN_WIDE 0
#endif
#ifndef Q38_FFN_DOWN_PANELS
#define Q38_FFN_DOWN_PANELS 4u
#endif
#if Q38_FFN_WIDE && !(Q38_GATEUP_PIPELINE && Q38_ASYNC)
#error "Wide FFN needs the async gate/up pipeline"
#endif
// Ordered K slices of the wide down projection. Each full-K rectangle runs as
// Q38_FFN_DOWN_SLICES jobs in ascending K on one engine, the later ones
// accumulating onto the earlier FP16 result through RedMulE's Y (= Z) input,
// so slice n+1's weights stream into the other half of the weight area while
// the engines consume slice n. The FFN intermediate is then slice-major so
// every slice is a contiguous X.
// MATMUL computes Z = X*W without reading Y, so a fresh job needs no zeroed Z
// tile; GEMM (Z = X*W + Y, Y read from Z) is kept for accumulating K slices.
#ifndef Q38_REDMULE_MATMUL
#define Q38_REDMULE_MATMUL 0
#endif
// Gate/up groups synchronized by dependency counters instead of whole-cluster
// barriers: an engine starts group g once its weights landed (and, for up
// rectangles, once group g-2's activation freed the up buffer); the last
// engine of a group publishes group g+2's weights; sibling cores activate a
// group as soon as all of its rectangles are copied out.
#ifndef Q38_FFN_DATAFLOW
#define Q38_FFN_DATAFLOW 0
#endif
#if Q38_FFN_DATAFLOW && !(Q38_FFN_WIDE && Q38_COORDINATOR_DMA)
#error "Dataflow gate/up needs the wide FFN and coordinator DMA"
#endif
// The same dependency-counter scheme for plain full-K projections.
#ifndef Q38_PROJ_DATAFLOW
#define Q38_PROJ_DATAFLOW 0
#endif
#if Q38_PROJ_DATAFLOW && !(Q38_COORDINATOR_DMA && Q38_GATEUP_PIPELINE &&       \
                           Q38_ASYNC && Q38_WEIGHT_PANELS > 0)
#error "Dataflow projections need the coordinator DMA queue and staged weights"
#endif
#ifndef Q38_FFN_DOWN_SLICES
#define Q38_FFN_DOWN_SLICES 1u
#endif
#if Q38_FFN_DOWN_SLICES > 1 && !Q38_FFN_WIDE
#error "Down K slices need the wide FFN"
#endif
#define Q38_KC (Q38_I / Q38_FFN_DOWN_SLICES)
#if Q38_GDN_READY_QUEUE &&                                                     \
    (Q38_KIND == 0 || !Q38_ASYNC || Q38_WEIGHT_PANELS == 0 ||                  \
     Q38_STAGE_SEPARATE || Q38_GDN_DOUBLE_CHUNK)
#error "GDN ready queue needs async GDN, staged weights and shared arenas"
#endif
#ifndef Q38_PROJECTION_ROWS
#define Q38_PROJECTION_ROWS MT
#endif
// Mixer activation window in token rows. Zero keeps the whole padded sequence
// resident, which is the control this bound is verified against.
#ifndef Q38_MIXER_ROWS
#define Q38_MIXER_ROWS 64u
#endif
#if Q38_MIXER_ROWS != 0 && Q38_MIXER_ROWS % 64 != 0
#error "Mixer row window must be zero or a positive multiple of 64"
#endif
#if Q38_MIXER_ROWS == 0 || Q38_MIXER_ROWS > Q38_R
#define Q38_WIN Q38_R
#else
#define Q38_WIN Q38_MIXER_ROWS
#endif
#ifndef Q38_SCHEDULE_PROBE
#define Q38_SCHEDULE_PROBE 0
#endif
#ifndef Q38_PROBE_BASE
#define Q38_PROBE_BASE 0u
#endif
#ifndef Q38_PROBE_ROWS
#define Q38_PROBE_ROWS 0u
#endif
#ifndef Q38_PROBE_TAIL
#define Q38_PROBE_TAIL 0
#endif
#ifndef Q38_PROBE_BURST_ONLY
#define Q38_PROBE_BURST_ONLY 0
#endif
#ifndef Q38_PROBE_LEAD_GROUPS
#define Q38_PROBE_LEAD_GROUPS 2u
#endif
#ifndef Q38_PROBE_BURST_ROUNDS
#define Q38_PROBE_BURST_ROUNDS 1u
#endif
// Model regression: the weight slot the burst stages and streams.
#ifndef Q38_PROBE_BURST_SLOT
#define Q38_PROBE_BURST_SLOT 0
#endif
#if Q38_PROBE_BURST_ONLY && !Q38_SCHEDULE_PROBE
#error "Burst-only execution requires a scheduling probe"
#endif
#ifndef Q38_PROBE_MIXER_WINDOWS
#define Q38_PROBE_MIXER_WINDOWS (Q38_KIND ? 1u : 2u)
#endif
#if Q38_SCHEDULE_PROBE
_Static_assert(Q38_PROBE_MIXER_WINDOWS > 0u &&
                   Q38_PROBE_ROWS >= Q38_PROBE_MIXER_WINDOWS * Q38_WIN &&
                   Q38_PROBE_ROWS % Q38_WIN == 0 &&
                   Q38_PROBE_ROWS % Q38_FFN_ROWS == 0 &&
                   Q38_PROBE_BASE % Q38_WIN == 0 &&
                   Q38_PROBE_BASE % Q38_FFN_ROWS == 0 &&
                   Q38_PROBE_BASE + Q38_PROBE_ROWS <= Q38_R,
               "Probe must cover aligned mixer and FFN buffer turnovers");
#define Q38_EXEC_START Q38_PROBE_BASE
#define Q38_EXEC_END (Q38_PROBE_BASE + Q38_PROBE_ROWS)
#define Q38_MIXER_EXEC_START                                                   \
  (Q38_PROBE_BASE +                                                            \
   (Q38_PROBE_TAIL ? Q38_PROBE_ROWS - Q38_PROBE_MIXER_WINDOWS * Q38_WIN : 0u))
#define Q38_MIXER_EXEC_END                                                     \
  (Q38_MIXER_EXEC_START + Q38_PROBE_MIXER_WINDOWS * Q38_WIN)
#else
#define Q38_EXEC_START 0u
#define Q38_EXEC_END Q38_R
#define Q38_MIXER_EXEC_START 0u
#define Q38_MIXER_EXEC_END Q38_R
#endif

// Gated mixer output width and the widest row block any stage processes.
#define Q38_MIXW (Q38_KIND ? Q38_GDIM : Q38_QDIM)
#define Q38_ROWS Q38_MAX(Q38_WIN, Q38_FFN_ROWS)
#if Q38_GDN_READY_QUEUE
#define Q38_GDN_STAGE_ROWS Q38_GDN_CHUNK_ROWS
#define Q38_GDN_SLOTS 1u
#elif Q38_GDN_DOUBLE_CHUNK
#define Q38_GDN_STAGE_ROWS Q38_GDN_CHUNK_ROWS
#if Q38_GDN_TRIPLE_PIPELINE
#define Q38_GDN_SLOTS 3u
#else
#define Q38_GDN_SLOTS 2u
#endif
_Static_assert(Q38_WIN % Q38_GDN_STAGE_ROWS == 0,
               "GDN stage slots must divide the mixer window");
_Static_assert(Q38_R % Q38_GDN_STAGE_ROWS == 0,
               "GDN stage slots must divide the padded sequence");
_Static_assert(Q38_GDN_STAGE_ROWS % Q38_PROJECTION_ROWS == 0,
               "GDN microchunks must hold whole projection row tiles");
#else
#define Q38_GDN_STAGE_ROWS Q38_WIN
#define Q38_GDN_SLOTS 1u
#endif
#if Q38_FFN_NORM_STREAM
_Static_assert(Q38_FFN_ROWS >= Q38_WIN,
               "FFN norm slot must hold one mixer window");
#endif
// RedMulE output tile width: attention PV writes head_dim, projections write
// one column panel.
#define Q38_ZW (Q38_KIND ? NT : Q38_MAX(NT, Q38_HD))
#if Q38_KIND == 0
#define Q38_ATTN_ENGINES Q38_MIN(NUM_REDMULE_TILES, Q38_QH *(Q38_WIN / MT))
#endif

_Static_assert(Q38_WIN % MT == 0 && Q38_FFN_ROWS % MT == 0,
               "Row blocks must hold whole RedMulE row tiles");
_Static_assert(Q38_PROJECTION_ROWS >= REDMULE_H && Q38_PROJECTION_ROWS <= MT &&
                   Q38_PROJECTION_ROWS % REDMULE_H == 0 &&
                   Q38_WIN % Q38_PROJECTION_ROWS == 0 &&
                   Q38_FFN_ROWS % Q38_PROJECTION_ROWS == 0,
               "Projection row tiles must fit the RedMulE and row blocks");
// K/V are packed per supertile when supertiles are on, else per window; the
// phased attention stores them per row tile.
_Static_assert(Q38_ATTN_VC ||
                   (Q38_ATTN_SUPERTILE ? Q38_ATTN_ST_ROWS : Q38_WIN) % BK == 0,
               "K/V packing units must hold whole attention key blocks");
_Static_assert(Q38_D % NT == 0 && Q38_I % NT == 0 && Q38_MIXW % NT == 0,
               "Projection widths must be whole column panels");

// Residual stream window. The stream itself lives in L2: `input_init` holds
// the layer input, `result_mixer` the post-mixer residual and `result_output`
// the layer output, so no L1 buffer here scales with the sequence.
// `residual` holds the window's incoming rows from the staging DMA until the
// matching residual addition; `mixed` receives the output projection.
#define Q38_AV_PRIVATE_LAYOUT                                                  \
  (Q38_ATTN_VC_SWEEP && Q38_PROBE_PHASE == 1 && !Q38_PROBE_BURST_ONLY)
_Static_assert(!Q38_AV_PRIVATE_LAYOUT ||
                   (!Q38_STAGE_SEPARATE && !Q38_BACKGROUND_FFN_NORM &&
                    !Q38_FFN_NORM_STREAM),
               "Private mixer layout cannot run background FFN helpers");
#if Q38_AV_PRIVATE_LAYOUT || Q38_FFN_L2
// The phased driver owns X/residual/Z in its reusable phase area. Generic
// projection/FFN buffers are unreachable in this mixer-only configuration.
static __fp16 residual[4] LOCAL;
#elif Q38_STAGE_SEPARATE
static __fp16 residual[Q38_WIN * Q38_D] LOCAL;
static __fp16 ffn_residual[Q38_FFN_ROWS * Q38_D] LOCAL;
#else
static __fp16 residual[Q38_ROWS * Q38_D] LOCAL;
#endif
#if Q38_AV_PRIVATE_LAYOUT || Q38_FFN_L2
static __fp16 mixed[4] LOCAL;
#elif Q38_GATEUP_PIPELINE && Q38_ASYNC && Q38_KIND == 0 && Q38_ATTN_SUPERTILE
// Supertiles project into stage.attention.out, and the pipelined FFN never
// uses `mixed`: only the probe burst's 16-row DMA target remains.
static __fp16 mixed[Q38_PROJECTION_ROWS * Q38_D] LOCAL;
#elif Q38_GATEUP_PIPELINE && Q38_ASYNC
static __fp16 mixed[Q38_WIN * Q38_D] LOCAL;
#else
static __fp16 mixed[Q38_ROWS * Q38_D] LOCAL;
#endif

// Shared GEMM input window: normalized rows first, then the gated mixer
// output that feeds the output projection.
#if Q38_AV_PRIVATE_LAYOUT || Q38_FFN_L2
static __fp16 a[4] LOCAL;
#elif Q38_GATEUP_PIPELINE && Q38_ASYNC
#if Q38_STAGE_SEPARATE
static __fp16 a[Q38_WIN * Q38_MAX(Q38_D, Q38_MIXW)] LOCAL;
static __fp16 ffn_a[Q38_FFN_ROWS * Q38_D] LOCAL;
#else
static __fp16
    a[Q38_MAX(Q38_WIN * Q38_MAX(Q38_D, Q38_MIXW), Q38_FFN_ROWS *Q38_D)] LOCAL;
#endif
#else
static __fp16 a[Q38_ROWS * Q38_MAX(Q38_D, Q38_MIXW)] LOCAL;
#endif
static __fp16 gamma[Q38_D] LOCAL;
#if Q38_STAGE_SEPARATE
static __fp16 ffn_gamma[Q38_D] LOCAL;
#endif
#if Q38_GATEUP_PIPELINE && Q38_ASYNC
#define Q38_PROJECTION_Z_SLOT (Q38_PROJECTION_ROWS * NT)
#else
#define Q38_PROJECTION_Z_SLOT (MT * Q38_ZW)
#endif
#if Q38_GATEUP_PIPELINE && Q38_ASYNC
_Static_assert(Q38_WEIGHT_PANELS > 0,
               "Gate/up overlap needs staged weight panels");
#endif
static __fp16 gemm_z[(Q38_AV_PRIVATE_LAYOUT || Q38_FFN_L2)
                         ? 4u
                         : NUM_REDMULE_TILES * 2 * Q38_PROJECTION_Z_SLOT] LOCAL;
// Dataflow gate/up weight ring: Q38_GU_SLOTS slots of Q38_GU_PANELS panels
// (gate and up). Two slots keep the historical layout (one per weights[]
// half). A deeper ring of smaller slots, carved from the same area, keeps a
// slot loading while enough groups compute to give every engine a job.
#ifndef Q38_GU_SLOTS
#define Q38_GU_SLOTS 2u
#endif
#ifndef Q38_GU_PANELS
#define Q38_GU_PANELS Q38_WEIGHT_PANELS
#endif
// Ordered K slices of the gate/up projections (1: full-K jobs). Above one, a
// step-synchronous double-buffered loop stages one K slice of Q38_GU_PANELS
// gate and up panels per step, and the FFN input is kept slice-major.
#ifndef Q38_FFN_GU_SLICES
#define Q38_FFN_GU_SLICES 1u
#endif
// One dataflow pipeline over all gate/up then down steps (see ffn_pipeline).
#ifndef Q38_FFN_PIPE
#define Q38_FFN_PIPE 0
#endif
// Weight slots per 2x2 VC of the VC-scoped FFN (q38_ffn.c); 0 disables it.
#ifndef Q38_FFN_VC
#define Q38_FFN_VC 0
#endif
// These phases carve their own L1 layout at phase start.
#define Q38_FFN_OWN_LAYOUT (Q38_FFN_PIPE || Q38_FFN_VC || Q38_ATTN_VC)
// Verify each staged pipeline step against its L2 source (debug).
#ifndef Q38_FP_CHECK
#define Q38_FP_CHECK 0
#endif
#define Q38_GUK (Q38_D / Q38_FFN_GU_SLICES)
#define Q38_GU_SLOT_ELEMS (2u * Q38_GU_PANELS * Q38_GUK * NT)
#if Q38_WEIGHT_PANELS > 0
#if Q38_FFN_OWN_LAYOUT
// The FFN carves its own buffers (ffn_pipeline, vc_setup).
#define Q38_WEIGHT_HALF (Q38_FFN_L2 ? 4u : Q38_MAX_K * NT * Q38_WEIGHT_PANELS)
#else
#define Q38_WEIGHT_HALF                                                        \
  Q38_MAX(Q38_MAX_K *NT *Q38_WEIGHT_PANELS,                                    \
          (Q38_GU_SLOTS * Q38_GU_SLOT_ELEMS + 1u) / 2u)
#endif
#if Q38_FFN_OWN_LAYOUT
_Static_assert(!Q38_FFN_NORM_STREAM && !Q38_STAGE_SEPARATE,
               "The FFN normalizes its own input in the phase area");
// Phase scratch: the mixer's weights and stage arena sit last in the static
// L1 image, and the FFN pipeline overlays them and everything above.
#define Q38_PHASE_AREA                                                         \
  __attribute__((section(".l1_tail"), aligned(4 * NUM_BANKS)))
static __fp16 weights[2][Q38_WEIGHT_HALF] Q38_PHASE_AREA;
#else
static __fp16 weights[2][Q38_WEIGHT_HALF] LOCAL;
#endif
#else
static __fp16 weights[2][Q38_MAX_WEIGHT] LOCAL;
#endif
#if Q38_STAGE_SEPARATE
static __fp16
    mixer_weights[2][Q38_MAX(Q38_D, Q38_MIXW) * NT * Q38_WEIGHT_PANELS] LOCAL;
static __fp16 mixer_z[NUM_REDMULE_TILES * 2 * Q38_PROJECTION_Z_SLOT] LOCAL;
#endif

typedef struct {
  __fp16 *slot[2];
  uint32_t slot_elems;
  __fp16 *z;
} q38_projection_buffers_t;

#if Q38_STAGE_SEPARATE
static const q38_projection_buffers_t mixer_buffers = {
    {mixer_weights[0], mixer_weights[1]},
    Q38_MAX(Q38_D, Q38_MIXW) * NT *Q38_WEIGHT_PANELS,
    mixer_z};
#else
static const q38_projection_buffers_t mixer_buffers = {
    {weights[0], weights[1]}, sizeof(weights[0]) / sizeof(__fp16), gemm_z};
#endif
static const q38_projection_buffers_t ffn_buffers = {
    {weights[0], weights[1]}, sizeof(weights[0]) / sizeof(__fp16), gemm_z};
#if Q38_STAGE_SEPARATE
#define Q38_FFN_A ffn_a
#define Q38_FFN_RESIDUAL ffn_residual
#define Q38_FFN_GAMMA ffn_gamma
#else
#define Q38_FFN_A a
#define Q38_FFN_RESIDUAL residual
#define Q38_FFN_GAMMA gamma
#endif
static uint32_t export_cycles[NUM_CORES] LOCAL;
#if Q38_SCHEDULE_PROBE
static uint32_t probe_mixer_windows LOCAL;
static uint32_t probe_ffn_blocks LOCAL;
static uint32_t probe_gdn_chunks LOCAL;
static uint32_t probe_key_blocks LOCAL;
static uint32_t probe_projection_groups LOCAL;
static uint32_t probe_gateup_groups LOCAL;
static uint32_t probe_mixer_end LOCAL;
static uint32_t probe_ffn_end LOCAL;
static uint32_t probe_burst_start LOCAL;
static uint32_t probe_burst_end LOCAL;
#endif
#if Q38_COORDINATOR_DMA && Q38_GATEUP_PIPELINE && Q38_ASYNC
static q38_dma_stage_queue_t gateup_dma_queue LOCAL;
static uint32_t gateup_dma_next[NUM_CORES] LOCAL;
#endif
#if Q38_BACKGROUND_FFN_NORM
static q38_dma_stage_queue_t ffn_norm_dma_queue LOCAL;
#endif

#if Q38_KIND == 0
_Static_assert(2 * Q38_QDIM % NT == 0 && Q38_KVDIM % NT == 0,
               "Attention projection widths must be whole column panels");

// Key/value window stage: the projection landing buffer and its head-major
// rearrangement, both dead once the window reaches the L2 cache.
#if Q38_ATTN_SUPERTILE
#define Q38_KV_ROWS Q38_ATTN_ST_ROWS
#else
#define Q38_KV_ROWS Q38_WIN
#endif
typedef struct {
  __fp16 projected[Q38_KV_ROWS * Q38_KVDIM];
  __fp16 heads[2 * Q38_KVH * Q38_KV_ROWS * Q38_HD];
} q38_kv_stage_t;

// Query window stage: one owner keeps every statistic of its rows for the
// whole causal key sweep, so no reduction crosses owners.
#if Q38_ATTN_LOCAL
// The cluster-wide sweep state lives in the group arenas instead.
#define Q38_SWEEP_ROWS 1u
#else
#define Q38_SWEEP_ROWS (Q38_QH * Q38_WIN)
#endif
// Shrunk sweep fields (8 B of softmax state) would misalign every field after
// them: a DMA beat must not straddle a tile's 256 B, and a RedMulE operand
// row starting mid-way through a 128 B wide access reads its second half from
// the neighbouring block. Align every field to a tile slice.
#define Q38_BEAT_ALIGNED __attribute__((aligned(256)))
typedef struct {
  // The phased attention stages its queries and gates in L2 instead.
  __fp16 qgate[Q38_ATTN_VC ? 1u : Q38_KV_ROWS * 2 * Q38_QDIM] Q38_BEAT_ALIGNED;
  __fp16 q[Q38_SWEEP_ROWS * Q38_HD] Q38_BEAT_ALIGNED;
  __fp16 context[Q38_ATTN_VC_SWEEP
                     ? 1u
                     : Q38_QH * Q38_WIN * Q38_HD] Q38_BEAT_ALIGNED;
  float accumulated[Q38_SWEEP_ROWS * Q38_HD] Q38_BEAT_ALIGNED;
  qwen38_softmax_state_t states[Q38_SWEEP_ROWS] Q38_BEAT_ALIGNED;
} q38_query_stage_t;
#if Q38_GATEUP_PIPELINE && Q38_ASYNC
typedef struct {
  q38_query_stage_t query;
#if Q38_ATTN_LOCAL
#define Q38_SWEEP_TILES 1u
#else
#define Q38_SWEEP_TILES (Q38_ATTN_ENGINES * MT)
#endif
  __fp16 z[Q38_SWEEP_TILES * Q38_HD] Q38_BEAT_ALIGNED;
  __fp16 kv_tile[2 * Q38_KVH * Q38_HD * BK] Q38_BEAT_ALIGNED;
  __fp16 scores[Q38_SWEEP_TILES * BK] Q38_BEAT_ALIGNED;
  __fp16 probs[Q38_SWEEP_TILES * BK] Q38_BEAT_ALIGNED;
  float corrections[Q38_SWEEP_TILES] Q38_BEAT_ALIGNED;
#if Q38_ATTN_SUPERTILE
  // Gated context and output projection of the whole supertile.
  __fp16 gated[Q38_ATTN_ST_ROWS * Q38_QDIM] Q38_BEAT_ALIGNED;
  __fp16 out[Q38_ATTN_ST_ROWS * Q38_D] Q38_BEAT_ALIGNED;
#endif
} q38_attention_stage_t;
#endif
#else
_Static_assert(Q38_QKVDIM % NT == 0 && Q38_QKVDIM % 32 == 0,
               "GDN projection width must be whole column panels");
#define Q38_CTRL ((2 * Q38_NV + NT - 1) / NT * NT)

#if Q38_GDN_READY_QUEUE
// Queued QKV, Z and control panels form one global panel list. A panel group
// holds up to Q38_QP panels of one matrix and yields (rows/PR)*count jobs.
#define Q38_QP Q38_GDN_QUEUE_PANELS
#define Q38_PR Q38_PROJECTION_ROWS
#define Q38_GB Q38_GDN_PROJECT_ROWS
#define Q38_P_QKV (Q38_QKVDIM / NT)
#define Q38_P_Z (Q38_GDIM / NT)
#define Q38_P_CTRL (Q38_CTRL / NT)
#define Q38_G_QKV ((Q38_P_QKV + Q38_QP - 1) / Q38_QP)
#define Q38_G_Z ((Q38_P_Z + Q38_QP - 1) / Q38_QP)
#define Q38_G_CTRL ((Q38_P_CTRL + Q38_QP - 1) / Q38_QP)
#define Q38_GROUPS (Q38_G_QKV + Q38_G_Z + Q38_G_CTRL)
#define Q38_STAGED_COLS (Q38_QKVDIM + Q38_GDIM + Q38_CTRL)
// Elements of one group in one L2 spill generation: [row tile][panel][PR][NT].
#define Q38_SPILL_GROUP (Q38_GB * Q38_QP * NT)
_Static_assert(Q38_GB % Q38_WIN == 0 && Q38_GB % Q38_GDN_CHUNK_ROWS == 0 &&
                   Q38_GDN_CHUNK_ROWS % Q38_PR == 0 && Q38_R % Q38_PR == 0 &&
                   Q38_R % Q38_GDN_CHUNK_ROWS == 0 && Q38_QP > 0 &&
                   Q38_GDIM % NT == 0 && NT % 32 == 0 &&
                   Q38_QP * Q38_MAX(Q38_D, Q38_GDIM) <=
                       Q38_MAX_K * Q38_WEIGHT_PANELS,
               "GDN ready queue: supertile, chunk and panel slots must align");
#endif

// GDN window stage. The recurrent state itself is head/column bounded and
// lives outside this arena because it must survive to the end-of-run check.
typedef struct {
#if Q38_GDN_READY_QUEUE
  // Staged projections, grouped as spilled: see q38_staged_at.
  __fp16 projected[Q38_GDN_STAGE_ROWS * Q38_STAGED_COLS];
#else
  __fp16 projected[Q38_GDN_STAGE_ROWS * Q38_QKVDIM];
#endif
  __fp16 convolution[Q38_GDN_STAGE_ROWS * Q38_QKVDIM];
#if !Q38_GDN_READY_QUEUE
  __fp16 zgate[Q38_GDN_STAGE_ROWS * Q38_GDIM];
#endif
  __fp16 context[Q38_NV * Q38_GDN_STAGE_ROWS * Q38_VD];
#if !Q38_GDN_READY_QUEUE
  __fp16 controls[Q38_GDN_STAGE_ROWS * Q38_CTRL];
#endif
  float q32[Q38_GDN_STAGE_ROWS * Q38_NK * Q38_KD];
  float k32[Q38_GDN_STAGE_ROWS * Q38_NK * Q38_KD];
  float decay[Q38_GDN_STAGE_ROWS * Q38_NV];
  float beta[Q38_GDN_STAGE_ROWS * Q38_NV];
} q38_gdn_slot_t;

typedef struct {
  q38_gdn_slot_t slot[Q38_GDN_SLOTS];
  __fp16 tail[4 * Q38_QKVDIM];
#if Q38_GDN_READY_QUEUE
  // Normalized input of the queued supertile, then the residual rows of the
  // supertile being output-projected: the queue drains between the two.
  __fp16 xin[Q38_GB * Q38_MAX(Q38_D, Q38_GDIM)];
  __fp16 gated[Q38_GB * Q38_GDIM];
  __fp16 out[Q38_GB * Q38_D];
  __fp16 landing[2][Q38_GB * Q38_QP * NT];
#endif
} q38_mixer_stage_t;
#endif

// The VC FFN carves its own buffers: its stage only backs the schedule
// probe's scratch rows, two panels wide.
#if Q38_FFN_VC
#define Q38_FFN_GATE_ROW (2u * NT)
#else
#define Q38_FFN_GATE_ROW Q38_I
#endif
typedef struct {
  __fp16 gate[Q38_FFN_ROWS * Q38_FFN_GATE_ROW];
#if Q38_GATEUP_PIPELINE && Q38_ASYNC
  __fp16
      up_panel[Q38_MAX(2u, Q38_GU_SLOTS)]
              [Q38_FFN_ROWS * Q38_MAX(Q38_WEIGHT_PANELS, Q38_GU_PANELS) * NT];
#else
  __fp16 up[Q38_FFN_ROWS * Q38_I];
#endif
} q38_ffn_stage_t;

#if Q38_STAGE_SEPARATE
// The K/V and query phases retain their existing exclusive lifetime. FFN
// buffers live beside the mixer arena so later workers may overlap stages.
static struct {
#if Q38_KIND == 0
  union {
    q38_kv_stage_t kv;
    q38_attention_stage_t attention;
  };
#else
  q38_mixer_stage_t mixer;
#endif
  q38_ffn_stage_t ffn;
} stage LOCAL;
#else
// Sequential stages share one arena until the independent-worker mode is set.
static union {
#if Q38_KIND == 0
  q38_kv_stage_t kv;
#if Q38_GATEUP_PIPELINE && Q38_ASYNC
  q38_attention_stage_t attention;
#else
  q38_query_stage_t query;
#endif
#else
  q38_mixer_stage_t mixer;
#endif
  q38_ffn_stage_t ffn;
#if Q38_FFN_OWN_LAYOUT
} stage Q38_PHASE_AREA;
#else
} stage LOCAL;
#endif
#endif

typedef struct {
  dma_collective_op_t op;
  const __fp16 *source;
  uint32_t slot;
  uint32_t valid;
} weight_prefetch_t;
static weight_prefetch_t prefetch[NUM_CORES] LOCAL;

static void schedule_delay(uint32_t tag) {
  if (Q38_DELAY_SEED)
    mempool_wait(((tag ^ (mempool_get_core_id() * 2654435761u)) *
                  (uint32_t)Q38_DELAY_SEED) &
                 63u);
}

// Record a DPI check with the tolerance as raw bits. mempool_check_dpi_* take
// a float that their F build moves through an FP register (fmv.s.x/fsw):
// illegal on the rv32ima core 0 of the 1 + 3 tile, the core that records.
static void q38_dpi(uint32_t type, const void *result, const void *golden,
                    uint32_t count, uint32_t tolerance_bits, uint32_t verbose) {
  // Domain builds: core ids are domain-local, so only domain 0 records (its EOC
  // waits for every domain); otherwise four domains race for the 4 slots.
  if (mempool_get_core_id() != 0u || mempool_get_domain_id() != 0u)
    return;
  const uint32_t id = mempool_dpi_check_count;
  if (id >= MEMPOOL_DPI_CHECK_MAX)
    return;
  mempool_dpi_checks[id].type = type;
  mempool_dpi_checks[id].count = count;
  mempool_dpi_checks[id].tolerance = tolerance_bits;
  mempool_dpi_checks[id].result_addr = (uint32_t)result;
  mempool_dpi_checks[id].golden_addr = (uint32_t)golden;
  mempool_dpi_checks[id].verbose = verbose;
  mempool_dpi_check_count = id + 1u;
}

// The legacy atomic + wake-CSR barrier, only at the start and the end of the
// program. Dropping every subscription first also drops whatever event the
// computing phases left pending (a subscribe write masks pending), so its
// single wfi is only released by its own wake.
static void q38_edge_barrier(void) {
#if FSYNC_TPG_SIDE != 0
  fsync_subscribe_self(0u);
#endif
  __sync_synchronize();
  mempool_barrier(NUM_CORES);
}

#if FSYNC_TPG_SIDE != 0
// fsync_barrier, but released only by a response of its own id and level:
// the tile's epoch also moves on another barrier's response, e.g. a VC
// barrier of this tile's agent while its workers already wait here. With one
// participant the caller arrives for its whole tile.
static NOSTACK void q38_barrier_n(uint32_t id, uint32_t aggr,
                                  uint32_t participants) {
  const uint32_t tile = mempool_get_tile_id();
  const uint32_t level = fsync_level_of_aggr(aggr);
  uint32_t epoch = fsync_epoch();
  if (participants == 1u) {
    fsync_trigger(id, aggr);
  } else if (__atomic_fetch_add(&fsync_tile_counters[tile].count, 1,
                                __ATOMIC_RELAXED) == participants - 1u) {
    fsync_tile_counters[tile].count = 0;
    fsync_trigger(id, aggr);
  }
  for (;;) {
    while (fsync_epoch() == epoch)
      mempool_wfi();
    // Epoch first: a response after this read moves it again.
    epoch = fsync_epoch();
    const uint32_t status = fsync_status();
    if (fsync_status_id(status) == id && fsync_status_level(status) == level)
      return;
  }
}

static void q38_barrier(uint32_t id, uint32_t aggr) {
  q38_barrier_n(id, aggr, NUM_CORES_PER_TILE);
}
#endif

// Every core of the machine meets, inside the run: the fractal_sync tree on
// the vertical ID (independent VCs may still retire horizontal barriers),
// isolated from agent DMA/engine events. Leaves the core subscribed to
// barrier events only.
static NOSTACK void sync_all(void) {
#if FSYNC_TPG_SIDE != 0
  fsync_subscribe_self(0u);
  fsync_subscribe_self(FSYNC_EVT_FSYNC);
  __sync_synchronize();
  q38_barrier(FSYNC_ID_V, fsync_aggr_global());
  __sync_synchronize();
#else
  q38_edge_barrier();
#endif
}

// The per-core stack is 512 bytes of the sequential region
// (software/runtime/arch.ld.c:23) and cannot grow without moving the hardware
// seq_mem_size. A printf costs 208 bytes of it, so fatal reporting from deep
// inside the projection chain writes the console directly instead.
static void report_fatal(const char *tag, uint32_t value) {
  while (*tag)
    _putchar(*tag++);
  for (int32_t shift = 28; shift >= 0; shift -= 4)
    _putchar("0123456789abcdef"[(value >> shift) & 15u]);
  _putchar('\n');
}

// __builtin_trap() is a no-op in GVSoC: the core runs on into whatever
// follows. This prints the cause and ends the run with a nonzero retval.
static void q38_fatal(const char *tag, uint32_t value) {
  report_fatal(tag, value);
  *(volatile uint32_t *)(CONTROL_REGISTER_OFFSET +
                         CONTROL_REGISTERS_EOC_REG_OFFSET) = (value << 1) | 1u;
  while (1)
    mempool_wfi();
}

static void transfer(void *dst, const void *src, uint32_t bytes) {
  const uint32_t error = dma_collective_memcpy_blocking(dst, src, bytes);
  if (error) {
    report_fatal("[qwen38_dma_error] core=", mempool_get_core_id());
    report_fatal("[qwen38_dma_error] status=", error);
    __builtin_trap();
  }
}

// Diagnostic tensor exports are attributed separately from the layer cycles.
static void export_tensor(void *dst, const void *src, uint32_t bytes) {
  const uint32_t start = mempool_get_timer();
  transfer(dst, src, bytes);
  export_cycles[mempool_get_core_id()] += mempool_get_timer() - start;
}

// Vector cores: every core but a tile's first NUM_SCALAR_CORES_PER_TILE
// (Snitch-only: the scheduler of the 1 + 3 tile). Vector and FP work is split
// over these; a scalar core's index is ~0u, which no loop bound reaches.
#ifndef NUM_SCALAR_CORES_PER_TILE
#define NUM_SCALAR_CORES_PER_TILE 0
#endif
#define Q38_VEC_PER_TILE (NUM_CORES_PER_TILE - NUM_SCALAR_CORES_PER_TILE)
#define Q38_VEC_CORES (NUM_CORES / NUM_CORES_PER_TILE * Q38_VEC_PER_TILE)
static inline uint32_t q38_vec_id(void) {
  const uint32_t c = mempool_get_core_id(), l = c % NUM_CORES_PER_TILE;
  return l < NUM_SCALAR_CORES_PER_TILE
             ? ~0u
             : c / NUM_CORES_PER_TILE * Q38_VEC_PER_TILE + l -
                   NUM_SCALAR_CORES_PER_TILE;
}

// Each vector core zeroes one contiguous slice with vector stores; one scalar
// store per word cost ~60k cycles for a 1.5 MB accumulator.
static void clear_all(void *ptr, uint32_t bytes) {
  const uint32_t words = (bytes + 3u) / 4u, core = q38_vec_id();
  uint32_t *p = (uint32_t *)ptr + words * core / Q38_VEC_CORES;
  uint32_t avl = core == ~0u ? 0u
                             : words * (core + 1u) / Q38_VEC_CORES -
                                   words * core / Q38_VEC_CORES;
  while (avl) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vmv.v.i v0, 0");
    asm volatile("vse32.v v0, (%0)" : : "r"(p) : "memory");
    p += vl;
    avl -= vl;
  }
  sync_all();
}

#if Q38_SCHEDULE_PROBE
#ifndef Q38_PROBE_NO_SEED
#define Q38_PROBE_NO_SEED 0 // timing-only: skip the scalar L2 fill (1M cycles)
#endif
static NOSTACK void probe_fill16(__fp16 *dst, uint32_t elements) {
  for (uint32_t i = q38_vec_id(); !Q38_PROBE_NO_SEED && i < elements;
       i += Q38_VEC_CORES)
    dst[i] = (__fp16)0.125f;
  sync_all();
}

#if Q38_KIND != 0 && Q38_PROBE_BASE > 0
static NOSTACK void probe_fill32(float *dst, uint32_t elements, float value) {
  for (uint32_t i = q38_vec_id(); i < elements; i += Q38_VEC_CORES)
    dst[i] = value;
  sync_all();
}
#endif
#endif

#if Q38_GDN_RECUR_PROJECTION_OVERLAP
// The collective DMA handles L1<->L2 staging, not copies between two L1
// slots. Keep the latter on Spatz with disjoint 32-element ownership.
static NOSTACK void copy_all16(__fp16 *dst, const __fp16 *src,
                               uint32_t elements) {
  for (uint32_t first = q38_vec_id() * 32u;
       q38_vec_id() != ~0u && first < elements; first += Q38_VEC_CORES * 32u) {
    const uint32_t count = elements - first < 32u ? elements - first : 32u;
    q38_copy16(dst + first, src + first, count);
  }
  sync_all();
}
#endif

#if Q38_SAMPLED_CHECKS
// Bounded checking (plan section 7). At full geometry a golden tensor fits
// neither the host's compute budget nor L2 beside the weights, so the guest
// gathers a fixed, anchored set of contiguous runs out of its own result and
// only that set is checked. Raw 16/32-bit moves: nothing is re-rounded on the
// way, so a gathered element is bit-for-bit the element the layer wrote.
//
// ONE CORE GATHERS. Spreading the scatter/gather over every core aborts the
// GVSoC model: the group AXI interface asserts "downstream sent a new
// response beat while this output still holds a denied one" once 128 cores
// issue independent scalar L2 reads at the same time. These gathers run after
// the measured interval closes, so serialising them costs nothing reported.
static NOSTACK void gather_runs_f16(const __fp16 *src, const uint32_t *start,
                                    __fp16 *dst, uint32_t runs, uint32_t len) {
  // Keep the trip counts opaque. With constant bounds clang unrolls the whole
  // gather -- 6163 instructions and 2 KB of spill for one call -- which does
  // not fit the 512-byte stack.
  asm volatile("" : "+r"(runs), "+r"(len));
  if (mempool_get_core_id() == 0)
    for (uint32_t j = 0; j < runs; ++j)
      for (uint32_t i = 0; i < len; ++i) {
        uint32_t word;
        asm volatile("lhu %0, 0(%1)"
                     : "=r"(word)
                     : "r"(src + start[j] + i)
                     : "memory");
        asm volatile("sh %0, 0(%1)"
                     :
                     : "r"(word), "r"(dst + j * len + i)
                     : "memory");
      }
  sync_all();
}

static NOSTACK void gather_runs_f32(const float *src, const uint32_t *start,
                                    float *dst, uint32_t runs, uint32_t len) {
  // Keep the trip counts opaque. With constant bounds clang unrolls the whole
  // gather -- 6163 instructions and 2 KB of spill for one call -- which does
  // not fit the 512-byte stack.
  asm volatile("" : "+r"(runs), "+r"(len));
  if (mempool_get_core_id() == 0)
    for (uint32_t j = 0; j < runs; ++j)
      for (uint32_t i = 0; i < len; ++i) {
        uint32_t word;
        asm volatile("lw %0, 0(%1)"
                     : "=r"(word)
                     : "r"(src + start[j] + i)
                     : "memory");
        asm volatile("sw %0, 0(%1)"
                     :
                     : "r"(word), "r"(dst + j * len + i)
                     : "memory");
      }
  sync_all();
}

// The mixer context is diagnostic: the layer consumes it in L1 and throws it
// away. Writing the whole tensor out so it could be checked put real DMA
// inside the measured interval and cost tens of megabytes of L2 at full
// geometry. Instead each window copies only the runs that fall in its own
// rows, straight out of the L1 window buffer, and the tensor never exists in
// L2.
//
// The copies are single vector load/stores on one core, not collective DMA.
// A run is a couple of hundred bytes, so a transfer's barrier costs far more
// than its payload; going around the collective removes a fixed cost per run
// from the measured interval, which is the whole point of gathering here.
static NOSTACK void gather_context_window(const __fp16 *window, uint32_t base,
                                          uint32_t rows, uint32_t width) {
  const uint32_t start_time = mempool_get_timer();
  uint32_t runs = Q38_RUNS_CONTEXT;
  uint32_t length = Q38_RUNLEN_CONTEXT;
  asm volatile("" : "+r"(runs), "+r"(length));
  if (mempool_get_core_id() == 0)
    for (uint32_t j = 0; j < runs; ++j) {
      const uint32_t flat = run_start_context[j];
      const uint32_t row = (flat / width) % Q38_R;
      if (row < base || row >= base + rows)
        continue;
      const uint32_t head = flat / (Q38_R * width);
      const __fp16 *const source =
          window + (head * rows + row - base) * width + flat % width;
      __fp16 *const target = result_sample_context + j * Q38_RUNLEN_CONTEXT;
      // Scalar, like every other gather here. The vector copy that was here
      // wrote values from elsewhere in the window for ~18% of the elements
      // while the same data checked exactly as a whole tensor; the three
      // tensors gathered with `lhu`/`sh` never did.
      for (uint32_t i = 0; i < length; ++i) {
        uint32_t word;
        asm volatile("lhu %0, 0(%1)" : "=r"(word) : "r"(source + i) : "memory");
        asm volatile("sh %0, 0(%1)" : : "r"(word), "r"(target + i) : "memory");
      }
    }
  sync_all();
  export_cycles[mempool_get_core_id()] += mempool_get_timer() - start_time;
}
#endif

static uint32_t block_rows(uint32_t base, uint32_t bound) {
  const uint32_t left = Q38_R - base;
  return left < bound ? left : bound;
}

// GEMM reads its Y addend from Z, so a job that keeps Z continues the FP16
// accumulation of the job that wrote it.
static uint8_t start_job_y(const __fp16 *in, const __fp16 *weight, __fp16 *out,
                           uint32_t m, uint32_t k, uint32_t n, uint32_t zero) {
  schedule_delay((uint32_t)out);
  if (zero && !Q38_REDMULE_MATMUL)
    llm_zero_f16(out, m * n);
  const int32_t target = tnc_redmule_acquire_target();
  redmule_cfg((uint32_t)in, (uint32_t)weight, (uint32_t)out, (uint16_t)m,
              (uint16_t)k, (uint16_t)n, 0,
              zero && Q38_REDMULE_MATMUL ? MATMUL : GEMM, Float16);
  redmule_x_shift_set(0);
  tnc_redmule_trigger();
  return (uint8_t)target;
}

static uint8_t start_job(const __fp16 *in, const __fp16 *weight, __fp16 *out,
                         uint32_t m, uint32_t k, uint32_t n) {
  return start_job_y(in, weight, out, m, k, n, 1u);
}

static void job(const __fp16 *in, const __fp16 *weight, __fp16 *out, uint32_t m,
                uint32_t k, uint32_t n) {
  tnc_redmule_wait(start_job(in, weight, out, m, k, n));
}

#if Q38_WEIGHT_PANELS == 0
// All column panels are staged once per projection call. Jobs partition BOTH
// rows and columns; each owner writes its disjoint output rectangle after EOC.
// With more than one activation window this restages the matrix per window:
// that is the cost this control measures, not a recommended mapping.
static void project_rows(const __fp16 *in, const __fp16 *packed_weight,
                         __fp16 *out, uint32_t k, uint32_t n,
                         const __fp16 *next_weight, uint32_t next_k,
                         uint32_t next_n, uint32_t rows,
                         const q38_projection_buffers_t *buffers) {
  const uint32_t padded_n = (n + NT - 1) / NT * NT;
  weight_prefetch_t *pending = prefetch + mempool_get_core_id();
  uint32_t current_slot = 0;
  if (Q38_ASYNC && pending->valid) {
    if (dma_collective_wait(&pending->op) != 0u || pending->op.status != 0u)
      __builtin_trap();
    if (pending->source != packed_weight)
      __builtin_trap();
    current_slot = pending->slot;
    pending->valid = 0;
  } else {
    transfer(buffers->slot[current_slot], packed_weight, k * padded_n * 2);
  }
  if (Q38_ASYNC && next_weight != 0) {
    pending->slot = current_slot ^ 1u;
    pending->source = next_weight;
    pending->op = dma_collective_memcpy_nonblocking(
        buffers->slot[pending->slot], next_weight,
        next_k * ((next_n + NT - 1) / NT * NT) * 2);
    pending->valid = 1;
  }
  if (tnc_is_tile_agent()) {
    const uint32_t engine = mempool_get_redmule_id();
    const uint32_t columns = padded_n / NT;
    const uint32_t tasks = (rows / Q38_PROJECTION_ROWS) * columns;
    __fp16 *z = buffers->z + engine * 2 * MT * Q38_ZW;
    uint32_t slot = 0;
    uint8_t target = 0;
    if (Q38_ASYNC && engine < tasks) {
      target =
          start_job(in + (engine / columns * Q38_PROJECTION_ROWS) * k,
                    buffers->slot[current_slot] + (engine % columns) * k * NT,
                    z, Q38_PROJECTION_ROWS, k, NT);
    }
    for (uint32_t task = engine; task < tasks; task += NUM_REDMULE_TILES) {
      const uint32_t row = task / columns * Q38_PROJECTION_ROWS;
      const uint32_t column = task % columns;
      __fp16 *completed = z + slot * MT * Q38_ZW;
      if (Q38_ASYNC) {
        tnc_redmule_wait(target);
        const uint32_t next = task + NUM_REDMULE_TILES;
        if (next < tasks) {
          target = start_job(
              in + (next / columns * Q38_PROJECTION_ROWS) * k,
              buffers->slot[current_slot] + (next % columns) * k * NT,
              z + (slot ^ 1u) * MT * Q38_ZW, Q38_PROJECTION_ROWS, k, NT);
        }
      } else {
        job(in + row * k, buffers->slot[current_slot] + column * k * NT,
            completed, Q38_PROJECTION_ROWS, k, NT);
      }
      for (uint32_t r = 0; r < Q38_PROJECTION_ROWS; ++r)
        q38_copy16(out + (row + r) * padded_n + column * NT, completed + r * NT,
                   NT);
      schedule_delay(task + 1);
      if (Q38_ASYNC)
        slot ^= 1u;
    }
  }
  sync_all();
}

#else
// This path stages full-K jobs. Ordered K chunks with the prior FP16 output as
// GEMM Y could preserve feedback, but need a separate qualified scheduler.
// A buffer is reused only after every engine has consumed its column group.
static NOSTACK void
project_panel_jobs(const __fp16 *in, const __fp16 *staged_weight, __fp16 *out,
                   uint32_t k, uint32_t stride, uint32_t rows, uint32_t first,
                   uint32_t count, const q38_projection_buffers_t *buffers) {
  if (!tnc_is_tile_agent())
    return;
  const uint32_t engine = mempool_get_redmule_id();
  const uint32_t tasks = (rows / Q38_PROJECTION_ROWS) * count;
  __fp16 *z = buffers->z + engine * 2 * Q38_PROJECTION_Z_SLOT;
  uint32_t z_slot = 0;
  uint8_t target = 0;
  if (Q38_ASYNC && engine < tasks)
    target = start_job(in + (engine / count * Q38_PROJECTION_ROWS) * k,
                       staged_weight + (engine % count) * k * NT, z,
                       Q38_PROJECTION_ROWS, k, NT);
  for (uint32_t task = engine; task < tasks; task += NUM_REDMULE_TILES) {
    const uint32_t row = task / count * Q38_PROJECTION_ROWS;
    const uint32_t column = task % count;
    __fp16 *completed = z + z_slot * Q38_PROJECTION_Z_SLOT;
    if (Q38_ASYNC) {
      tnc_redmule_wait(target);
      const uint32_t next = task + NUM_REDMULE_TILES;
      if (next < tasks)
        target = start_job(in + (next / count * Q38_PROJECTION_ROWS) * k,
                           staged_weight + (next % count) * k * NT,
                           z + (z_slot ^ 1u) * Q38_PROJECTION_Z_SLOT,
                           Q38_PROJECTION_ROWS, k, NT);
    } else {
      job(in + row * k, staged_weight + column * k * NT, completed,
          Q38_PROJECTION_ROWS, k, NT);
    }
    for (uint32_t r = 0; r < Q38_PROJECTION_ROWS; ++r)
      q38_copy16(out + (row + r) * stride + (first + column) * NT,
                 completed + r * NT, NT);
    schedule_delay(first + task + 1);
    if (Q38_ASYNC)
      z_slot ^= 1u;
  }
}

static uint32_t next_panel_group(uint32_t first, uint32_t count,
                                 uint32_t columns, uint32_t group_panels) {
  uint32_t next = first + count;
#if Q38_SCHEDULE_PROBE
  // Leading adjacent groups exercise prefetch, slot reuse and steady state;
  // the final group exercises drain and any short column tail.
  const uint32_t final = ((columns - 1u) / group_panels) * group_panels;
  if (next >= Q38_PROBE_LEAD_GROUPS * group_panels && next < final)
    next = final;
#else
  (void)columns;
  (void)group_panels;
#endif
  return next;
}

static void project_rows_from_panel(const __fp16 *in,
                                    const __fp16 *packed_weight, __fp16 *out,
                                    uint32_t k, uint32_t n, uint32_t rows,
                                    uint32_t first_panel, uint32_t group_panels,
                                    const q38_projection_buffers_t *buffers) {
  const uint32_t columns = (n + NT - 1) / NT;
  const uint32_t stride = columns * NT;
  const uint32_t panel_bytes = k * NT * sizeof(__fp16);
  if (first_panel >= columns)
    return;
  uint32_t count = columns - first_panel;
  if (count > group_panels)
    count = group_panels;
  transfer(buffers->slot[0], packed_weight + first_panel * k * NT,
           count * panel_bytes);
  uint32_t slot = 0;
  for (uint32_t first = first_panel; first < columns;) {
#if Q38_SCHEDULE_PROBE
    if (mempool_get_core_id() == 0u)
      ++probe_projection_groups;
#endif
    count = columns - first;
    if (count > group_panels)
      count = group_panels;
    const uint32_t next = next_panel_group(first, count, columns, group_panels);
    uint32_t next_count = columns - next;
    if (next_count > group_panels)
      next_count = group_panels;
    dma_collective_op_t pending;
    if (Q38_ASYNC && next_count)
      pending = dma_collective_memcpy_nonblocking(buffers->slot[slot ^ 1u],
                                                  packed_weight + next * k * NT,
                                                  next_count * panel_bytes);
    project_panel_jobs(in, buffers->slot[slot], out, k, stride, rows, first,
                       count, buffers);
    sync_all();
    if (next_count) {
      if (Q38_ASYNC) {
        if (dma_collective_wait(&pending) != 0u || pending.status != 0u)
          __builtin_trap();
      } else {
        transfer(buffers->slot[slot ^ 1u], packed_weight + next * k * NT,
                 next_count * panel_bytes);
      }
    }
    slot ^= 1u;
    first = next;
  }
}

static void project_rows(const __fp16 *in, const __fp16 *packed_weight,
                         __fp16 *out, uint32_t k, uint32_t n,
                         const __fp16 *next_weight, uint32_t next_k,
                         uint32_t next_n, uint32_t rows,
                         const q38_projection_buffers_t *buffers) {
  (void)next_weight;
  (void)next_k;
  (void)next_n;
  const uint32_t columns = (n + NT - 1) / NT;
  const uint32_t panel_bytes = k * NT * sizeof(__fp16);
#if Q38_ADAPTIVE_PANELS
  // The slot is sized for the largest K. Shorter reductions can stage more
  // independent output-column panels without changing the L1 footprint.
  const uint32_t capacity = buffers->slot_elems * sizeof(__fp16) / panel_bytes;
  const uint32_t group_panels = columns < capacity ? columns : capacity;
#else
  const uint32_t group_panels = Q38_WEIGHT_PANELS;
#endif
  project_rows_from_panel(in, packed_weight, out, k, n, rows, 0u, group_panels,
                          buffers);
}
#endif

// The VC helpers come first: both phases build on them.
#include "q38_vc.c"
#if Q38_G2_ITEM_KEEP && Q38_X8_BATCH
// Defined with the resident parameter layout in q38_gdn2.c; f2 must stop here.
static uint32_t g2_item_limit(void);
#endif
// Then the projection machinery and the phases.
#if Q38_PROBE_PHASE != 1
#include "q38_ffn.c"
#endif
#if Q38_GDN_V2
#include "q38_gdn2.c"
#endif
#if !Q38_FFN_L2 || Q38_ATTN_L2_LAYER
#include "q38_projection.c"
#if Q38_KIND == 0
#include "q38_attention.c"
#include "q38_attn_vc.c"
#else
#include "q38_gdn.c"
#endif
#endif

#if Q38_SCHEDULE_PROBE && !Q38_PROBE_SKIP_BURST
#define Q38_BURST_ROWS (2u * Q38_FFN_ROWS)
#define Q38_BURST_ROW_TILES (Q38_BURST_ROWS / Q38_PROJECTION_ROWS)
#define Q38_BURST_PANELS (NUM_REDMULE_TILES / Q38_BURST_ROW_TILES)
#if Q38_STAGE_SEPARATE
#define Q38_BURST_X0 ffn_a
#define Q38_BURST_X1 ffn_residual
#define Q38_BURST_Z mixer_z
#define Q38_BURST_DMA_DEST residual
#else
#define Q38_BURST_X0 a
#define Q38_BURST_X1 residual
#define Q38_BURST_Z gemm_z
#define Q38_BURST_DMA_DEST mixed
#endif
_Static_assert(Q38_BURST_ROWS <= Q38_PROBE_ROWS &&
                   Q38_BURST_ROWS % Q38_PROJECTION_ROWS == 0 &&
                   NUM_REDMULE_TILES % Q38_BURST_ROW_TILES == 0 &&
                   Q38_BURST_PANELS > 0 &&
                   Q38_BURST_PANELS * Q38_D * NT <=
                       sizeof(weights) / sizeof(weights[0][0]) &&
                   2u * Q38_FFN_ROWS >= NUM_CORES && Q38_WEIGHT_PANELS >= 2,
               "Probe burst must fit L1 and offer one job per RedMulE");

// A bounded output-task capacity test, distinct from the following layer
// schedule. It reuses the idle FFN arenas as X rows and stages real full-K
// weights before issuing exactly one output rectangle on each engine.
static NOSTACK void probe_projection_burst(void) {
  const uint32_t core = mempool_get_core_id();
#if Q38_KIND == 0
  const __fp16 *const source_weight = w_q_gate;
#else
  const __fp16 *const source_weight = w_qkv;
#endif
  transfer(Q38_BURST_X0, input_init + Q38_EXEC_START * Q38_D,
           Q38_FFN_ROWS * Q38_D * 2u);
  transfer(Q38_BURST_X1, input_init + (Q38_EXEC_START + Q38_FFN_ROWS) * Q38_D,
           Q38_FFN_ROWS * Q38_D * 2u);
  transfer(weights[Q38_PROBE_BURST_SLOT], source_weight,
           Q38_BURST_PANELS * Q38_D * NT * 2u);
  sync_all();
  if (core == 0u)
    probe_burst_start = mempool_get_timer();
  sync_all();
  uint8_t target = 0u;
  if (tnc_is_tile_agent()) {
    const uint32_t engine = mempool_get_redmule_id();
    const uint32_t row_tile = engine / Q38_BURST_PANELS;
    const uint32_t row = row_tile * Q38_PROJECTION_ROWS;
    const __fp16 *const input =
        row < Q38_FFN_ROWS ? Q38_BURST_X0 + row * Q38_D
                           : Q38_BURST_X1 + (row - Q38_FFN_ROWS) * Q38_D;
    target = start_job(input,
                       weights[Q38_PROBE_BURST_SLOT] +
                           (engine % Q38_BURST_PANELS) * Q38_D * NT,
                       Q38_BURST_Z + engine * 2u * Q38_PROJECTION_Z_SLOT,
                       Q38_PROJECTION_ROWS, Q38_D, NT);
  }
  sync_all();
  dma_collective_op_t pending = dma_collective_memcpy_nonblocking(
      Q38_BURST_DMA_DEST, input_init + Q38_MIXER_EXEC_START * Q38_D,
      Q38_PROJECTION_ROWS * Q38_D * 2u);
  // Two independent column pieces per row expose a vector task on every core.
  const uint32_t row = core / 2u, piece = core % 2u;
  q38_activation16(stage.ffn.gate + row * Q38_FFN_GATE_ROW + piece * NT,
                   stage.ffn.up_panel[0] + row * Q38_WEIGHT_PANELS * NT +
                       piece * NT,
                   stage.ffn.gate + row * Q38_FFN_GATE_ROW + piece * NT, NT, 1);
  sync_all();
  if (dma_collective_wait(&pending) != 0u || pending.status != 0u)
    __builtin_trap();
  if (tnc_is_tile_agent()) {
    tnc_redmule_wait(target);
    // Model regression: further back-to-back rounds of the same 64 jobs.
    for (uint32_t round = 1u; round < Q38_PROBE_BURST_ROUNDS; ++round) {
      const uint32_t engine = mempool_get_redmule_id();
      const uint32_t row = engine / Q38_BURST_PANELS * Q38_PROJECTION_ROWS;
      tnc_redmule_wait(start_job(
          row < Q38_FFN_ROWS ? Q38_BURST_X0 + row * Q38_D
                             : Q38_BURST_X1 + (row - Q38_FFN_ROWS) * Q38_D,
          weights[Q38_PROBE_BURST_SLOT] +
              (engine % Q38_BURST_PANELS) * Q38_D * NT,
          Q38_BURST_Z + engine * 2u * Q38_PROJECTION_Z_SLOT,
          Q38_PROJECTION_ROWS, Q38_D, NT));
    }
  }
  sync_all();
  if (core == 0u)
    probe_burst_end = mempool_get_timer();
}
#endif

#if Q38_SCHEDULE_PROBE
static NOSTACK void probe_seed_inputs(void) {
  // FFN rows not produced by the selected mixer remain finite and explicit.
  probe_fill16(result_mixer + Q38_EXEC_START * Q38_D, Q38_PROBE_ROWS * Q38_D);
#if !Q38_PROBE_SKIP_BURST
  const uint32_t core = mempool_get_core_id();
  const uint32_t row = core / 2u, piece = core % 2u;
  for (uint32_t col = 0u; col < NT; ++col) {
    stage.ffn.gate[row * Q38_FFN_GATE_ROW + piece * NT + col] = (__fp16)0.125f;
    stage.ffn.up_panel[0][row * Q38_WEIGHT_PANELS * NT + piece * NT + col] =
        (__fp16)0.125f;
  }
#endif
  sync_all();
#if Q38_PROBE_BASE > 0
#if Q38_KIND == 0
  // Prefix keys are not projected in the tail probe. Give the causal sweep
  // finite, deterministic inputs without paying for those omitted windows.
  probe_fill16(result_cache, 2u * Q38_MIXER_EXEC_START * Q38_KVDIM);
#else
  probe_fill16(history, sizeof(history) / sizeof(history[0]));
  probe_fill16(stage.mixer.tail,
               sizeof(stage.mixer.tail) / sizeof(stage.mixer.tail[0]));
  probe_fill32(recurrent, sizeof(recurrent) / sizeof(recurrent[0]), 0.0625f);
#endif
#endif
}
#endif

#if Q38_SCHEDULE_PROBE
static NOSTACK void probe_report(void) {
  printf("[qwen38_probe] mixer_start_row=%u mixer_rows=%u "
         "ffn_start_row=%u ffn_rows=%u mixer_windows=%u "
         "ffn_blocks=%u gdn_chunks=%u key_blocks=%u "
         "projection_groups=%u gateup_groups=%u recurrent_k=%u "
         "burst_only=%u burst_start=%u burst_end=%u "
         "burst_jobs=%u burst_spatz_ops=%u "
         "mixer_end=%u ffn_end=%u numerical_checked=0\n",
         Q38_MIXER_EXEC_START, Q38_MIXER_EXEC_END - Q38_MIXER_EXEC_START,
         Q38_EXEC_START, Q38_PROBE_ROWS, probe_mixer_windows, probe_ffn_blocks,
         probe_gdn_chunks, probe_key_blocks, probe_projection_groups,
         probe_gateup_groups,
         Q38_PROBE_BURST_ONLY ? 0u : (Q38_KIND ? Q38_RECUR_K : 0u),
         Q38_PROBE_BURST_ONLY, probe_burst_start, probe_burst_end,
         Q38_PROBE_SKIP_BURST ? 0u : NUM_REDMULE_TILES,
         Q38_PROBE_SKIP_BURST ? 0u : NUM_CORES, probe_mixer_end, probe_ffn_end);
#if Q38_FFN_VC && !Q38_FFN_L2 && Q38_VC_PROFILE && Q38_PROBE_PHASE != 1
  for (uint32_t k = 0; k < VCP_COUNT; ++k) {
    uint32_t sum = 0, most = 0;
    for (uint32_t e = 0; e < NUM_REDMULE_TILES; ++e) {
      sum += vc_prof[k][e];
      most = vc_prof[k][e] > most ? vc_prof[k][e] : most;
    }
    printf("[qwen38_ffn_vc_agent] part=%u mean=%u max=%u\n", k,
           sum / NUM_REDMULE_TILES, most);
  }
#endif
#if Q38_ATTN_VC && Q38_KIND == 0
  printf("[qwen38_attn_vc] qkv_start=%u sweep_start=%u out_start=%u "
         "out_end=%u\n",
         av_bounds[0], av_bounds[1], av_bounds[2], av_bounds[3]);
  for (uint32_t ph = 0; ph < 2; ++ph)
    for (uint32_t k = 0; k < AVP_COUNT; ++k) {
      uint32_t sum = 0, most = 0;
      for (uint32_t e = 0; e < NUM_REDMULE_TILES; ++e) {
        sum += av_prof[ph][k][e];
        most = av_prof[ph][k][e] > most ? av_prof[ph][k][e] : most;
      }
      printf("[qwen38_attn_vc_agent] phase=%u part=%u mean=%u max=%u\n", ph, k,
             sum / NUM_REDMULE_TILES, most);
    }
#endif
#if Q38_FFN_PIPE
  for (uint32_t k = 0; k < FPP_COUNT; ++k) {
    uint32_t sum = 0, most = 0;
    for (uint32_t e = 0; e < NUM_REDMULE_TILES; ++e) {
      sum += fp_prof[k][e];
      most = fp_prof[k][e] > most ? fp_prof[k][e] : most;
    }
    printf("[qwen38_ffn_agent] part=%u mean=%u max=%u\n", k,
           sum / NUM_REDMULE_TILES, most);
  }
#endif
#if Q38_KIND == 0 && !Q38_FFN_L2
  // Totals over the probed key blocks and windows; divide by their counts.
  for (uint32_t phase = 0; phase < KP_COUNT; ++phase) {
    uint32_t sum = 0, most = 0;
    for (uint32_t c = 0; c < NUM_CORES; ++c) {
      sum += kp_cycles[phase][c];
      most = kp_cycles[phase][c] > most ? kp_cycles[phase][c] : most;
    }
    printf("[qwen38_key_phase] phase=%u mean=%u max=%u\n", phase,
           sum / NUM_CORES, most);
  }
#endif
#if Q38_GDN_READY_QUEUE
  printf("[qwen38_queue] supertile_rows=%u panels=%u groups=%u "
         "queue_groups=%u queue_jobs=%u overlap_start=%u overlap_end=%u "
         "prologue_cycles=%u recur_cycles=%u drain_cycles=%u "
         "output_cycles=%u\n",
         Q38_GB, Q38_QP, Q38_GROUPS, probe_queue_groups, probe_queue_jobs,
         probe_overlap_start, probe_overlap_end, probe_prologue_cycles,
         probe_recur_cycles, probe_drain_cycles, probe_output_cycles);
#endif
}
#endif

#if DOMAINS > 1
// Domain builds: the domain-fold DMA paths take the tile agents (core 0 of a
// tile) past the 512 B sequential-region stack, which cannot grow
// (stackdepth.py). The agents run on Q38_AGENT_STACK_ROWS tile-window rows of
// their own tile instead: contiguous and local. Simulation only: the stack
// limit CSR is left as crt0 set it (the model does not check it), and no DMA
// may target a stack variable (the tile window is compute-only).
// Five rows give 640 B per tile at this geometry, above the 512 B gate
// plus the 16 B top guard. The stack still uses its owner's tile window.
#define Q38_AGENT_STACK_ROWS                                                   \
  ((STACK_SIZE + 16u + VC_TILE_ROW_BYTES - 1u) / VC_TILE_ROW_BYTES)
_Static_assert(Q38_AGENT_STACK_ROWS *VC_TILE_ROW_BYTES >= STACK_SIZE + 16u,
               "The agent stack includes the gate budget and top guard");
static uint8_t q38_agent_stack[Q38_AGENT_STACK_ROWS * VC_GLOBAL_ROW_BYTES]
    __attribute__((section(".l1_rows"), aligned(VC_GLOBAL_ROW_BYTES)));
int q38_call_on_stack(int (*fn)(void), uint32_t top);
__asm__(".globl q38_call_on_stack\n"
        "q38_call_on_stack:\n"
        "  addi sp, sp, -16\n"
        "  sw ra, 12(sp)\n"
        "  sw s0, 8(sp)\n"
        "  mv s0, sp\n"
        "  mv sp, a1\n"
        "  jalr a0\n"
        "  mv sp, s0\n"
        "  lw s0, 8(sp)\n"
        "  lw ra, 12(sp)\n"
        "  addi sp, sp, 16\n"
        "  ret\n");
static int q38_main(void);
#endif
#if Q38_GDN_V2 && Q38_X8_BATCH
// The batch-major GDN layer: in-projection of batch b+1 after the FFN of
// batch b, so that a later overlap can move it under the chunk core of b.
#define X8_BATCHES (Q38_R / Q38_FFN_ROWS)
// Debug (numeric coverage): run the chunk core in windows of this many rows
// within each batch, so that a short sequence also runs middle windows (state
// in and out).
#ifndef Q38_X8_CHUNK_ROWS
#define Q38_X8_CHUNK_ROWS Q38_FFN_ROWS
#endif
_Static_assert(Q38_FFN_ROWS % Q38_X8_CHUNK_ROWS == 0u &&
                   Q38_X8_CHUNK_ROWS % 128u == 0u,
               "Chunk windows divide the batch and keep the z-buffer parity");
// Core 0's phase stamps per batch: in_proj (since the previous FFN), chunks,
// out_proj, FFN; printed after the layer.
#if !Q38_X8_PIPE
static uint32_t x8_stamp[X8_BATCHES][4] Q38_VC_SMALL;
static NOSTACK void x8_gdn_layer(void) {
  const uint32_t core = mempool_get_core_id();
#if Q38_G2_ITEM_KEEP
  g2_item_begin();
#endif
  uint32_t t0 = mempool_get_timer();
  gdn2_project(0u, Q38_FFN_ROWS);
#pragma clang loop unroll(disable)
  for (uint32_t b = 0u; b < X8_BATCHES; ++b) {
    const uint32_t w0 = b * Q38_FFN_ROWS, w1 = w0 + Q38_FFN_ROWS;
    const uint32_t t1 = mempool_get_timer();
#pragma clang loop unroll(disable)
    for (uint32_t c0 = w0; c0 < w1; c0 += Q38_X8_CHUNK_ROWS)
      gdn2_chunks(c0, c0 + Q38_X8_CHUNK_ROWS);
    const uint32_t t2 = mempool_get_timer();
    gdn2_output(w0, w1);
    const uint32_t t3 = mempool_get_timer();
    ffn_rows(w0, w1);
    const uint32_t t4 = mempool_get_timer();
    if (core == 0u) {
      x8_stamp[b][0] = t1 - t0, x8_stamp[b][1] = t2 - t1;
      x8_stamp[b][2] = t3 - t2, x8_stamp[b][3] = t4 - t3;
    }
    t0 = t4;
    if (w1 < Q38_R)
      gdn2_project(w1, w1 + Q38_FFN_ROWS);
  }
}

static void x8_report(void) {
  uint32_t t[4] = {0u};
  for (uint32_t b = 0u; b < X8_BATCHES; ++b) {
    printf("[x8_batch] rows=%u in_proj=%u chunks=%u out_proj=%u ffn=%u\n",
           b * Q38_FFN_ROWS, x8_stamp[b][0], x8_stamp[b][1], x8_stamp[b][2],
           x8_stamp[b][3]);
    for (uint32_t i = 0u; i < 4u; ++i)
      t[i] += x8_stamp[b][i];
  }
  printf("[gdn2] in_proj=%u chunks=%u out_proj=%u total=%u ffn=%u\n", t[0],
         t[1], t[2], t[0] + t[1] + t[2], t[3]);
}
#else
#include "q38_x8_pipe.c"
#endif
#endif
#ifndef Q38_AV_DOMAIN_REPORT
#define Q38_AV_DOMAIN_REPORT 0
#endif
#if Q38_AV_DOMAIN_REPORT
#if !(DOMAINS > 1 && Q38_ATTN_VC && Q38_AV_PROFILE && Q38_KIND == 0)
#error "Q38_AV_DOMAIN_REPORT reports the attention bounds of a domain build"
#endif
// Only domain 0 prints (serial.c), so every domain's core 0 posts its
// attention bounds and layer span here (shared L2 image), after a done word,
// and domain 0's core 0 prints them all.
#define AV_DOM_NOT_READY 0x5a5a5a5au
#define AV_DOM_DONE(d) (0x6a000000u | (d))
static volatile uint32_t av_dom_rep[DOMAINS][7] = {
    [0 ... DOMAINS - 1] = {[0 ... 6] = AV_DOM_NOT_READY}};

static NOSTACK void av_domain_report(uint32_t start, uint32_t end) {
  const uint32_t d = mempool_get_domain_id();
  for (uint32_t i = 0u; i < 4u; ++i)
    av_dom_rep[d][1u + i] = av_bounds[i];
  av_dom_rep[d][5] = start;
  av_dom_rep[d][6] = end;
  __sync_synchronize();
  av_dom_rep[d][0] = AV_DOM_DONE(d);
  __sync_synchronize();
  if (d != 0u)
    return;
  for (uint32_t k = 0u; k < DOMAINS; ++k) {
    const volatile uint32_t *const r = av_dom_rep[k];
    while (r[0] != AV_DOM_DONE(k))
      mempool_wait(256u);
    __sync_synchronize();
    printf("[qwen38_av_domain] domain=%u qkv=%u sweep=%u out=%u start=%u "
           "qkv_end=%u sweep_end=%u end=%u layer_start=%u layer_end=%u\n",
           k, r[2] - r[1], r[3] - r[2], r[4] - r[3], r[1], r[2], r[3], r[4],
           r[5], r[6]);
  }
}
#endif
// GVSoC sends a trapping core (e.g. an FP or vector instruction on a scalar
// agent) back through the boot ROM into main, silently: it then waits in a
// barrier the others never reach. Initialized data, so loaded with the ELF.
// Domain builds run one image per domain on domain-local core ids.
#define Q38_BOOT_MARKS (DOMAINS * NUM_CORES)
static volatile uint32_t q38_boot_mark[Q38_BOOT_MARKS] = {
    [0 ... Q38_BOOT_MARKS - 1] = 1u};
static void q38_boot_check(void) {
  const uint32_t i =
      mempool_get_domain_id() * NUM_CORES + mempool_get_core_id();
  if (q38_boot_mark[i] != 1u)
    q38_fatal("[qwen38_fatal] core re-entered main=", i);
  q38_boot_mark[i] = 2u;
}
#ifndef Q38_WAIT_SIDECAR
#define Q38_WAIT_SIDECAR 0
#endif
#if Q38_WAIT_SIDECAR
#if !Q38_SAMPLED_CHECKS
#error "Q38_WAIT_SIDECAR polls the last sampled-check gold of the fixture"
#endif
// Numeric runs: the model's loader keeps writing the data sidecar, in
// program-header order, for ~24k cycles after the cores start, and the layer
// reads its tail (norm gammas, RoPE) earlier. Core 0 waits for the last
// segment, the last check's gold samples, before the start barrier.
static NOSTACK void q38_wait_sidecar(void) {
#if Q38_KIND == 0
  const volatile uint32_t *last = (const volatile uint32_t *)gold_sample_mixer;
  const uint32_t words = Q38_RUNS_MIXER * Q38_RUNLEN_MIXER / 2u;
#else
  const volatile uint32_t *last =
      (const volatile uint32_t *)gold_sample_recurrent;
  const uint32_t words = Q38_RUNS_RECURRENT * Q38_RUNLEN_RECURRENT;
#endif
  for (;;) {
    for (uint32_t i = 0u; i < words; ++i)
      if (last[i])
        return;
    mempool_wait(1000u);
  }
}
#endif
#if DOMAINS > 1
int main(void) {
  q38_boot_check();
  if (mempool_get_core_id() % NUM_CORES_PER_TILE != 0u)
    return q38_main();
  const uint32_t rows = VC_GLOBAL_ROW(q38_agent_stack) + Q38_AGENT_STACK_ROWS;
  return q38_call_on_stack(
      q38_main, (uint32_t)vc_tile_ptr(rows * VC_TILE_ROW_BYTES) - 16u);
}
static int q38_main(void) {
#else
int main(void) {
  q38_boot_check();
#endif
  const uint32_t core = mempool_get_core_id();
#if Q38_WAIT_SIDECAR
  if (core == 0u)
    q38_wait_sidecar();
#endif
  mempool_barrier_init(core);
#if Q38_ATTN_VC && Q38_AV_PROFILE && Q38_KIND == 0
  av_profile_init();
#if Q38_ATTN_VC_SWEEP
  avs_profile_init();
#endif
#endif
#if !(Q38_G2_ITEM_KEEP && Q38_X8_BATCH) &&                                     \
    !(Q38_KIND == 0 && Q38_FFN_L2 && Q38_ATTN_VC && Q38_AV_QKV_F2 &&           \
      Q38_AV_OUT_F2)
  // The f2-only attention path and the batch-major GDN-v2 path uses f2's DMA
  // state, never these legacy buffers. Omitting their only live references
  // frees L1 for layer params.
  prefetch[core].valid = 0;
  prefetch[core].slot = 0;
#endif
  export_cycles[core] = 0;
#if Q38_COORDINATOR_DMA && Q38_GATEUP_PIPELINE && Q38_ASYNC &&                 \
    !(Q38_G2_ITEM_KEEP && Q38_X8_BATCH) &&                                     \
    !(Q38_KIND == 0 && Q38_FFN_L2 && Q38_ATTN_VC && Q38_AV_QKV_F2 &&           \
      Q38_AV_OUT_F2)
  gateup_dma_next[core] = 1u;
  q38_dma_stage_init(&gateup_dma_queue);
#endif
#if Q38_BACKGROUND_FFN_NORM
  q38_dma_stage_init(&ffn_norm_dma_queue);
#endif
#if Q38_GDN_READY_QUEUE
  if (core == 0u) {
    q38_dma_stage_init(&gdn_weight_queue);
    q38_dma_stage_init(&gdn_spill_queue);
    gdn_queue.active = 0u;
    gdn_queue.next = 0u;
    gdn_queue.completed = 0u;
    for (uint32_t i = 0u; i < 4u; ++i)
      gdn_queue.done[i] = 0u;
#if Q38_SCHEDULE_PROBE
    probe_queue_groups = 0u;
    probe_queue_jobs = 0u;
    probe_overlap_start = 0u;
    probe_overlap_end = 0u;
    probe_prologue_cycles = 0u;
    probe_recur_cycles = 0u;
    probe_drain_cycles = 0u;
    probe_output_cycles = 0u;
#endif
  }
#endif
#if Q38_GDN_TRIPLE_PIPELINE
  gdn_dma_epoch[core] = 0u;
  gdn_raw_epoch[core] = 0u;
  q38_dma_stage_init(&gdn_input_dma_queue);
#endif
#if Q38_SCHEDULE_PROBE
  if (core == 0u) {
    probe_mixer_windows = 0u;
    probe_ffn_blocks = 0u;
    probe_gdn_chunks = 0u;
    probe_key_blocks = 0u;
    probe_projection_groups = 0u;
    probe_gateup_groups = 0u;
    probe_burst_start = 0u;
    probe_burst_end = 0u;
    probe_mixer_end = 0u;
    probe_ffn_end = 0u;
  }
#endif
#if Q38_FFN_L2 && Q38_GDN_ENTRY_INIT
  f2_init_records();
#endif
  q38_edge_barrier();
#if Q38_SCHEDULE_PROBE
  probe_seed_inputs();
#elif Q38_PROBE_PHASE == 2 && Q38_FFN_L2
  f2_layout(0u);
  sync_all();
  for (uint32_t base = 0u; base < Q38_R; base += Q38_FFN_ROWS) {
    transfer(f2.residual, gold_mixer + base * Q38_D,
             Q38_FFN_ROWS * Q38_D * sizeof(__fp16));
    transfer(result_mixer + base * Q38_D, f2.residual,
             Q38_FFN_ROWS * Q38_D * sizeof(__fp16));
  }
#elif Q38_PROBE_PHASE == 2
// Seed through L1 so the phase-only check uses the qualified DMA paths.
for (uint32_t base = 0u; base < Q38_R; base += Q38_FFN_ROWS) {
  const uint32_t rows = Q38_MIN(Q38_FFN_ROWS, Q38_R - base);
  transfer(Q38_FFN_RESIDUAL, gold_mixer + base * Q38_D,
           rows * Q38_D * sizeof(__fp16));
  transfer(result_mixer + base * Q38_D, Q38_FFN_RESIDUAL,
           rows * Q38_D * sizeof(__fp16));
}
#endif
  const uint32_t start = mempool_get_timer();
#if Q38_SCHEDULE_PROBE && !Q38_PROBE_SKIP_BURST
  probe_projection_burst();
#endif
#if !Q38_PROBE_BURST_ONLY
#if Q38_PROBE_PHASE != 2
#if Q38_KIND == 0
  attention();
#elif Q38_GDN_V2 && Q38_X8_PIPE && Q38_X8_PIPE_TEST
  x8_pipe_test();
#elif Q38_GDN_V2 && Q38_X8_PIPE
  x8_pipe();
#elif Q38_GDN_V2 && Q38_X8_BATCH
  x8_gdn_layer();
#elif Q38_GDN_V2
  gdn2();
#if Q38_GDN2_DEBUG_TOKENS || Q38_GDN2_DEBUG_OUT_ROWS || Q38_GDN2_DEBUG_NO_FFN
  q38_edge_barrier();
  return 0;
#endif
#else
  gdn();
#endif
#endif
#if Q38_SCHEDULE_PROBE
  sync_all();
  if (core == 0u)
    probe_mixer_end = mempool_get_timer();
#endif
#if Q38_F2_C1
  f2_c1();
#elif Q38_PROBE_PHASE != 1 && !Q38_X8_BATCH
#if Q38_AV_STOP_PHASE
  if (!q38_av_stop)
#endif
    ffn();
#endif
#if Q38_SCHEDULE_PROBE
  sync_all();
  if (core == 0u)
    probe_ffn_end = mempool_get_timer();
#endif
#if Q38_KIND != 0 && !Q38_GDN_V2 && (!Q38_FFN_L2 || Q38_ATTN_L2_LAYER)
  transfer(result_conv, history, sizeof(history));
  transfer(result_recurrent, recurrent, sizeof(recurrent));
#endif
#else
  if (core == 0u) {
    probe_mixer_end = probe_burst_end;
    probe_ffn_end = probe_burst_end;
  }
#endif
  const uint32_t end = mempool_get_timer();
  const uint32_t cycles = end - start;
#if Q38_SCHEDULE_PROBE ||                                                      \
    (Q38_X8_PIPE && (!Q38_X8_PIPE_RELAY || DOMAINS > 1 || Q38_X8_PIPE_SNAP ||  \
                     Q38_X8_STOP_PHASE || Q38_X8_PIPE_TEST)) ||                \
    Q38_AV_STOP_PHASE
  // Numerical tensors are deliberately incomplete in this scheduling run (or
  // in a layer pipeline other than the one-domain relay chain).
#else
#if Q38_SAMPLED_CHECKS
  // Descriptor order must match the fixture's tensor order: the harness maps
  // descriptor index to tensor name by position.
  gather_runs_f16(result_output, run_start_output, result_sample_output,
                  Q38_RUNS_OUTPUT, Q38_RUNLEN_OUTPUT);
  q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, result_sample_output, gold_sample_output,
          Q38_RUNS_OUTPUT * Q38_RUNLEN_OUTPUT, 0x3b83126fu /* 0.004f */,
          Q38_VERBOSE_CHECKS);
  __sync_synchronize();
  // Context was gathered window by window out of L1; nothing to do here.
#if Q38_KIND == 0
  q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, result_sample_context,
          gold_sample_context, Q38_RUNS_CONTEXT * Q38_RUNLEN_CONTEXT,
          0x3b449ba6u /* 0.003f */, Q38_VERBOSE_CHECKS);
  __sync_synchronize();
  gather_runs_f16(result_cache, run_start_cache, result_sample_cache,
                  Q38_RUNS_CACHE, Q38_RUNLEN_CACHE);
  q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, result_sample_cache, gold_sample_cache,
          Q38_RUNS_CACHE * Q38_RUNLEN_CACHE, 0x3b449ba6u /* 0.003f */,
          Q38_VERBOSE_CHECKS);
  __sync_synchronize();
  gather_runs_f16(result_mixer, run_start_mixer, result_sample_mixer,
                  Q38_RUNS_MIXER, Q38_RUNLEN_MIXER);
  q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, result_sample_mixer, gold_sample_mixer,
          Q38_RUNS_MIXER * Q38_RUNLEN_MIXER, 0x3b83126fu /* 0.004f */,
          Q38_VERBOSE_CHECKS);
#else
  q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, result_sample_context,
          gold_sample_context, Q38_RUNS_CONTEXT * Q38_RUNLEN_CONTEXT,
          0x3a03126fu /* 0.0005f */, Q38_VERBOSE_CHECKS);
  __sync_synchronize();
  gather_runs_f16(result_conv, run_start_conv, result_sample_conv,
                  Q38_RUNS_CONV, Q38_RUNLEN_CONV);
  q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, result_sample_conv, gold_sample_conv,
          Q38_RUNS_CONV * Q38_RUNLEN_CONV, 0x3b449ba6u /* 0.003f */,
          Q38_VERBOSE_CHECKS);
  __sync_synchronize();
  gather_runs_f32(result_recurrent, run_start_recurrent,
                  result_sample_recurrent, Q38_RUNS_RECURRENT,
                  Q38_RUNLEN_RECURRENT);
  q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F32, result_sample_recurrent,
          gold_sample_recurrent, Q38_RUNS_RECURRENT * Q38_RUNLEN_RECURRENT,
          0x3a03126fu /* 0.0005f */, Q38_VERBOSE_CHECKS);
#endif
#else
#if Q38_PROBE_PHASE != 1
  q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, result_output, gold_output, Q38_R * Q38_D,
          0x3b83126fu /* 0.004f */, Q38_VERBOSE_CHECKS);
  __sync_synchronize();
#endif
#if Q38_PROBE_PHASE != 2
#if Q38_KIND == 0
  q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, result_context, gold_context,
          Q38_R * Q38_QDIM, 0x3b449ba6u /* 0.003f */, Q38_VERBOSE_CHECKS);
  __sync_synchronize();
  q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, result_cache, gold_cache,
          2 * Q38_R * Q38_KVDIM, 0x3b449ba6u /* 0.003f */, Q38_VERBOSE_CHECKS);
  __sync_synchronize();
  q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, result_mixer, gold_mixer, Q38_R * Q38_D,
          0x3b83126fu /* 0.004f */, Q38_VERBOSE_CHECKS);
#else
  q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, result_context, gold_context,
          Q38_R * Q38_GDIM, 0x3a03126fu /* 0.0005f */, Q38_VERBOSE_CHECKS);
  __sync_synchronize();
  q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F16, result_conv, gold_conv, 4 * Q38_QKVDIM,
          0x3b449ba6u /* 0.003f */, Q38_VERBOSE_CHECKS);
  __sync_synchronize();
  q38_dpi(MEMPOOL_DPI_CHECK_TYPE_F32, result_recurrent, gold_recurrent,
          Q38_NV * Q38_KD * Q38_VD, 0x3a03126fu /* 0.0005f */,
          Q38_VERBOSE_CHECKS);
#endif
#endif
#endif
#endif
  __sync_synchronize();
  q38_edge_barrier();
  if (core == 0) {
#if Q38_SCHEDULE_PROBE
    probe_report();
#endif
#if Q38_GDN_V2 && Q38_X8_PIPE
    if (!Q38_X8_PIPE_TEST)
      x8_pipe_report();
#elif Q38_GDN_V2 && Q38_X8_BATCH
    x8_report();
#endif
#if Q38_FFN_L2
    printf("[qwen38_ffn_l2] gu_batch=%u down_batch=%u gu_tile=%u "
           "down_tile=%u columns=%u l1_peak=%u g_l2=%u phased=1 "
           "down_slices=%u down_k=%u g_rows=%u arena_base=%u\n",
           Q38_FFN_ROWS, Q38_FFN_DN_ROWS, Q38_FFN_GU_ROWS, Q38_FFN_DN_TILE,
           Q38_FFN_COLUMNS, f2.peak, (uint32_t)F2_G_L2, Q38_FFN_DOWN_SLICES,
           Q38_KC, Q38_R, f2.arena_base);
#endif
#if Q38_ATTN_VC && Q38_AV_PROFILE && Q38_KIND == 0
    av_timing_report();
#if Q38_ATTN_VC_SWEEP
    avs_report();
#endif
#endif
    printf("[qwen38_layer] kind=%u tokens=%u rows=%u cores=%u schedule=%u "
           "start=%u end=%u cycles=%u\n",
           Q38_KIND, Q38_S, Q38_R, NUM_CORES, Q38_ASYNC, start, end, cycles);
#if Q38_AV_DOMAIN_REPORT
    av_domain_report(start, end);
#endif
    // Diagnostic context export is inside the measured interval because the
    // window buffer is reused; subtract it for a layer-only comparison.
    printf("[qwen38_bounds] mixer_rows=%u ffn_rows=%u weight_panels=%u "
           "context_export_cycles=%u\n",
           Q38_WIN, Q38_FFN_ROWS, Q38_WEIGHT_PANELS, export_cycles[0]);
#if Q38_KIND == 0 && !Q38_FFN_L2
    const uint32_t copies =
        Q38_KV_REPLICATED ? NUM_GROUPS / (1u << (2 * Q38_KV_LEVEL)) : 1u;
    // Causal windowing bounds each window's key sweep, so the block loads are
    // counted rather than derived from the sequence length.
    printf("[qwen38_kv] replicas=%u level=%u requested_read_bytes=%u "
           "load_join_cycles=%u\n",
           copies, Q38_KV_LEVEL,
           kv_block_loads[0] * (uint32_t)sizeof(kv_tile) * copies,
           kv_wait_cycles[0]);
#endif
  }
  return 0;
}
