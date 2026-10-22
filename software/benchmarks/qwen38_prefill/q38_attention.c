// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// The attention mixer (Q38_KIND == 0): K/V cache windows, query sweeps and
// the output projection. Included by layer_resident.c; not a translation unit.

#if Q38_GATEUP_PIPELINE && Q38_ASYNC
#define kv_tile stage.attention.kv_tile
#define scores stage.attention.scores
#define probs stage.attention.probs
#define corrections stage.attention.corrections
#define query_stage stage.attention.query
#else
static __fp16 kv_tile[2 * Q38_KVH * Q38_HD * BK] LOCAL;
#define query_stage stage.query
#endif
static uint32_t kv_wait_cycles[NUM_CORES] LOCAL;
static uint32_t kv_block_loads[NUM_CORES] LOCAL;
#if Q38_SCHEDULE_PROBE
// Per-core cycles in each key-block phase: QK jobs, QK join, softmax, softmax
// join, PV jobs, PV join, accumulate, block end (KV wait and join), then the
// per-window query prep (its query norms and rotations counted apart) and
// finish.
enum {
  KP_QK,
  KP_QK_JOIN,
  KP_SOFTMAX,
  KP_SOFTMAX_JOIN,
  KP_PV,
  KP_PV_JOIN,
  KP_ACCUMULATE,
  KP_BLOCK_END,
  KP_PREP,
  KP_FINISH,
  KP_PREP_NORM,
  KP_PREP_ROTARY,
  KP_COUNT
};
static uint32_t kp_cycles[KP_COUNT][NUM_CORES] LOCAL;
static uint32_t kp_last[NUM_CORES] LOCAL;
static void kp_mark(uint32_t phase) {
  const uint32_t core = mempool_get_core_id(), now = mempool_get_timer();
  if (phase < KP_COUNT)
    kp_cycles[phase][core] += now - kp_last[core];
  kp_last[core] = now;
}
#define KP(phase) kp_mark(phase)
#else
#define KP(phase) ((void)0)
#endif
#if Q38_ATTN_LOCAL_PIPE
// The pipelined sweep's four K/V tiles live in the first projection weight
// slot, which is idle between the Q/gate and output projections.
#define Q38_KV_ELEMS (2u * Q38_KVH * Q38_HD * BK)
_Static_assert(Q38_ATTN_VC_SWEEP ||
                   4u * Q38_KV_ELEMS <= sizeof(weights[0]) / sizeof(__fp16),
               "The K/V ring must fit one projection weight slot");
#define kv_tile_next (weights[0] + Q38_KV_ELEMS)
#elif Q38_KV_PREFETCH
static __fp16 kv_tile_next[2 * Q38_KVH * Q38_HD * BK] LOCAL;
#endif
#define Q38_HALF (Q38_ROT / 2)
static __fp16 cosines[Q38_ROWS * Q38_HALF] LOCAL;
static __fp16 sines[Q38_ROWS * Q38_HALF] LOCAL;

// Stage this window's rotary table; `rotary` then indexes the window locally.
static void stage_rotary(uint32_t base, uint32_t rows) {
  transfer(cosines, rope_cos + base * Q38_HALF, rows * Q38_HALF * 2);
  transfer(sines, rope_sin + base * Q38_HALF, rows * Q38_HALF * 2);
}
#if !(Q38_GATEUP_PIPELINE && Q38_ASYNC)
static __fp16 scores[NUM_REDMULE_TILES * MT * BK] LOCAL;
static __fp16 probs[NUM_REDMULE_TILES * MT * BK] LOCAL;
static float corrections[NUM_REDMULE_TILES * MT] LOCAL;
#endif

static void rotary(__fp16 *values, uint32_t token) {
  const uint32_t half = Q38_HALF;
  const float zh = llm_f16box(0.0f);
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(half));
  asm volatile("vle16.v v0, (%0)" : : "r"(values) : "memory");
  asm volatile("vle16.v v1, (%0)" : : "r"(values + half) : "memory");
  asm volatile("vfwadd.vf v4, v0, %0" : : "f"(zh));
  asm volatile("vfwadd.vf v6, v1, %0" : : "f"(zh));
  asm volatile("vle16.v v0, (%0)" : : "r"(cosines + token * half) : "memory");
  asm volatile("vle16.v v1, (%0)" : : "r"(sines + token * half) : "memory");
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

#if Q38_ATTN_FUSED_SOFTMAX
#include "q38_attn_vc_softmax.c"
// A block row fits one register group: BK <= VLEN/32 * 8 elements.
static float softmax_row(const __fp16 *score, __fp16 *prob, uint32_t visible,
                         qwen38_softmax_state_t *state) {
#if Q38_ATTN_VC_SWEEP && Q38_AV_SWEEP_REGISTER && Q38_BK > 128
  return avs_register_softmax_row(score, prob, visible, state);
#elif Q38_ATTN_VC_SWEEP && Q38_AV_SWEEP_LEAN && Q38_BK > 128
  return avs_lean_softmax_row(score, prob, visible, state);
#elif Q38_BK > 128
  // Reduce a block-wide maximum, then emit probabilities in register chunks.
  // FP32 m8 holds 128 elements at VLEN512; one online update covers all chunks.
  const float zh = llm_f16box(0.0f), zero = 0.0f;
  if (!visible) {
    llm_zero_f16(prob, BK);
    return 1.0f;
  }
  float maximum = state->maximum;
#pragma clang loop unroll(full)
  for (uint32_t off = 0; off < BK; off += 128u) {
    if (off >= visible)
      continue;
    const uint32_t count = Q38_MIN(128u, visible - off);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(128u));
    asm volatile("vle16.v v8, (%0)" : : "r"(score + off) : "memory");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
    asm volatile("vfwadd.vf v0, v8, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count));
    asm volatile("vfmul.vf v0, v0, %0" : : "f"(llm_rsqrt((float)Q38_HD)));
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
    asm volatile("vfncvt.f.f.w v8, v0");
    asm volatile("vfwadd.vf v0, v8, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count));
#if Q38_ATTN_VC_SWEEP && Q38_AV_SWEEP_TREE_MAX
    maximum = avs_tree_max(count, maximum);
#else
    asm volatile("vfmv.s.f v24, %0" : : "f"(maximum));
    asm volatile("vfredmax.vs v24, v0, v24");
    asm volatile("vfmv.f.s %0, v24" : "=f"(maximum));
#endif
  }
  const float correction =
      state->denominator == 0.0f ? 0.0f : llm_expf(state->maximum - maximum);
  float denominator = 0.0f;
#pragma clang loop unroll(full)
  for (uint32_t off = 0; off < BK; off += 128u) {
    if (off >= visible) {
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(128u));
      asm volatile("vmv.v.i v16, 0");
      asm volatile("vse16.v v16, (%0)" : : "r"(prob + off) : "memory");
      continue;
    }
    const uint32_t count = Q38_MIN(128u, visible - off);
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(128u));
    asm volatile("vle16.v v8, (%0)" : : "r"(score + off) : "memory");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
    asm volatile("vfwadd.vf v0, v8, %0" : : "f"(zh));
    if (count < 128u) {
      // Raw valid values are now in v0: an aligned clear also works in place.
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(128u));
      asm volatile("vmv.v.i v16, 0");
      asm volatile("vse16.v v16, (%0)" : : "r"(prob + off) : "memory");
    }
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count));
    asm volatile("vfmul.vf v0, v0, %0" : : "f"(llm_rsqrt((float)Q38_HD)));
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
    asm volatile("vfncvt.f.f.w v8, v0");
    asm volatile("vfwadd.vf v0, v8, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(count));
    asm volatile("vfsub.vf v0, v0, %0" : : "f"(maximum));
    llm_rvv_expf_m8();
    asm volatile("vfmv.s.f v24, %0" : : "f"(zero));
    asm volatile("vfredusum.vs v24, v0, v24");
    float sum;
    asm volatile("vfmv.f.s %0, v24" : "=f"(sum));
    denominator += sum;
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(count));
    asm volatile("vfncvt.f.f.w v8, v0");
    asm volatile("vse16.v v8, (%0)" : : "r"(prob + off) : "memory");
  }
  state->maximum = maximum;
  state->denominator = correction * state->denominator + denominator;
  return correction;
#else
#if Q38_ATTN_VC_SWEEP
  const float zh = llm_f16box(0.0f), zero = 0.0f;
  if (!visible) {
    llm_zero_f16(prob, BK);
    return 1.0f;
  }
  float maximum = state->maximum;
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(BK));
  asm volatile("vle16.v v8, (%0)" : : "r"(score) : "memory");
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(visible));
  asm volatile("vfwadd.vf v0, v8, %0" : : "f"(zh));
  if (visible < BK) {
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(BK));
    asm volatile("vmv.v.i v16, 0");
    asm volatile("vse16.v v16, (%0)" : : "r"(prob) : "memory");
  }
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(visible));
#else
  const float zh = llm_f16box(0.0f), zero = 0.0f;
  // Zero the whole row from its aligned start: zeroing only the tail from
  // prob + visible (2 B aligned) left stale values on half the rows in GVSoC.
  if (visible < BK) {
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(BK));
    asm volatile("vmv.v.i v8, 0");
    asm volatile("vse16.v v8, (%0)" : : "r"(prob) : "memory");
  }
  if (visible == 0u)
    return 1.0f;
  float maximum = state->maximum;
  // Load the whole (padded) row: a Spatz burst load must be whole words, and
  // GVSoC hangs on an odd FP16 count. Only the visible prefix is used.
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(BK));
  asm volatile("vle16.v v8, (%0)" : : "r"(score) : "memory");
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(visible));
  asm volatile("vfwadd.vf v0, v8, %0" : : "f"(zh));
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(visible));
#endif
  asm volatile("vfmul.vf v0, v0, %0" : : "f"(llm_rsqrt((float)Q38_HD)));
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(visible));
  asm volatile("vfncvt.f.f.w v8, v0");
  asm volatile("vfwadd.vf v0, v8, %0" : : "f"(zh));
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(visible));
  asm volatile("vfmv.s.f v24, %0" : : "f"(maximum));
  asm volatile("vfredmax.vs v24, v0, v24");
  asm volatile("vfmv.f.s %0, v24" : "=f"(maximum));

  const float correction =
      state->denominator == 0.0f ? 0.0f : llm_expf(state->maximum - maximum);
  asm volatile("vfmv.s.f v24, %0" : : "f"(zero));
  asm volatile("vfsub.vf v0, v0, %0" : : "f"(maximum));
  llm_rvv_expf_m8();
  asm volatile("vfredusum.vs v24, v0, v24");
  asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(visible));
  asm volatile("vfncvt.f.f.w v8, v0");
  asm volatile("vse16.v v8, (%0)" : : "r"(prob) : "memory");
  float denominator;
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(visible));
  asm volatile("vfmv.f.s %0, v24" : "=f"(denominator));
  state->maximum = maximum;
  state->denominator = correction * state->denominator + denominator;
  return correction;
#endif
}

#endif

// Read a row of n FP16 values back and wait for the result.
static void q38_drain16(const __fp16 *p, uint32_t n) {
  float sink;
  for (uint32_t off = 0u; off < n;) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(n - off));
    asm volatile("vle16.v v0, (%0)" : : "r"(p + off) : "memory");
    asm volatile("vfmv.f.s %0, v0" : "=f"(sink));
    off += vl;
  }
  (void)sink;
}

static void accumulate(float *out, const __fp16 *value, float scale) {
  const float zh = llm_f16box(0.0f);
  for (uint32_t offset = 0; offset < Q38_HD;) {
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma"
                 : "=r"(vl)
                 : "r"(Q38_HD - offset));
    asm volatile("vle16.v v0, (%0)" : : "r"(value + offset) : "memory");
    asm volatile("vfwadd.vf v16, v0, %0" : : "f"(zh));
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    asm volatile("vle32.v v8, (%0)" : : "r"(out + offset) : "memory");
    asm volatile("vfmul.vf v8, v8, %0" : : "f"(scale));
    asm volatile("vfadd.vv v8, v8, v16");
    asm volatile("vse32.v v8, (%0)" : : "r"(out + offset) : "memory");
    offset += vl;
  }
}

// One token window of keys and values, normalized, rotated and persisted to
// the blocked L2 cache. Padded rows project to zero and stay zero.
static NOSTACK void build_kv_window(uint32_t base, uint32_t rows,
                                    uint32_t last) {
  const uint32_t core = mempool_get_core_id();
  __fp16 *const projected = stage.kv.projected;
  __fp16 *const heads = stage.kv.heads;
  transfer(residual, input_init + base * Q38_D, rows * Q38_D * 2);
  norm_rows(residual, w_input_norm, a, rows, Q38_D, 0, gamma);
  stage_rotary(base, rows);
  project_rows(a, w_key, projected, Q38_D, Q38_KVDIM, w_value, Q38_D, Q38_KVDIM,
               rows, &mixer_buffers);
  for (uint32_t i = core; i < rows * Q38_KVH; i += NUM_CORES) {
    const uint32_t t = i / Q38_KVH, h = i % Q38_KVH;
    q38_copy16(heads + (h * rows + t) * Q38_HD, projected + i * Q38_HD, Q38_HD);
  }
  sync_all();
  project_rows(a, w_value, projected, Q38_D, Q38_KVDIM,
               last ? w_q_gate : (const __fp16 *)0, Q38_D, 2 * Q38_QDIM, rows,
               &mixer_buffers);
  for (uint32_t i = core; i < rows * Q38_KVH; i += NUM_CORES) {
    const uint32_t t = i / Q38_KVH, h = i % Q38_KVH;
    q38_copy16(heads + ((Q38_KVH + h) * rows + t) * Q38_HD,
               projected + i * Q38_HD, Q38_HD);
  }
  sync_all();
  transfer(gamma, w_k_norm, Q38_HD * 2);
  for (uint32_t i = core; i < Q38_KVH * rows; i += NUM_CORES) {
    q38_norm16(heads + i * Q38_HD, gamma, heads + i * Q38_HD, Q38_HD, 0);
    rotary(heads + i * Q38_HD, i % rows);
  }
  sync_all();
  q38_pack_kv_to_l2(heads, kv_tile,
                    result_cache + base / BK * 2 * Q38_KVH * Q38_HD * BK, rows,
                    Q38_KVH, Q38_HD, BK);
}

// A query window runs as three bounded steps. They are separate functions
// because the per-core stack is 512 bytes: one frame spanning the projection
// call, the key sweep and the finalize loop does not fit under project_rows.
static NOSTACK void query_project(uint32_t base, uint32_t rows) {
  const uint32_t core = mempool_get_core_id();
  __fp16 *const qgate = query_stage.qgate;
  __fp16 *const queries = query_stage.q;
  qwen38_softmax_state_t *const states = query_stage.states;
#if !Q38_ATTN_REUSE_NORM
  transfer(residual, input_init + base * Q38_D, rows * Q38_D * 2);
  norm_rows(residual, w_input_norm, a, rows, Q38_D, 0, gamma);
  stage_rotary(base, rows);
#endif
  project_rows(a, w_q_gate, qgate, Q38_D, 2 * Q38_QDIM, w_out, Q38_QDIM, Q38_D,
               rows, &mixer_buffers);
  transfer(gamma, w_q_norm, Q38_HD * 2);
  for (uint32_t i = core; i < Q38_QH * rows; i += NUM_CORES) {
    const uint32_t h = i / rows, t = i % rows;
    q38_norm16(qgate + t * 2 * Q38_QDIM + h * 2 * Q38_HD, gamma,
               queries + i * Q38_HD, Q38_HD, 0);
    rotary(queries + i * Q38_HD, t);
    qwen38_softmax_init(states + i);
  }
  sync_all();
  clear_all(query_stage.accumulated,
            Q38_QH * rows * Q38_HD * (uint32_t)sizeof(float));
}

#if Q38_BACKGROUND_FFN_NORM
static NOSTACK void ffn_norm_dma_begin(uint32_t epoch, void *dest,
                                       const void *src, uint32_t bytes,
                                       uint32_t wait_ready) {
  if (mempool_get_core_id() == 0u &&
      q38_dma_stage_publish(&ffn_norm_dma_queue, epoch, dest, src, bytes, 0u) !=
          Q38_DMA_QUEUE_OK)
    __builtin_trap();
  if (mempool_is_group_coordinator()) {
    q38_dma_group_progress_t *const state =
        &ffn_norm_dma_queue.slot[epoch % Q38_DMA_QUEUE_SLOTS]
             .group[mempool_get_group_id()];
    while (__atomic_load_n(&state->issued_epoch, __ATOMIC_ACQUIRE) != epoch)
      (void)q38_dma_stage_progress(&ffn_norm_dma_queue);
  }
  if (wait_ready) {
    uint32_t status;
    do {
      if (mempool_is_group_coordinator())
        (void)q38_dma_stage_progress(&ffn_norm_dma_queue);
      else
        mempool_wait(4u);
      status = q38_dma_stage_ready(&ffn_norm_dma_queue, epoch);
    } while (status == Q38_DMA_QUEUE_PENDING);
    if (status != Q38_DMA_QUEUE_OK)
      __builtin_trap();
  }
}

static NOSTACK void ffn_norm_background(uint32_t rows, uint32_t epoch) {
  if (tnc_is_tile_agent())
    return;
  uint32_t status;
  do {
    mempool_wait(4u);
    status = q38_dma_stage_ready(&ffn_norm_dma_queue, epoch);
  } while (status == Q38_DMA_QUEUE_PENDING);
  if (status != Q38_DMA_QUEUE_OK)
    __builtin_trap();
  const uint32_t worker = mempool_get_core_id() / NUM_CORES_PER_TILE;
  const uint32_t workers = NUM_CORES / NUM_CORES_PER_TILE;
  for (uint32_t row = worker; row < rows; row += workers)
    q38_norm16(Q38_FFN_RESIDUAL + row * Q38_D, Q38_FFN_GAMMA,
               Q38_FFN_A + row * Q38_D, Q38_D, 0);
}

static NOSTACK void ffn_norm_dma_finish(uint32_t epoch) {
  uint32_t status;
  do {
    if (mempool_is_group_coordinator())
      (void)q38_dma_stage_progress(&ffn_norm_dma_queue);
    else
      mempool_wait(4u);
    status = q38_dma_stage_ready(&ffn_norm_dma_queue, epoch + 1u);
  } while (status == Q38_DMA_QUEUE_PENDING);
  if (status != Q38_DMA_QUEUE_OK)
    __builtin_trap();
  if (mempool_get_core_id() == 0u &&
      (q38_dma_stage_release(&ffn_norm_dma_queue, epoch) != Q38_DMA_QUEUE_OK ||
       q38_dma_stage_release(&ffn_norm_dma_queue, epoch + 1u) !=
           Q38_DMA_QUEUE_OK))
    __builtin_trap();
}
#endif

static NOSTACK void query_key_jobs(uint32_t base, uint32_t rows, uint32_t key,
                                   const __fp16 *kt) {
  if (!tnc_is_tile_agent())
    return;
  __fp16 *const queries = query_stage.q;
  float *const accumulated = query_stage.accumulated;
  qwen38_softmax_state_t *const states = query_stage.states;
  const uint32_t engine = mempool_get_redmule_id();
  __fp16 *score = scores + engine * MT * BK;
  __fp16 *p = probs + engine * MT * BK;
#if Q38_GATEUP_PIPELINE && Q38_ASYNC
  __fp16 *z = stage.attention.z + engine * MT * Q38_HD;
#else
  __fp16 *z = gemm_z + engine * 2 * MT * Q38_ZW;
#endif
  float *alpha = corrections + engine * MT;
  const uint32_t blocks = rows / MT;
  for (uint32_t task = engine; task < Q38_QH * blocks;
       task += NUM_REDMULE_TILES) {
    const uint32_t h = task / blocks, row = task % blocks * MT;
    const uint32_t kh = h / (Q38_QH / Q38_KVH);
    const uint32_t first = base + row;
    if (first >= Q38_S || first + MT <= key)
      continue;
    job(queries + (h * rows + row) * Q38_HD, kt + kh * Q38_HD * BK, score, MT,
        Q38_HD, BK);
    llm_zero_f16(p, MT * BK);
    for (uint32_t r = 0; r < MT; ++r) {
      const uint32_t token = first + r;
      uint32_t visible = token < Q38_S && token >= key ? token + 1 - key : 0;
      if (visible > BK)
        visible = BK;
      llm_scale_f16_f32(score + r * BK, BK, llm_rsqrt((float)Q38_HD));
      for (uint32_t j = visible; j < BK; ++j)
        q38_write16(score + r * BK + j, -65504.0f);
      alpha[r] = qwen38_softmax_update_f16(score + r * BK, p + r * BK, visible,
                                           states + h * rows + row + r);
    }
    __sync_synchronize();
    job(p, kt + (Q38_KVH + kh) * BK * Q38_HD, z, MT, BK, Q38_HD);
    for (uint32_t r = 0; r < MT; ++r)
      accumulate(accumulated + (h * rows + row + r) * Q38_HD, z + r * Q38_HD,
                 alpha[r]);
  }
}

#if Q38_ATTN_SPREAD
_Static_assert(Q38_QH *(Q38_WIN / MT) <= Q38_ATTN_ENGINES,
               "Spread attention keeps one score/probability tile per owner");
static uint32_t query_task_live(uint32_t base, uint32_t rows, uint32_t key,
                                uint32_t task) {
  const uint32_t first = base + task % (rows / MT) * MT;
  return first < Q38_S && first + MT > key;
}

static NOSTACK void query_key_spread(uint32_t base, uint32_t rows, uint32_t key,
                                     const __fp16 *kt) {
  const uint32_t core = mempool_get_core_id();
  const uint32_t blocks = rows / MT, tasks = Q38_QH * blocks;
  __fp16 *const queries = query_stage.q;
  if (tnc_is_tile_agent())
    for (uint32_t task = mempool_get_redmule_id(); task < tasks;
         task += NUM_REDMULE_TILES) {
      const uint32_t h = task / blocks, row = task % blocks * MT;
      if (query_task_live(base, rows, key, task))
        job(queries + (h * rows + row) * Q38_HD,
            kt + h / (Q38_QH / Q38_KVH) * Q38_HD * BK, scores + task * MT * BK,
            MT, Q38_HD, BK);
    }
  KP(KP_QK);
  sync_all();
  KP(KP_QK_JOIN);
  for (uint32_t i = core; i < tasks * MT; i += NUM_CORES) {
    const uint32_t task = i / MT, r = i % MT;
    if (!query_task_live(base, rows, key, task))
      continue;
    const uint32_t h = task / blocks, row = task % blocks * MT;
    const uint32_t token = base + row + r;
    uint32_t visible = token < Q38_S && token >= key ? token + 1 - key : 0;
    if (visible > BK)
      visible = BK;
    __fp16 *const score = scores + i * BK;
#if Q38_ATTN_FUSED_SOFTMAX
    corrections[i] = softmax_row(score, probs + i * BK, visible,
                                 query_stage.states + h * rows + row + r);
#else
    llm_zero_f16(probs + i * BK, BK);
    llm_scale_f16_f32(score, BK, llm_rsqrt((float)Q38_HD));
    for (uint32_t j = visible; j < BK; ++j)
      q38_write16(score + j, -65504.0f);
    corrections[i] =
        qwen38_softmax_update_f16(score, probs + i * BK, visible,
                                  query_stage.states + h * rows + row + r);
#endif
  }
  KP(KP_SOFTMAX);
  sync_all();
  KP(KP_SOFTMAX_JOIN);
  if (tnc_is_tile_agent())
    for (uint32_t task = mempool_get_redmule_id(); task < tasks;
         task += NUM_REDMULE_TILES)
      if (query_task_live(base, rows, key, task))
        job(probs + task * MT * BK,
            kt + (Q38_KVH + task / blocks / (Q38_QH / Q38_KVH)) * BK * Q38_HD,
            stage.attention.z + task * MT * Q38_HD, MT, BK, Q38_HD);
  KP(KP_PV);
  sync_all();
  KP(KP_PV_JOIN);
  for (uint32_t i = core; i < tasks * MT; i += NUM_CORES) {
    const uint32_t task = i / MT, r = i % MT;
    if (!query_task_live(base, rows, key, task))
      continue;
    const uint32_t h = task / blocks, row = task % blocks * MT;
    accumulate(query_stage.accumulated + (h * rows + row + r) * Q38_HD,
               stage.attention.z + i * Q38_HD, corrections[i]);
  }
  KP(KP_ACCUMULATE);
}
#endif

#if Q38_ATTN_LOCAL
#define Q38_LOC_TILES (Q38_WIN / Q38_PROJECTION_ROWS)
#define Q38_LOC_TOTAL (Q38_QH * Q38_LOC_TILES)
#define Q38_LOC_OWNERS ((Q38_LOC_TOTAL + NUM_GROUPS - 1u) / NUM_GROUPS)
#define Q38_LOC_ROWS (Q38_LOC_OWNERS * Q38_PROJECTION_ROWS)
#define Q38_LOC_CORES (NUM_CORES / NUM_GROUPS)
#define Q38_LOC_ENGINES (NUM_REDMULE_TILES / NUM_GROUPS)
_Static_assert(Q38_LOC_ENGINES > 0, "Group-local owners need engines");
// RedMulE addresses this arena through the level-0 VC window, where each
// owner tile is contiguous. Scalar and Spatz accesses use the global alias of
// the same words instead (GVSoC does not complete core stores to the window):
// each 1 KB window row is one contiguous 1 KB run globally, so every field is
// 1 KB aligned and every per-row record stays within one window row.
#define Q38_ROW_ALIGNED __attribute__((aligned(VC_ROW_BYTES)))
// RedMulE accumulation keeps the context in z; u is then unused.
#define Q38_LOC_U_ROWS (Q38_ATTN_RM_ACCUM ? 1u : Q38_LOC_ROWS)
// Tile geometry of the arena: a 1 KB VC row is one 256 B slice per tile, and
// a score or probability row of BK FP16 values is Q38_RPS of a slice.
#define Q38_SLICE_BYTES (VC_ROW_BYTES / NUM_TILES_PER_GROUP)
#define Q38_RPS (Q38_SLICE_BYTES / (BK * 2u))
#define Q38_TILE_OF(i) ((i) / Q38_RPS % NUM_TILES_PER_GROUP)
#define Q38_SLOT_OF(i)                                                         \
  ((i) / (Q38_RPS * NUM_TILES_PER_GROUP) * Q38_RPS + (i) % Q38_RPS)
#if Q38_ATTN_TILE_LOCAL
#define Q38_LOC_STATES                                                         \
  (NUM_TILES_PER_GROUP * (Q38_SLICE_BYTES / sizeof(qwen38_softmax_state_t)))
#define Q38_LOC_ALPHAS (NUM_TILES_PER_GROUP * (Q38_SLICE_BYTES / sizeof(float)))
#define Q38_ST(i)                                                              \
  (Q38_TILE_OF(i) * (Q38_SLICE_BYTES / sizeof(qwen38_softmax_state_t)) +       \
   Q38_SLOT_OF(i))
#define Q38_AL(i)                                                              \
  (Q38_TILE_OF(i) * (Q38_SLICE_BYTES / sizeof(float)) + Q38_SLOT_OF(i))
#else
#define Q38_LOC_STATES Q38_LOC_ROWS
#define Q38_LOC_ALPHAS Q38_LOC_ROWS
#define Q38_ST(i) (i)
#define Q38_AL(i) (i)
#endif
// The pipeline alternates score and probability tiles between two slots.
#define Q38_LOC_SLOTS (Q38_ATTN_LOCAL_PIPE ? 2u : 1u)
typedef struct {
  float u[Q38_LOC_U_ROWS * Q38_HD] Q38_ROW_ALIGNED;
  __fp16 q[Q38_LOC_ROWS * Q38_HD] Q38_ROW_ALIGNED;
  __fp16 z[Q38_LOC_ROWS * Q38_HD] Q38_ROW_ALIGNED;
  __fp16 score_tile[Q38_LOC_SLOTS * Q38_LOC_ROWS * BK] Q38_ROW_ALIGNED;
  __fp16 prob_tile[Q38_LOC_SLOTS * Q38_LOC_ROWS * BK] Q38_ROW_ALIGNED;
  float alpha[Q38_LOC_ALPHAS] Q38_ROW_ALIGNED;
  qwen38_softmax_state_t states[Q38_LOC_STATES] Q38_ROW_ALIGNED;
  mempool_poll_slot_t slot Q38_ROW_ALIGNED;
  __fp16 zg[(Q38_ATTN_RM_FOLD ? Q38_LOC_ROWS : 1u) * Q38_HD] Q38_ROW_ALIGNED;
  __fp16 eye[Q38_PROJECTION_ROWS * Q38_PROJECTION_ROWS] Q38_ROW_ALIGNED;
#if Q38_ATTN_VC
  // Phased attention: the owners' gates from L2, their gated context rows.
  __fp16 gate[Q38_LOC_ROWS * Q38_HD] Q38_ROW_ALIGNED;
  __fp16 gated[Q38_LOC_ROWS * Q38_HD] Q38_ROW_ALIGNED;
#endif
} q38_local_sweep_t;
// Widest per-row record the sweep touches: the fp32 accumulator u, or with
// RedMulE accumulation (u unused) the fp16 q / z rows.
#define Q38_LOC_ROW_BYTES                                                      \
  (Q38_HD * (Q38_ATTN_RM_ACCUM ? sizeof(__fp16) : sizeof(float)))
_Static_assert(Q38_LOC_ROW_BYTES <= VC_ROW_BYTES &&
                   VC_ROW_BYTES % Q38_LOC_ROW_BYTES == 0 &&
                   VC_ROW_BYTES % (BK * sizeof(__fp16)) == 0 &&
                   VC_ROW_BYTES % sizeof(qwen38_softmax_state_t) == 0,
               "Per-row sweep records must stay inside one VC row");
// alpha / states are touched one element at a time through Q38_LA, so they may
// span VC rows; the tile-local slice map needs each to fit a single row.
_Static_assert(!Q38_ATTN_TILE_LOCAL ||
                   (Q38_LOC_ALPHAS * sizeof(float) <= VC_ROW_BYTES &&
                    Q38_LOC_STATES * sizeof(qwen38_softmax_state_t) <=
                        VC_ROW_BYTES),
               "Tile-local alpha / state slices must fit one VC row");
_Static_assert(!Q38_ATTN_TILE_LOCAL ||
                   (Q38_SLICE_BYTES % (BK * 2u) == 0u &&
                    Q38_LOC_ROWS % (Q38_RPS * NUM_TILES_PER_GROUP) == 0u &&
                    Q38_LOC_ROWS / NUM_TILES_PER_GROUP <=
                        Q38_SLICE_BYTES / sizeof(qwen38_softmax_state_t)),
               "Tile-local rows need whole slice periods and slice room");
// Same VC-local address in every group, each resolving to its own banks.
static q38_local_sweep_t *q38_loc LOCAL;
// The query-norm weight, loaded once per layer rather than once per window.
static __fp16 q38_q_gamma[Q38_HD] LOCAL;

// Global alias of a level-0 window address for this core's group: the one
// helper (q38_vc.c), which adds VC_GLOBAL_BASE for domain builds.
#define q38_la(window) vc_group_global(window)
#define Q38_LA(ptr) Q38_VC_GLOBAL(ptr)

static NOSTACK void q38_local_init(void) {
  vc_init();
  // GVSoC lets vc_reserve's load of this core's (remote) row counters overtake
  // vc_init's still-pending stores to them and read uninitialized L1.
  __sync_synchronize();
  const uint32_t bytes = (sizeof(q38_local_sweep_t) + vc_row_stride(0) - 1u) /
                         vc_row_stride(0) * vc_row_stride(0);
  // The arena takes whole bank rows above the static L1 image.
  if (vc_reserve(0, bytes) != 0)
    q38_fatal("[qwen38_fatal] vc arena rows=", bytes / vc_row_stride(0));
  q38_local_sweep_t *const loc = (q38_local_sweep_t *)vc_malloc(0, bytes);
  if (loc == 0)
    q38_fatal("[qwen38_fatal] vc_malloc bytes=", bytes);
  const vc_rect_t rect = vc_rect(0);
  // vc_global_addr is the base-less DMA form; core pointers add the base.
  if ((uint32_t)q38_la(&loc->slot) !=
      VC_GLOBAL_BASE +
          vc_global_addr(&rect, (uint32_t)&loc->slot - VC_WINDOW_BASE_0))
    q38_fatal("[qwen38_fatal] vc alias slot=", (uint32_t)&loc->slot);
  if (mempool_get_core_id() == 0u)
    q38_loc = loc;
  if (mempool_is_group_coordinator()) {
    mempool_barrier_poll_init(Q38_LA(&loc->slot));
    uint16_t *const eye = (uint16_t *)Q38_LA(loc->eye + 0);
    for (uint32_t r = 0; r < Q38_PROJECTION_ROWS; ++r)
      for (uint32_t c = 0; c < Q38_PROJECTION_ROWS; ++c)
        eye[r * Q38_PROJECTION_ROWS + c] = r == c ? 0x3c00u : 0u;
  }
  sync_all();
}

static void q38_group_sync(void) {
  __sync_synchronize();
  mempool_barrier_poll_slot(Q38_LA(&q38_loc->slot), Q38_LOC_CORES);
}

// Owner j of this group: head and 16-row tile of the window, dealt round
// robin over the groups (it may not exist when owners do not split evenly).
static uint32_t q38_loc_owner(uint32_t j) {
  return j * NUM_GROUPS + mempool_get_group_id();
}

static NOSTACK void local_query_prep(uint32_t offset) {
  q38_local_sweep_t *const loc = q38_loc;
  const __fp16 *const qgate = query_stage.qgate + offset * 2 * Q38_QDIM;
  const uint32_t lc = mempool_get_core_id() % Q38_LOC_CORES;
  for (uint32_t i = lc; i < Q38_LOC_ROWS; i += Q38_LOC_CORES) {
    const uint32_t t = q38_loc_owner(i / Q38_PROJECTION_ROWS);
    if (t >= Q38_LOC_TOTAL)
      continue;
    const uint32_t h = t / Q38_LOC_TILES;
    const uint32_t row =
        t % Q38_LOC_TILES * Q38_PROJECTION_ROWS + i % Q38_PROJECTION_ROWS;
    __fp16 *const q = Q38_LA(loc->q + i * Q38_HD);
    q38_norm16(qgate + row * 2 * Q38_QDIM + h * 2 * Q38_HD, q38_q_gamma, q,
               Q38_HD, 0);
    KP(KP_PREP_NORM);
    rotary(q, offset + row);
    KP(KP_PREP_ROTARY);
    qwen38_softmax_init(Q38_LA(loc->states + Q38_ST(i)));
#if !Q38_ATTN_RM_ACCUM
    float *const u = Q38_LA(loc->u + i * Q38_HD);
    for (uint32_t d = 0; d < Q38_HD;) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                   : "=r"(vl)
                   : "r"(Q38_HD - d));
      asm volatile("vmv.v.i v0, 0");
      asm volatile("vse32.v v0, (%0)" : : "r"(u + d) : "memory");
      d += vl;
    }
#endif
  }
  q38_group_sync();
}

static uint32_t q38_loc_live(uint32_t base, uint32_t key, uint32_t j) {
  if (q38_loc_owner(j) >= Q38_LOC_TOTAL)
    return 0u;
  const uint32_t first =
      base + q38_loc_owner(j) % Q38_LOC_TILES * Q38_PROJECTION_ROWS;
  return first < Q38_S && first + Q38_PROJECTION_ROWS > key;
}

#define Q38_LOC_SLOT (Q38_LOC_ROWS * BK)
#define Q38_LOC_AGENT                                                          \
  (tnc_is_tile_agent() && mempool_get_redmule_id() < NUM_REDMULE_TILES)

// QK or PV jobs of this engine's live owners for one key block. An engine's
// jobs queue back to back and its completion counter is monotonic, so the
// caller waits once for the last target. Returns whether any job started.
// pv: 0 = QK, 1 = PV, 3 = fold Zg into Z.
static uint32_t local_jobs(uint32_t base, uint32_t key, const __fp16 *kt,
                           uint32_t slot, uint32_t pv, uint8_t *last) {
  q38_local_sweep_t *const loc = q38_loc;
  uint32_t issued = 0u;
  if (Q38_ATTN_ISOLATE & 1u)
    return 0u;
  for (uint32_t j = mempool_get_redmule_id() % Q38_LOC_ENGINES;
       j < Q38_LOC_OWNERS; j += Q38_LOC_ENGINES) {
    if (!q38_loc_live(base, key, j))
      continue;
    const uint32_t kv = q38_loc_owner(j) / Q38_LOC_TILES / (Q38_QH / Q38_KVH);
    if (pv == 3u)
      *last = start_job_y(loc->eye, loc->zg + j * Q38_PROJECTION_ROWS * Q38_HD,
                          loc->z + j * Q38_PROJECTION_ROWS * Q38_HD,
                          Q38_PROJECTION_ROWS, Q38_PROJECTION_ROWS, Q38_HD,
                          key == 0u);
    else if (pv && Q38_ATTN_RM_FOLD)
      *last = start_job(loc->prob_tile + slot * Q38_LOC_SLOT +
                            j * Q38_PROJECTION_ROWS * BK,
                        kt + (Q38_KVH + kv) * BK * Q38_HD,
                        loc->zg + j * Q38_PROJECTION_ROWS * Q38_HD,
                        Q38_PROJECTION_ROWS, BK, Q38_HD);
    else if (pv)
      // RedMulE accumulation adds into Z after key block 0.
      *last = start_job_y(
          loc->prob_tile + slot * Q38_LOC_SLOT + j * Q38_PROJECTION_ROWS * BK,
          kt + (Q38_KVH + kv) * BK * Q38_HD,
          loc->z + j * Q38_PROJECTION_ROWS * Q38_HD, Q38_PROJECTION_ROWS, BK,
          Q38_HD, !Q38_ATTN_RM_ACCUM || key == 0u);
    else
      *last = start_job(
          loc->q + j * Q38_PROJECTION_ROWS * Q38_HD, kt + kv * Q38_HD * BK,
          loc->score_tile + slot * Q38_LOC_SLOT + j * Q38_PROJECTION_ROWS * BK,
          Q38_PROJECTION_ROWS, Q38_HD, BK);
    issued = 1u;
  }
  return issued;
}

static void local_jobs_wait(uint32_t base, uint32_t key, const __fp16 *kt,
                            uint32_t slot, uint32_t pv) {
  uint8_t last = 0u;
  if (Q38_LOC_AGENT && local_jobs(base, key, kt, slot, pv, &last))
    tnc_redmule_wait(last);
}

// Softmax rows of one key block. Without the pipeline, RedMulE accumulation
// rescales moved rows here, since the previous PV has completed; otherwise
// the correction is kept for local_rescale or accumulate.
static void local_softmax_row(uint32_t base, uint32_t key, uint32_t slot,
                              uint32_t i);

// The k-th softmax row of this core. Tile-local: rows of the core's own tile,
// dealt over its tile's cores; with two cores per tile this is lc + 8k, the
// plain order. Returns Q38_LOC_ROWS past the end.
static uint32_t local_row(uint32_t k) {
  const uint32_t lc = mempool_get_core_id() % Q38_LOC_CORES;
#if Q38_ATTN_TILE_LOCAL
  const uint32_t t = lc / NUM_CORES_PER_TILE;
  const uint32_t n = k * NUM_CORES_PER_TILE + lc % NUM_CORES_PER_TILE;
  const uint32_t i =
      n / Q38_RPS * (Q38_RPS * NUM_TILES_PER_GROUP) + t * Q38_RPS + n % Q38_RPS;
#else
  const uint32_t i = lc + k * Q38_LOC_CORES;
#endif
  return i < Q38_LOC_ROWS ? i : Q38_LOC_ROWS;
}

static NOSTACK void local_softmax(uint32_t base, uint32_t key, uint32_t slot) {
  for (uint32_t k = 0u, i; (i = local_row(k)) < Q38_LOC_ROWS; ++k)
    if (q38_loc_live(base, key, i / Q38_PROJECTION_ROWS))
      local_softmax_row(base, key, slot, i);
}

static void local_softmax_row(uint32_t base, uint32_t key, uint32_t slot,
                              uint32_t i) {
  q38_local_sweep_t *const loc = q38_loc;
  const uint32_t j = i / Q38_PROJECTION_ROWS;
  const uint32_t token =
      base + q38_loc_owner(j) % Q38_LOC_TILES * Q38_PROJECTION_ROWS +
      i % Q38_PROJECTION_ROWS;
  uint32_t visible = token < Q38_S && token >= key ? token + 1 - key : 0;
  if (visible > BK)
    visible = BK;
  __fp16 *const score = Q38_LA(loc->score_tile + slot * Q38_LOC_SLOT + i * BK);
  __fp16 *const prob = Q38_LA(loc->prob_tile + slot * Q38_LOC_SLOT + i * BK);
#if Q38_ATTN_FUSED_SOFTMAX
  const float alpha =
      softmax_row(score, prob, visible, Q38_LA(loc->states + Q38_ST(i)));
#else
  llm_zero_f16(prob, BK);
  llm_scale_f16_f32(score, BK, llm_rsqrt((float)Q38_HD));
  for (uint32_t k = visible; k < BK; ++k)
    q38_write16(score + k, -65504.0f);
  const float alpha = qwen38_softmax_update_f16(
      score, prob, visible, Q38_LA(loc->states + Q38_ST(i)));
#endif
#if Q38_ATTN_RM_ACCUM && !Q38_ATTN_LOCAL_PIPE
  // Key block 0 is visible to every row and its PV job overwrites Z.
  if (key != 0u && alpha != 1.0f)
    llm_scale_f16_f32(Q38_LA(loc->z + i * Q38_HD), Q38_HD, alpha);
#else
  *Q38_LA(loc->alpha + Q38_AL(i)) = alpha;
  if (Q38_ATTN_DRAIN)
    q38_drain16(prob, BK);
#endif
}

static NOSTACK void query_key_local(uint32_t base, uint32_t key,
                                    const __fp16 *kt) {
  local_jobs_wait(base, key, kt, 0u, 0u);
  KP(KP_QK);
  q38_group_sync();
  KP(KP_QK_JOIN);
  local_softmax(base, key, 0u);
  KP(KP_SOFTMAX);
  q38_group_sync();
  KP(KP_SOFTMAX_JOIN);
  local_jobs_wait(base, key, kt, 0u, 1u);
  KP(KP_PV);
  q38_group_sync();
  KP(KP_PV_JOIN);
#if !Q38_ATTN_RM_ACCUM
  q38_local_sweep_t *const loc = q38_loc;
  for (uint32_t i = mempool_get_core_id() % Q38_LOC_CORES; i < Q38_LOC_ROWS;
       i += Q38_LOC_CORES)
    if (q38_loc_live(base, key, i / Q38_PROJECTION_ROWS))
      accumulate(Q38_LA(loc->u + i * Q38_HD), Q38_LA(loc->z + i * Q38_HD),
                 *Q38_LA(loc->alpha + Q38_AL(i)));
#endif
  KP(KP_ACCUMULATE);
}

#if Q38_ATTN_LOCAL_PIPE
static uint32_t next_key_block(uint32_t key, uint32_t limit);

// Apply block `key`'s corrections to Z once PV(n-1), and its fold, have
// landed and before PV(n) starts.
static NOSTACK void local_rescale(uint32_t base, uint32_t key) {
  q38_local_sweep_t *const loc = q38_loc;
  if (key == 0u)
    return;
  for (uint32_t k = 0u, i; (i = local_row(k)) < Q38_LOC_ROWS; ++k) {
    if (!q38_loc_live(base, key, i / Q38_PROJECTION_ROWS))
      continue;
    const float alpha = *Q38_LA(loc->alpha + Q38_AL(i));
    if (alpha != 1.0f) {
      llm_scale_f16_f32(Q38_LA(loc->z + i * Q38_HD), Q38_HD, alpha);
      if (Q38_ATTN_DRAIN)
        q38_drain16(Q38_LA(loc->z + i * Q38_HD), Q38_HD);
    }
  }
}

// Software-pipelined key sweep of one window. Step n runs softmax(n) on Spatz
// while RedMulE runs PV(n-1) and QK(n+1); block n's corrections are applied
// to Z after PV(n-1) and before PV(n). Scores and probabilities alternate
// between two slots, and a four-buffer K/V ring keeps V(n-1) and K(n+1)
// resident while K/V(n+2) streams in.
static NOSTACK void query_sweep_pipe(uint32_t base) {
  const uint32_t core = mempool_get_core_id();
  const uint32_t reach = base + Q38_WIN < Q38_S ? base + Q38_WIN : Q38_S;
  const uint32_t limit = base >= Q38_S ? 0u : reach;
  if (limit == 0u)
    return;
  __fp16 *const ring[4] = {weights[0], weights[0] + Q38_KV_ELEMS,
                           weights[0] + 2u * Q38_KV_ELEMS,
                           weights[0] + 3u * Q38_KV_ELEMS};
#define Q38_KV_AT(key) (result_cache + (key) / BK * 2 * Q38_KVH * Q38_HD * BK)
  const uint32_t load_start = mempool_get_timer();
  const uint32_t dma = !(Q38_ATTN_ISOLATE & 2u);
  if (dma)
    transfer(ring[0], Q38_KV_AT(0u), sizeof(kv_tile));
  uint32_t prev = 0u, cur = 0u, next = next_key_block(0u, limit);
  dma_collective_op_t pending;
  uint32_t loading = dma && next < limit;
  if (loading)
    pending = dma_collective_memcpy_nonblocking(ring[1], Q38_KV_AT(next),
                                                sizeof(kv_tile));
  kv_wait_cycles[core] += mempool_get_timer() - load_start;
  local_jobs_wait(base, cur, ring[0], 0u, 0u);
  q38_group_sync();
  uint32_t n = 0u;
  for (; cur < limit; ++n) {
#if Q38_SCHEDULE_PROBE
    if (core == 0u)
      ++probe_key_blocks;
#endif
    kv_block_loads[core] += 1u;
    KP(KP_COUNT);
    const uint32_t wait_start = mempool_get_timer();
    if (loading &&
        (dma_collective_wait(&pending) != 0u || pending.status != 0u))
      __builtin_trap();
    kv_wait_cycles[core] += mempool_get_timer() - wait_start;
    const uint32_t after = next < limit ? next_key_block(next, limit) : limit;
    loading = dma && after < limit;
    if (loading)
      pending = dma_collective_memcpy_nonblocking(
          ring[(n + 2u) % 4u], Q38_KV_AT(after), sizeof(kv_tile));
    KP(KP_BLOCK_END);
    uint8_t last = 0u;
    uint32_t issued = 0u;
    if (Q38_LOC_AGENT) {
      if (n > 0u) {
        issued |= local_jobs(base, prev, ring[(n + 3u) % 4u], (n + 1u) % 2u, 1u,
                             &last);
        if (Q38_ATTN_RM_FOLD)
          local_jobs(base, prev, ring[(n + 3u) % 4u], (n + 1u) % 2u, 3u, &last);
      }
      if (next < limit)
        issued |= local_jobs(base, next, ring[(n + 1u) % 4u], (n + 1u) % 2u, 0u,
                             &last);
    }
    KP(KP_QK);
    local_softmax(base, cur, n % 2u);
    KP(KP_SOFTMAX);
    if (issued)
      tnc_redmule_wait(last);
    KP(KP_PV);
    q38_group_sync();
    KP(KP_PV_JOIN);
    local_rescale(base, cur);
    q38_group_sync();
    KP(KP_ACCUMULATE);
    prev = cur;
    cur = next;
    next = after;
  }
  local_jobs_wait(base, prev, ring[(n + 3u) % 4u], (n + 1u) % 2u, 1u);
  if (Q38_ATTN_RM_FOLD)
    local_jobs_wait(base, prev, ring[(n + 3u) % 4u], (n + 1u) % 2u, 3u);
  q38_group_sync();
#undef Q38_KV_AT
}
#endif

#if !Q38_ATTN_VC
// Normalize and gate the group's owner rows into the supertile's gated rows,
// and copy the raw context out for the sampled checks.
static NOSTACK void local_query_finish(uint32_t base, uint32_t offset) {
  q38_local_sweep_t *const loc = q38_loc;
  const uint32_t lc = mempool_get_core_id() % Q38_LOC_CORES;
  const __fp16 *const qgate = query_stage.qgate + offset * 2 * Q38_QDIM;
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
#if Q38_ATTN_RM_ACCUM
    const __fp16 *const z = Q38_LA(loc->z + i * Q38_HD);
    const float zh = llm_f16box(0.0f);
#endif
#if !Q38_ATTN_RM_ACCUM
    const float *const u = Q38_LA(loc->u + i * Q38_HD);
#endif
    for (uint32_t o = 0; o < Q38_HD;) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                   : "=r"(vl)
                   : "r"(Q38_HD - o));
#if Q38_ATTN_RM_ACCUM
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
      asm volatile("vle16.v v0, (%0)" : : "r"(z + o) : "memory");
      asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
#else
      asm volatile("vle32.v v8, (%0)" : : "r"(u + o) : "memory");
#endif
      asm volatile("vfmul.vf v8, v8, %0" : : "f"(inverse));
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
      asm volatile("vfncvt.f.f.w v0, v8");
      asm volatile("vse16.v v0, (%0)" : : "r"(context + o) : "memory");
      o += vl;
    }
    q38_activation16(
        qgate + row * 2 * Q38_QDIM + h * 2 * Q38_HD + Q38_HD, context,
        stage.attention.gated + (offset + row) * Q38_QDIM + h * Q38_HD, Q38_HD,
        0);
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
#endif
#endif

// Its owner engines hold every row's softmax statistics and unnormalized
// output for the whole causal key sweep: no reduction and no rescaling
// crosses owners.
static uint32_t next_key_block(uint32_t key, uint32_t limit) {
  uint32_t next = key + BK;
#if Q38_SCHEDULE_PROBE
  // The first two blocks turn the prefetch ping-pong; the last tests the
  // distant causal address and drain without repeating every middle key.
  if (next >= 2u * BK && next < limit - BK)
    next = limit - BK;
#else
  (void)limit;
#endif
  return next;
}

static NOSTACK void query_sweep(uint32_t base, uint32_t rows,
                                __fp16 *active_kv) {
  const uint32_t core = mempool_get_core_id();
  // Causality bounds the sweep: a window never reads a key block that starts
  // after its last row.
  const uint32_t reach = base + rows < Q38_S ? base + rows : Q38_S;
  const uint32_t limit = base >= Q38_S ? 0u : reach;
#if Q38_KV_PREFETCH
  __fp16 *current_kv = active_kv;
  __fp16 *next_kv = kv_tile_next;
  if (limit) {
    const uint32_t load_start = mempool_get_timer();
    transfer(current_kv, result_cache, sizeof(kv_tile));
    kv_wait_cycles[core] += mempool_get_timer() - load_start;
  }
#endif
#if Q38_BACKGROUND_FFN_NORM
  const uint32_t has_prior = base >= Q38_MIXER_EXEC_START + Q38_WIN;
  const uint32_t prior_base = base - (has_prior ? Q38_WIN : 0u);
  const uint32_t norm_epoch =
      has_prior ? 2u * ((base - Q38_MIXER_EXEC_START) / Q38_WIN - 1u) + 1u : 0u;
  if (has_prior)
    ffn_norm_dma_begin(norm_epoch, Q38_FFN_RESIDUAL,
                       result_mixer + prior_base * Q38_D, Q38_WIN * Q38_D * 2,
                       0u);
#endif
  for (uint32_t key = 0; key < limit;) {
#if Q38_SCHEDULE_PROBE
    if (mempool_get_core_id() == 0u)
      ++probe_key_blocks;
#endif
    KP(KP_COUNT);
#if Q38_KV_PREFETCH
    const uint32_t next_key = next_key_block(key, limit);
    dma_collective_op_t pending;
    if (next_key < limit)
      pending = dma_collective_memcpy_nonblocking(
          next_kv, result_cache + next_key / BK * 2 * Q38_KVH * Q38_HD * BK,
          sizeof(kv_tile));
    const __fp16 *kt = current_kv;
#else
    const uint32_t load_start = mempool_get_timer();
    const __fp16 *l2_tile = result_cache + key / BK * 2 * Q38_KVH * Q38_HD * BK;
#if Q38_KV_REPLICATED
    q38_load_kv_replica(active_kv, l2_tile, sizeof(kv_tile));
#else
    transfer(active_kv, l2_tile, sizeof(kv_tile));
#endif
    sync_all();
    kv_wait_cycles[core] += mempool_get_timer() - load_start;
    const __fp16 *kt = active_kv;
#endif
    kv_block_loads[core] += 1u;
#if Q38_BACKGROUND_FFN_NORM
    if (has_prior && key == 0u)
      ffn_norm_background(Q38_WIN, norm_epoch);
#endif
#if Q38_ATTN_LOCAL
    (void)rows;
    query_key_local(base, key, kt);
#elif Q38_ATTN_SPREAD
    query_key_spread(base, rows, key, kt);
#else
    query_key_jobs(base, rows, key, kt);
#endif
#if Q38_BACKGROUND_FFN_NORM
    if (has_prior && key == 0u && mempool_is_group_coordinator()) {
      uint32_t status;
      do {
        (void)q38_dma_stage_progress(&ffn_norm_dma_queue);
        status = q38_dma_stage_ready(&ffn_norm_dma_queue, norm_epoch);
      } while (status == Q38_DMA_QUEUE_PENDING);
      if (status != Q38_DMA_QUEUE_OK)
        __builtin_trap();
    }
#endif
#if Q38_KV_PREFETCH
    if (next_key < limit) {
      const uint32_t wait_start = mempool_get_timer();
      if (dma_collective_wait(&pending) != 0u || pending.status != 0u)
        __builtin_trap();
      kv_wait_cycles[core] += mempool_get_timer() - wait_start;
      __fp16 *const prior = current_kv;
      current_kv = next_kv;
      next_kv = prior;
    } else {
      sync_all();
    }
#else
    sync_all();
#endif
#if Q38_BACKGROUND_FFN_NORM
    if (has_prior && key == 0u)
      ffn_norm_dma_begin(norm_epoch + 1u, result_output + prior_base * Q38_D,
                         Q38_FFN_A, Q38_WIN * Q38_D * 2, 0u);
#endif
    KP(KP_BLOCK_END);
    key = next_key_block(key, limit);
  }
#if Q38_BACKGROUND_FFN_NORM
  if (has_prior)
    ffn_norm_dma_finish(norm_epoch);
#endif
}

static NOSTACK void query_finish(uint32_t base, uint32_t rows, uint32_t last) {
  const uint32_t core = mempool_get_core_id();
  __fp16 *const qgate = query_stage.qgate;
  __fp16 *const context = query_stage.context;
  const float *const accumulated = query_stage.accumulated;
  const qwen38_softmax_state_t *const states = query_stage.states;
  for (uint32_t i = core; i < Q38_QH * rows; i += NUM_CORES) {
    const uint32_t h = i / rows, t = i % rows;
    const float inverse =
        base + t < Q38_S ? llm_recip_pos(states[i].denominator) : 0.0f;
    for (uint32_t offset = 0; offset < Q38_HD;) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                   : "=r"(vl)
                   : "r"(Q38_HD - offset));
      asm volatile("vle32.v v8, (%0)"
                   :
                   : "r"(accumulated + i * Q38_HD + offset)
                   : "memory");
      asm volatile("vfmul.vf v8, v8, %0" : : "f"(inverse));
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
      asm volatile("vfncvt.f.f.w v0, v8");
      asm volatile("vse16.v v0, (%0)"
                   :
                   : "r"(context + i * Q38_HD + offset)
                   : "memory");
      offset += vl;
    }
    q38_activation16(qgate + t * 2 * Q38_QDIM + h * 2 * Q38_HD + Q38_HD,
                     context + i * Q38_HD, a + t * Q38_QDIM + h * Q38_HD,
                     Q38_HD, 0);
  }
  sync_all();
#if Q38_SAMPLED_CHECKS
  gather_context_window(context, base, rows, Q38_HD);
#else
  for (uint32_t h = 0; h < Q38_QH; ++h)
    export_tensor(result_context + (h * Q38_R + base) * Q38_HD,
                  context + h * rows * Q38_HD, rows * Q38_HD * 2);
#endif
  project_rows(a, w_out, mixed, Q38_QDIM, Q38_D,
               last ? w_gate : (const __fp16 *)0, Q38_D, Q38_I, rows,
               &mixer_buffers);
  llm_residual_add_f16(mixed, residual, rows * Q38_D, core, NUM_CORES);
  sync_all();
  transfer(result_mixer + base * Q38_D, mixed, rows * Q38_D * 2);
}

static NOSTACK void query_window(uint32_t base, uint32_t rows, uint32_t last,
                                 __fp16 *active_kv) {
  query_project(base, rows);
  query_sweep(base, rows, active_kv);
  query_finish(base, rows, last);
}

#if Q38_ATTN_SUPERTILE
#define Q38_AST_PANELS                                                         \
  Q38_MIN(Q38_MAX(1u, NUM_REDMULE_TILES /                                      \
                          (Q38_ATTN_ST_ROWS / Q38_PROJECTION_ROWS)),           \
          Q38_MAX_K *Q38_WEIGHT_PANELS / Q38_MAX(Q38_D, Q38_QDIM))
_Static_assert(Q38_ATTN_ST_ROWS % Q38_WIN == 0 && Q38_ATTN_ST_ROWS % BK == 0 &&
                   Q38_ATTN_ST_ROWS <= Q38_ROWS &&
                   Q38_ATTN_ST_ROWS * Q38_D <= sizeof(a) / sizeof(a[0]) &&
                   Q38_AST_PANELS > 0,
               "Attention supertile must fit the row buffers and weight slots");

#if Q38_PROJ_DATAFLOW
#define Q38_AST_PROJECT(in, w, out, k, n, rows)                                \
  project_dataflow(in, w, out, k, n, rows, Q38_AST_PANELS)
#else
#define Q38_AST_PROJECT(in, w, out, k, n, rows)                                \
  project_rows_from_panel(in, w, out, k, n, rows, 0u, Q38_AST_PANELS,          \
                          &mixer_buffers)
#endif

// K and V of the whole supertile, normalized, rotated and packed to L2.
static NOSTACK void ast_kv(uint32_t base, uint32_t rows) {
  const uint32_t core = mempool_get_core_id();
  __fp16 *const projected = stage.kv.projected;
  __fp16 *const heads = stage.kv.heads;
  for (uint32_t pass = 0u; pass < 2u; ++pass) {
    Q38_AST_PROJECT(a, pass ? w_value : w_key, projected, Q38_D, Q38_KVDIM,
                    rows);
    for (uint32_t i = core; i < rows * Q38_KVH; i += NUM_CORES) {
      const uint32_t t = i / Q38_KVH, h = i % Q38_KVH;
      q38_copy16(heads + ((pass * Q38_KVH + h) * rows + t) * Q38_HD,
                 projected + i * Q38_HD, Q38_HD);
    }
    sync_all();
  }
  transfer(gamma, w_k_norm, Q38_HD * 2);
  for (uint32_t i = core; i < Q38_KVH * rows; i += NUM_CORES) {
    q38_norm16(heads + i * Q38_HD, gamma, heads + i * Q38_HD, Q38_HD, 0);
    rotary(heads + i * Q38_HD, i % rows);
  }
  sync_all();
  q38_pack_kv_to_l2(heads, kv_tile,
                    result_cache + base / BK * 2 * Q38_KVH * Q38_HD * BK, rows,
                    Q38_KVH, Q38_HD, BK);
}

// Query heads of one window, from the supertile's Q/gate projection.
static NOSTACK void ast_query_prep(uint32_t offset, uint32_t rows) {
  const uint32_t core = mempool_get_core_id();
  const __fp16 *const qgate = query_stage.qgate + offset * 2 * Q38_QDIM;
  transfer(gamma, w_q_norm, Q38_HD * 2);
  for (uint32_t i = core; i < Q38_QH * rows; i += NUM_CORES) {
    const uint32_t h = i / rows, t = i % rows;
    q38_norm16(qgate + t * 2 * Q38_QDIM + h * 2 * Q38_HD, gamma,
               query_stage.q + i * Q38_HD, Q38_HD, 0);
    rotary(query_stage.q + i * Q38_HD, offset + t);
    qwen38_softmax_init(query_stage.states + i);
  }
  sync_all();
  clear_all(query_stage.accumulated,
            Q38_QH * rows * Q38_HD * (uint32_t)sizeof(float));
}

// Normalize and gate one window's context into the supertile's gated rows.
static NOSTACK void ast_query_finish(uint32_t base, uint32_t offset,
                                     uint32_t rows) {
  const uint32_t core = mempool_get_core_id();
  const __fp16 *const qgate = query_stage.qgate + offset * 2 * Q38_QDIM;
  __fp16 *const context = query_stage.context;
  for (uint32_t i = core; i < Q38_QH * rows; i += NUM_CORES) {
    const uint32_t h = i / rows, t = i % rows;
    const float inverse = base + t < Q38_S
                              ? llm_recip_pos(query_stage.states[i].denominator)
                              : 0.0f;
    for (uint32_t o = 0; o < Q38_HD;) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                   : "=r"(vl)
                   : "r"(Q38_HD - o));
      asm volatile("vle32.v v8, (%0)"
                   :
                   : "r"(query_stage.accumulated + i * Q38_HD + o)
                   : "memory");
      asm volatile("vfmul.vf v8, v8, %0" : : "f"(inverse));
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(vl));
      asm volatile("vfncvt.f.f.w v0, v8");
      asm volatile("vse16.v v0, (%0)"
                   :
                   : "r"(context + i * Q38_HD + o)
                   : "memory");
      o += vl;
    }
    q38_activation16(qgate + t * 2 * Q38_QDIM + h * 2 * Q38_HD + Q38_HD,
                     context + i * Q38_HD,
                     stage.attention.gated + (offset + t) * Q38_QDIM +
                         h * Q38_HD,
                     Q38_HD, 0);
  }
  sync_all();
#if Q38_SAMPLED_CHECKS
  gather_context_window(context, base, rows, Q38_HD);
#else
  for (uint32_t h = 0; h < Q38_QH; ++h)
    export_tensor(result_context + (h * Q38_R + base) * Q38_HD,
                  context + h * rows * Q38_HD, rows * Q38_HD * 2);
#endif
}

static NOSTACK void ast_output(uint32_t base, uint32_t rows) {
  Q38_AST_PROJECT(stage.attention.gated, w_out, stage.attention.out, Q38_QDIM,
                  Q38_D, rows);
  sync_all();
  llm_residual_add_f16(stage.attention.out, residual, rows * Q38_D,
                       mempool_get_core_id(), NUM_CORES);
  sync_all();
  transfer(result_mixer + base * Q38_D, stage.attention.out, rows * Q38_D * 2);
}

static NOSTACK void attention_supertiles(void) {
  const uint32_t core = mempool_get_core_id();
  kv_wait_cycles[core] = 0;
  kv_block_loads[core] = 0;
#if Q38_ATTN_LOCAL
  q38_local_init();
  transfer(q38_q_gamma, w_q_norm, Q38_HD * 2);
#endif
  for (uint32_t base = Q38_MIXER_EXEC_START; base < Q38_MIXER_EXEC_END;
       base += Q38_ATTN_ST_ROWS) {
    const uint32_t rows = Q38_MIN(Q38_ATTN_ST_ROWS, Q38_MIXER_EXEC_END - base);
    transfer(residual, input_init + base * Q38_D, rows * Q38_D * 2);
    norm_rows(residual, w_input_norm, a, rows, Q38_D, 0, gamma);
    stage_rotary(base, rows);
    ast_kv(base, rows);
    Q38_AST_PROJECT(a, w_q_gate, query_stage.qgate, Q38_D, 2 * Q38_QDIM, rows);
    for (uint32_t offset = 0u; offset < rows; offset += Q38_WIN) {
#if Q38_ATTN_LOCAL
      KP(KP_COUNT);
      local_query_prep(offset);
      KP(KP_PREP);
#if Q38_ATTN_LOCAL_PIPE
      query_sweep_pipe(base + offset);
#else
      query_sweep(base + offset, Q38_WIN, kv_tile);
#endif
      KP(KP_COUNT);
      local_query_finish(base + offset, offset);
#else
      KP(KP_COUNT);
      ast_query_prep(offset, Q38_WIN);
      KP(KP_PREP);
      query_sweep(base + offset, Q38_WIN, kv_tile);
      KP(KP_COUNT);
      ast_query_finish(base + offset, offset, Q38_WIN);
#endif
      KP(KP_FINISH);
#if Q38_SCHEDULE_PROBE
      if (core == 0u)
        ++probe_mixer_windows;
#endif
    }
    ast_output(base, rows);
  }
}
#endif

#if Q38_ATTN_VC
static NOSTACK void attention_phased(void);
#endif

static NOSTACK void attention(void) {
  const uint32_t core = mempool_get_core_id();
  kv_wait_cycles[core] = 0;
  kv_block_loads[core] = 0;
#if Q38_SCHEDULE_PROBE
  for (uint32_t phase = 0; phase < KP_COUNT; ++phase)
    kp_cycles[phase][core] = 0;
  kp_last[core] = 0;
#endif
#if Q38_ATTN_VC
  attention_phased();
#elif Q38_ATTN_SUPERTILE
  attention_supertiles();
#elif Q38_INTERLEAVE_KV_QUERY
  __fp16 *active_kv = kv_tile;
#if Q38_KV_REPLICATED
  active_kv = q38_allocate_kv_replica(Q38_KV_LEVEL, sizeof(kv_tile));
#endif
#if Q38_BACKGROUND_FFN_NORM
  transfer(Q38_FFN_GAMMA, w_ffn_norm, Q38_D * 2);
#endif
  for (uint32_t base = Q38_MIXER_EXEC_START; base < Q38_MIXER_EXEC_END;
       base += Q38_WIN) {
    const uint32_t rows = block_rows(base, Q38_WIN);
    build_kv_window(base, rows, base + rows >= Q38_MIXER_EXEC_END);
    query_window(base, rows, base + rows >= Q38_MIXER_EXEC_END, active_kv);
#if Q38_SCHEDULE_PROBE
    if (core == 0u)
      ++probe_mixer_windows;
#endif
#if Q38_FFN_NORM_STREAM && !Q38_BACKGROUND_FFN_NORM
    ffn_norm_window(base, rows);
#endif
  }
#if Q38_BACKGROUND_FFN_NORM
  ffn_norm_window(Q38_MIXER_EXEC_END - Q38_WIN, Q38_WIN);
#endif
#else
  for (uint32_t base = Q38_MIXER_EXEC_START; base < Q38_MIXER_EXEC_END;
       base += Q38_WIN) {
    const uint32_t rows = block_rows(base, Q38_WIN);
    build_kv_window(base, rows, base + rows >= Q38_MIXER_EXEC_END);
  }
  __fp16 *active_kv = kv_tile;
#if Q38_KV_REPLICATED
  active_kv = q38_allocate_kv_replica(Q38_KV_LEVEL, sizeof(kv_tile));
#endif
  for (uint32_t base = Q38_MIXER_EXEC_START; base < Q38_MIXER_EXEC_END;
       base += Q38_WIN) {
    const uint32_t rows = block_rows(base, Q38_WIN);
    query_window(base, rows, base + rows >= Q38_MIXER_EXEC_END, active_kv);
#if Q38_SCHEDULE_PROBE
    if (core == 0u)
      ++probe_mixer_windows;
#endif
#if Q38_FFN_NORM_STREAM
    ffn_norm_window(base, rows);
#endif
  }
#endif
}
