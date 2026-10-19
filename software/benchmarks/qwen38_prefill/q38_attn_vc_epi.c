// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Included by q38_attn_vc.c. Raw heads stay in registers, not row scratch.
#if Q38_AV_PANEL_AWARE
_Static_assert(((VLEN == 512u && (NT == 32u || NT == 64u)) ||
                (VLEN == 1024u && NT == 64u)) &&
                   (Q38_HD == 64u || Q38_HD == 128u || Q38_HD == 256u) &&
                   Q38_HALF <= 32u,
               "Panel epilogues need VLEN512 or VLEN1024/NT64");
#define AVP_REDUCE_ELEMS (VLEN / 32u * 8u)

static inline __attribute__((always_inline)) void
avp_load(const __fp16 *zu, uint32_t row, uint32_t off, uint32_t upper) {
  const __fp16 *p = Q38_VC_GLOBAL(zu + off / NT * Q38_AV_TILE + row * NT);
#if VLEN == 1024
  // One NT64 panel fills one register; keep adjacent panels adjacent in VRF.
  (void)upper;
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(64u));
  asm volatile("vle16.v v0, (%0)" : : "r"(p) : "memory");
#if Q38_HD >= 128
  p = Q38_VC_GLOBAL(zu + (off / NT + 1u) * Q38_AV_TILE + row * NT);
  asm volatile("vle16.v v1, (%0)" : : "r"(p) : "memory");
#endif
#if Q38_HD == 256
  p = Q38_VC_GLOBAL(zu + (off / NT + 2u) * Q38_AV_TILE + row * NT);
  asm volatile("vle16.v v2, (%0)" : : "r"(p) : "memory");
  p = Q38_VC_GLOBAL(zu + (off / NT + 3u) * Q38_AV_TILE + row * NT);
  asm volatile("vle16.v v3, (%0)" : : "r"(p) : "memory");
#endif
#elif NT == 32
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(32u));
  if (upper)
    asm volatile("vle16.v v4, (%0)" : : "r"(p) : "memory");
  else
    asm volatile("vle16.v v0, (%0)" : : "r"(p) : "memory");
  p = Q38_VC_GLOBAL(zu + (off / NT + 1u) * Q38_AV_TILE + row * NT);
  if (upper)
    asm volatile("vle16.v v5, (%0)" : : "r"(p) : "memory");
  else
    asm volatile("vle16.v v1, (%0)" : : "r"(p) : "memory");
#if Q38_HD >= 128
  p = Q38_VC_GLOBAL(zu + (off / NT + 2u) * Q38_AV_TILE + row * NT);
  if (upper)
    asm volatile("vle16.v v6, (%0)" : : "r"(p) : "memory");
  else
    asm volatile("vle16.v v2, (%0)" : : "r"(p) : "memory");
  p = Q38_VC_GLOBAL(zu + (off / NT + 3u) * Q38_AV_TILE + row * NT);
  if (upper)
    asm volatile("vle16.v v7, (%0)" : : "r"(p) : "memory");
  else
    asm volatile("vle16.v v3, (%0)" : : "r"(p) : "memory");
#endif
#else
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(64u));
  if (upper)
    asm volatile("vle16.v v4, (%0)" : : "r"(p) : "memory");
  else
    asm volatile("vle16.v v0, (%0)" : : "r"(p) : "memory");
#if Q38_HD >= 128
  p = Q38_VC_GLOBAL(zu + (off / NT + 1u) * Q38_AV_TILE + row * NT);
  if (upper)
    asm volatile("vle16.v v6, (%0)" : : "r"(p) : "memory");
  else
    asm volatile("vle16.v v2, (%0)" : : "r"(p) : "memory");
#endif
#endif
}

static inline __attribute__((always_inline)) void
avp_store_row(uint32_t engine, uint32_t slot, uint32_t row) {
  asm volatile("vsetvli zero, %0, e16, m8, ta, ma" : : "r"(Q38_HD));
  asm volatile("vse16.v v0, (%0)"
               :
               : "r"(av_stage_at(engine, slot, row * Q38_HD * 2u))
               : "memory");
}

static inline __attribute__((always_inline)) void avp_prefetch(const __fp16 *zu,
                                                               uint32_t row) {
  const __fp16 *p = Q38_VC_GLOBAL(zu + row * NT);
#if VLEN == 1024
  // v4..v7 are free: the current HD256 head occupies only v0..v3.
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(NT));
  asm volatile("vle16.v v4, (%0)" : : "r"(p) : "memory");
#if Q38_HD >= 128
  p = Q38_VC_GLOBAL(zu + Q38_AV_TILE + row * NT);
  asm volatile("vle16.v v5, (%0)" : : "r"(p) : "memory");
#endif
#if Q38_HD == 256
  p = Q38_VC_GLOBAL(zu + 2u * Q38_AV_TILE + row * NT);
  asm volatile("vle16.v v6, (%0)" : : "r"(p) : "memory");
  p = Q38_VC_GLOBAL(zu + 3u * Q38_AV_TILE + row * NT);
  asm volatile("vle16.v v7, (%0)" : : "r"(p) : "memory");
#endif
#else
  // v28..v31 retain the next lower128 chunk through the current row kernel.
#if NT == 32
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(NT));
  asm volatile("vle16.v v28, (%0)" : : "r"(p) : "memory");
  p = Q38_VC_GLOBAL(zu + Q38_AV_TILE + row * NT);
  asm volatile("vle16.v v29, (%0)" : : "r"(p) : "memory");
#if Q38_HD >= 128
  p = Q38_VC_GLOBAL(zu + 2u * Q38_AV_TILE + row * NT);
  asm volatile("vle16.v v30, (%0)" : : "r"(p) : "memory");
  p = Q38_VC_GLOBAL(zu + 3u * Q38_AV_TILE + row * NT);
  asm volatile("vle16.v v31, (%0)" : : "r"(p) : "memory");
#endif
#else
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(NT));
  asm volatile("vle16.v v28, (%0)" : : "r"(p) : "memory");
#if Q38_HD >= 128
  p = Q38_VC_GLOBAL(zu + Q38_AV_TILE + row * NT);
  asm volatile("vle16.v v30, (%0)" : : "r"(p) : "memory");
#endif
#endif
#endif
}

static inline __attribute__((always_inline)) void avp_take_prefetch(void) {
#if VLEN == 1024
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(NT));
  asm volatile("vmv.v.v v0, v4");
#if Q38_HD >= 128
  asm volatile("vmv.v.v v1, v5");
#endif
#if Q38_HD == 256
  asm volatile("vmv.v.v v2, v6");
  asm volatile("vmv.v.v v3, v7");
#endif
#elif NT == 32
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(NT));
  asm volatile("vmv.v.v v0, v28");
  asm volatile("vmv.v.v v1, v29");
#if Q38_HD >= 128
  asm volatile("vmv.v.v v2, v30");
  asm volatile("vmv.v.v v3, v31");
#endif
#else
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(NT));
  asm volatile("vmv.v.v v0, v28");
#if Q38_HD >= 128
  asm volatile("vmv.v.v v2, v30");
#endif
#endif
}

static NOSTACK void avp_epi_row_pipe(const __fp16 *zu, uint32_t u,
                                     uint32_t row0, uint32_t engine,
                                     uint32_t slot, uint32_t row,
                                     uint32_t prefetched, uint32_t nextrow) {
  const uint32_t heads = 2u * Q38_QH;
  const uint32_t key = u >= heads && u < heads + Q38_KVH;
  const uint32_t rotate = u < heads ? !(u & 1u) : key;
  if (!rotate) {
#if Q38_AV_STATIC_CORES
    avp_load(zu, row, 0u, 0u);
#if Q38_HD > AVP_REDUCE_ELEMS
    avp_load(zu, row, AVP_REDUCE_ELEMS, 1u);
#endif
    avp_store_row(engine, slot, row);
#else
#pragma clang loop unroll(full)
    for (uint32_t p = 0; p < Q38_AV_U; ++p)
      q38_copy16(
          (__fp16 *)av_stage_at(engine, slot, (row * Q38_HD + p * NT) * 2u),
          Q38_VC_GLOBAL(zu + p * Q38_AV_TILE + row * NT), NT);
#endif
    return;
  }
  const float zero = 0.0f, zh = llm_f16box(0.0f);
  const __fp16 *const gamma = u < heads ? av_qg : av_kg;
  const uint32_t chunk = Q38_MIN(Q38_HD, AVP_REDUCE_ELEMS);
  // Match q38_norm16's accumulator width/order for the selected VLEN.
  asm volatile("vsetvli zero, %0, e32, m8, tu, ma" : : "r"(AVP_REDUCE_ELEMS));
  asm volatile("vmv.v.i v8, 0");
  if (prefetched)
    avp_take_prefetch();
  else
    avp_load(zu, row, 0u, 0u);
  asm volatile("vsetvli zero, %0, e16, m4, tu, ma" : : "r"(chunk));
  asm volatile("vfwmacc.vv v8, v0, v0");
#if Q38_HD > AVP_REDUCE_ELEMS
  // Fetch the upper raw chunk while the lower chunk's multiply is pending.
  avp_load(zu, row, AVP_REDUCE_ELEMS, 1u);
  asm volatile("vsetvli zero, %0, e16, m4, tu, ma" : : "r"(AVP_REDUCE_ELEMS));
  asm volatile("vfwmacc.vv v8, v4, v4");
#endif
  if (nextrow != ~0u)
    avp_prefetch(zu, nextrow);
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(AVP_REDUCE_ELEMS));
  float sum;
#if Q38_AV_CACHE_GAMMA
  asm volatile("vfmv.s.f v16, %0" : : "f"(zero));
  asm volatile("vfredusum.vs v16, v8, v16");
  asm volatile("vfmv.f.s %0, v16" : "=f"(sum));
#else
  asm volatile("vfmv.s.f v24, %0" : : "f"(zero));
  asm volatile("vfredusum.vs v24, v8, v24");
  asm volatile("vfmv.f.s %0, v24" : "=f"(sum));
#endif
#if Q38_AV_SWEEP_QSCALE
  // Q carries the power-of-two 1/sqrt(HD); every later rounding commutes.
  const float inverse = llm_rsqrt(sum * llm_recip_pos((float)Q38_HD) + 1e-6f) *
                        (u < heads ? Q38_QK_SCALE : 1.0f);
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
#if Q38_AV_CACHE_GAMMA
    asm volatile("vmv.v.v v24, v28");
#else
    asm volatile("vle16.v v24, (%0)" : : "r"(gamma + off) : "memory");
#endif
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
  // Extract rotary pairs from the retained, rounded FP16 head.
  // Keep v28..v31 intact when the S1 prefetch shares this destination group.
  asm volatile("vsetvli zero, %0, e16, m8, tu, ma" : : "r"(Q38_HALF));
  asm volatile("vslidedown.vx v24, v0, %0" : : "r"(Q38_HALF));
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(Q38_HALF));
  asm volatile("vfwadd.vf v8, v0, %0" : : "f"(zh));
  asm volatile("vfwadd.vf v10, v24, %0" : : "f"(zh));
  asm volatile("vle16.v v24, (%0)"
               :
               : "r"(av.cos + (row0 + row) * Q38_HALF)
               : "memory");
  asm volatile("vfwadd.vf v12, v24, %0" : : "f"(zh));
  asm volatile("vle16.v v24, (%0)"
               :
               : "r"(av.sin + (row0 + row) * Q38_HALF)
               : "memory");
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
    avp_store_row(engine, slot, row);
    return;
  }
  const uint32_t run = VC_ROW_BYTES / (Q38_PROJECTION_ROWS * 2u);
  for (uint32_t d = 0; d < Q38_HD; d += run) {
    asm volatile("vsetvli zero, %0, e16, m8, tu, ma" : : "r"(run));
    asm volatile("vslidedown.vx v24, v0, %0" : : "r"(d));
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(run));
    asm volatile(
        "vsse16.v v24, (%0), %1"
        :
        : "r"(av_stage_at(engine, slot, (d * Q38_PROJECTION_ROWS + row) * 2u)),
          "r"(Q38_PROJECTION_ROWS * 2u)
        : "memory");
  }
}

static inline __attribute__((always_inline)) void avp_gamma(uint32_t u) {
#if Q38_AV_CACHE_GAMMA
  if (!av_copy_unit(u)) {
    const __fp16 *gamma = u < 2u * Q38_QH ? av_qg : av_kg;
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" : : "r"(Q38_HD));
    asm volatile("vle16.v v28, (%0)" : : "r"(gamma) : "memory");
  }
#endif
  (void)u;
}

static void avp_epi_row(const __fp16 *zu, uint32_t u, uint32_t row0,
                        uint32_t engine, uint32_t slot, uint32_t row) {
  avp_gamma(u);
  avp_epi_row_pipe(zu, u, row0, engine, slot, row, 0u, ~0u);
}

static NOSTACK void avp_epi_rows(const __fp16 *zu, uint32_t u, uint32_t row0,
                                 uint32_t engine, uint32_t slot, uint32_t first,
                                 uint32_t last) {
  // S2's next raw head is in v4..v7; v28..v31 retain gamma over this range.
  avp_gamma(u);
#pragma clang loop unroll_count(Q38_AV_ROW_UNROLL)
  for (uint32_t row = first; row < last; ++row)
    avp_epi_row_pipe(zu, u, row0, engine, slot, row,
                     row != first && !av_copy_unit(u),
                     row + 1u < last && !av_copy_unit(u) ? row + 1u : ~0u);
}
#if Q38_AV_DUAL
#if Q38_AV_STATIC_CORES
// A scalar agent (Snitch only, the 1 + 3 tile) has no FPU or vector unit: it
// takes no rows and only publishes; the vector cores split them all.
#define AVP_AGENT_ROWS (NUM_SCALAR_CORES_PER_TILE == 0)
_Static_assert(AVP_AGENT_ROWS || Q38_AV_STATIC_CORES >= 2u,
               "A scalar agent needs vector epilogue cores");
static void avp_range(uint32_t u, uint32_t core, uint32_t *first,
                      uint32_t *last) {
  const uint32_t divisor =
      Q38_AV_STATIC_CORES == 2u ? (av.kind || av_copy_unit(u) ? 4u : 3u) : 8u;
  const uint32_t agent =
      AVP_AGENT_ROWS ? Q38_MAX(1u, Q38_PROJECTION_ROWS / divisor) : 0u;
  const uint32_t rest = Q38_PROJECTION_ROWS - agent;
  *first = core ? agent + rest * (core - 1u) / (Q38_AV_STATIC_CORES - 1u) : 0u;
  *last = core ? agent + rest * core / (Q38_AV_STATIC_CORES - 1u) : agent;
}

static void avp_publish(uint32_t engine, uint32_t slot, uint32_t core,
                        const __fp16 *zu, uint32_t u, uint32_t row0,
                        uint32_t last) {
  const __fp16 *p =
      av.kind ? av.res + (row0 + last - 1u) * Q38_D + u * NT
      : av_copy_unit(av_head_unit(u)) || av_head_unit(u) < 2u * Q38_QH
          ? (const __fp16 *)av_stage_at(engine, slot, (last - 1u) * Q38_HD * 2u)
          : (const __fp16 *)av_stage_at(engine, slot, (last - 1u) * 2u);
  (void)zu;
  // Scalar fences alone do not drain pending vector stores.
  if (AVP_AGENT_ROWS || core)
    q38_drain16(p, 1u);
  __sync_synchronize();
  __atomic_fetch_or(&av_progress[engine].completed[slot], 1u << core,
                    __ATOMIC_RELEASE);
}
#endif

static void avp_wait_unit(uint32_t engine, uint32_t value) {
  while (__atomic_load_n(&av_progress[engine].unit_done, __ATOMIC_ACQUIRE) <
         value) {
    // Once the final panel is armed, its event can precede the producer's
    // software publish by a row kernel. Poll the tile-local word through it.
    if (__atomic_load_n(&av_progress[engine].unit_armed, __ATOMIC_ACQUIRE) >=
        value) {
      mempool_wait(16u);
    } else
      vc_sleep(5u, value);
  }
}

// Bounded cooperative work: the caller services DMA/RM between row kernels.
#if Q38_AV_STATIC_CORES
static uint32_t avp_service(q38_av_work_t *work, uint32_t compute) {
  if (work->unit >= av.unit_steps)
    return 0u;
  const uint32_t engine = mempool_get_redmule_id();
  const uint32_t unit = work->block * av.unit_steps + work->unit;
  uint32_t tile;
  const uint32_t u = av_unit_of(engine, work->unit, &tile);
  if (u != ~0u) {
    if (__atomic_load_n(&av_progress[engine].unit_done, __ATOMIC_ACQUIRE) <
        unit + 1u)
      return 0u;
    if (!av.kind && !(Q38_AV_DIRECT_DMA && av_copy_unit(av_head_unit(u))) &&
        unit >= 2u &&
        __atomic_load_n(&av_progress[engine].sent, __ATOMIC_ACQUIRE) <
            unit - 1u)
      return 0u;
    if (!(Q38_AV_DIRECT_DMA && !av.kind && av_copy_unit(av_head_unit(u)))) {
      uint32_t first, last;
      avp_range(av_head_unit(u), 0u, &first, &last);
      const __fp16 *const z =
          av.z + (engine % NUM_TILES_PER_GROUP * 2u + (unit & 1u)) *
                     av.upanels * Q38_AV_TILE;
      const uint32_t row0 = tile * Q38_PROJECTION_ROWS;
      if (work->row < last) {
        if (!compute)
          return 0u;
        const uint32_t start = mempool_get_timer();
        // Keep the controller responsive between individual rows. Its next
        // raw row and gamma remain live through scalar-only RM/DMA service.
        const uint32_t count = 1u;
        if (av.kind) {
#pragma clang loop unroll(full)
          for (uint32_t j = 0u; j < Q38_AV_ROW_UNROLL; ++j) {
            if (j >= count)
              break;
            const uint32_t row = work->row + j;
            llm_residual_add_f16(av.res + (row0 + row) * Q38_D + u * NT,
                                 Q38_VC_GLOBAL(z + row * NT), NT, 0u, 1u);
          }
        } else {
          if (work->row == 0u)
            avp_gamma(av_head_unit(u));
          avp_epi_row_pipe(
              z, av_head_unit(u), row0, engine, unit & 1u, work->row,
              work->row != 0u && !av_copy_unit(av_head_unit(u)),
              work->row + 1u < last && !av_copy_unit(av_head_unit(u))
                  ? work->row + 1u
                  : ~0u);
        }
        work->row += count;
        const uint32_t elapsed = mempool_get_timer() - start;
        work->elapsed += elapsed;
        work->worst = Q38_MAX(work->worst, elapsed);
#if Q38_AV_PROFILE
        av_agent_times[engine].rows[av_time_kind(u)] += count;
#endif
        return 1u;
      }
      const uint32_t done = __atomic_load_n(
          &av_progress[engine].completed[unit & 1u], __ATOMIC_ACQUIRE);
      if (!(done & 1u))
        avp_publish(engine, unit & 1u, 0u, z, u, row0, last);
      if ((done | 1u) != (1u << Q38_AV_STATIC_CORES) - 1u)
        return 0u;
    }
#if Q38_AV_PROFILE
    av_time_add(&av_agent_times[engine].unit[av_time_kind(u)], work->elapsed);
#endif
  }
  __atomic_store_n(&av_progress[engine].epi_done, unit + 1u, __ATOMIC_RELEASE);
  ++work->unit;
  work->row = work->elapsed = 0u;
  return 1u;
}
#else
static uint32_t avp_service(q38_av_work_t *work, uint32_t compute) {
  if (work->unit >= av.unit_steps)
    return 0u;
  const uint32_t engine = mempool_get_redmule_id();
  const uint32_t unit = work->block * av.unit_steps + work->unit;
  uint32_t tile;
  const uint32_t u = av_unit_of(engine, work->unit, &tile);
  if (u != ~0u) {
    if (__atomic_load_n(&av_progress[engine].unit_done, __ATOMIC_ACQUIRE) <
        unit + 1u)
      return 0u;
#if Q38_AV_DIRECT_DMA
    if (!av.kind && av_copy_unit(av_head_unit(u))) {
      __atomic_store_n(&av_progress[engine].epi_done, unit + 1u,
                       __ATOMIC_RELEASE);
      ++work->unit;
      work->row = work->elapsed = 0u;
      return 1u;
    }
#endif
    if (!av.kind && unit >= 2u &&
        __atomic_load_n(&av_progress[engine].sent, __ATOMIC_ACQUIRE) <
            unit - 1u)
      return 0u;
    if (compute && __atomic_load_n(&av_progress[engine].claimed[unit & 1u],
                                   __ATOMIC_ACQUIRE) < Q38_PROJECTION_ROWS) {
      const uint32_t row = __atomic_fetch_add(
          &av_progress[engine].claimed[unit & 1u], 1u, __ATOMIC_ACQ_REL);
      if (row >= Q38_PROJECTION_ROWS)
        return 0u;
      const uint32_t start = mempool_get_timer();
      const __fp16 *const z =
          av.z + engine % NUM_TILES_PER_GROUP * 2u * av.upanels * Q38_AV_TILE +
          (unit & 1u) * av.upanels * Q38_AV_TILE;
      const uint32_t row0 = tile * Q38_PROJECTION_ROWS;
      if (av.kind)
        llm_residual_add_f16(av.res + (row0 + row) * Q38_D + u * NT,
                             Q38_VC_GLOBAL(z + row * NT), NT, 0u, 1u);
      else
        avp_epi_row(z, av_head_unit(u), row0, engine, unit & 1u, row);
      __sync_synchronize();
      __atomic_fetch_add(&av_progress[engine].rows_done[unit & 1u], 1u,
                         __ATOMIC_ACQ_REL);
      const uint32_t elapsed = mempool_get_timer() - start;
      work->elapsed += elapsed;
      work->worst = Q38_MAX(work->worst, elapsed);
#if Q38_AV_PROFILE
      ++av_agent_times[engine].rows[av_time_kind(u)];
#endif
      return 1u;
    }
    if (__atomic_load_n(&av_progress[engine].rows_done[unit & 1u],
                        __ATOMIC_ACQUIRE) < Q38_PROJECTION_ROWS)
      return 0u;
    if (__atomic_load_n(&av_progress[engine].worker_done, __ATOMIC_ACQUIRE) <
        unit + 1u)
      return 0u;
#if Q38_AV_PROFILE
    av_time_add(&av_agent_times[engine].unit[av_time_kind(u)], work->elapsed);
#endif
  }
  __atomic_store_n(&av_progress[engine].epi_done, unit + 1u, __ATOMIC_RELEASE);
  ++work->unit;
  work->row = work->elapsed = 0u;
  return 1u;
}
#endif
#endif
#endif
