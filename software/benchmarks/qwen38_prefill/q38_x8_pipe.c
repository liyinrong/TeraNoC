// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Layer pipeline (8x8 P2, Q38_X8_PIPE): layer l runs the batch-major GDN layer
// on domain l % DOMAINS. Batch b of layer l starts once layer l - 1 has
// published (l - 1, b): its output rows are in L2. Included by
// layer_resident.c after x8_gdn_layer; not a translation unit.
//
// One prompt only. Streamed prompts (T2) need (1) a prompt epoch in the flag
// value, so that prompt p's flag does not release prompt p + 1, and (2) the
// producer to wait for the previous occupant's consumer flag (l_prev + 1, b)
// before it overwrites slot b of its region.
//
// The flags are initialized data in the program image, which every domain
// shares: core loads and stores are not folded per domain (DOMAIN_L2_HOME
// folds DMA addresses only), so all domains see one table.

#define X8_NOT_READY 0x5a5a5a5au
#define X8_READY(l, b) (0x60000000u | (l) << 12 | (b))
_Static_assert(X8_BATCHES < 4096u && Q38_X8_LAYERS < 4096u,
               "Ready values are distinct");
static volatile uint32_t x8_ready[Q38_X8_LAYERS][X8_BATCHES] = {
    [0 ... Q38_X8_LAYERS - 1] = {[0 ... X8_BATCHES - 1] = X8_NOT_READY}};

// Every core of the domain, after the rows of (l, b) have landed in L2 (the
// FFN's last down slice waits for its output DMAs before its final barrier;
// this barrier covers any other producer).
static NOSTACK void x8_publish(uint32_t l, uint32_t b) {
  sync_all();
  if (mempool_get_core_id() == 0u) {
    __sync_synchronize();
    x8_ready[l][b] = X8_READY(l, b);
    __sync_synchronize();
  }
}

// Every core of the domain: core 0 polls, the others wait in the barrier.
static NOSTACK void x8_wait(uint32_t l, uint32_t b) {
  if (mempool_get_core_id() == 0u) {
    while (x8_ready[l][b] != X8_READY(l, b))
      mempool_wait(64u);
    __sync_synchronize();
  }
  sync_all();
}

#if Q38_X8_SYNC
// Lock-step: every core of the chip meets on the fractal_sync tree's chip
// level. A domain barrier on the same id stops lower in the tree and matches
// its own response level (q38_barrier_n), and the leaf nodes hold one
// register per network, so no other id exists for it.
static NOSTACK void x8_chip_sync(void) {
#if DOMAINS > 1
  fsync_subscribe_self(0u);
  fsync_subscribe_self(FSYNC_EVT_FSYNC);
  __sync_synchronize();
  q38_barrier(FSYNC_ID_V, fsync_aggr_chip());
  __sync_synchronize();
#else
  sync_all();
#endif
}
// Domain d runs its k-th item, layer d + DOMAINS * (k / NB) at batch k % NB,
// in step k + d; the step barrier then publishes its rows, which the next
// layer reads in step k + d + 1, and orders every region reuse.
#define X8_ITEMS(d)                                                            \
  ((Q38_X8_LAYERS + DOMAINS - 1u - (d)) / DOMAINS * X8_BATCHES)
#define X8_STEPS (X8_ITEMS(0u) + DOMAINS - 1u)
// Layer l + 1 batch b runs one step after layer l batch b only if a domain's
// batches cover the other domains' steps: fewer batches read unwritten rows.
_Static_assert(X8_BATCHES >= DOMAINS,
               "Lock-step P2 needs X8_BATCHES >= DOMAINS (use flags mode)");
#endif

// Domain reports. Only domain 0 prints (serial.c), so every domain's core 0
// posts its numbers here, after a done word, and domain 0's core 0 prints
// them all.
// Debug (ring test, SYNC=1): every core's cycle at its chip-barrier release.
#ifndef Q38_X8_REL_STAMP
#define Q38_X8_REL_STAMP 0
#endif
#if Q38_X8_REL_STAMP
#define X8_REL_STEP 1u // the step whose per-group latest release is reported
#define X8_REP_WORDS (4u + 4u * X8_STEPS + NUM_GROUPS)
#else
#define X8_REP_WORDS (6u + 6u * X8_BATCHES)
#endif
#define X8_REP_DONE(d) X8_READY(0xfffu, (d))
static volatile uint32_t x8_rep[DOMAINS][1u + X8_REP_WORDS] = {
    [0 ... DOMAINS - 1] = {[0 ... X8_REP_WORDS] = X8_NOT_READY}};

// Copies n words (none: the caller wrote them), then sets the done word.
static void x8_rep_post(const uint32_t *w, uint32_t n) {
  const uint32_t d = mempool_get_domain_id();
  for (uint32_t i = 0u; i < n && i < X8_REP_WORDS; ++i)
    x8_rep[d][1u + i] = w[i];
  __sync_synchronize();
  x8_rep[d][0] = X8_REP_DONE(d);
  __sync_synchronize();
}

static const volatile uint32_t *x8_rep_get(uint32_t d) {
  while (x8_rep[d][0] != X8_REP_DONE(d))
    mempool_wait(256u);
  __sync_synchronize();
  return &x8_rep[d][1];
}

#if Q38_X8_PIPE_RELAY
// Rows [w0, w1) of the layer input over the output rows just written: the
// next layer then sees this layer's input. Through L1 (f2's residual), the
// qualified DMA path.
static NOSTACK void x8_relay(uint32_t w0, uint32_t w1) {
  f2_layout(0u);
  sync_all();
  transfer(f2.residual, input_init + w0 * Q38_D,
           (w1 - w0) * Q38_D * sizeof(__fp16));
  transfer(X8_OUT + w0 * Q38_D, f2.residual,
           (w1 - w0) * Q38_D * sizeof(__fp16));
}
#endif

// Core 0's stamps of the domain's last layer, per batch: wait for the input,
// in-proj, chunk core, out-proj, FFN, publish.
static uint32_t x8p_stamp[X8_BATCHES][6] Q38_VC_SMALL;
static uint32_t x8p_layer Q38_VC_SMALL, x8p_first Q38_VC_SMALL;
static uint32_t x8p_last Q38_VC_SMALL;
static uint32_t x8p_t0 Q38_VC_SMALL, x8p_t1 Q38_VC_SMALL;

static void x8p_set_io(uint32_t l) {
#if Q38_G2_ITEM_KEEP
  // Every group's coordinator resets once per layer, including a snapshot
  // whose first item has w0 != 0. The following input barrier publishes it.
  g2_item_begin();
#endif
  if (mempool_get_core_id() != 0u)
    return;
  const uint32_t dom = mempool_get_domain_id();
  x8_io.in = l ? X8_ACT_L2((l - 1u) % DOMAINS) : input_init;
  x8_io.mix = X8_MIX_L2(dom);
  // Home builds: the last layer's output stays home, like every handoff.
  x8_io.out = l + 1u < Q38_X8_LAYERS || Q38_L2_HOME_SLOTS ? X8_ACT_L2(dom)
                                                          : result_output;
}

#define X8P_STAMP(b, i, t)                                                     \
  do {                                                                         \
    const uint32_t now_ = mempool_get_timer();                                 \
    if (core == 0u)                                                            \
      x8p_stamp[b][i] = now_ - (t);                                            \
    (t) = now_;                                                                \
  } while (0)
// Debug stop after phase k of the first item.
#define X8P_STOP(k, b)                                                         \
  if (Q38_X8_STOP_PHASE == (k)) {                                              \
    if (core == 0u)                                                            \
      x8p_last = (b);                                                          \
    return;                                                                    \
  }

static NOSTACK void x8_pipe_layer(uint32_t l, uint32_t b0) {
  const uint32_t core = mempool_get_core_id();
  x8p_set_io(l);
  if (core == 0u)
    x8p_layer = l, x8p_first = b0, x8p_last = X8_BATCHES - 1u;
  uint32_t t = mempool_get_timer();
  if (l)
    x8_wait(l - 1u, b0);
  else
    sync_all(); // x8_io is published
  X8P_STAMP(b0, 0, t);
  if (!Q38_X8_START_FFN)
    gdn2_project(b0 * Q38_FFN_ROWS, (b0 + 1u) * Q38_FFN_ROWS);
  X8P_STAMP(b0, 1, t);
  X8P_STOP(1, b0);
#pragma clang loop unroll(disable)
  for (uint32_t b = b0; b < X8_BATCHES; ++b) {
    const uint32_t w0 = b * Q38_FFN_ROWS, w1 = w0 + Q38_FFN_ROWS;
    const uint32_t mixer = !Q38_X8_START_FFN || b != b0;
    if (mixer)
      gdn2_chunks(w0, w1);
    X8P_STAMP(b, 2, t);
    X8P_STOP(2, b);
    if (mixer)
      gdn2_output(w0, w1);
    X8P_STAMP(b, 3, t);
#if Q38_X8_WARM_CHUNKS
    // Debug (timing only): the next batch's chunk core right after this
    // out-projection, its in-projection and this FFN skipped, so a warm item
    // (resident data, imported state) is timed. Its stamps: chunks only.
    _Static_assert(Q38_X8_PIPE_SNAP && Q38_X8_STOP_PHASE == 3,
                   "Warm chunks: an A+B snapshot window");
    if (w1 < Q38_R) {
      if (core == 0u)
        for (uint32_t i = 0u; i < 6u; ++i)
          x8p_stamp[b + 1u][i] = 0u;
      gdn2_chunks(w1, w1 + Q38_FFN_ROWS);
      X8P_STAMP(b + 1u, 2, t);
    }
    if (core == 0u)
      x8p_last = w1 < Q38_R ? b + 1u : b;
    return;
#endif
    X8P_STOP(3, b);
    ffn_rows(w0, w1);
    X8P_STAMP(b, 4, t);
    X8P_STOP(4, b);
    if (l + 1u < Q38_X8_LAYERS) {
#if Q38_X8_PIPE_RELAY
      x8_relay(w0, w1);
#endif
      x8_publish(l, b);
    }
    X8P_STAMP(b, 5, t);
    if (w1 < Q38_R) {
      if (l)
        x8_wait(l - 1u, b + 1u);
      X8P_STAMP(b + 1u, 0, t);
      gdn2_project(w1, w1 + Q38_FFN_ROWS);
      X8P_STAMP(b + 1u, 1, t);
    }
  }
}

#if Q38_X8_SYNC
// Lock-step pipeline. Stamps per batch of the domain's last layer: in-proj,
// chunk core, out-proj, FFN, and the step barrier after the item (slot 5).
static NOSTACK void x8_pipe(void) {
  const uint32_t core = mempool_get_core_id(), dom = mempool_get_domain_id();
  const uint32_t items = X8_ITEMS(dom);
  if (core == 0u)
    x8p_t0 = mempool_get_timer(), x8p_first = 0u, x8p_last = X8_BATCHES - 1u;
  uint32_t t = mempool_get_timer();
#pragma clang loop unroll(disable)
  for (uint32_t step = 0u; step < X8_STEPS; ++step) {
    const uint32_t k = step - dom; // wraps below dom: no item
    if (step >= dom && k < items) {
      const uint32_t l = dom + DOMAINS * (k / X8_BATCHES), b = k % X8_BATCHES;
      const uint32_t w0 = b * Q38_FFN_ROWS, w1 = w0 + Q38_FFN_ROWS;
      if (b == 0u) {
        x8p_set_io(l);
        if (core == 0u)
          x8p_layer = l;
      }
      if (core == 0u)
        x8p_stamp[b][0] = 0u;
      gdn2_project(w0, w1);
      X8P_STAMP(b, 1, t);
      gdn2_chunks(w0, w1);
      X8P_STAMP(b, 2, t);
      gdn2_output(w0, w1);
      X8P_STAMP(b, 3, t);
      ffn_rows(w0, w1);
#if Q38_X8_PIPE_RELAY
      if (l + 1u < Q38_X8_LAYERS)
        x8_relay(w0, w1);
#endif
      X8P_STAMP(b, 4, t);
      x8_chip_sync();
      X8P_STAMP(b, 5, t);
    } else {
      x8_chip_sync();
      t = mempool_get_timer();
    }
  }
  if (core == 0u)
    x8p_t1 = mempool_get_timer();
}
#else
// Layers d, d + DOMAINS, ... on domain d.
static NOSTACK void x8_pipe(void) {
  const uint32_t dom = mempool_get_domain_id();
  if (mempool_get_core_id() == 0u)
    x8p_t0 = mempool_get_timer();
#pragma clang loop unroll(disable)
  for (uint32_t l = dom; l < Q38_X8_LAYERS; l += DOMAINS) {
#if Q38_X8_PIPE_SNAP
    // Steady state: layer l at batch NB-1-l, its input taken as published.
    _Static_assert(Q38_X8_LAYERS <= X8_BATCHES && Q38_X8_LAYERS <= DOMAINS,
                   "One snapshot layer per domain, one batch apart");
    const uint32_t b0 = X8_BATCHES - 1u - l;
    if (l && mempool_get_core_id() == 0u)
      x8_ready[l - 1u][b0] = X8_READY(l - 1u, b0);
#else
    const uint32_t b0 = 0u;
#endif
    x8_pipe_layer(l, b0);
  }
  if (mempool_get_core_id() == 0u)
    x8p_t1 = mempool_get_timer();
}
#endif

// Core 0 of every domain; domain 0 prints all domains. Written straight into
// the table (a 6 x 16-word copy would not fit the 512-byte stack).
static void x8_pipe_report(void) {
  const uint32_t dom = mempool_get_domain_id();
  volatile uint32_t *const r = &x8_rep[dom][1];
  r[0] = x8p_layer, r[1] = x8p_first, r[2] = x8p_last;
  r[3] = x8p_t0, r[4] = x8p_t1;
  for (uint32_t i = 0u; i < 6u * X8_BATCHES; ++i)
    r[5u + i] = ((const uint32_t *)x8p_stamp)[i];
  x8_rep_post(0, 0u);
  if (dom != 0u)
    return;
  for (uint32_t d = 0u; d < DOMAINS; ++d) {
    const volatile uint32_t *const q = x8_rep_get(d);
    printf("[x8_pipe] domain=%u layer=%u first=%u last=%u start=%u end=%u "
           "cycles=%u\n",
           d, q[0], q[1], q[2], q[3], q[4], q[4] - q[3]);
    for (uint32_t b = q[1]; b <= q[2] && b < X8_BATCHES; ++b) {
      const volatile uint32_t *const t = &q[5u + 6u * b];
      printf("[x8_pipe_batch] domain=%u layer=%u rows=%u wait=%u in_proj=%u "
             "chunks=%u out_proj=%u ffn=%u publish=%u\n",
             d, q[0], b * Q38_FFN_ROWS, t[0], t[1], t[2], t[3], t[4], t[5]);
    }
  }
}

#if Q38_X8_PIPE_TEST
// Handoff test, the P2 pattern without layers: "layer" l on domain
// l % DOMAINS waits on (l - 1, b), reads the rows layer l - 1 left in its
// domain's region, checks them, and publishes its own batch b in its domain's
// region. With Q38_X8_LAYERS = 2 x DOMAINS the ring wraps (layer 4 on domain
// 0 reads domain 3's region) and every region is reused one lap later, as in
// the driver. Layer l's batch b holds input rows (b + l) % T, so a read of a
// wrong, shared or stale buffer mismatches.
#ifndef Q38_X8_TEST_ROWS
#define Q38_X8_TEST_ROWS 64u
#endif
#ifndef Q38_X8_TEST_BATCHES
#define Q38_X8_TEST_BATCHES 4u
#endif
_Static_assert(Q38_X8_TEST_BATCHES <= X8_BATCHES,
               "The test uses flags (l, b) of the layer table");
#define X8T_ELEMS (Q38_X8_TEST_ROWS * Q38_D)
static __fp16 x8t_buf[X8T_ELEMS]
    __attribute__((section(".l1"), aligned(VC_GLOBAL_ROW_BYTES)));
static uint32_t x8t_bad Q38_VC_SMALL, x8t_seen Q38_VC_SMALL;
static uint32_t x8t_layers Q38_VC_SMALL;
// Domain d's handoff rows, below the L2 top (clear of the fixture).
#define X8T_L2(d)                                                              \
  ((__fp16 *)(L2_BASE + L2_SIZE -                                              \
              ((d) + 1u) * Q38_X8_TEST_BATCHES * X8T_ELEMS * sizeof(__fp16)))

#if Q38_X8_SYNC
// Core 0's cycle at its step-barrier arrival and release, per step.
_Static_assert(Q38_X8_TEST_BATCHES == X8_BATCHES &&
                   4u + 2u * X8_STEPS <= X8_REP_WORDS,
               "The lock-step test reports every step");
static uint32_t x8t_arr[X8_STEPS] Q38_VC_SMALL, x8t_rel[X8_STEPS] Q38_VC_SMALL;
#if Q38_X8_REL_STAMP
static uint32_t x8t_all[X8_STEPS][NUM_CORES] __attribute__((section(".l1")));
#endif
#endif

// One item of the test: check layer l - 1's batch b, write layer l's.
static NOSTACK void x8_test_item(uint32_t l, uint32_t b) {
  const uint32_t core = mempool_get_core_id(), dom = mempool_get_domain_id();
  const uint32_t T = Q38_X8_TEST_BATCHES, bytes = X8T_ELEMS * sizeof(__fp16);
  if (l) {
#if !Q38_X8_SYNC
    x8_wait(l - 1u, b);
#endif
    transfer(x8t_buf, X8T_L2((l - 1u) % DOMAINS) + b * X8T_ELEMS, bytes);
    sync_all();
    const volatile uint16_t *const got = (const volatile uint16_t *)x8t_buf;
    const volatile uint16_t *const ref =
        (const volatile uint16_t *)(input_init + (b + l - 1u) % T * X8T_ELEMS);
    uint32_t bad = 0u;
    for (uint32_t i = core; i < X8T_ELEMS; i += NUM_CORES)
      bad += got[i] != ref[i];
    __atomic_fetch_add(&x8t_bad, bad, __ATOMIC_RELAXED);
    __atomic_fetch_add(&x8t_seen, 1u, __ATOMIC_RELAXED);
    sync_all();
  }
  if (l + 1u < Q38_X8_LAYERS) {
    transfer(x8t_buf, input_init + (b + l) % T * X8T_ELEMS, bytes);
    transfer(X8T_L2(dom) + b * X8T_ELEMS, x8t_buf, bytes);
#if !Q38_X8_SYNC
    x8_publish(l, b);
#endif
  }
  if (core == 0u && b == T - 1u)
    ++x8t_layers;
}

static NOSTACK void x8_pipe_test(void) {
  const uint32_t core = mempool_get_core_id(), dom = mempool_get_domain_id();
  const uint32_t T = Q38_X8_TEST_BATCHES;
  if (core == 0u)
    x8t_bad = 0u, x8t_seen = 0u, x8t_layers = 0u;
  sync_all();
#if Q38_X8_SYNC
  const uint32_t items = X8_ITEMS(dom);
#pragma clang loop unroll(disable)
  for (uint32_t step = 0u; step < X8_STEPS; ++step) {
    const uint32_t k = step - dom;
    if (step >= dom && k < items)
      x8_test_item(dom + DOMAINS * (k / T), k % T);
    if (core == 0u)
      x8t_arr[step] = mempool_get_timer();
    x8_chip_sync();
#if Q38_X8_REL_STAMP
    x8t_all[step][core] = mempool_get_timer();
#endif
    if (core == 0u)
      x8t_rel[step] = mempool_get_timer();
  }
#else
#pragma clang loop unroll(disable)
  for (uint32_t l = dom; l < Q38_X8_LAYERS; l += DOMAINS) {
    for (uint32_t b = 0u; b < T; ++b) {
      x8_test_item(l, b);
    }
  }
#endif
  sync_all();
  if (core != 0u)
    return;
    // checked = core-batch checks of a domain (NUM_CORES per consumed batch).
#if Q38_X8_SYNC
  volatile uint32_t *const r = &x8_rep[dom][1];
  r[0] = x8t_layers, r[1] = x8t_seen, r[2] = x8t_bad, r[3] = X8_STEPS;
  for (uint32_t i = 0u; i < X8_STEPS; ++i)
    r[4u + i] = x8t_arr[i], r[4u + X8_STEPS + i] = x8t_rel[i];
#if Q38_X8_REL_STAMP
  // Per step: the domain's earliest and latest core release; per group, the
  // latest at step X8_REL_STEP.
  for (uint32_t i = 0u; i < X8_STEPS; ++i) {
    uint32_t lo = ~0u, hi = 0u;
    for (uint32_t c = 0u; c < NUM_CORES; ++c) {
      const uint32_t t = x8t_all[i][c];
      lo = t < lo ? t : lo, hi = t > hi ? t : hi;
    }
    r[4u + 2u * X8_STEPS + i] = lo, r[4u + 3u * X8_STEPS + i] = hi;
  }
  for (uint32_t g = 0u; g < NUM_GROUPS; ++g) {
    uint32_t hi = 0u;
    for (uint32_t c = g * NUM_CORES_PER_GROUP;
         c < (g + 1u) * NUM_CORES_PER_GROUP; ++c)
      hi = x8t_all[X8_REL_STEP][c] > hi ? x8t_all[X8_REL_STEP][c] : hi;
    r[4u + 4u * X8_STEPS + g] = hi;
  }
#endif
  x8_rep_post(0, 0u);
#else
  const uint32_t w[3] = {x8t_layers, x8t_seen, x8t_bad};
  x8_rep_post(w, 3u);
#endif
  if (dom != 0u)
    return;
  uint32_t fail = 0u;
  for (uint32_t d = 0u; d < DOMAINS; ++d) {
    const volatile uint32_t *const r = x8_rep_get(d);
    const uint32_t want =
        (Q38_X8_LAYERS - d + DOMAINS - 1u) / DOMAINS * T * NUM_CORES -
        (d == 0u ? T * NUM_CORES : 0u);
    fail |= r[2] != 0u || r[1] != want;
    printf("[x8_test] domain=%u layers=%u batches=%u rows=%u checked=%u "
           "(expect %u) bad=%u\n",
           d, r[0], T, Q38_X8_TEST_ROWS, r[1], want, r[2]);
  }
#if Q38_X8_SYNC
  // Step barrier cost: release (domain 0) minus the last arrival.
  for (uint32_t i = 0u; i < X8_STEPS; ++i) {
    uint32_t last = 0u, first = ~0u;
    for (uint32_t d = 0u; d < DOMAINS; ++d) {
      const uint32_t a = x8_rep_get(d)[4u + i];
      last = a > last ? a : last, first = a < first ? a : first;
    }
    const uint32_t rel = x8_rep_get(0u)[4u + X8_STEPS + i];
    printf("[x8_step] step=%u first_arrival=%u last_arrival=%u release=%u "
           "barrier=%u skew=%u\n",
           i, first, last, rel, rel - last, last - first);
  }
#if Q38_X8_REL_STAMP
  // Core release spread: chip earliest / latest, and per domain.
  for (uint32_t i = 0u; i < X8_STEPS; ++i) {
    uint32_t lo = ~0u, hi = 0u;
    for (uint32_t d = 0u; d < DOMAINS; ++d) {
      const volatile uint32_t *const q = x8_rep_get(d);
      const uint32_t a = q[4u + 2u * X8_STEPS + i],
                     z = q[4u + 3u * X8_STEPS + i];
      lo = a < lo ? a : lo, hi = z > hi ? z : hi;
    }
    printf("[x8_rel] step=%u earliest=%u latest=%u spread=%u", i, lo, hi,
           hi - lo);
    for (uint32_t d = 0u; d < DOMAINS; ++d) {
      const volatile uint32_t *const q = x8_rep_get(d);
      printf(" d%u=%u..%u", d, q[4u + 2u * X8_STEPS + i] - lo,
             q[4u + 3u * X8_STEPS + i] - lo);
    }
    printf("\n");
  }
  for (uint32_t d = 0u; d < DOMAINS; ++d) {
    const volatile uint32_t *const q = x8_rep_get(d);
    printf("[x8_rel_group] step=%u domain=%u latest_minus_earliest_core:",
           X8_REL_STEP, d);
    for (uint32_t g = 0u; g < NUM_GROUPS; ++g)
      printf(" g%u=%u", g,
             q[4u + 4u * X8_STEPS + g] - q[4u + 2u * X8_STEPS + X8_REL_STEP]);
    printf("\n");
  }
#endif
#endif
  printf("[x8_test] result=%s\n", fail ? "FAIL" : "PASS");
}
#endif
