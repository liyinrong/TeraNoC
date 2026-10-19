// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Exact domain fold (DOMAIN_L2_EXACT): a descriptor in domain addresses is
// issued so that every byte lands at dma_domain_addr(byte). One the fold maps
// as a whole -- no burst crosses a domain row, one constant row carry, no home
// wrap crossed -- goes out unchanged; any other is split by stride residue
// class or by run into row-local pieces, each again exact. Needs dma_fold.h
// and DMA_FOLD_EMIT(stream, d, s, len, d2, s2, r2, d3, s3, r3, dims), which
// issues one descriptor and returns its id. Integer only.

#ifndef _DMA_FOLD_EXACT_H_
#define _DMA_FOLD_EXACT_H_

#define DMA_FOLD_ROW DMA_DOMAIN_ROW_BYTES
// Domain-offset period of the home wrap: a whole number of rows.
#define DMA_FOLD_WRAPO (DMA_DOMAIN_HOME_WRAP / DOMAINS)
// Past this many residue classes, one descriptor per row-local piece.
#ifndef DMA_FOLD_MAX_CLASSES
#define DMA_FOLD_MAX_CLASSES 16u
#endif
extern uint32_t volatile dma_domain_splits;

// Side kinds: 0 untranslated, 1 domain L1, 2 home L2.
static inline uint32_t dma_fold_kind(uint32_t a) {
  return dma_domain_home(a) ? 2u : dma_domain_l1(a) ? 1u : 0u;
}

// Whether n bursts of len bytes at domain offsets o, o + t, ... (t modulo
// 2^32; the row is a power of two) stay inside rows with one row carry.
static inline uint32_t dma_fold_seq(uint32_t o, uint32_t t, uint32_t n,
                                    uint32_t len) {
  const uint32_t q = o % DMA_FOLD_ROW, b = t % DMA_FOLD_ROW;
  if (len > DMA_FOLD_ROW || q + len > DMA_FOLD_ROW)
    return 0u;
  if (n <= 1u || b == 0u)
    return 1u;
  if (n - 1u > DMA_FOLD_ROW)
    return 0u;
  return q + (n - 1u) * b + len <= DMA_FOLD_ROW ||
         (n - 1u) * (DMA_FOLD_ROW - b) <= q;
}

// Whether a home-side walk stays in one wrap period. Its burst offsets are
// affine in (plane, row), so the four corner bursts bound them.
static inline uint32_t dma_fold_in_wrap(uint32_t x, uint32_t len,
                                        uint32_t seg) {
  return x / DMA_FOLD_WRAPO == seg && (x + len - 1u) / DMA_FOLD_WRAPO == seg;
}
static inline uint32_t dma_fold_nowrap(uint32_t o, uint32_t len, uint32_t t2,
                                       uint32_t r2, uint32_t p, uint32_t r3) {
  const uint32_t seg = o / DMA_FOLD_WRAPO;
  const uint32_t a = (r2 - 1u) * t2, c = (r3 - 1u) * p;
  return dma_fold_in_wrap(o, len, seg) && dma_fold_in_wrap(o + a, len, seg) &&
         dma_fold_in_wrap(o + c, len, seg) &&
         dma_fold_in_wrap(o + a + c, len, seg);
}

// Whether one side of a walk (r3 planes of r2 bursts, row stride t2, stride
// t3 from a plane's last burst) maps exactly as one descriptor. Planes repeat
// one row pattern only when the plane step is a whole number of rows.
static inline uint32_t dma_fold_side_exact(uint32_t a, uint32_t len,
                                           uint32_t t2, uint32_t r2,
                                           uint32_t t3, uint32_t r3) {
  const uint32_t k = dma_fold_kind(a);
  if (!k)
    return 1u;
  const uint32_t o = dma_domain_off(a), p = (r2 - 1u) * t2 + t3;
  if (!dma_fold_seq(o, t2, r2, len) || (r3 > 1u && p % DMA_FOLD_ROW))
    return 0u;
  return k == 1u || dma_fold_nowrap(o, len, t2, r2, p, r3);
}

// A class walk (n bursts at stride t) is exact once cut at rows if t is whole
// rows, or uncut if its bursts keep one row carry, unless a home side crosses
// its wrap.
static inline uint32_t dma_fold_class_ok(uint32_t a, uint32_t t, uint32_t n,
                                         uint32_t len) {
  const uint32_t k = dma_fold_kind(a);
  if (!k || n <= 1u)
    return 1u;
  const uint32_t o = dma_domain_off(a);
  if (t % DMA_FOLD_ROW && !dma_fold_seq(o, t, n, len))
    return 0u;
  return k == 1u || dma_fold_nowrap(o, len, t, n, 0u, 1u);
}

// Residue classes of a folded side's row stride: the bursts j, j + m, ...
// share their offset inside a row (the row is a power of two).
static inline uint32_t dma_fold_period(uint32_t a, uint32_t t) {
  const uint32_t b = t % DMA_FOLD_ROW;
  return dma_fold_kind(a) && b ? DMA_FOLD_ROW / (b & (0u - b)) : 1u;
}

// How many more bursts after the one at a (stride t) keep its row carry:
// all stay in rows without an extra carry, or all take one. ~0 when the side
// is not folded or the stride is whole rows; 0 when the burst crosses a row.
static inline uint32_t dma_fold_run(uint32_t a, uint32_t t, uint32_t len) {
  if (!dma_fold_kind(a))
    return ~0u;
  const uint32_t q = dma_domain_off(a) % DMA_FOLD_ROW, b = t % DMA_FOLD_ROW;
  if (len > DMA_FOLD_ROW || q + len > DMA_FOLD_ROW)
    return 0u;
  if (!b)
    return ~0u;
  const uint32_t k1 = (DMA_FOLD_ROW - len - q) / b;
  const uint32_t k2 = q / (DMA_FOLD_ROW - b);
  return k1 > k2 ? k1 : k2;
}

// Whether a side's planes all repeat plane 0's row pattern (plane step a
// whole number of rows) without crossing the home wrap.
static inline uint32_t dma_fold_planes_ok(uint32_t a, uint32_t len, uint32_t t2,
                                          uint32_t r2, uint32_t p,
                                          uint32_t r3) {
  const uint32_t k = dma_fold_kind(a);
  if (!k)
    return 1u;
  if (p % DMA_FOLD_ROW)
    return 0u;
  return k == 1u || dma_fold_nowrap(dma_domain_off(a), len, t2, r2, p, r3);
}

// One descriptor's arguments (strides and reps normalized to its dims), the
// pieces issued so far, and (DMA_DOMAIN_CHECK) the caller's return address.
typedef struct {
  uint32_t stream, d, s, len, d2, s2, r2, d3, s3, r3, n, dims, ra;
} dma_fold_t;

// One walk to issue: planes of n bursts at stride td/ts, plane steps pd/ps
// (first burst to first burst).
typedef struct {
  uint32_t stream, len, pd, ps, planes;
} dma_fold_shape_t;

// The bursts at d, s (n of them, strides td, ts, in every plane) cut at each
// row boundary of a folded side; each piece is one descriptor. Counts the
// pieces, and issues them when id is non-null (the last one's id there).
static inline __attribute__((always_inline)) uint32_t
dma_fold_pieces(const dma_fold_shape_t *f, uint32_t d, uint32_t s, uint32_t td,
                uint32_t ts, uint32_t n, dma_id_t *id) {
  uint32_t nd =
      dma_fold_kind(d) ? DMA_FOLD_ROW - dma_domain_off(d) % DMA_FOLD_ROW : ~0u;
  uint32_t ns =
      dma_fold_kind(s) ? DMA_FOLD_ROW - dma_domain_off(s) % DMA_FOLD_ROW : ~0u;
  uint32_t pieces = 0u;
  for (uint32_t k = 0u; k < f->len; ++pieces) {
    uint32_t e = f->len;
    e = nd < e ? nd : e;
    e = ns < e ? ns : e;
    if (id) {
      if (f->planes > 1u && n > 1u)
        *id = DMA_FOLD_EMIT(f->stream, d + k, s + k, e - k, td, ts, n,
                            f->pd - (n - 1u) * td, f->ps - (n - 1u) * ts,
                            f->planes, 3u);
      else if (f->planes > 1u)
        *id = DMA_FOLD_EMIT(f->stream, d + k, s + k, e - k, f->pd, f->ps,
                            f->planes, 0u, 0u, 1u, 2u);
      else
        *id = DMA_FOLD_EMIT(f->stream, d + k, s + k, e - k, td, ts, n, 0u, 0u,
                            1u, n > 1u ? 2u : 1u);
    }
    nd += e == nd ? DMA_FOLD_ROW : 0u;
    ns += e == ns ? DMA_FOLD_ROW : 0u;
    k = e;
  }
  return pieces;
}

// One plane by residue classes: every m-th burst shares its in-row offset on
// a side whose period is m, so a class is cut at that side's rows only; the
// other side must keep one row carry along the class.
static inline __attribute__((always_inline)) uint32_t
dma_fold_classes(const dma_fold_shape_t *f, uint32_t d, uint32_t s, uint32_t d2,
                 uint32_t s2, uint32_t r2, uint32_t m, dma_id_t *id) {
  uint32_t pieces = 0u;
  m = m < r2 ? m : r2;
  for (uint32_t j = 0u; j < m; ++j) {
    const uint32_t n = (r2 - j + m - 1u) / m;
    const uint32_t dj = d + j * d2, sj = s + j * s2;
    if (m <= DMA_FOLD_MAX_CLASSES && dma_fold_class_ok(dj, m * d2, n, f->len) &&
        dma_fold_class_ok(sj, m * s2, n, f->len))
      pieces += dma_fold_pieces(f, dj, sj, m * d2, m * s2, n, id);
    else
      for (uint32_t i = 0u; i < n; ++i)
        pieces += dma_fold_pieces(f, dj + i * m * d2, sj + i * m * s2, 0u, 0u,
                                  1u, id);
  }
  return pieces;
}

// One plane by runs: consecutive bursts with one row carry on every folded
// side; a burst that crosses a row is cut on its own.
static inline __attribute__((always_inline)) uint32_t
dma_fold_runs(const dma_fold_shape_t *f, uint32_t d, uint32_t s, uint32_t d2,
              uint32_t s2, uint32_t r2, dma_id_t *id) {
  uint32_t pieces = 0u;
  for (uint32_t i = 0u; i < r2;) {
    const uint32_t di = d + i * d2, si = s + i * s2;
    uint32_t e = dma_fold_run(di, d2, f->len);
    const uint32_t es = dma_fold_run(si, s2, f->len);
    e = es < e ? es : e;
    e = r2 - 1u - i < e ? r2 - 1u - i : e;
    // Unmerged planes: a home piece must also stay in one wrap period.
    if (f->planes == 1u && e &&
        ((dma_fold_kind(di) == 2u &&
          !dma_fold_nowrap(dma_domain_off(di), f->len, d2, e + 1u, 0u, 1u)) ||
         (dma_fold_kind(si) == 2u &&
          !dma_fold_nowrap(dma_domain_off(si), f->len, s2, e + 1u, 0u, 1u))))
      e = 0u;
    pieces += dma_fold_pieces(f, di, si, d2, s2, e + 1u, id);
    i += e + 1u;
  }
  return pieces;
}

// A descriptor the fold cannot map as a whole: row-local pieces, by residue
// classes (of either side's period) or by runs, whichever plane 0 needs
// fewest of. When every folded
// side's plane step is whole rows each piece covers all planes, else the
// planes are walked one by one. Returns the last piece's id (one stream, in
// order).
static __attribute__((noinline)) dma_id_t dma_fold_walk(dma_fold_t *w) {
  const uint32_t d = w->d, s = w->s, d2 = w->d2, s2 = w->s2, r2 = w->r2;
  const uint32_t r3 = w->r3;
  dma_fold_shape_t f = {w->stream, w->len, (r2 - 1u) * d2 + w->d3,
                        (r2 - 1u) * s2 + w->s3, 1u};
  const uint32_t merged = r3 > 1u &&
                          dma_fold_planes_ok(d, f.len, d2, r2, f.pd, r3) &&
                          dma_fold_planes_ok(s, f.len, s2, r2, f.ps, r3);
  if (merged)
    f.planes = r3;
  // m = 0 walks by runs, else by classes of either side's period.
  const uint32_t md = dma_fold_period(d, d2), ms = dma_fold_period(s, s2);
  uint32_t m = 0u, best = dma_fold_runs(&f, d, s, d2, s2, r2, 0);
  uint32_t c = dma_fold_classes(&f, d, s, d2, s2, r2, md, 0);
  if (c < best)
    best = c, m = md;
  c = ms != md ? dma_fold_classes(&f, d, s, d2, s2, r2, ms, 0) : ~0u;
  if (c < best)
    m = ms;
  dma_id_t id = 0;
  uint32_t pieces = 0u;
  for (uint32_t p = 0u; p < (merged ? 1u : r3); ++p) {
    const uint32_t dp = d + p * f.pd, sp = s + p * f.ps;
    pieces += m ? dma_fold_classes(&f, dp, sp, d2, s2, r2, m, &id)
                : dma_fold_runs(&f, dp, sp, d2, s2, r2, &id);
  }
  w->n = pieces;
  __atomic_fetch_add(&dma_domain_splits, pieces - 1u, __ATOMIC_RELAXED);
  return id;
}

#endif // _DMA_FOLD_EXACT_H_
