#!/usr/bin/env python3

# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Exhaustive self-check of the WP3 virtual-cluster permuter reference
# (scripts/vc/vc_permute.py).
#
# It answers the acceptance questions of the Track A task:
#   1. BIJECTION      - every VC-relative offset of a level maps to a distinct
#                       physical word, and the rectangles of one level tile the
#                       whole L1 exactly once.
#   2. ALIAS EQUIV.   - the bit-OR permuter agrees with an independent
#                       arithmetic (+, *, //) reconstruction of the same
#                       address, and with the ND-strided descriptor a DMA
#                       would program.  This is the oracle the RTL gate of
#                       docs/plan/wp2-wp4-implementation-roadmap.md section
#                       T2.3 item 1 is later checked against.
#   3. VIOLATIONS     - a flat group mask (invariant #4) and a misaligned
#                       chunk_base (invariant #5) both produce a DETECTED
#                       collision, with the failing pair printed.
#   4. CONFINEMENT    - a VC-local sweep emits ZERO physical addresses outside
#                       the SxS bounding box of the requester's rectangle
#                       (the property section T2.3 item 3 asserts with SVA).
#   5. LEGALITY       - illegal descriptors are rejected, not silently mapped.
#
# Sweep granularity, stated so the claims are not overread:
#   - A geometry whose whole-L1 pass fits BYTE_SWEEP_BUDGET is swept at BYTE
#     granularity: every byte offset of every rectangle of every level.
#   - A larger geometry is swept at WORD granularity by default, and the byte
#     lane is covered by `check_byte_lane_identity`, which proves exhaustively
#     over lanes that the low ByteOffset bits are a pass-through.
#     Word-exhaustive plus that lemma is equivalent to byte-exhaustive.
#   - `--byte-exhaustive` forces the byte sweep anyway, at 4x the probes and an
#     8-byte-per-byte collision map (537 MB/level on the 8x8).  It has been run
#     clean on monsterpool8x8: 4 levels x 67108864 byte probes, 233 s, so that
#     geometry's byte-level result does not depend on the lemma either.
#   The choice made for each geometry is printed in the report.
#
# Usage:
#   python3 scripts/vc/vc_selfcheck.py                # terapool + 8x8
#   python3 scripts/vc/vc_selfcheck.py --all          # every fixture
#   python3 scripts/vc/vc_selfcheck.py --quick        # sampled, for iteration
#   python3 scripts/vc/vc_selfcheck.py -g terapool

import argparse
import os
import sys
import time
from array import array
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vc_permute import (  # noqa: E402
    GEOMETRIES, Geometry, LevelRule, MODE_FLAT_MASK_AXIS,
    MODE_FLAT_MASK_PACKED, MODE_NO_MASK, MODE_SPEC, SpecViolation, clog2,
)

# Odd multiplier for every low-discrepancy offset walk here and in
# vc_golden.py.  With a power-of-two space N, i -> (i*STRIDE) mod N is a
# bijection on [0, N), so the walk never repeats and touches every bit
# position.  Being a plain multiply-and-mask, an RTL testbench or a C model
# can regenerate the same offsets without Python.
STRIDE = 0x9E3779B1

# Whole-L1 passes cheaper than this many probes run at byte granularity.
BYTE_SWEEP_BUDGET = 1 << 23
# Requester invariance is exhaustive below this many probes, sampled above.
INVARIANCE_EXHAUSTIVE_BUDGET = 1 << 24
INVARIANCE_SAMPLE = 1 << 13


class Report:
    """Collects PASS/FAIL lines and the evidence behind them."""

    def __init__(self, verbose: bool = False):
        self.rows: List[Tuple[str, bool, str]] = []
        self.verbose = verbose

    def add(self, name: str, ok: bool, detail: str = '') -> bool:
        self.rows.append((name, ok, detail))
        print('  [%s] %-56s %s' % ('PASS' if ok else 'FAIL', name, detail))
        return ok

    def info(self, text: str) -> None:
        print('  [info] %s' % text)

    @property
    def failed(self) -> List[Tuple[str, bool, str]]:
        return [r for r in self.rows if not r[1]]

    def summary(self) -> int:
        bad = self.failed
        print()
        print('%d checks, %d failed' % (len(self.rows), len(bad)))
        for name, _, detail in bad:
            print('  FAIL %s  %s' % (name, detail))
        return 1 if bad else 0


# ---------------------------------------------------------------------------
# Offset sets
# ---------------------------------------------------------------------------
def stride_walk(space: int, count: int) -> Iterable[int]:
    """`count` distinct values of [0, space), spread by an odd multiplier."""
    for i in range(min(count, space)):
        yield (i * STRIDE) % space


def boundary_offsets(rule: LevelRule) -> List[int]:
    """Every field edge of a level's offset space.

    Included: the first and last offset; for every field, the offset that
    sets only that field's low bit, only its high bits, and the carry point
    just below each; and the first and last offset of chunk-row planes
    {0, 1, 2, chunk_rows/2, chunk_rows-2, chunk_rows-1}.
    """
    g = rule.geom
    n = rule.offset_bytes
    out = {0, n - 1}

    fields = (
        (0, g.byte_bits),
        (rule.o_bank_pos, g.bank_bits),
        (rule.o_tile_pos, g.tile_bits),
        (rule.o_vcgy_pos, rule.k),
        (rule.o_vcgx_pos, rule.k),
        (rule.o_row_pos, rule.row_bits),
    )
    for pos, width in fields:
        if width == 0:
            continue
        lo = 1 << pos
        hi = ((1 << width) - 1) << pos
        for v in (lo, hi, hi | (lo - 1), lo - 1, hi + lo - 1):
            out.add(v & (n - 1))

    plane = 1 << rule.o_row_pos
    planes = {0, 1, 2, rule.chunk_rows // 2, rule.chunk_rows - 2,
              rule.chunk_rows - 1}
    for c in sorted(planes):
        if 0 <= c < rule.chunk_rows:
            out.add(c * plane)
            out.add(c * plane + plane - 1)
    return sorted(out)


def probe_offsets(rule: LevelRule, count: int) -> List[int]:
    """Boundary set plus a stride walk, deduplicated and sorted."""
    seen = set(boundary_offsets(rule))
    seen.update(stride_walk(rule.offset_bytes, count))
    return sorted(seen)


def group_field_offsets(rule: LevelRule) -> List[int]:
    """Offsets covering EVERY (vc_gx, vc_gy) pair of a level.

    The requester's coordinates interact with the offset only through those
    two fields, so crossing all of them with a handful of patterns in the
    remaining fields makes the requester-invariance check exhaustive in the
    dimension that matters even when the full aperture is too large to sweep.
    """
    span = ((1 << (2 * rule.k)) - 1) << rule.o_vcgy_pos
    other_mask = (rule.offset_bytes - 1) & ~span
    others = {0, other_mask}
    others.update(v & other_mask
                  for v in stride_walk(rule.offset_bytes, 8))
    out = []
    for base in sorted(others):
        for gxv in range(rule.side):
            for gyv in range(rule.side):
                out.append(base | (gxv << rule.o_vcgx_pos)
                           | (gyv << rule.o_vcgy_pos))
    return out


def all_requesters(geom: Geometry) -> List[Tuple[int, int]]:
    return [(gx, gy) for gx in range(geom.num_x) for gy in range(geom.num_y)]


def rectangle_corners(geom: Geometry, k: int) -> List[Tuple[int, int]]:
    """One representative requester per rectangle: the rectangle's corner."""
    s = 1 << k
    return [(gx, gy)
            for gx in range(0, geom.num_x, s)
            for gy in range(0, geom.num_y, s)]


def chunk_configs(geom: Geometry, k: int) -> List[Tuple[int, int]]:
    """Legal (chunk_base, chunk_rows) pairs worth probing at a level."""
    w = geom.words_per_bank
    out = [(0, w)]
    quarter = w >> 2
    if quarter >= 1:
        out += [(0, quarter), (quarter, quarter), (w - quarter, quarter)]
    return out


# ---------------------------------------------------------------------------
# Structural / spec-fixture checks
# ---------------------------------------------------------------------------
def check_layout_fixture(rep: Report) -> None:
    """terapool must reproduce the roadmap's canonical layout exactly.

    docs/plan/wp2-wp4-implementation-roadmap.md section "Canonical layout":
    byte[1:0], bank[5:2], tile[9:6], gy[11:10], gx[13:12], word above.
    """
    g = GEOMETRIES['terapool']
    got = {n: (hi, lo) for n, hi, lo in g.describe_fields()}
    want = {'byte': (1, 0), 'bank': (5, 2), 'tile': (9, 6), 'gy': (11, 10),
            'gx': (13, 12)}
    rep.add('canonical layout matches roadmap (terapool)',
            all(got[k] == v for k, v in want.items()),
            ' '.join('%s[%d:%d]' % (n, hi, lo)
                     for n, hi, lo in g.describe_fields()))

    word_hi, word_lo = got['word']
    rep.add('word-in-bank is TCDMAddrMemWidth wide (roadmap says [31:14])',
            (word_lo, word_hi - word_lo + 1) == (14, clog2(g.words_per_bank)),
            'word[%d:%d] = %d bits, not 18'
            % (word_hi, word_lo, word_hi - word_lo + 1))


def check_doc_stride_example(rep: Report) -> None:
    """Reproduce the worked example in the design rationale.

    docs/plan/wp2-wp4-design-rationale.md section "VC address model",
    subsection "Cross-VC L1<->L1 DMA addressing": a 2x2 terapool VC is groups
    {0,1,4,5}, its global image is the runs [0x0000..0x07FC] and
    [0x1000..0x17FC], the x-group stride is 0x1000, the word-plane stride
    0x4000.
    """
    g = GEOMETRIES['terapool']
    rule = LevelRule(g, 1)
    dims = rule.nd_descriptor(0, 0)['dims']
    rep.add('rationale 2x2 terapool stride example',
            dims[0]['count'] == 0x800 and dims[1]['stride'] == 0x1000
            and dims[1]['count'] == 2 and dims[2]['stride'] == 0x4000,
            'run=0x%x xstride=0x%x wstride=0x%x'
            % (dims[0]['count'], dims[1]['stride'], dims[2]['stride']))

    ids = sorted(g.group_id(gx, gy) for gx in (0, 1) for gy in (0, 1))
    rep.add('rationale 2x2 group ids are {0,1,4,5} (gy low)',
            ids == [0, 1, 4, 5], str(ids))

    # VC offsets 0x000..0x7FC are the first run; the next VC offset, 0x800,
    # is the first word of the second run because it increments vc_gx.
    got = [rule.permute_addr(o, 0, 0) for o in (0x0, 0x7FC, 0x800, 0xFFC)]
    rep.add('rationale run boundaries 0x0..0x7FC then 0x1000..0x17FC',
            got == [0x0, 0x7FC, 0x1000, 0x17FC],
            ' '.join('0x%x' % v for v in got))


def check_split_agrees(rep: Report, geom: Geometry) -> None:
    """`split`, `permute`, `permute_addr` and `decode` describe one map."""
    bad = None
    n = 0
    for k in range(geom.max_level + 1):
        rule = LevelRule(geom, k)
        for gx, gy in rectangle_corners(geom, k)[:4]:
            for off in probe_offsets(rule, 4096):
                f = rule.split(off)
                p = rule.permute(off, gx, gy)
                a = rule.permute_addr(off, gx, gy)
                n += 1
                if (f['bank'], f['tile'], f['byte']) != (p.bank, p.tile,
                                                         p.byte):
                    bad = ('split vs permute', k, gx, gy, hex(off))
                elif geom.compose(p.word, p.gx, p.gy, p.tile, p.bank,
                                  p.byte) != a:
                    bad = ('permute vs permute_addr', k, gx, gy, hex(off))
                elif geom.decode(a) != p:
                    bad = ('decode vs permute', k, gx, gy, hex(off))
                if bad:
                    break
            if bad:
                break
        if bad:
            break
    rep.add('split / permute / permute_addr / decode agree', bad is None,
            'n=%d %s' % (n, '' if bad is None else str(bad)))


def check_byte_lane_identity(rep: Report, geom: Geometry) -> None:
    """The low ByteOffset bits are a pass-through, exhaustive over lanes.

    Word-granular bijection plus this lemma is equivalent to byte-granular
    bijection; it is how the geometries too large for a byte sweep are
    covered.
    """
    lanes = 1 << geom.byte_bits
    bad = None
    n = 0
    for k in range(geom.max_level + 1):
        rule = LevelRule(geom, k)
        bases = [o & ~(lanes - 1) for o in probe_offsets(rule, 4096)]
        for gx, gy in rectangle_corners(geom, k):
            for base in bases:
                ref = rule.permute_addr(base, gx, gy)
                for lane in range(lanes):
                    n += 1
                    if rule.permute_addr(base | lane, gx, gy) != (ref | lane):
                        bad = (k, gx, gy, hex(base), lane)
                        break
                if bad:
                    break
            if bad:
                break
        if bad:
            break
    rep.add('byte lane is a pass-through (%d lanes)' % lanes, bad is None,
            'n=%d %s' % (n, '' if bad is None else str(bad)))


def check_top_level_is_identity(rep: Report, geom: Geometry) -> None:
    """The whole-chip level with chunk_base=0 IS today's global map.

    Invariant #10 says the global region stays byte-identical so existing
    non-VC software runs unchanged; on the permuter that means the top level
    must be the identity for every requester.
    """
    k = geom.max_level
    if geom.num_x != geom.num_y:
        return
    rule = LevelRule(geom, k)
    bad = None
    for gx, gy in all_requesters(geom):
        for off in probe_offsets(rule, 4096):
            if rule.permute_addr(off, gx, gy) != off:
                bad = (gx, gy, hex(off))
                break
        if bad:
            break
    rep.add('whole-chip level k=%d is the identity map' % k, bad is None,
            '' if bad is None else str(bad))


def check_rect_id_roundtrip(rep: Report, geom: Geometry) -> None:
    """(rx,ry) = (gx>>k, gy>>k) and base = coord<<k must round-trip."""
    bad = None
    for k in range(geom.max_level + 1):
        rule = LevelRule(geom, k)
        for gx, gy in all_requesters(geom):
            rx, ry = rule.rect_id(gx, gy)
            if (rx << k, ry << k) != rule.base_of(gx, gy):
                bad = (k, gx, gy, rx, ry)
                break
        if bad:
            break
    rep.add('rect id (gx>>k, gy>>k) round-trips to the base', bad is None,
            '' if bad is None else str(bad))


def check_nd_descriptor(rep: Report, geom: Geometry) -> None:
    """The ND-strided DMA walk must land where the permuter lands.

    The descriptor is the SW/DMA half of the alias-equivalence contract
    (rationale section "VC address model"); if it disagrees with the compute
    path, a cross-VC copy silently corrupts.
    """
    bad = None
    n = 0
    for k in range(geom.max_level + 1):
        for cbase, crows in chunk_configs(geom, k):
            rule = LevelRule(geom, k, chunk_base=cbase, chunk_rows=crows)
            for gx, gy in all_requesters(geom):
                for off in probe_offsets(rule, 1024):
                    n += 1
                    if rule.nd_addr(off, gx, gy) != rule.permute_addr(
                            off, gx, gy):
                        bad = (k, cbase, crows, gx, gy, hex(off))
                        break
                if bad:
                    break
            if bad:
                break
        if bad:
            break
    rep.add('ND-strided descriptor == permuter', bad is None,
            'n=%d %s' % (n, '' if bad is None else str(bad)))


# ---------------------------------------------------------------------------
# The exhaustive sweep
# ---------------------------------------------------------------------------
def sweep_level(geom: Geometry, k: int, byte_granular: bool,
                chunk_base: int = 0, chunk_rows: Optional[int] = None,
                owner: Optional[array] = None) -> Dict[str, object]:
    """Sweep every rectangle of level `k` over its whole aperture.

    `owner` maps a physical slot (byte address or word index, matching the
    granularity) to `(pass_index+1) << 32 | offset`, so a collision reports
    both colliding offsets and both requesters.  Passing the same array across
    calls turns the coverage counter into a disjointness proof.
    """
    rule = LevelRule(geom, k, chunk_base=chunk_base, chunk_rows=chunk_rows)
    step = 1 if byte_granular else geom.be_width
    slots = geom.total_bytes if byte_granular else geom.total_words
    shift = 0 if byte_granular else geom.byte_bits
    if owner is None:
        owner = array('q', bytes(8 * slots))
    corners = rectangle_corners(geom, k)

    res: Dict[str, object] = {
        'rule': rule, 'probes': 0, 'covered': 0, 'rectangles': len(corners),
        'collision': None, 'alias': None, 'escape': None,
    }

    side = rule.side
    gx_pos, gy_pos = geom.gx_pos, geom.gy_pos
    gx_msk = (1 << geom.gx_bits) - 1
    gy_msk = (1 << geom.gy_bits) - 1
    limit = 1 << geom.addr_bits

    for pass_idx, (gx_i, gy_i) in enumerate(corners):
        base_gx, base_gy = gx_i & ~(side - 1), gy_i & ~(side - 1)
        pa, aa = rule.permute_addr, rule.alias_addr
        token = (pass_idx + 1) << 32
        for off in range(0, rule.offset_bytes, step):
            addr = pa(off, gx_i, gy_i)
            res['probes'] += 1

            if res['alias'] is None and addr != aa(off, gx_i, gy_i):
                res['alias'] = (gx_i, gy_i, hex(off), hex(addr),
                                hex(aa(off, gx_i, gy_i)))

            ogx = (addr >> gx_pos) & gx_msk
            ogy = (addr >> gy_pos) & gy_msk
            if res['escape'] is None and not (
                    base_gx <= ogx < base_gx + side
                    and base_gy <= ogy < base_gy + side):
                res['escape'] = {'requester': (gx_i, gy_i), 'offset': off,
                                 'out_group': (ogx, ogy),
                                 'box': (base_gx, base_gy, side)}
            if res['escape'] is None and addr >= limit:
                res['escape'] = {'requester': (gx_i, gy_i), 'offset': off,
                                 'addr_out_of_l1': hex(addr)}

            slot = addr >> shift
            if slot >= slots:
                continue
            prev = owner[slot]
            if prev:
                if res['collision'] is None:
                    res['collision'] = {
                        'phys': geom.decode(addr if byte_granular
                                            else slot << geom.byte_bits),
                        'a': {'requester': corners[(prev >> 32) - 1],
                              'offset': prev & 0xFFFFFFFF},
                        'b': {'requester': (gx_i, gy_i), 'offset': off},
                    }
                continue
            owner[slot] = token | off
            res['covered'] += 1
        if res['collision'] is not None:
            break
    return res


def fmt_collision(c: Dict[str, object]) -> str:
    p = c['phys']
    return ('offset 0x%x from group(%d,%d) and offset 0x%x from group(%d,%d) '
            'BOTH hit gx=%d gy=%d tile=%d bank=%d word=%d'
            % (c['a']['offset'], c['a']['requester'][0],
               c['a']['requester'][1], c['b']['offset'],
               c['b']['requester'][0], c['b']['requester'][1],
               p.gx, p.gy, p.tile, p.bank, p.word))


def check_exhaustive(rep: Report, geom: Geometry, byte_granular: bool,
                     levels: Sequence[int]) -> None:
    gran = 'byte' if byte_granular else 'word'
    slots = geom.total_bytes if byte_granular else geom.total_words
    for k in levels:
        owner = array('q', bytes(8 * slots))
        t0 = time.time()
        res = sweep_level(geom, k, byte_granular, owner=owner)
        del owner
        rule = res['rule']
        dt = time.time() - t0
        tag = 'k=%d (%dx%d, %d rect, %s-exhaustive)' % (
            k, rule.side, rule.side, res['rectangles'], gran)

        rep.add('%s bijection' % tag, res['collision'] is None,
                'probes=%d distinct=%d %.1fs%s'
                % (res['probes'], res['covered'], dt,
                   '' if res['collision'] is None
                   else '  ' + fmt_collision(res['collision'])))
        rep.add('%s alias equivalence (OR == +)' % tag, res['alias'] is None,
                '' if res['alias'] is None else str(res['alias']))
        rep.add('%s confinement to the SxS box' % tag, res['escape'] is None,
                '' if res['escape'] is None else str(res['escape']))
        rep.add('%s rectangles tile all of L1 exactly once' % tag,
                res['covered'] == slots,
                'covered %d of %d %ss' % (res['covered'], slots, gran))


def check_requester_invariance(rep: Report, geom: Geometry) -> None:
    """Every member of a rectangle must see the same map as its corner.

    This is what invariant #4's per-axis mask buys and what makes a kernel
    position independent: the same VC-local offset must resolve identically
    for all S*S participants.
    """
    for k in range(1, geom.max_level + 1):
        rule = LevelRule(geom, k)
        side = rule.side
        exhaustive = (geom.num_groups * rule.offset_words
                      <= INVARIANCE_EXHAUSTIVE_BUDGET)
        if exhaustive:
            offs = range(0, rule.offset_bytes, geom.be_width)
            label = 'exhaustive'
        else:
            offs = sorted(set(probe_offsets(rule, INVARIANCE_SAMPLE))
                          | set(group_field_offsets(rule)))
            label = 'exhaustive over (vc_gx,vc_gy), sampled elsewhere'
        bad = None
        n = 0
        for gx_i, gy_i in all_requesters(geom):
            cx, cy = gx_i & ~(side - 1), gy_i & ~(side - 1)
            if (cx, cy) == (gx_i, gy_i):
                continue
            for off in offs:
                n += 1
                a = rule.permute_addr(off, gx_i, gy_i)
                if a != rule.permute_addr(off, cx, cy):
                    bad = (gx_i, gy_i, cx, cy, hex(off))
                    break
            if bad:
                break
        rep.add('k=%d requester invariance inside a rectangle' % k,
                bad is None,
                'n=%d %s %s' % (n, label, '' if bad is None else str(bad)))


def check_level_heaps_disjoint(rep: Report, geom: Geometry) -> None:
    """Coexisting per-level heaps over one budget must never overlap.

    rationale section "VC address model", subsection "Allocator model": each
    group's L1 is partitioned into DISJOINT physical ranges, one per level,
    and all levels coexist (no runtime repartitioning).  This lays every level
    of one budget into a single map: any shared physical word is a bug.
    """
    levels = list(range(geom.max_level + 1))
    rows = geom.words_per_bank // len(levels)
    rows = 1 << (rows.bit_length() - 1)  # pow2, so invariant #5 is satisfiable
    budget = [(k, i * rows, rows) for i, k in enumerate(levels)]
    if budget[-1][1] + rows > geom.words_per_bank:
        rep.add('per-level heap budget fits the bank depth', False,
                'needs %d rows of %d' % (budget[-1][1] + rows,
                                         geom.words_per_bank))
        return

    owner = array('q', bytes(8 * geom.total_words))
    total, overlap, alias_bad, escape = 0, None, None, None
    for k, cbase, crows in budget:
        res = sweep_level(geom, k, False, chunk_base=cbase, chunk_rows=crows,
                          owner=owner)
        total += res['covered']
        overlap = overlap or (res['collision'] and (k, res['collision']))
        alias_bad = alias_bad or (res['alias'] and (k, res['alias']))
        escape = escape or (res['escape'] and (k, res['escape']))
    del owner
    desc = ' '.join('k%d@row%d+%d' % b for b in budget)
    want = len(levels) * rows * geom.words_per_group_plane * geom.num_groups
    rep.add('coexisting per-level heaps are disjoint', not overlap,
            '%s: %d distinct words%s'
            % (desc, total, '' if not overlap else '  ' + str(overlap)))
    rep.add('per-level heaps cover their whole budget', total == want,
            '%d of %d words' % (total, want))
    rep.add('budgeted heaps keep alias equivalence', not alias_bad,
            '' if not alias_bad else str(alias_bad))
    rep.add('budgeted heaps keep confinement', not escape,
            '' if not escape else str(escape))


# ---------------------------------------------------------------------------
# Injected violations
# ---------------------------------------------------------------------------
def characterise(rule: LevelRule, gx_i: int, gy_i: int,
                 mode: str) -> Dict[str, object]:
    """Collision, escape and image size of one (level, requester, mode).

    `rule` is expected to carry a single chunk row: an invariant-#4 failure
    lives entirely in the group field, so one word plane is enough to expose
    it and the sweep stays small even on the 8x8 mesh.
    """
    g = rule.geom
    side = rule.side
    bx, by = gx_i & ~(side - 1), gy_i & ~(side - 1)
    seen: Dict[int, int] = {}
    out: Dict[str, object] = {'collision': None, 'escape': None}
    for off in range(0, rule.offset_bytes, g.be_width):
        addr = rule.permute_addr(off, gx_i, gy_i, mode)
        p = g.decode(addr)
        if out['escape'] is None and not (bx <= p.gx < bx + side
                                          and by <= p.gy < by + side):
            out['escape'] = {'requester': (gx_i, gy_i), 'offset': off,
                             'out_group': (p.gx, p.gy),
                             'box': (bx, by, side)}
        w = addr >> g.byte_bits
        if w in seen:
            if out['collision'] is None:
                out['collision'] = {
                    'phys': p,
                    'a': {'requester': (gx_i, gy_i), 'offset': seen[w]},
                    'b': {'requester': (gx_i, gy_i), 'offset': off},
                }
        else:
            seen[w] = off
    out['image'] = len(seen)
    out['aperture'] = rule.offset_words
    return out


def check_flat_mask_violation(rep: Report, geom: Geometry) -> None:
    """INVARIANT #4: a flat group mask must be caught, not tolerated.

    Three mis-implementations of roadmap section T2.1 are injected:
      flat_mask_packed - mask the flat row-major group id and OR one packed
                         2k-bit vc_group field at the bottom of the group
                         field (the reading that yields the S x 1 column of
                         invariant #4);
      flat_mask_axis   - keep the per-axis field placement but derive both
                         bases by masking the flat id;
      no_mask          - drop the mask entirely.
    Each must produce a detected collision somewhere in the requester sweep.
    The SPEC mode over the identical sweep is the control and must be clean.
    """
    modes = (MODE_SPEC, MODE_FLAT_MASK_PACKED, MODE_FLAT_MASK_AXIS,
             MODE_NO_MASK)
    stats: Dict[str, Dict[str, object]] = {}
    for mode in modes:
        collided, escaped, worst = [], [], 1.0
        for k in range(1, geom.max_level + 1):
            rule = LevelRule(geom, k, chunk_base=0, chunk_rows=1)
            for gx_i, gy_i in all_requesters(geom):
                r = characterise(rule, gx_i, gy_i, mode)
                if r['collision']:
                    collided.append((k, r['collision']))
                if r['escape']:
                    escaped.append((k, r['escape']))
                worst = min(worst, r['image'] / float(r['aperture']))
        sweeps = geom.num_groups * geom.max_level
        stats[mode] = {'collided': collided, 'escaped': escaped,
                       'worst': worst, 'sweeps': sweeps}

    ctrl = stats[MODE_SPEC]
    rep.add('control: the specified permuter has no collision or escape',
            not ctrl['collided'] and not ctrl['escaped']
            and ctrl['worst'] == 1.0,
            '%d requester sweeps, min image/aperture = %.3f'
            % (ctrl['sweeps'], ctrl['worst']))

    for mode in modes[1:]:
        s = stats[mode]
        ok = bool(s['collided'])
        if s['collided']:
            k, c = s['collided'][0]
            detail = ('k=%d: %s  [%d/%d sweeps collide, %d escape the box, '
                      'min image/aperture %.3f]'
                      % (k, fmt_collision(c), len(s['collided']), s['sweeps'],
                         len(s['escaped']), s['worst']))
        else:
            detail = 'no collision found - THE TEST IS BLIND'
        rep.add('INVARIANT #4 injected (%s) is DETECTED' % mode, ok, detail)

    # The packed reading is the one invariant #4 names: it must additionally
    # break confinement (the "S x 1 column" / half the VC unreachable).
    s = stats[MODE_FLAT_MASK_PACKED]
    ok = bool(s['escaped'])
    detail = 'no escape found'
    if s['escaped']:
        k, e = s['escaped'][0]
        detail = ('k=%d: group(%d,%d) offset 0x%x lands on group(%d,%d), '
                  'outside the %dx%d box at (%d,%d)  [%d escapes]'
                  % (k, e['requester'][0], e['requester'][1], e['offset'],
                     e['out_group'][0], e['out_group'][1], e['box'][2],
                     e['box'][2], e['box'][0], e['box'][1],
                     len(s['escaped'])))
    rep.add('INVARIANT #4 injected (packed) breaks CONFINEMENT', ok, detail)


def check_chunk_base_violation(rep: Report, geom: Geometry) -> None:
    """INVARIANT #5: a misaligned chunk_base must be caught, not tolerated."""
    w = geom.words_per_bank
    crows = max(2, w >> 2)
    misaligned = crows // 2  # not a multiple of crows

    rejected, why = False, ''
    try:
        LevelRule(geom, 1, chunk_base=misaligned, chunk_rows=crows)
    except SpecViolation as exc:
        rejected, why = True, str(exc)
    rep.add('INVARIANT #5: misaligned chunk_base is rejected', rejected, why)

    # Built anyway (validate=False, i.e. the bug shipped): it must fold.
    rule = LevelRule(geom, 1, chunk_base=misaligned, chunk_rows=crows,
                     validate=False)
    r = characterise(rule, 0, 0, MODE_SPEC)
    ok = r['collision'] is not None
    detail = 'no collision found - THE TEST IS BLIND'
    if ok:
        detail = ('chunk_base=%d, extent=%d rows: %s  [image %d of %d words]'
                  % (misaligned, crows, fmt_collision(r['collision']),
                     r['image'], r['aperture']))
    rep.add('INVARIANT #5 injected (misaligned chunk_base) is DETECTED',
            ok, detail)

    mismatch = None
    for off in probe_offsets(rule, 4096):
        if rule.permute_addr(off, 0, 0) != rule.alias_addr(off, 0, 0):
            mismatch = off
            break
    rep.add('INVARIANT #5 injected breaks OR == + (alias equivalence)',
            mismatch is not None,
            '' if mismatch is None else 'first at offset 0x%x' % mismatch)


def check_illegal_descriptors(rep: Report, geom: Geometry) -> None:
    """Roadmap section T2.3 item 4: illegal descriptors must be refused."""
    w = geom.words_per_bank
    q = max(2, w >> 2)
    cases = [
        ('non-pow2 chunk extent', dict(k=1, chunk_base=0, chunk_rows=3)),
        ('misaligned chunk_base', dict(k=1, chunk_base=1, chunk_rows=q)),
        ('over-capacity chunk', dict(k=1, chunk_base=w, chunk_rows=w)),
        ('chunk_base + extent past the bank',
         dict(k=1, chunk_base=w - q + q // 2, chunk_rows=q)),
        ('negative chunk_base', dict(k=1, chunk_base=-q, chunk_rows=q)),
        ('level above the mesh',
         dict(k=geom.max_level + 1, chunk_base=0, chunk_rows=w)),
    ]
    accepted = []
    for name, kwargs in cases:
        try:
            LevelRule(geom, **kwargs)
            accepted.append(name)
        except SpecViolation:
            pass
    rep.add('illegal descriptors rejected', not accepted,
            'all %d cases' % len(cases) if not accepted
            else 'ACCEPTED: %s' % ', '.join(accepted))

    ok = True
    try:
        r = LevelRule(geom, 1, chunk_base=w - q, chunk_rows=q)
        ok = r.highest_legal_chunk_base() == w - q
    except SpecViolation:
        ok = False
    rep.add('highest legal chunk_base accepted', ok,
            'chunk_base=%d extent=%d rows' % (w - q, q))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
DEFAULT_GEOMETRIES = ('terapool', 'monsterpool8x8')


def run_geometry(rep: Report, geom: Geometry, quick: bool,
                 force_byte: bool = False) -> None:
    print()
    print('=== %s ===' % geom.name)
    print('    %r' % geom)
    print('    config source: %s' % geom.source)
    print('    canonical layout: %s' % ' '.join(
        '%s[%d:%d]' % (n, hi, lo) for n, hi, lo in geom.describe_fields()))
    print('    L1 = %d words = %d bytes; levels k=0..%d'
          % (geom.total_words, geom.total_bytes, geom.max_level))

    byte_granular = force_byte or geom.total_bytes <= BYTE_SWEEP_BUDGET
    print('    sweep granularity: %s (one whole-L1 pass = %d probes)'
          % ('byte' if byte_granular else 'word',
             geom.total_bytes if byte_granular else geom.total_words))

    check_split_agrees(rep, geom)
    check_byte_lane_identity(rep, geom)
    check_rect_id_roundtrip(rep, geom)
    check_top_level_is_identity(rep, geom)
    check_nd_descriptor(rep, geom)
    check_illegal_descriptors(rep, geom)
    check_chunk_base_violation(rep, geom)
    check_flat_mask_violation(rep, geom)

    levels = list(range(geom.max_level + 1))
    if quick:
        levels = levels[:2]
    check_exhaustive(rep, geom, byte_granular, levels)
    if not quick:
        check_requester_invariance(rep, geom)
        check_level_heaps_disjoint(rep, geom)


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Exhaustive self-check of the WP3 VC permuter reference.')
    ap.add_argument('-g', '--geometry', action='append',
                    choices=sorted(GEOMETRIES),
                    help='geometry to check (repeatable); default %s'
                         % ', '.join(DEFAULT_GEOMETRIES))
    ap.add_argument('--all', action='store_true',
                    help='check every geometry fixture')
    ap.add_argument('--quick', action='store_true',
                    help='fewer levels, skip the slow passes')
    ap.add_argument('--byte-exhaustive', action='store_true',
                    help='force a BYTE-granular whole-L1 sweep even when it '
                         'exceeds BYTE_SWEEP_BUDGET; removes the reliance on '
                         'the byte-lane lemma at the cost of 4x the probes '
                         'and an 8-byte-per-byte collision map')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    if args.all:
        names = sorted(GEOMETRIES)
    elif args.geometry:
        names = args.geometry
    else:
        names = list(DEFAULT_GEOMETRIES)

    rep = Report(args.verbose)
    t0 = time.time()
    print('WP3 virtual-cluster permuter - reference self-check')
    print('reference: scripts/vc/vc_permute.py')
    print('spec: docs/plan/wp2-wp4-implementation-roadmap.md sections T2.1, '
          '2.2 (invariants 4,5,10), "Canonical layout"')
    print()
    print('=== spec fixtures ===')
    check_layout_fixture(rep)
    check_doc_stride_example(rep)
    for name in names:
        run_geometry(rep, GEOMETRIES[name], args.quick,
                     args.byte_exhaustive)
    print()
    print('elapsed %.1fs' % (time.time() - t0))
    return rep.summary()


if __name__ == '__main__':
    raise SystemExit(main())
