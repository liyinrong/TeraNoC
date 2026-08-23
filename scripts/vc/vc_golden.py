#!/usr/bin/env python3

# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Golden-vector generator for the WP3 virtual-cluster permuter.
#
# Emits, per geometry, two versioned plain-text files that an RTL testbench
# (or the GVSoC model, or a C runtime unit test) can replay line by line:
#
#   scripts/vc/golden/vc_golden_<geometry>_v1.txt
#       legal (level, requester, chunk, offset) -> physical address vectors
#   scripts/vc/golden/vc_illegal_<geometry>_v1.txt
#       illegal descriptors that the elaboration assert of roadmap section
#       T2.1 / the gate of section T2.3 item 4 must reject
#
# The vectors are WINDOW-RELATIVE: the input column is the VC-relative offset
# O and the level k, not a full address.  The window base per level is Track
# C's to choose (roadmap section T2.2 reserves "a FIXED address window per
# level" but does not fix the constants), so binding the vectors to a base
# would date them.  An RTL testbench drives `window_base[k] + O`.
#
# Usage:
#   python3 scripts/vc/vc_golden.py            # regenerate the shipped files
#   python3 scripts/vc/vc_golden.py --verify   # re-derive and check them
#   python3 scripts/vc/vc_golden.py --self-test  # prove the verifier bites
#   python3 scripts/vc/vc_golden.py --all      # every geometry fixture
#   python3 scripts/vc/vc_golden.py --profile general
#   python3 scripts/vc/vc_golden.py -g terapool -o DIR

import argparse
import os
import shutil
import sys
import tempfile
import zlib
from typing import Dict, List, Sequence, TextIO, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vc_permute import (  # noqa: E402
    GEOMETRIES, Geometry, LevelRule, SpecViolation,
)
from vc_selfcheck import STRIDE, boundary_offsets  # noqa: E402

FORMAT = 'teranoc-vc-golden'
VERSION = 1
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'golden')
# The geometries whose vector files are committed.  Every other fixture can be
# generated on demand with --all or -g; they are not kept in the repository so
# the tree does not carry a megabyte of derived text.
SHIPPED = ('terapool', 'monsterpool8x8')

# Sampling knobs.  Changing any of them changes the file, so they are written
# into the header and the reader checks them back.
#
# PROFILES.  'adopted' matches the shipped design after the 2026-08-23 review:
# a level's window is the FULL L1 of its rectangle, so chunk_rows is always the
# whole bank depth and chunk_base can only ever be 0 (see README finding S12).
# Vectors with a non-zero chunk_base are unprogrammable on that hardware, so
# they are not shipped -- but 'general' still emits them, because the reference
# keeps the parameter and the original acceptance evidence rests on it.
PROFILES = {'adopted': 1, 'general': 3}
WALK_COUNT = {'adopted': 80, 'general': 16}
COLUMNS = ('k', 'gx_i', 'gy_i', 'chunk_base', 'chunk_rows', 'offset', 'phys',
           'gx', 'gy', 'tile', 'bank', 'word', 'byte')


def hexw(bits: int) -> int:
    """Hex digits needed for a `bits`-wide field, at least one."""
    return max(1, (bits + 3) // 4)


# ---------------------------------------------------------------------------
# The sampling rule.  Everything below is restated verbatim in the file header
# so the set can be regenerated without running this script.
# ---------------------------------------------------------------------------
def axis_set(n: int, side: int) -> List[int]:
    """Requester coordinates probed on one axis at level side=2^k.

    {0, 1, side-1, side, n-side, n-1} clipped to [0, n).  It contains the
    corner of the first rectangle, two interior members of it (so the per-axis
    mask is exercised, not just the aligned case), the corner of the second
    rectangle, and the corner and far member of the last rectangle.
    """
    return sorted({v for v in (0, 1, side - 1, side, n - side, n - 1)
                   if 0 <= v < n})


def requester_set(geom: Geometry, k: int) -> List[Tuple[int, int]]:
    if geom.num_groups <= 16:
        return [(gx, gy) for gx in range(geom.num_x)
                for gy in range(geom.num_y)]
    side = 1 << k
    return [(gx, gy) for gx in axis_set(geom.num_x, side)
            for gy in axis_set(geom.num_y, side)]


def chunk_set(geom: Geometry, profile: str) -> List[Tuple[int, int]]:
    """(chunk_base, chunk_rows) probed at every level.

    'adopted': the whole bank at base 0 -- the only descriptor the shipped
    hardware can express.
    'general': additionally a quarter-bank chunk at a non-zero aligned base and
    that extent at its HIGHEST legal chunk_base.
    """
    w = geom.words_per_bank
    if profile == 'adopted':
        return [(0, w)]
    q = max(1, w >> 2)
    out = [(0, w), (q, q), (w - q, q)]
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def golden_offsets(rule: LevelRule, profile: str) -> List[int]:
    """Boundary set (vc_selfcheck.boundary_offsets) plus a stride walk.

    boundary_offsets covers: offset 0 and the last offset of the level; for
    every field, the offset setting only its low bit, only its high bits, and
    the carry point just below each; and the first and last offset of chunk
    planes {0, 1, 2, chunk_rows/2, chunk_rows-2, chunk_rows-1}.
    The walk is offset_i = (i * 0x9E3779B1) mod aperture_bytes, i in
    [0, WALK_COUNT); the aperture is a power of two and the multiplier is odd,
    so the walk never repeats.
    """
    out = set(boundary_offsets(rule))
    for i in range(WALK_COUNT[profile]):
        out.add((i * STRIDE) % rule.offset_bytes)
    return sorted(out)


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
def header_params(geom: Geometry, profile: str) -> List[Tuple[str, str]]:
    fields = {n: (hi, lo) for n, hi, lo in geom.describe_fields()}
    params = [
        ('format', FORMAT),
        ('version', str(VERSION)),
        ('geometry', geom.name),
        ('profile', profile),
        ('config_source', geom.source),
        ('num_x', str(geom.num_x)),
        ('num_y', str(geom.num_y)),
        ('num_groups', str(geom.num_groups)),
        ('tiles_per_group', str(geom.tiles_per_group)),
        ('banks_per_tile', str(geom.banks_per_tile)),
        ('bank_size_bytes', str(geom.bank_size_bytes)),
        ('words_per_bank', str(geom.words_per_bank)),
        ('data_width_bits', str(geom.data_width_bits)),
        ('byte_offset_bits', str(geom.byte_bits)),
        ('addr_bits', str(geom.addr_bits)),
        ('total_bytes', str(geom.total_bytes)),
        ('max_level', str(geom.max_level)),
        ('group_id_rule', 'group_id = gx*num_y + gy (gy is the LOW half)'),
        ('window_base', '0 (vectors are VC-window relative; see the notes)'),
        ('walk_stride', '0x%08X' % STRIDE),
        ('walk_count', str(WALK_COUNT[profile])),
        ('columns', ' '.join(COLUMNS)),
    ]
    for name in ('byte', 'bank', 'tile', 'gy', 'gx', 'word'):
        hi, lo = fields[name]
        params.append(('field_' + name, '[%d:%d]' % (hi, lo)))
    return params


NOTES = """
What this file is
-----------------
Golden vectors for the TeraNoC WP3 virtual-cluster address permuter.  The
reference implementation that produced them is scripts/vc/vc_permute.py; it is
verified exhaustively by scripts/vc/vc_selfcheck.py.  The permuter itself is
specified by
  docs/plan/wp2-wp4-implementation-roadmap.md section T2.1 (pseudo-code),
  the same document's section "Canonical layout" (physical field order),
  and its section 2.2 invariants 4 (per-axis base mask) and 5 (pow2-aligned
  chunk_base).

How to replay
-------------
Skip every line whose first non-blank character is '#'.  Every remaining line
holds the columns named by @columns, whitespace separated, ALL IN HEX WITHOUT
A 0x PREFIX:

  k           level index; the virtual cluster is a (2^k)x(2^k) square
  gx_i gy_i   the REQUESTING group's own mesh coordinates
  chunk_base  the level's chunk base, in WORD-PLANES (not bytes)
  chunk_rows  the level's chunk extent, in WORD-PLANES
  offset      the VC-relative byte offset O driven into the permuter
  phys        expected TCDM-relative physical byte address
  gx gy tile bank word byte
              expected canonical decode of `phys`, for a field-by-field
              mismatch message

The vectors are WINDOW-RELATIVE.  Roadmap section T2.2 reserves a fixed
address window per level but does not fix the constants, so a testbench drives
window_base[k] + offset and expects TCDMBaseAddr + phys.

The sampling rule
-----------------
Every combination of
  level      k = 0 .. max_level
  requester  all groups when num_groups <= 16, otherwise the cross product of
             {0, 1, S-1, S, N-S, N-1} clipped to [0, N) on each axis, S = 2^k
  chunk      profile 'adopted': (0, W) only, W = words_per_bank -- a level's
             window is the full L1 of its rectangle, so the whole bank depth
             is addressable from the offset and chunk_base can only be 0.
             profile 'general': additionally (W/4, W/4) and (W-W/4, W/4), the
             latter the HIGHEST legal chunk_base for that extent.
  offset     the boundary set below, plus a stride walk
The boundary set is: offset 0; the last offset of the level; for each field
(byte, bank, tile, vc_gy, vc_gx, chunk_row) the offset setting only its low
bit, only its high bits, the value one below each of those, and their sum; and
the first and last offset of chunk planes {0, 1, 2, chunk_rows/2,
chunk_rows-2, chunk_rows-1}.
The stride walk is offset_i = (i * @walk_stride) mod aperture_bytes for i in
[0, @walk_count).  The aperture is a power of two and the multiplier is odd,
so the walk is injective.  @profile records which chunk set was used.
Duplicates are removed and offsets are emitted in ascending order.

Integrity
---------
The checksum parameter below is the CRC-32 (zlib) of the concatenated data
lines, each including its trailing newline, with the header excluded.
""".strip()


def format_row(geom: Geometry, widths: Dict[str, int], k: int,
               gx_i: int, gy_i: int, rule: LevelRule, off: int) -> str:
    addr = rule.permute_addr(off, gx_i, gy_i)
    p = geom.decode(addr)
    vals = {
        'k': k, 'gx_i': gx_i, 'gy_i': gy_i,
        'chunk_base': rule.chunk_base, 'chunk_rows': rule.chunk_rows,
        'offset': off, 'phys': addr, 'gx': p.gx, 'gy': p.gy, 'tile': p.tile,
        'bank': p.bank, 'word': p.word, 'byte': p.byte,
    }
    return ' '.join('%0*x' % (widths[c], vals[c]) for c in COLUMNS)


def column_widths(geom: Geometry) -> Dict[str, int]:
    row_w = hexw(geom.word_bits + 1)  # chunk_rows can equal words_per_bank
    return {
        'k': 1,
        'gx_i': hexw(geom.gx_bits), 'gy_i': hexw(geom.gy_bits),
        'chunk_base': row_w, 'chunk_rows': row_w,
        'offset': hexw(geom.addr_bits), 'phys': hexw(geom.addr_bits),
        'gx': hexw(geom.gx_bits), 'gy': hexw(geom.gy_bits),
        'tile': hexw(geom.tile_bits), 'bank': hexw(geom.bank_bits),
        'word': hexw(geom.word_bits), 'byte': hexw(geom.byte_bits),
    }


def build_rows(geom: Geometry, profile: str) -> List[str]:
    widths = column_widths(geom)
    rows: List[str] = []
    for k in range(geom.max_level + 1):
        for cbase, crows in chunk_set(geom, profile):
            rule = LevelRule(geom, k, chunk_base=cbase, chunk_rows=crows)
            offs = golden_offsets(rule, profile)
            for gx_i, gy_i in requester_set(geom, k):
                for off in offs:
                    rows.append(format_row(geom, widths, k, gx_i, gy_i,
                                           rule, off))
    return rows


def write_header(fh: TextIO, geom: Geometry, rows: Sequence[str],
                 checksum: int, profile: str) -> None:
    widths = column_widths(geom)
    fh.write('# Copyright 2026 ETH Zurich and University of Bologna.\n')
    fh.write('# Licensed under the Apache License, Version 2.0, see LICENSE '
             'for details.\n')
    fh.write('# SPDX-License-Identifier: Apache-2.0\n')
    fh.write('#\n')
    fh.write('# GENERATED by scripts/vc/vc_golden.py - do not edit by hand.\n')
    fh.write('#\n')
    for line in NOTES.splitlines():
        fh.write(('# ' + line).rstrip() + '\n')
    fh.write('#\n')
    for key, value in header_params(geom, profile):
        fh.write('# @%-16s %s\n' % (key, value))
    fh.write('# @%-16s %d\n' % ('count', len(rows)))
    fh.write('# @%-16s crc32=0x%08x\n' % ('checksum', checksum))
    fh.write('#\n')
    fh.write('# %s\n' % ' '.join(
        c.ljust(widths[c]) for c in COLUMNS).rstrip())


def emit(geom: Geometry, out_dir: str, profile: str) -> str:
    rows = build_rows(geom, profile)
    body = ''.join(r + '\n' for r in rows)
    checksum = zlib.crc32(body.encode('ascii')) & 0xFFFFFFFF
    path = os.path.join(out_dir, 'vc_golden_%s_v%d.txt' % (geom.name,
                                                           VERSION))
    with open(path, 'w') as fh:
        write_header(fh, geom, rows, checksum, profile)
        fh.write(body)
    return path


# ---------------------------------------------------------------------------
# Illegal descriptors (roadmap section T2.3 item 4)
# ---------------------------------------------------------------------------
def illegal_cases(geom: Geometry) -> List[Tuple[str, int, int, int]]:
    w = geom.words_per_bank
    q = max(2, w >> 2)
    return [
        ('non_pow2_chunk_extent', 1, 0, 3),
        ('non_pow2_chunk_extent', 1, 0, w - 1),
        ('misaligned_chunk_base', 1, 1, q),
        ('misaligned_chunk_base', 1, q // 2, q),
        ('misaligned_chunk_base', 0, q + 1, q),
        # Aligned bases that still do not fit: only the capacity check
        # rejects these.
        ('over_capacity', 1, w, q),
        ('over_capacity', 1, 2 * w, w),
        ('over_capacity', 1, 0, 2 * w),
        ('level_above_mesh', geom.max_level + 1, 0, w),
    ]


def emit_illegal(geom: Geometry, out_dir: str) -> str:
    path = os.path.join(out_dir,
                        'vc_illegal_%s_v%d.txt' % (geom.name, VERSION))
    rows = []
    for reason, k, cbase, crows in illegal_cases(geom):
        try:
            LevelRule(geom, k, chunk_base=cbase, chunk_rows=crows)
        except SpecViolation as exc:
            rows.append((reason, k, cbase, crows, str(exc)))
            continue
        raise AssertionError(
            'illegal case %s (k=%d base=%d rows=%d) was accepted'
            % (reason, k, cbase, crows))

    with open(path, 'w') as fh:
        fh.write('# Copyright 2026 ETH Zurich and University of Bologna.\n')
        fh.write('# Licensed under the Apache License, Version 2.0, see '
                 'LICENSE for details.\n')
        fh.write('# SPDX-License-Identifier: Apache-2.0\n')
        fh.write('#\n')
        fh.write('# GENERATED by scripts/vc/vc_golden.py - do not edit.\n')
        fh.write('#\n')
        fh.write('# Descriptors that the level-rule programming path MUST\n')
        fh.write('# refuse.  Roadmap section T2.1 requires an elaboration\n')
        fh.write('#\n')
        fh.write('# NOTE: under the adopted design (README S12) a level\n')
        fh.write('# window is the full L1 of its rectangle, chunk_rows is\n')
        fh.write('# fixed at the bank depth and chunk_base is always 0, so\n')
        fh.write('# the non-pow2 / misaligned / over-capacity classes below\n')
        fh.write('# are unreachable by construction and only the level\n')
        fh.write('# bound survives as a real elaboration check.  They are\n')
        fh.write('# kept because the reference still models the params.\n')
        fh.write('# assert on chunk_base alignment; section T2.3 item 4 is\n')
        fh.write('# the gate: an illegal-descriptor SVA must fire on\n')
        fh.write('# unaligned / non-pow2 chunk_base / over-capacity.\n')
        fh.write('#\n')
        fh.write('# @format           teranoc-vc-illegal\n')
        fh.write('# @version          %d\n' % VERSION)
        fh.write('# @geometry         %s\n' % geom.name)
        fh.write('# @words_per_bank   %d\n' % geom.words_per_bank)
        fh.write('# @max_level        %d\n' % geom.max_level)
        fh.write('# @columns          reason k chunk_base chunk_rows\n')
        fh.write('# k / chunk_base / chunk_rows are DECIMAL here; chunk_base'
                 ' and\n')
        fh.write('# chunk_rows are word-planes.  The trailing text after '
                 '\'|\' is the\n')
        fh.write('# reference diagnostic, informative only.\n')
        fh.write('#\n')
        for reason, k, cbase, crows, why in rows:
            fh.write('%-22s %2d %6d %6d | %s\n'
                     % (reason, k, cbase, crows, why))
    return path


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
def parse_header(path: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            body = line[1:].strip()
            if body.startswith('@'):
                key, _, value = body[1:].partition(' ')
                params[key] = value.strip()
    return params


def verify(path: str) -> Tuple[int, List[str]]:
    """Re-derive every vector from the reference; returns (count, errors)."""
    params = parse_header(path)
    errors: List[str] = []
    if params.get('format') != FORMAT:
        return 0, ['%s: not a %s file' % (path, FORMAT)]
    geom = GEOMETRIES.get(params.get('geometry', ''))
    if geom is None:
        return 0, ['%s: unknown geometry %r' % (path,
                                                params.get('geometry'))]
    for key, want in (('num_x', geom.num_x), ('num_y', geom.num_y),
                      ('tiles_per_group', geom.tiles_per_group),
                      ('banks_per_tile', geom.banks_per_tile),
                      ('words_per_bank', geom.words_per_bank),
                      ('addr_bits', geom.addr_bits)):
        if int(params.get(key, -1)) != want:
            errors.append('%s: header %s=%s but the fixture says %d'
                          % (path, key, params.get(key), want))
    cols = params.get('columns', '').split()
    if tuple(cols) != COLUMNS:
        errors.append('%s: column list %r != %r' % (path, cols, COLUMNS))
        return 0, errors

    body_lines: List[str] = []
    with open(path) as fh:
        for line in fh:
            if line.lstrip().startswith('#') or not line.strip():
                continue
            body_lines.append(line)

    crc = zlib.crc32(''.join(body_lines).encode('ascii')) & 0xFFFFFFFF
    want_crc = params.get('checksum', '')
    if want_crc != 'crc32=0x%08x' % crc:
        errors.append('%s: checksum %s, recomputed crc32=0x%08x'
                      % (path, want_crc, crc))
    if int(params.get('count', -1)) != len(body_lines):
        errors.append('%s: @count %s but %d data lines'
                      % (path, params.get('count'), len(body_lines)))

    cache: Dict[Tuple[int, int, int], LevelRule] = {}
    offsets_seen: Dict[Tuple[int, int, int], set] = {}
    for lineno, line in enumerate(body_lines, 1):
        v = [int(t, 16) for t in line.split()]
        rec = dict(zip(COLUMNS, v))
        key = (rec['k'], rec['chunk_base'], rec['chunk_rows'])
        rule = cache.get(key)
        if rule is None:
            rule = LevelRule(geom, rec['k'], chunk_base=rec['chunk_base'],
                             chunk_rows=rec['chunk_rows'])
            cache[key] = rule
            offsets_seen[key] = set()
        offsets_seen[key].add(rec['offset'])
        addr = rule.permute_addr(rec['offset'], rec['gx_i'], rec['gy_i'])
        if addr != rec['phys']:
            errors.append('%s:%d phys 0x%x, reference says 0x%x'
                          % (path, lineno, rec['phys'], addr))
            continue
        if addr != rule.alias_addr(rec['offset'], rec['gx_i'], rec['gy_i']):
            errors.append('%s:%d fails alias equivalence' % (path, lineno))
        p = geom.decode(addr)
        if (p.gx, p.gy, p.tile, p.bank, p.word, p.byte) != (
                rec['gx'], rec['gy'], rec['tile'], rec['bank'], rec['word'],
                rec['byte']):
            errors.append('%s:%d decoded fields disagree' % (path, lineno))
        side = 1 << rec['k']
        bx = rec['gx_i'] & ~(side - 1)
        by = rec['gy_i'] & ~(side - 1)
        if not (bx <= p.gx < bx + side and by <= p.gy < by + side):
            errors.append('%s:%d escapes the %dx%d box at (%d,%d)'
                          % (path, lineno, side, side, bx, by))
        if len(errors) > 20:
            errors.append('%s: ... more errors suppressed' % path)
            break

    errors.extend(audit_coverage(path, geom, cache, offsets_seen,
                                 params.get('profile', 'general')))
    return len(body_lines), errors


def audit_coverage(path: str, geom: Geometry,
                   cache: Dict[Tuple[int, int, int], LevelRule],
                   seen: Dict[Tuple[int, int, int], set],
                   profile: str) -> List[str]:
    """The boundary cases the acceptance list names must actually be there.

    Every level must appear; every (level, chunk) must carry its first and
    last offset and every chunk-plane boundary the rule promises; and the
    highest legal chunk_base must be exercised at every level.
    """
    errors: List[str] = []
    levels = {k for k, _, _ in cache}
    want_levels = set(range(geom.max_level + 1))
    if levels != want_levels:
        errors.append('%s: levels %s, expected %s'
                      % (path, sorted(levels), sorted(want_levels)))
    for k in sorted(levels):
        bases = {(cb, cr) for kk, cb, cr in cache if kk == k}
        if profile == 'adopted':
            # The window is the full L1 of the rectangle, so (0, whole bank)
            # is the only descriptor the hardware can express.
            if bases != {(0, geom.words_per_bank)}:
                errors.append('%s: level %d carries %s, but the adopted '
                              'profile allows only (0, %d)'
                              % (path, k, sorted(bases),
                                 geom.words_per_bank))
            continue
        # A partial extent must be exercised at its highest legal base; the
        # full-bank extent only has base 0, so it proves nothing.
        partial = {cr for _, cr in bases if cr < geom.words_per_bank}
        if not partial:
            errors.append('%s: level %d has no partial chunk extent' %
                          (path, k))
        for cr in sorted(partial):
            if (geom.words_per_bank - cr, cr) not in bases:
                errors.append('%s: level %d never uses the highest legal '
                              'chunk_base %d for extent %d'
                              % (path, k, geom.words_per_bank - cr, cr))
    for key, rule in cache.items():
        got = seen[key]
        for want, what in ((0, 'first offset'),
                           (rule.offset_bytes - 1, 'last offset')):
            if want not in got:
                errors.append('%s: (k=%d base=%d rows=%d) misses the %s 0x%x'
                              % (path, key[0], key[1], key[2], what, want))
        plane = 1 << rule.o_row_pos
        for c in {0, 1, 2, rule.chunk_rows // 2, rule.chunk_rows - 2,
                  rule.chunk_rows - 1}:
            if not 0 <= c < rule.chunk_rows:
                continue
            for want in (c * plane, c * plane + plane - 1):
                if want not in got:
                    errors.append(
                        '%s: (k=%d base=%d rows=%d) misses chunk-plane '
                        'boundary 0x%x' % (path, key[0], key[1], key[2],
                                           want))
    return errors


def verify_illegal(path: str) -> Tuple[int, List[str]]:
    params = parse_header(path)
    errors: List[str] = []
    geom = GEOMETRIES.get(params.get('geometry', ''))
    if geom is None:
        return 0, ['%s: unknown geometry' % path]
    n = 0
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            if line.lstrip().startswith('#') or not line.strip():
                continue
            head = line.split('|')[0].split()
            reason, k, cbase, crows = (head[0], int(head[1]), int(head[2]),
                                       int(head[3]))
            n += 1
            try:
                LevelRule(geom, k, chunk_base=cbase, chunk_rows=crows)
                errors.append('%s:%d %s was ACCEPTED (k=%d base=%d rows=%d)'
                              % (path, lineno, reason, k, cbase, crows))
            except SpecViolation:
                pass
    return n, errors


# ---------------------------------------------------------------------------
# Negative self-test: prove the verifier is not blind
# ---------------------------------------------------------------------------
def _rewrite(path: str, keep, mutate=None) -> None:
    """Rewrite a golden file, dropping/mutating data lines and re-stamping."""
    head, body = [], []
    with open(path) as fh:
        for ln in fh:
            (head if ln.startswith('#') else body).append(ln)
    out = []
    for ln in body:
        if not keep(ln):
            continue
        out.append(mutate(ln) if mutate else ln)
    crc = zlib.crc32(''.join(out).encode('ascii')) & 0xFFFFFFFF
    stamped = []
    for ln in head:
        if ln.startswith('# @count'):
            ln = '# @%-16s %d\n' % ('count', len(out))
        elif ln.startswith('# @checksum'):
            ln = '# @%-16s crc32=0x%08x\n' % ('checksum', crc)
        stamped.append(ln)
    with open(path, 'w') as fh:
        fh.writelines(stamped + out)


def self_test(src_dir: str) -> int:
    """Corrupt a copy of a golden file four ways; each must be reported.

    Without this, "the verifier passes" says nothing: a verifier that accepts
    everything also passes.
    """
    name = SHIPPED[0]
    src = os.path.join(src_dir, 'vc_golden_%s_v%d.txt' % (name, VERSION))
    if not os.path.exists(src):
        print('self-test needs %s; run the generator first' % src)
        return 1
    geom = GEOMETRIES[name]
    cases = []

    def case(label, keep, mutate=None, restamp=True):
        cases.append((label, keep, mutate, restamp))

    case('a dropped vector (count + checksum)',
         lambda ln: not ln.startswith('0 0 0 '), None, False)
    case('a corrupted phys column',
         lambda ln: True,
         lambda ln: _flip_phys(ln) if ln.split()[5].endswith('3c') else ln)
    case('a missing first/last boundary offset',
         lambda ln: not (ln.split()[0] == '0'
                         and int(ln.split()[5], 16) == 0
                         and int(ln.split()[4], 16) == geom.words_per_bank))
    plane = 1 << LevelRule(geom, 0).o_row_pos
    case('a missing mid-chunk plane boundary',
         lambda ln: (int(ln.split()[5], 16)
                     != (geom.words_per_bank // 2) * plane))

    rc = 0
    tmp = tempfile.mkdtemp(prefix='vc_golden_selftest_')
    try:
        for label, keep, mutate, restamp in cases:
            dst = os.path.join(tmp, os.path.basename(src))
            shutil.copyfile(src, dst)
            if restamp:
                _rewrite(dst, keep, mutate)
            else:
                _rewrite_raw(dst, keep)
            _, errs = verify(dst)
            ok = bool(errs)
            print('  [%s] verifier reports %s%s'
                  % ('PASS' if ok else 'FAIL', label,
                     '' if not ok else '  -> ' + errs[0].split(': ', 1)[-1]))
            rc |= 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rc


def _rewrite_raw(path: str, keep) -> None:
    """Drop lines WITHOUT re-stamping, so count and checksum go stale."""
    with open(path) as fh:
        lines = fh.readlines()
    with open(path, 'w') as fh:
        fh.writelines(ln for ln in lines if ln.startswith('#') or keep(ln))


def _flip_phys(line: str) -> str:
    f = line.split()
    f[6] = '%0*x' % (len(f[6]), int(f[6], 16) ^ 4)
    return ' '.join(f) + '\n'


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description='Generate or verify WP3 VC permuter golden vectors.')
    ap.add_argument('-g', '--geometry', action='append',
                    choices=sorted(GEOMETRIES),
                    help='geometry (repeatable); default: %s'
                         % ', '.join(SHIPPED))
    ap.add_argument('--all', action='store_true',
                    help='every geometry fixture, not just the shipped two')
    ap.add_argument('-o', '--out-dir', default=DEFAULT_DIR)
    ap.add_argument('--profile', default='adopted', choices=sorted(PROFILES),
                    help='adopted = the shipped design (chunk_base is always '
                         '0, see README S12); general = also emit partial '
                         'chunks at non-zero bases')
    ap.add_argument('--verify', action='store_true',
                    help='re-check the files on disk instead of writing')
    ap.add_argument('--self-test', action='store_true',
                    help='prove the verifier catches corrupted vectors')
    args = ap.parse_args()

    if args.all:
        names: Sequence[str] = sorted(GEOMETRIES)
    else:
        names = args.geometry or list(SHIPPED)
    os.makedirs(args.out_dir, exist_ok=True)
    rc = 0

    if args.self_test:
        print('golden-vector verifier self-test')
        return self_test(args.out_dir)

    if args.verify:
        for name in names:
            for kind, fn in (('vc_golden', verify),
                             ('vc_illegal', verify_illegal)):
                path = os.path.join(args.out_dir,
                                    '%s_%s_v%d.txt' % (kind, name, VERSION))
                if not os.path.exists(path):
                    print('MISSING %s' % path)
                    rc = 1
                    continue
                n, errs = fn(path)
                print('%-7s %s  (%d vectors)'
                      % ('FAIL' if errs else 'OK', path, n))
                for e in errs:
                    print('   %s' % e)
                rc |= 1 if errs else 0
        return rc

    for name in names:
        geom = GEOMETRIES[name]
        p1 = emit(geom, args.out_dir, args.profile)
        p2 = emit_illegal(geom, args.out_dir)
        for p in (p1, p2):
            with open(p) as fh:
                lines = sum(1 for ln in fh if not ln.startswith('#'))
            print('wrote %-56s %6d vectors  %7d B'
                  % (p, lines, os.path.getsize(p)))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
