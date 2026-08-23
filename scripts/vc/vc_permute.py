#!/usr/bin/env python3

# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Reference (oracle) implementation of the TeraNoC WP3 virtual-cluster
# address permuter.
#
# Specification sources (cite these, not this file, for *what* is built):
#   docs/plan/wp2-wp4-implementation-roadmap.md
#     - section "Canonical layout"  : physical field order and widths
#     - section T2.1                : the per-level permuter pseudo-code
#     - section 2.2, invariants 4/5 : per-axis base mask, pow2 chunk_base
#   docs/plan/wp2-wp4-design-rationale.md
#     - section "VC address model"  : fixed fractal pow2-square levels, one
#                                     rule per level, computed base, per-level
#                                     heaps, and the cross-VC DMA stride shape
#
# RTL cross-references (the branch Track C adds must match this file):
#   hardware/src/mempool_addr_scrambler.sv:71   (the always_comb the VC branch
#                                                joins; mutually exclusive with
#                                                the tile-id-remap branch)
#   hardware/src/mempool_tile_remote_req_router.sv:56,66
#                                               (canonical decode: the group
#                                                field sits at ByteOffset +
#                                                clog2(NumBanksPerTile) +
#                                                clog2(NumTilesPerGroup), the
#                                                word-in-bank field above it)
#   hardware/src/mempool_cluster_floonoc_wrapper.sv:249
#                                               (group_id_i = {gx, gy}: gy is
#                                                the LOW half of the group id)
#   hardware/src/mempool_pkg.sv:64-66           (DataWidth 32 => ByteOffset 2)
#
# Nothing here is hardcoded to a flavour: every field width is derived from
# the geometry parameters. The terapool numbers live only in GEOMETRIES.

import argparse
from typing import Dict, List, NamedTuple, Optional, Tuple


class SpecViolation(ValueError):
    """Raised when a descriptor breaks a documented invariant."""


def clog2(value: int) -> int:
    """Ceiling log2, matching SystemVerilog's $clog2 for value >= 1."""
    if value < 1:
        raise SpecViolation('clog2 needs a positive argument, got %r' % value)
    return (value - 1).bit_length()


def is_pow2(value: int) -> bool:
    return value >= 1 and (value & (value - 1)) == 0


class Geometry:
    """One cluster instance: group mesh, tiles, banks and bank depth.

    All field positions of the canonical layout

        [ word_in_bank | gx | gy | tile | bank | byte ]

    are derived here.  `gy` is the LOW half of the group field and `gx` the
    high half, because the RTL builds the group id as ``{group_id.x,
    group_id.y}`` (hardware/src/mempool_cluster_floonoc_wrapper.sv:249).  That
    ordering is what makes the two y-adjacent groups of a 2x2 rectangle
    contiguous in the global alias.
    """

    def __init__(self, name: str, num_x: int, num_y: int,
                 tiles_per_group: int, banks_per_tile: int,
                 bank_size_bytes: int, data_width_bits: int = 32,
                 source: str = ''):
        self.name = name
        self.num_x = num_x
        self.num_y = num_y
        self.num_groups = num_x * num_y
        self.tiles_per_group = tiles_per_group
        self.banks_per_tile = banks_per_tile
        self.bank_size_bytes = bank_size_bytes
        self.data_width_bits = data_width_bits
        self.source = source

        for label, value in (('num_x', num_x), ('num_y', num_y),
                             ('tiles_per_group', tiles_per_group),
                             ('banks_per_tile', banks_per_tile)):
            if not is_pow2(value):
                raise SpecViolation(
                    '%s: %s must be a power of two, got %d'
                    % (name, label, value))

        # BeWidth = DataWidth/8, ByteOffset = clog2(BeWidth).
        # hardware/src/mempool_pkg.sv:64-66
        self.be_width = data_width_bits // 8
        self.byte_bits = clog2(self.be_width)
        if self.bank_size_bytes % self.be_width:
            raise SpecViolation(
                '%s: bank size %d is not a multiple of BeWidth %d'
                % (name, bank_size_bytes, self.be_width))
        self.words_per_bank = bank_size_bytes // self.be_width
        if not is_pow2(self.words_per_bank):
            raise SpecViolation(
                '%s: words_per_bank must be a power of two, got %d'
                % (name, self.words_per_bank))

        self.bank_bits = clog2(banks_per_tile)
        self.tile_bits = clog2(tiles_per_group)
        self.gy_bits = clog2(num_y)
        self.gx_bits = clog2(num_x)
        self.word_bits = clog2(self.words_per_bank)

        self.bank_pos = self.byte_bits
        self.tile_pos = self.bank_pos + self.bank_bits
        self.gy_pos = self.tile_pos + self.tile_bits
        self.gx_pos = self.gy_pos + self.gy_bits
        self.word_pos = self.gx_pos + self.gx_bits
        self.addr_bits = self.word_pos + self.word_bits

        # Words per (group, word-plane): the contiguous run the DMA sees.
        self.words_per_group_plane = tiles_per_group * banks_per_tile
        self.total_words = (self.words_per_group_plane * self.num_groups
                            * self.words_per_bank)
        self.total_bytes = self.total_words * self.be_width

        # Sanity: the packed fields must tile the whole L1 exactly.
        assert self.total_bytes == 1 << self.addr_bits

        self.max_level = min(clog2(num_x), clog2(num_y))

    # -- canonical layout -------------------------------------------------
    def compose(self, word: int, gx: int, gy: int, tile: int, bank: int,
                byte: int = 0) -> int:
        """Pack canonical fields into a TCDM-relative byte address."""
        return ((word << self.word_pos) | (gx << self.gx_pos)
                | (gy << self.gy_pos) | (tile << self.tile_pos)
                | (bank << self.bank_pos) | byte)

    def decode(self, addr: int) -> 'Phys':
        """Independent field slice of a TCDM-relative byte address."""
        return Phys(
            word=(addr >> self.word_pos) & ((1 << self.word_bits) - 1),
            gx=(addr >> self.gx_pos) & ((1 << self.gx_bits) - 1),
            gy=(addr >> self.gy_pos) & ((1 << self.gy_bits) - 1),
            tile=(addr >> self.tile_pos) & ((1 << self.tile_bits) - 1),
            bank=(addr >> self.bank_pos) & ((1 << self.bank_bits) - 1),
            byte=addr & ((1 << self.byte_bits) - 1),
        )

    def group_id(self, gx: int, gy: int) -> int:
        """Flat group id, {gx, gy} with gy low.

        hardware/src/mempool_cluster_floonoc_wrapper.sv:249-250 builds both the
        group_id_i port and the FlooNoC node id as ``x*NumY + y``.
        """
        return (gx << self.gy_bits) | gy

    def describe_fields(self) -> List[Tuple[str, int, int]]:
        """(name, high_bit, low_bit) of every canonical field, low first."""
        out = []
        for name, pos, width in (
                ('byte', 0, self.byte_bits),
                ('bank', self.bank_pos, self.bank_bits),
                ('tile', self.tile_pos, self.tile_bits),
                ('gy', self.gy_pos, self.gy_bits),
                ('gx', self.gx_pos, self.gx_bits),
                ('word', self.word_pos, self.word_bits)):
            out.append((name, pos + width - 1, pos))
        return out

    def __repr__(self) -> str:
        return ('Geometry(%s: %dx%d groups, %d tiles/group, %d banks/tile, '
                '%d words/bank)' % (self.name, self.num_x, self.num_y,
                                    self.tiles_per_group,
                                    self.banks_per_tile,
                                    self.words_per_bank))


class Phys(NamedTuple):
    """A physical location in the canonical layout."""
    word: int
    gx: int
    gy: int
    tile: int
    bank: int
    byte: int

    def bank_key(self) -> Tuple[int, int, int, int]:
        """The physical bank, ignoring the word inside it."""
        return (self.gx, self.gy, self.tile, self.bank)


# Injectable deviations from the specification.  They exist so the self-check
# can prove that the invariants are load-bearing: a test that only asserts the
# good path is not evidence.
MODE_SPEC = 'spec'
# Invariant #4 broken, reading 1: mask the FLAT row-major group id and OR a
# single packed 2k-bit vc_group field at the bottom of the group field.
MODE_FLAT_MASK_PACKED = 'flat_mask_packed'
# Invariant #4 broken, reading 2: keep the per-axis field placement but derive
# both axis bases by masking the flat group id.
MODE_FLAT_MASK_AXIS = 'flat_mask_axis'
# No mask at all: base = the requester's own coordinates.
MODE_NO_MASK = 'no_mask'

ALL_MODES = (MODE_SPEC, MODE_FLAT_MASK_PACKED, MODE_FLAT_MASK_AXIS,
             MODE_NO_MASK)


class LevelRule:
    """One fixed fractal level: an SxS square of groups plus its chunk.

    Per docs/plan/wp2-wp4-design-rationale.md section "VC address model", a
    level's rule carries `{enable, chunk_base}` as PROGRAMMABLE state and
    `{window-range, k}` as FIXED state; `base` is COMPUTED from the
    requester's own group coordinates.  `chunk_rows` is the level's physical
    budget expressed in word-planes: each participating group contributes
    `chunk_rows * tiles_per_group * banks_per_tile` words to the level heap.
    """

    def __init__(self, geom: Geometry, k: int, chunk_base: int = 0,
                 chunk_rows: Optional[int] = None, validate: bool = True):
        self.geom = geom
        self.k = k
        self.side = 1 << k
        self.chunk_base = chunk_base
        self.chunk_rows = (geom.words_per_bank if chunk_rows is None
                           else chunk_rows)

        # Offset field layout: identical to the canonical layout except that
        # the group field is 2k bits wide (roadmap section T2.1).
        g = geom
        self.o_bank_pos = g.byte_bits
        self.o_tile_pos = self.o_bank_pos + g.bank_bits
        self.o_vcgy_pos = self.o_tile_pos + g.tile_bits
        self.o_vcgx_pos = self.o_vcgy_pos + k
        self.o_row_pos = self.o_vcgx_pos + k
        self.row_bits = clog2(self.chunk_rows) if self.chunk_rows else 0
        self.offset_bits = self.o_row_pos + self.row_bits
        self.offset_bytes = 1 << self.offset_bits
        self.offset_words = self.offset_bytes >> g.byte_bits

        if validate:
            self.validate()

    # -- descriptor legality (roadmap section T2.3 item 4) ----------------
    def validate(self) -> None:
        g = self.geom
        if self.k < 0 or self.k > g.max_level:
            raise SpecViolation(
                'level k=%d out of range for a %dx%d mesh (max %d)'
                % (self.k, g.num_x, g.num_y, g.max_level))
        if not is_pow2(self.chunk_rows):
            raise SpecViolation(
                'chunk extent must be a power of two, got %d rows'
                % self.chunk_rows)
        if self.chunk_base % self.chunk_rows:
            raise SpecViolation(
                'INVARIANT #5: chunk_base %d is not aligned to the %d-row '
                'chunk extent; chunk_base|chunk_row would fold the chunk '
                'onto itself' % (self.chunk_base, self.chunk_rows))
        if self.chunk_base < 0:
            raise SpecViolation('chunk_base must be non-negative')
        if self.chunk_base + self.chunk_rows > g.words_per_bank:
            raise SpecViolation(
                'over-capacity: chunk_base %d + %d rows exceeds the %d-word '
                'bank depth' % (self.chunk_base, self.chunk_rows,
                                g.words_per_bank))

    @property
    def aperture_bytes(self) -> int:
        return self.offset_bytes

    def highest_legal_chunk_base(self) -> int:
        return self.geom.words_per_bank - self.chunk_rows

    def rect_id(self, gx: int, gy: int) -> Tuple[int, int]:
        """Rectangle-grid coordinate (rx, ry) = (gx>>k, gy>>k).

        The ABI id encoding of roadmap section T2.1 ("id encoding") and
        rationale section "VC address model" ("id encoding = rectangle-grid
        coord"); the base is recovered as coord<<k.
        """
        return (gx >> self.k, gy >> self.k)

    def base_of(self, gx: int, gy: int,
                mode: str = MODE_SPEC) -> Tuple[int, int]:
        """The rectangle base, per axis (INVARIANT #4)."""
        g = self.geom
        mask = ~(self.side - 1)
        if mode == MODE_SPEC:
            return (gx & mask, gy & mask)
        if mode == MODE_NO_MASK:
            return (gx, gy)
        flat = g.group_id(gx, gy) & mask
        if mode == MODE_FLAT_MASK_AXIS:
            return (flat >> g.gy_bits, flat & ((1 << g.gy_bits) - 1))
        if mode == MODE_FLAT_MASK_PACKED:
            return (flat, 0)  # interpreted as a packed group id, see permute
        raise SpecViolation('unknown mode %r' % mode)

    def split(self, offset: int) -> Dict[str, int]:
        """Slice a VC-relative byte offset into its fields."""
        g = self.geom
        km = (1 << self.k) - 1
        rm = (1 << self.row_bits) - 1
        return {
            'byte': offset & ((1 << g.byte_bits) - 1),
            'bank': (offset >> self.o_bank_pos) & ((1 << g.bank_bits) - 1),
            'tile': (offset >> self.o_tile_pos) & ((1 << g.tile_bits) - 1),
            'vc_gy': (offset >> self.o_vcgy_pos) & km,
            'vc_gx': (offset >> self.o_vcgx_pos) & km,
            'chunk_row': (offset >> self.o_row_pos) & rm,
        }

    # -- the permuter -----------------------------------------------------
    def permute(self, offset: int, gx_i: int, gy_i: int,
                mode: str = MODE_SPEC) -> Phys:
        """Map a VC-relative offset onto a physical location.

        This is the roadmap section T2.1 pseudo-code, verbatim in structure:
        slice the offset like the global interleave but with a 2k-bit group
        field, compute the base from the requester's own coordinates, and
        compose with OR.  The field slicing is inlined rather than routed
        through `split` only because the exhaustive sweeps call this tens of
        millions of times; `test_split_agrees` in vc_selfcheck.py pins the two
        together.
        """
        g = self.geom
        km = (1 << self.k) - 1
        byte = offset & ((1 << g.byte_bits) - 1)
        bank = (offset >> self.o_bank_pos) & ((1 << g.bank_bits) - 1)
        tile = (offset >> self.o_tile_pos) & ((1 << g.tile_bits) - 1)
        vc_gy = (offset >> self.o_vcgy_pos) & km
        vc_gx = (offset >> self.o_vcgx_pos) & km
        chunk_row = (offset >> self.o_row_pos) & ((1 << self.row_bits) - 1)

        if mode == MODE_SPEC:
            side1 = self.side - 1
            out_gx = (gx_i & ~side1) | vc_gx
            out_gy = (gy_i & ~side1) | vc_gy
        elif mode == MODE_FLAT_MASK_PACKED:
            # The "mask the flat group id" bug in its most literal form: one
            # 2k-bit vc_group field ORed at the bottom of the group field.
            flat_base = g.group_id(gx_i, gy_i) & ~(self.side - 1)
            out_group = flat_base | (vc_gx << self.k) | vc_gy
            out_gx = out_group >> g.gy_bits
            out_gy = out_group & ((1 << g.gy_bits) - 1)
        else:
            base_gx, base_gy = self.base_of(gx_i, gy_i, mode)
            out_gx = base_gx | vc_gx
            out_gy = base_gy | vc_gy

        out_word = self.chunk_base | chunk_row
        return Phys(word=out_word, gx=out_gx, gy=out_gy, tile=tile,
                    bank=bank, byte=byte)

    def permute_addr(self, offset: int, gx_i: int, gy_i: int,
                     mode: str = MODE_SPEC) -> int:
        """`permute`, returning the packed TCDM-relative byte address."""
        g = self.geom
        km = (1 << self.k) - 1
        bank = (offset >> self.o_bank_pos) & ((1 << g.bank_bits) - 1)
        tile = (offset >> self.o_tile_pos) & ((1 << g.tile_bits) - 1)
        vc_gy = (offset >> self.o_vcgy_pos) & km
        vc_gx = (offset >> self.o_vcgx_pos) & km
        chunk_row = (offset >> self.o_row_pos) & ((1 << self.row_bits) - 1)

        if mode == MODE_SPEC:
            side1 = self.side - 1
            out_gx = (gx_i & ~side1) | vc_gx
            out_gy = (gy_i & ~side1) | vc_gy
            return (((self.chunk_base | chunk_row) << g.word_pos)
                    | (out_gx << g.gx_pos) | (out_gy << g.gy_pos)
                    | (tile << g.tile_pos) | (bank << g.bank_pos)
                    | (offset & ((1 << g.byte_bits) - 1)))
        p = self.permute(offset, gx_i, gy_i, mode)
        return g.compose(p.word, p.gx, p.gy, p.tile, p.bank, p.byte)

    # -- independent arithmetic twin (the alias-equivalence oracle) -------
    def alias_addr(self, offset: int, gx_i: int, gy_i: int) -> int:
        """The same address, computed with + and * instead of | and shifts.

        This is the DMA / global-alias view: the runtime knows the rectangle
        base and the chunk base and reconstructs the global address by
        arithmetic (docs/plan/wp2-wp4-design-rationale.md section "VC address
        model", subsection "Cross-VC L1<->L1 DMA addressing").  It agrees with
        `permute_addr` exactly when invariants #4 and #5 hold, which is the
        whole point of both invariants: they are what makes OR equal +.
        """
        g = self.geom
        byte = offset % g.be_width
        word_off = offset // g.be_width

        bank = word_off % g.banks_per_tile
        rest = word_off // g.banks_per_tile
        tile = rest % g.tiles_per_group
        rest = rest // g.tiles_per_group
        vc_gy = rest % self.side
        rest = rest // self.side
        vc_gx = rest % self.side
        chunk_row = rest // self.side
        if chunk_row >= self.chunk_rows:
            raise SpecViolation(
                'offset 0x%x is outside the level-%d aperture (%d chunk rows)'
                % (offset, self.k, self.chunk_rows))

        base_gx = (gx_i // self.side) * self.side
        base_gy = (gy_i // self.side) * self.side

        out_gx = base_gx + vc_gx
        out_gy = base_gy + vc_gy
        out_word = self.chunk_base + chunk_row

        flat_word = (out_word * g.num_groups * g.words_per_group_plane
                     + out_gx * g.num_y * g.words_per_group_plane
                     + out_gy * g.words_per_group_plane
                     + tile * g.banks_per_tile
                     + bank)
        return flat_word * g.be_width + byte

    # -- the strided descriptor the DMA actually programs ------------------
    def nd_descriptor(self, gx_i: int, gy_i: int) -> Dict[str, object]:
        """The global ND-strided shape of this level's aperture.

        rationale section "Cross-VC L1<->L1 DMA addressing" states the 2x2
        terapool case as: dim0 = size_y*256 contiguous words, dim1 = x-group
        stride 0x1000, dim2 = word-plane stride 0x4000.  This generalises it.
        """
        g = self.geom
        base_gx = (gx_i // self.side) * self.side
        base_gy = (gy_i // self.side) * self.side
        start = g.compose(self.chunk_base, base_gx, base_gy, 0, 0, 0)
        run_bytes = self.side * g.words_per_group_plane * g.be_width
        return {
            'start': start,
            'dims': [
                {'name': 'contiguous', 'stride': 1, 'count': run_bytes},
                {'name': 'x_group', 'stride': 1 << g.gx_pos,
                 'count': self.side},
                {'name': 'word_plane', 'stride': 1 << g.word_pos,
                 'count': self.chunk_rows},
            ],
        }

    def nd_addr(self, offset: int, gx_i: int, gy_i: int) -> int:
        """Walk `nd_descriptor` with a linear index, mixed-radix."""
        desc = self.nd_descriptor(gx_i, gy_i)
        addr = desc['start']
        rest = offset
        for dim in desc['dims']:
            idx = rest % dim['count']
            rest //= dim['count']
            addr += idx * dim['stride']
        if rest:
            raise SpecViolation(
                'offset 0x%x overflows the level-%d descriptor' %
                (offset, self.k))
        return addr

    def __repr__(self) -> str:
        return ('LevelRule(k=%d, %dx%d groups, chunk_base=%d, chunk_rows=%d, '
                'aperture=%d B)' % (self.k, self.side, self.side,
                                    self.chunk_base, self.chunk_rows,
                                    self.aperture_bytes))


# ---------------------------------------------------------------------------
# Geometry fixtures.  These are the ONLY place flavour numbers appear.
# Every value below was read out of the make configuration, not assumed:
#   make -f <probe> show config=<flavour>  =>  num_cores, num_cores_per_tile,
#   num_groups, num_x, banking_factor, num_fus_per_core, l1_bank_size
#   NumTiles         = num_cores / num_cores_per_tile
#   NumTilesPerGroup = NumTiles / num_groups
#   NumBanks         = num_cores * num_fus_per_core * banking_factor
#                                        (hardware/src/mempool_pkg.sv:70)
#   NumBanksPerTile  = NumBanks / NumTiles
#                                        (hardware/src/mempool_pkg.sv:71)
# ---------------------------------------------------------------------------
GEOMETRIES: Dict[str, Geometry] = {
    # config/terapool.mk: 1024 cores, 4 cores/tile, 16 groups, num_x=4,
    # banking_factor=4, n_ipu=n_fpu=1 => 256 tiles, 16 tiles/group,
    # 4096 banks, 16 banks/tile; config/config.mk:41 l1_bank_size=1024 B.
    # Matches the roadmap's canonical-layout example exactly:
    # byte[1:0] bank[5:2] tile[9:6] gy[11:10] gx[13:12] word[21:14].
    'terapool': Geometry(
        name='terapool', num_x=4, num_y=4, tiles_per_group=16,
        banks_per_tile=16, bank_size_bytes=1024,
        source='config/terapool.mk'),

    # config/dse_s1_8x8_winner.mk: 512 cores, 64 groups, num_x=8,
    # num_cores_per_tile=2 and banking_factor=8 inherited from
    # config/dse_s1_common.mk, n_ipu=n_fpu=4 from
    # config/tensorpool256_spatz4.mk, l1_bank_size=4096 B from
    # config/dse_s1_common.mk.  => 256 tiles, 4 tiles/group,
    # 16384 banks, 64 banks/tile, 1024 words/bank, 64 MiB L1.
    # NOTE: 64 banks/tile, not 32 - see scripts/vc/README.md.
    'monsterpool8x8': Geometry(
        name='monsterpool8x8', num_x=8, num_y=8, tiles_per_group=4,
        banks_per_tile=64, bank_size_bytes=4096,
        source='config/dse_s1_8x8_winner.mk'),

    # config/dse_s1_winner.mk, the 4x4 stage-1 winner: same group internals as
    # the 8x8 above but 16 groups.  Cheap, and it exercises a geometry where
    # tiles/group (4) < banks/tile (64).
    'dse_s1_winner': Geometry(
        name='dse_s1_winner', num_x=4, num_y=4, tiles_per_group=4,
        banks_per_tile=64, bank_size_bytes=4096,
        source='config/dse_s1_winner.mk'),

    # config/monster_baseline.mk: 128 cores, 2/tile, 16 groups, num_x=4,
    # banking_factor=4, n_fus=4, l1_bank_size=2048 => 64 tiles,
    # 4 tiles/group, 2048 banks, 32 banks/tile.  Small; used for the
    # fast smoke run.
    'monster_baseline': Geometry(
        name='monster_baseline', num_x=4, num_y=4, tiles_per_group=4,
        banks_per_tile=32, bank_size_bytes=2048,
        source='config/monster_baseline.mk'),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Describe the VC permuter field layout of a geometry.')
    ap.add_argument('--geometry', default='terapool',
                    choices=sorted(GEOMETRIES), help='geometry fixture')
    args = ap.parse_args()

    g = GEOMETRIES[args.geometry]
    print(repr(g))
    print('source: %s' % g.source)
    print('L1 = %d words = %d bytes (%d address bits)'
          % (g.total_words, g.total_bytes, g.addr_bits))
    print('canonical layout (low field first):')
    for name, hi, lo in g.describe_fields():
        print('  %-5s [%2d:%2d]' % (name, hi, lo))
    print('levels:')
    for k in range(g.max_level + 1):
        rule = LevelRule(g, k)
        print('  k=%d  %dx%d groups  max aperture %d B  chunk_row bits %d'
              % (k, rule.side, rule.side, rule.aperture_bytes,
                 rule.row_bits))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
