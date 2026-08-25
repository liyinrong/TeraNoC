#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Step 0 of the Morton (Z-order) group-numbering change: prove the encoding
before any RTL is touched.

Today the group field of an address is the raster concatenation {gx, gy}, so a
level-k rectangle is NOT a contiguous range of group ids and the base mask has
to be applied per axis (INVARIANT #4,
docs/plan/wp2-wp4-implementation-roadmap.md:68-69).  Under Morton the field is
the bit interleave of gx and gy, which makes a rectangle a contiguous id range
and collapses the per-axis mask into one flat mask.

This module implements the Morton permuter independently of vc_permute.py and
checks the properties the raster oracle was proven to have, plus the two new
claims that motivate the change.  It is deliberately a separate file: an
independent derivation is worth more than an edit to the model it is meant to
cross-check.
"""

import sys
from typing import Dict, List, Tuple


def clog2(v: int) -> int:
    return (v - 1).bit_length()


def morton(gx: int, gy: int, bits: int) -> int:
    """Interleave, gy in the even (low) positions to match the raster layout's
    gy-low convention (mempool_cluster_floonoc_wrapper.sv:249)."""
    v = 0
    for i in range(bits):
        v |= ((gy >> i) & 1) << (2 * i)
        v |= ((gx >> i) & 1) << (2 * i + 1)
    return v


def unmorton(gid: int, bits: int) -> Tuple[int, int]:
    gx = gy = 0
    for i in range(bits):
        gy |= ((gid >> (2 * i)) & 1) << i
        gx |= ((gid >> (2 * i + 1)) & 1) << i
    return gx, gy


class Geom:
    def __init__(self, name, num_x, num_y, tiles_per_group, banks_per_tile,
                 words_per_bank):
        assert num_x == num_y, "Morton needs a square mesh"
        self.name = name
        self.num_x, self.num_y = num_x, num_y
        self.groups = num_x * num_y
        self.tiles_per_group = tiles_per_group
        self.banks_per_tile = banks_per_tile
        self.words_per_bank = words_per_bank
        self.byte_bits = 2
        self.bank_bits = clog2(banks_per_tile)
        self.tile_bits = clog2(tiles_per_group)
        self.axis_bits = clog2(num_x)
        self.group_bits = 2 * self.axis_bits
        self.word_bits = clog2(words_per_bank)
        # canonical: [ word | group | tile | bank | byte ]
        self.bank_pos = self.byte_bits
        self.tile_pos = self.bank_pos + self.bank_bits
        self.group_pos = self.tile_pos + self.tile_bits
        self.word_pos = self.group_pos + self.group_bits
        self.max_level = self.axis_bits

    def compose(self, word, gid, tile, bank, byte=0):
        return ((word << self.word_pos) | (gid << self.group_pos) |
                (tile << self.tile_pos) | (bank << self.bank_pos) | byte)

    def row_bytes(self):
        """One bank row of ONE group: the byte|bank|tile span."""
        return 1 << self.group_pos

    def l1_bytes(self):
        return self.groups * self.words_per_bank * self.row_bytes()

    def __repr__(self):
        return (f"{self.name}: {self.num_x}x{self.num_y} groups, "
                f"{self.tiles_per_group} tiles/grp, {self.banks_per_tile} "
                f"banks/tile, {self.words_per_bank} words/bank, "
                f"L1 {self.l1_bytes() >> 20} MiB")


class MortonRule:
    """One fractal level under Morton numbering.

    The offset is [ row | vc_group(2k) | tile | bank | byte ] -- the two k-bit
    per-axis fields of the raster form merge into ONE 2k-bit field, because a
    rectangle is now a contiguous id range.
    """

    def __init__(self, g: Geom, k: int):
        self.g, self.k = g, k
        self.side = 1 << k
        self.ngroups = 1 << (2 * k)
        self.vcg_pos = g.tile_pos + g.tile_bits      # == g.group_pos
        self.row_pos = self.vcg_pos + 2 * k
        self.aperture = 1 << (self.row_pos + g.word_bits)

    def row_stride(self) -> int:
        """Offset distance between consecutive bank rows of the RECTANGLE."""
        return self.g.row_bytes() << (2 * self.k)

    def permute(self, offset: int, gid_i: int) -> int:
        """The hardware path: one mask, one OR -- no per-axis split."""
        g = self.g
        byte = offset & ((1 << g.byte_bits) - 1)
        bank = (offset >> g.bank_pos) & ((1 << g.bank_bits) - 1)
        tile = (offset >> g.tile_pos) & ((1 << g.tile_bits) - 1)
        vcg = (offset >> self.vcg_pos) & (self.ngroups - 1)
        row = offset >> self.row_pos
        out_gid = (gid_i & ~(self.ngroups - 1)) | vcg
        return g.compose(row, out_gid, tile, bank, byte)

    def alias(self, offset: int, gid_i: int) -> int:
        """The software path, by independent arithmetic. This is what
        vc_global_addr computes; it must equal permute()."""
        g = self.g
        rb = g.row_bytes()
        byte_in_row = offset % rb
        rest = offset // rb
        vcg = rest % self.ngroups
        row = rest // self.ngroups
        base = gid_i - (gid_i % self.ngroups)
        return byte_in_row + (base + vcg) * rb + row * rb * g.groups

    def descriptor(self, gid_i: int, rows: int) -> Dict[str, int]:
        """Global ND descriptor. The claim under test: 2-D, with dim0 the WHOLE
        rectangle row, because the rectangle's groups are now contiguous."""
        base = self.alias(0, gid_i)
        return {"base": base, "len": self.row_stride(), "count": rows,
                "stride": self.g.row_bytes() * self.g.groups}


def check(g: Geom, k: int, word_step: int = 1) -> List[str]:
    r = MortonRule(g, k)
    fails: List[str] = []
    rb = g.row_bytes()

    # one representative requester per rectangle, plus every requester at k=0
    reps = ([gid for gid in range(g.groups)] if k == 0 else
            [b for b in range(0, g.groups, r.ngroups)] +
            [b + r.ngroups - 1 for b in range(0, g.groups, r.ngroups)])

    for gid_i in reps:
        seen: Dict[int, int] = {}
        base_gid = gid_i - (gid_i % r.ngroups)
        rect = set(range(base_gid, base_gid + r.ngroups))
        step_b = rb * word_step
        for off in range(0, r.aperture, step_b):
            a = r.permute(off, gid_i)
            b = r.alias(off, gid_i)
            if a != b:
                fails.append(f"{g.name} k={k} gid={gid_i} off={off:#x} "
                             f"permute={a:#x} != alias={b:#x}")
                break
            out_gid = (a >> g.group_pos) & ((1 << g.group_bits) - 1)
            if out_gid not in rect:
                fails.append(f"{g.name} k={k} gid={gid_i} off={off:#x} "
                             f"escaped to gid {out_gid}, rect {sorted(rect)}")
                break
            if a in seen:
                fails.append(f"{g.name} k={k} gid={gid_i} collision at "
                             f"{a:#x}: off {seen[a]:#x} and {off:#x}")
                break
            seen[a] = off
    return fails


def check_contiguity(g: Geom, k: int) -> List[str]:
    """The claim that buys the 2-D descriptor: a rectangle's group ids form
    a contiguous range, so one rectangle bank row is one contiguous run."""
    r = MortonRule(g, k)
    fails = []
    for b in range(0, g.groups, r.ngroups):
        ids = sorted(range(b, b + r.ngroups))
        if ids != list(range(ids[0], ids[0] + len(ids))):
            fails.append(f"{g.name} k={k}: rectangle at {b} not contiguous")
        coords = [unmorton(i, g.axis_bits) for i in ids]
        xs, ys = {c[0] for c in coords}, {c[1] for c in coords}
        if len(xs) != r.side or len(ys) != r.side:
            fails.append(f"{g.name} k={k}: rectangle at {b} is not "
                         f"{r.side}x{r.side}: {sorted(coords)}")
        bx, by = min(xs), min(ys)
        if any(x < bx or x >= bx + r.side or y < by or y >= by + r.side
               for x, y in coords):
            fails.append(f"{g.name} k={k}: rect at {b} not square")
    return fails


GEOMS = [
    Geom("terapool", 4, 4, 16, 16, 256),
    Geom("dse_s1_winner", 4, 4, 4, 64, 1024),
    Geom("monster_baseline", 4, 4, 4, 32, 512),
    Geom("monsterpool8x8", 8, 8, 4, 64, 1024),
]


def main() -> int:
    total = 0
    for g in GEOMS:
        print(f"\n{g!r}")
        for k in range(0, g.max_level):  # top level has no window
            r = MortonRule(g, k)
            # keep the sweep bounded on the big apertures
            step = 1 if r.aperture <= (1 << 22) else r.aperture >> 22
            f = check(g, k, step) + check_contiguity(g, k)
            d = r.descriptor(0, 4)
            shape = "2-D" if d["len"] == r.row_stride() else "3-D"
            print(f"  k={k} side={r.side} "
                  f"aperture={r.aperture >> 10:>7} KiB "
                  f"step={step:<5} dim0={d['len']:>6} B contiguous "
                  f"{shape}  "
                  f"{'OK' if not f else 'FAIL'}")
            for line in f[:3]:
                print("      " + line)
            total += len(f)
    print(f"\ntotal failures: {total}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
