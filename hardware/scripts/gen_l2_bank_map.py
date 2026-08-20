#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Generate the L2/HBM bank -> mesh perimeter port map for an NxN group mesh.

HBM sits on the cluster perimeter, so an NxN group mesh offers 4N channels for
N^2 groups.  Which bank attaches to which perimeter port is a placement choice,
and it matters: the distributed DMA gives group g the L1 stripe that the L2
fold puts in bank (g // fold) % nb_banks, so a bad placement makes every group
walk the mesh to reach its own bank.

This script picks the placement that minimises total group-to-bank hops, and
emits it as a SystemVerilog package that `terapool_cluster_floonoc_wrapper.sv`
indexes directly.  The same map must appear in the floogen YAML, which decides
where the endpoint actually lives in the network -- `--check-yml` verifies the
two agree, because a silent disagreement misroutes every L2 access.

Before this existed the wrapper hardcoded the 4x4 map as index arithmetic
(`y+12`, `5-x`, `x+6`, `13-x`).  Those constants only tile the channel space
when 4N == N^2, i.e. only at N=4; at N=8 they collide and the design does not
elaborate.

Usage:
    gen_l2_bank_map.py --n 8 -o hardware/generated/l2_bank_map_pkg.sv
    gen_l2_bank_map.py --n 8 --check-yml config/floo_noc_terapool_8x8.yml
    gen_l2_bank_map.py --report --n 8
"""

import argparse
import collections
import re
import sys

BIG = 10 ** 6

# floo_pkg direction encoding (hw/floo_pkg.sv:46-50).
DIR_CODE = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
DIR_NAME = {'N': 'North', 'E': 'East', 'S': 'South', 'W': 'West'}

# The map the RTL shipped for the 4x4 mesh, transcribed from the pre-generator
# wrapper. It is one of several equal-cost optima (8 hops); the solver finds a
# different one with a better worst case (max 1 vs 2). We PIN the shipped map
# so that regenerating for N=4 leaves every existing 4x4 build bit-identical --
# the DSE calibration was measured against it.
SHIPPED_4X4 = [('W', 0, 0), ('W', 0, 1), ('W', 0, 2), ('W', 0, 3),
               ('S', 1, 0), ('S', 0, 0), ('N', 0, 3), ('N', 1, 3),
               ('S', 2, 0), ('S', 3, 0), ('N', 3, 3), ('N', 2, 3),
               ('E', 3, 0), ('E', 3, 1), ('E', 3, 2), ('E', 3, 3)]


def hungarian(cost):
    """Minimum-cost perfect matching, O(n^3) (Jonker-Volgenant style)."""
    n = len(cost)
    inf = float('inf')
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], inf, -1
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    out = [0] * n
    for j in range(1, n + 1):
        out[p[j] - 1] = j - 1
    return out


def edge_ports(n):
    """Every perimeter port as (direction, router_x, router_y).

    Corner routers carry two: [0][0] has both a West and a South port.
    """
    return ([('W', 0, y) for y in range(n)]
            + [('E', n - 1, y) for y in range(n)]
            + [('S', x, 0) for x in range(n)]
            + [('N', x, n - 1) for x in range(n)])


def group_pos(n, g):
    """RTL group id -> router coord. g = x*N + y, so ids run down a column."""
    return g // n, g % n


def served_groups(n, bank, nb_banks, fold):
    """Groups whose DMA stripe lands in `bank`.

    Mirrors axi_L2_interleaver.sv: the bank-select field is taken from
    addr / (L2BankBeWidth * Interleave), and config.mk sizes that granule as
    groups_per_l2_bank * NumBanksPerGroup * 4, so group g lands in bank
    (g // fold) % nb_banks with fold = num_groups / l2_banks.
    """
    return [g for g in range(n * n) if (g // fold) % nb_banks == bank]


def solve(n, nb_banks, fold, max_dist=None):
    """Assign each bank a perimeter port.

    Banks may be fewer than perimeter ports (a 2x2 mesh has 8 ports but the
    small flavors declare 4 banks), so the cost matrix is padded to square with
    zero-cost dummy banks that soak up the leftover ports.
    """
    ports = edge_ports(n)
    if nb_banks > len(ports):
        raise SystemExit('%d banks do not fit on %d perimeter ports at N=%d'
                         % (nb_banks, len(ports), n))
    cost = []
    for b in range(nb_banks):
        served = [group_pos(n, g) for g in served_groups(n, b, nb_banks, fold)]
        row = []
        for _, px, py in ports:
            d = [abs(px - gx) + abs(py - gy) for gx, gy in served]
            # Lexicographic objective folded into one float: total hops first,
            # then spread (sum of squares), then a column tiebreak.
            tie = sum(e * e for e in d) * 100 + abs(px - served[0][0])
            c = sum(d) + tie * 1e-9
            if max_dist is not None and max(d) > max_dist:
                c += BIG
            row.append(c)
        cost.append(row)
    cost += [[0.0] * len(ports) for _ in range(len(ports) - nb_banks)]
    assign = hungarian(cost)
    return [ports[assign[b]] for b in range(nb_banks)]


def best(n, nb_banks, fold):
    """Solve, then tighten the worst-case distance as far as it will go."""
    place = solve(n, nb_banks, fold)
    base = total(distances(n, place, nb_banks, fold))
    worst = max(distances(n, place, nb_banks, fold).values())
    for t in range(1, worst):
        try_place = solve(n, nb_banks, fold, max_dist=t)
        d = distances(n, try_place, nb_banks, fold)
        if max(d.values()) <= t and total(d) == base:
            return try_place
    return place


def placement(n, nb_banks=None, fold=None):
    """The map to emit: pinned at N=4, solved otherwise."""
    nb_banks = nb_banks or 4 * n
    fold = fold or (n * n // nb_banks)
    if n == 4 and nb_banks == 16 and fold == 1:
        return list(SHIPPED_4X4)
    return best(n, nb_banks, fold)


def distances(n, place, nb_banks, fold):
    out = {}
    for b, (_, px, py) in enumerate(place):
        for g in served_groups(n, b, nb_banks, fold):
            gx, gy = group_pos(n, g)
            out[g] = abs(px - gx) + abs(py - gy)
    return out


def total(d):
    return sum(d.values())


def emit_sv(n, place):
    lines = [
        '// Copyright 2026 ETH Zurich and University of Bologna.',
        '// Solderpad Hardware License, Version 0.51, see LICENSE for'
        ' details.',
        '// SPDX-License-Identifier: SHL-0.51',
        '',
        '// GENERATED by hardware/scripts/gen_l2_bank_map.py -- do not edit.',
        '// L2/HBM bank -> mesh perimeter port map for a %dx%d group mesh.'
        % (n, n),
        '// Must stay identical to the connections in the floogen YAML.',
        '',
        'package l2_bank_map_pkg;',
        '',
        '  localparam int unsigned NumL2BankPorts = %d;' % len(place),
        '',
        '  // Router coordinates of the perimeter port each bank attaches to.',
        '  localparam int unsigned L2BankPortX [NumL2BankPorts] = \'{',
    ]
    lines.append('      ' + ', '.join(str(p[1]) for p in place))
    lines += ['  };', '',
              '  localparam int unsigned L2BankPortY [NumL2BankPorts] = \'{']
    lines.append('      ' + ', '.join(str(p[2]) for p in place))
    lines += ['  };', '',
              '  // floo_pkg::route_direction_e --'
              ' North 0, East 1, South 2, West 3.',
              '  localparam int unsigned L2BankPortDir [NumL2BankPorts] = \'{']
    lines.append('      ' + ', '.join(str(DIR_CODE[p[0]]) for p in place))
    lines += ['  };', '',
              '  // The host/peripheral chimney shares the channel on the',
              '  // South port of the corner router [0][0]. Which bank id',
              '  // that is depends on the placement (5 on the 4x4 mesh, 4',
              '  // on the 8x8), so it is derived rather than hardcoded.',
              '  localparam int unsigned HostL2Bank = %d;' % host_bank(place),
              '', 'endpackage']
    return '\n'.join(lines) + '\n'


def host_bank(place):
    """Bank id attached to South of router [0][0] -- the host/peripheral port.

    Falls back to 0 when nothing lands there, which happens only on the small
    flavors whose cluster wrapper has no perimeter HBM ring and never reads it.
    """
    for b, p in enumerate(place):
        if p == ('S', 0, 0):
            return b
    return 0


def parse_yml(path):
    """Pull the bank -> (dir, x, y) map out of a floogen YAML."""
    txt = open(path).read()
    out = {}
    pat = re.compile(
        r'-\s+src:\s*"(?:host_)?hbm(\d+)(?:_peri)?"\s*\n'
        r'\s+dst:\s*"group_router"\s*\n'
        r'\s+dst_idx:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*\n'
        r'\s+dst_dir:\s*"(North|East|South|West)"')
    for m in pat.finditer(txt):
        b, x, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        d = m.group(4)
        out[b] = (d[0] if d != 'North' else 'N', x, y)
    return out


def report(n, place, nb_banks, fold):
    d = distances(n, place, nb_banks, fold)
    hist = dict(sorted(collections.Counter(d.values()).items()))
    print('N=%d  %d groups on %d channels (fold %d)'
          % (n, n * n, len(place), fold))
    print('  hops: total %d, mean %.3f, max %d, histogram %s'
          % (total(d), total(d) / float(n * n), max(d.values()), hist))
    for b, p in enumerate(place):
        print('  bank %2d -> %-5s router [%d][%d]   groups %s'
              % (b, DIR_NAME[p[0]], p[1], p[2],
                 ','.join(str(g) for g in
                          served_groups(n, b, len(place), fold))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, required=True,
                    help='mesh side (groups per axis)')
    ap.add_argument('--banks', type=int, default=None,
                    help='L2 banks (default 4N)')
    ap.add_argument('--fold', type=int, default=None,
                    help='groups per L2 bank (default num_groups/banks)')
    ap.add_argument('-o', '--output', default=None,
                    help='SystemVerilog package to write')
    ap.add_argument('--check-yml', default=None,
                    help='verify against a floogen YAML')
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()

    nb_banks = args.banks or 4 * args.n
    fold = args.fold or (args.n * args.n // nb_banks)
    if args.n * args.n != nb_banks * fold:
        raise SystemExit('num_groups (%d) must equal banks (%d) * fold (%d)'
                         % (args.n * args.n, nb_banks, fold))
    place = placement(args.n, nb_banks, fold)

    if args.check_yml:
        ref = parse_yml(args.check_yml)
        if not ref:
            # The small flavors attach a single monolithic "hbm" endpoint
            # instead of a per-bank perimeter ring, and their cluster wrapper
            # never reads this map.
            print('%s declares no per-bank HBM ring; skipping map check'
                  % args.check_yml)
        elif len(ref) != len(place):
            raise SystemExit('%s: found %d bank connections, expected %d'
                             % (args.check_yml, len(ref), len(place)))
        else:
            bad = [b for b in range(len(place))
                   if ref.get(b) != tuple(place[b])]
            if bad:
                for b in bad:
                    print('bank %2d: yml %s != generated %s'
                          % (b, ref.get(b), place[b]), file=sys.stderr)
                raise SystemExit('%s disagrees with the generated map on'
                                 ' %d bank(s)'
                                 % (args.check_yml, len(bad)))
            print('%s matches the generated map (%d banks)'
                  % (args.check_yml, len(place)))

    if args.report:
        report(args.n, place, nb_banks, fold)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(emit_sv(args.n, place))


if __name__ == '__main__':
    main()
