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

`--check-yml` also doubles as the system-address-map guard.  A wrong address
map elaborates with zero errors and produces X, so given `--l2-base`,
`--l2-size`, `--bootrom-base` and `--periph-base` it additionally verifies that

  * the HBM endpoints tile [l2_base, l2_base + l2_size) into `--banks` equal
    slices, so the SAM and `l2_size` cannot drift apart,
  * the host/peripheral endpoint claims exactly [0, l2_base), i.e. TCDM, the
    bootrom and the peripheral window, and
  * nothing at all is mapped at or above l2_base + l2_size, which is what makes
    the L2 aperture contiguous.

`--check-sam` repeats those checks against the *generated* FlooNoC package
rather than the YAML.  That is the copy the RTL compiles, so it is the one that
catches a stale package -- the failure that cost a monsterpool batch on
2026-08-21, where a package left over from another flavor declared
NumEndpoints=32 for a 96-endpoint design.

`--check-bootrom` decodes the boot image's `auipc`/`addi` pair and checks that
it still computes `l2_base` from `--boot-addr`.  The image is PC-relative and
checked in, so moving the bootrom without regenerating it sends every core to a
garbage entry point with nothing to complain about.

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
    """RTL group id -> router coord, as mempool_pkg::group_xy_of.

    Square meshes number groups in Morton order (gy in the even bits, gx in the
    odd ones), so ids 2k, 2k+1 are still vertical neighbours but larger blocks
    are squares, not columns.
    """
    x = y = 0
    for i in range(max(n - 1, 0).bit_length()):
        y |= ((g >> (2 * i)) & 1) << i
        x |= ((g >> (2 * i + 1)) & 1) << i
    return x, y


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
    """The map to emit, solved for Morton ids at every mesh size.

    The 4x4 map was once pinned to the one solved for raster ids (8 hops
    then, 28 under Morton); the stable branch, which keeps raster ids, still
    ships it.
    """
    nb_banks = nb_banks or 4 * n
    fold = fold or (n * n // nb_banks)
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
              '  // that is depends on the placement, so it is derived',
              '  // rather than hardcoded.',
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


# System address map guard.
# A wrong address map elaborates with zero errors and produces X, so the YAML,
# the generated FlooNoC package and the boot image are each checked against
# config/config.mk on every build.


def _int(tok):
    """Parse a floogen address literal (0x8000_0000, 0x80000000, decimal)."""
    tok = tok.strip().replace('_', '')
    return int(tok, 16) if tok.lower().startswith('0x') else int(tok)


def hx(v):
    return '0x%08X' % v


def yml_name(path):
    """The floogen `name:` field, which names the emitted package."""
    for line in open(path):
        m = re.match(r'name:\s*"?([A-Za-z0-9_]+)"?\s*(#.*)?$', line)
        if m:
            return m.group(1)
    raise SystemExit('%s: no top-level `name:` field' % path)


def parse_endpoints(path):
    """Endpoint name -> {'array': [...], 'ranges': [(start, end), ...]}.

    floogen YAML is regular enough to read with an indentation walk, and the
    build environment has no PyYAML, so this stays dependency-free.
    """
    out = collections.OrderedDict()
    cur = None
    in_endpoints = False
    in_range = False
    pending = {}
    base = size = None
    for raw in open(path):
        line = raw.rstrip('\n')
        if re.match(r'^\S', line):
            in_endpoints = line.startswith('endpoints:')
            cur, in_range = None, False
            continue
        stripped = line.strip()
        if not in_endpoints or not stripped or stripped.startswith('#'):
            continue
        m = re.match(r'^  - name:\s*"?([A-Za-z0-9_]+)"?', line)
        if m:
            cur = {'array': None, 'ranges': []}
            out[m.group(1)] = cur
            in_range, pending, base, size = False, {}, None, None
            continue
        if cur is None:
            continue
        m = re.match(r'^    array:\s*\[([0-9,\s]+)\]', line)
        if m:
            cur['array'] = [int(v) for v in m.group(1).split(',')]
            continue
        if re.match(r'^    addr_range:', line):
            in_range = True
            continue
        if re.match(r'^    \w', line):          # any other endpoint key
            in_range = False
            continue
        if not in_range:
            continue
        m = re.match(r'^\s+-?\s*(start|end|base|size):\s*(\S+)', line)
        if not m:
            continue
        key, val = m.group(1), _int(m.group(2))
        if key == 'start':
            pending = {'start': val}
        elif key == 'end':
            cur['ranges'].append((pending['start'], val))
            pending = {}
        elif key == 'base':
            base = val
        elif key == 'size':
            size = val
            n = cur['array'][0] if cur['array'] else 1
            cur['ranges'] += [(base + i * size, base + (i + 1) * size)
                              for i in range(n)]
    return out


def bank_index(name):
    """Bank id an endpoint name declares, or None if it is not an L2 bank."""
    m = re.match(r'^(?:host_)?hbm(\d+)(?:_peri)?$', name)
    return int(m.group(1)) if m else None


def yml_bank_ranges(eps, l2_base, l2_size):
    """bank id -> (start, end), expanding the `array:` + base/size form."""
    banks = {}
    for name, ep in eps.items():
        if not name.startswith('hbm') and '_peri' not in name:
            continue
        idx = bank_index(name)
        if idx is not None:
            hits = [r for r in ep['ranges']
                    if l2_base <= r[0] < l2_base + l2_size]
            if len(hits) != 1:
                raise SystemExit('%s declares %d L2 ranges, expected 1'
                                 % (name, len(hits)))
            banks[idx] = hits[0]
        elif name == 'hbm':                      # array form, one endpoint
            for i, r in enumerate(ep['ranges']):
                banks[i] = r
    return banks


def host_endpoint(eps):
    for name in eps:
        if 'peri' in name:
            return name, eps[name]
    raise SystemExit('no host/peripheral endpoint found')


def check_addr_map(path, n, l2_base, l2_size, l2_banks,
                   bootrom_base, periph_base):
    """The YAML address map has to agree with config/config.mk."""
    eps = parse_endpoints(path)
    errs = []
    l2_end = l2_base + l2_size
    slice_sz = l2_size // l2_banks
    if slice_sz * l2_banks != l2_size:
        errs.append('l2_size %s does not divide into %d banks'
                    % (hx(l2_size), l2_banks))

    grp = eps.get('group')
    if grp is None or grp['array'] is None:
        errs.append('no `group` endpoint with an `array:` field')
    elif grp['array'] != [n, n]:
        errs.append('group array %s != mesh %dx%d' % (grp['array'], n, n))

    banks = yml_bank_ranges(eps, l2_base, l2_size)
    if sorted(banks) != list(range(l2_banks)):
        errs.append('banks declared %s, config says %d'
                    % (sorted(banks), l2_banks))
    for b in sorted(set(banks) & set(range(l2_banks))):
        want = (l2_base + b * slice_sz, l2_base + (b + 1) * slice_sz)
        if banks[b] != want:
            errs.append('bank %d is [%s, %s), config says [%s, %s)'
                        % (b, hx(banks[b][0]), hx(banks[b][1]),
                           hx(want[0]), hx(want[1])))

    hname, host = host_endpoint(eps)
    low = [r for r in host['ranges'] if r[0] < l2_base]
    if low != [(0, l2_base)]:
        errs.append('%s claims %s below L2, expected [%s, %s)'
                    % (hname, [(hx(a), hx(b)) for a, b in low],
                       hx(0), hx(l2_base)))
    for base in (bootrom_base, periph_base):
        if not 0 <= base < l2_base:
            errs.append('%s is not inside the host endpoint window [0, %s)'
                        % (hx(base), hx(l2_base)))
    if not bootrom_base < periph_base:
        errs.append('bootrom_base %s must sit below periph_base %s'
                    % (hx(bootrom_base), hx(periph_base)))

    above = [(a, b) for ep in eps.values() for a, b in ep['ranges']
             if b > l2_end]
    if above:
        errs.append('%d rule(s) mapped above L2 end %s: %s -- the L2 aperture'
                    ' must be the top of the map'
                    % (len(above), hx(l2_end),
                       [(hx(a), hx(b)) for a, b in above]))

    if errs:
        for e in errs:
            print('%s: %s' % (path, e), file=sys.stderr)
        raise SystemExit('%s disagrees with config/config.mk on %d point(s)'
                         % (path, len(errs)))
    print('%s address map matches the config (L2 [%s, %s) in %d banks,'
          ' bootrom %s, peripherals %s)'
          % (path, hx(l2_base), hx(l2_end), l2_banks,
             hx(bootrom_base), hx(periph_base)))


def parse_sam(pkg):
    """(NumEndpoints, [(endpoint, start, end)]) from a generated package."""
    txt = open(pkg).read()
    m = re.search(r'NumEndpoints\s*=\s*(\d+)', txt)
    if not m:
        raise SystemExit('%s: no NumEndpoints' % pkg)
    rules = [(name, int(a, 16), int(b, 16)) for a, b, name in re.findall(
        r"start_addr:\s*32'h([0-9a-fA-F]+),\s*end_addr:\s*32'h([0-9a-fA-F]+)"
        r"\}[,\s]*//\s*(\w+)", txt)]
    if not rules:
        raise SystemExit('%s: no Sam rules' % pkg)
    return int(m.group(1)), rules


def check_sam(gendir, yml, l2_base, l2_size, l2_banks,
              bootrom_base, periph_base):
    """The package the RTL actually compiles has to match the YAML.

    This is the copy that goes stale when the flavor changes, so it is checked
    separately from `--check-yml` and after floogen has run.
    """
    name = yml_name(yml)
    if name != 'terapool':
        raise SystemExit('%s: floogen `name:` is "%s"; the RTL imports'
                         ' floo_terapool_noc_pkg unconditionally, so every'
                         ' flavor must keep name: terapool' % (yml, name))
    pkg = '%s/floo_%s_noc_pkg.sv' % (gendir.rstrip('/'), name)
    num_eps, rules = parse_sam(pkg)

    eps = parse_endpoints(yml)
    want_eps = sum(ep['array'][0] * ep['array'][1] if ep['array'] and
                   len(ep['array']) == 2 else
                   (ep['array'][0] if ep['array'] else 1)
                   for ep in eps.values())
    errs = []
    if num_eps != want_eps:
        errs.append('NumEndpoints %d, but %s declares %d endpoints -- the'
                    ' package is stale, rerun `make update-floogen`'
                    % (num_eps, yml, want_eps))

    got = collections.defaultdict(list)
    for ep, a, b in rules:
        got[ep.lower()].append((a, b))
    want = collections.defaultdict(list)
    for epname, ep in eps.items():
        if not ep['ranges']:
            continue
        if ep['array'] and epname == 'hbm':
            for i, r in enumerate(ep['ranges']):
                want['hbm%d' % i].append(r)
        else:
            want[epname.replace('_', '')] += ep['ranges']
    for key in sorted(set(want) | set(got)):
        if sorted(want.get(key, [])) != sorted(got.get(key, [])):
            show = [[(hx(a), hx(b)) for a, b in sorted(d.get(key, []))]
                    for d in (got, want)]
            errs.append('%s: package %s, yml %s'
                        % (key, show[0], show[1]))

    slice_sz = l2_size // l2_banks
    flat = sorted((a, b) for _, a, b in rules)
    for b_id in range(l2_banks):
        r = (l2_base + b_id * slice_sz, l2_base + (b_id + 1) * slice_sz)
        if r not in flat:
            errs.append('no SAM rule for bank %d [%s, %s)'
                        % (b_id, hx(r[0]), hx(r[1])))
    if (0, l2_base) not in flat:
        errs.append('no SAM rule [%s, %s) covering TCDM, the bootrom at %s'
                    ' and the peripherals at %s'
                    % (hx(0), hx(l2_base), hx(bootrom_base), hx(periph_base)))
    above = [(a, b) for a, b in flat if b > l2_base + l2_size]
    if above:
        errs.append('SAM maps %s above the L2 aperture'
                    % [(hx(a), hx(b)) for a, b in above])

    if errs:
        for e in errs:
            print('%s: %s' % (pkg, e), file=sys.stderr)
        raise SystemExit('%s does not match %s / config.mk on %d point(s)'
                         % (pkg, yml, len(errs)))
    print('%s matches %s and the config (%d endpoints, %d SAM rules)'
          % (pkg, yml, num_eps, len(rules)))


def check_bootrom(pkg, boot_addr, l2_base):
    """The boot image is PC-relative, so it pins boot_addr against l2_base."""
    words = [int(w, 16) for w in
             re.findall(r"32'h([0-9a-fA-F]{8})", open(pkg).read())]
    if len(words) < 2:
        raise SystemExit('%s: expected at least 2 boot words' % pkg)
    # The package lists the packed array MSB-first, so word 0 is last.
    auipc, addi = words[-1], words[-2]
    if auipc & 0x7F != 0x17 or addi & 0x707F != 0x13:
        raise SystemExit('%s: boot image does not start with auipc/addi;'
                         ' teach this check the new bootrom.S' % pkg)
    imm12 = addi >> 20
    if imm12 & 0x800:
        imm12 -= 0x1000
    target = (boot_addr + (auipc & 0xFFFFF000) + imm12) & 0xFFFFFFFF
    if target != l2_base:
        raise SystemExit('%s: booting at %s jumps to %s, but l2_base is %s --'
                         ' rebuild the image (`rm software/runtime/bootrom.img'
                         ' && make -C hardware src/bootrom_image_pkg.sv`)'
                         % (pkg, hx(boot_addr), hx(target), hx(l2_base)))
    print('%s jumps from %s to %s (l2_base)'
          % (pkg, hx(boot_addr), hx(target)))


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
    ap.add_argument('--check-sam', default=None,
                    help='directory holding the generated FlooNoC package,'
                         ' verified against --check-yml and the config')
    ap.add_argument('--check-bootrom', default=None,
                    help='bootrom_image_pkg.sv to verify against --boot-addr')
    ap.add_argument('--l2-base', type=lambda v: int(v, 0), default=None,
                    help='config l2_base; enables the address-map checks')
    ap.add_argument('--l2-size', type=lambda v: int(v, 0), default=None)
    ap.add_argument('--bootrom-base', type=lambda v: int(v, 0), default=None)
    ap.add_argument('--periph-base', type=lambda v: int(v, 0), default=None)
    ap.add_argument('--boot-addr', type=lambda v: int(v, 0), default=None)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()

    addr_args = (args.l2_base, args.l2_size,
                 args.bootrom_base, args.periph_base)
    check_addrs = any(a is not None for a in addr_args)
    if check_addrs and None in addr_args:
        raise SystemExit('--l2-base, --l2-size, --bootrom-base and'
                         ' --periph-base go together')

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
        if check_addrs:
            check_addr_map(args.check_yml, args.n, args.l2_base, args.l2_size,
                           nb_banks, args.bootrom_base, args.periph_base)

    if args.check_sam:
        if not args.check_yml or not check_addrs:
            raise SystemExit('--check-sam needs --check-yml and the'
                             ' address-map arguments')
        check_sam(args.check_sam, args.check_yml, args.l2_base,
                  args.l2_size, nb_banks, args.bootrom_base, args.periph_base)

    if args.check_bootrom:
        if args.boot_addr is None or args.l2_base is None:
            raise SystemExit('--check-bootrom needs --boot-addr and --l2-base')
        check_bootrom(args.check_bootrom, args.boot_addr, args.l2_base)

    if args.report:
        report(args.n, place, nb_banks, fold)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(emit_sv(args.n, place))


if __name__ == '__main__':
    main()
