#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Author: Yinrong Li, ETH Zurich
"""Interconnect performance summary in PLAIN TEXT -- both ends of the path.

The text counterpart of what ``scripts/perfetto_gen.py`` draws from
``noc_profiling/`` + ``spm_profiling/`` (needs ``noc_profiling=1``, plus
``spm_profiling=1`` for the bank view). Two halves:

**Requesters** -- one round-trip latency distribution per requester class,
built by pairing each request P line with its response P line on the same port
(key = meta_id, which is unique among a port's outstanding requests):

    snitch scalar   the core's own load/store port    (core_ idx % ppc == 0)
    spatz vector    the Spatz VLSU's memory ports      (core_ idx % ppc != 0)
    redmule         the RedMulE master ports           (rm_)

``rm_`` is the RedMulE family on EITHER plane. One record is 32 bits on the
narrow plane and WIDE_TCDM_DATA_WIDTH on the WP2 wide one, so request counts
are not comparable across a plane A/B -- the file's ``D <bits>`` header carries
the width so a consumer can normalise to bytes.

Because all three are measured the SAME way, they answer the question a model
calibration actually asks: if scalar and RedMulE round trips match the model
but the vector ones do not, the error is in the vector-engine model, not in the
memory system underneath it (and vice versa). Each class is also split by
address class (local tile / intra-group / inter-group / SoC), so a NUMA effect
cannot masquerade as a latency constant.

**Fabric** -- every port side's time split into idle / stall / read / write,
aggregated by port class, plus the hottest individual ports:

    busy   = accepted handshakes            (states read + write)
    stall  = valid asserted, ready low      (back-pressure the port SEES)
    util   = busy + stall                   (cycles the port is not idle)

Log families (S lines are run-length ``start end state``, state
0=idle 1=stall 2=read 3=write; P lines are per-packet)::

    router_g<g>_{req,resp}.log  <rid> S <dir 0..4> <io> <start> <end> <state>
    tile_g<g>_t<t>.log          S <net> <io> <port> <start> <end> <state>
    core_g<g>_t<t>.log          D <data_width_bits>
                                <port> S <net> <io> 0 <start> <end> <state>
                                <port> P 1 0 <cyc> <wen> <addr> <mid> <dst>
                                <port> P 0 0 <cyc> <mid>
    rm_g<g>_t<t>.log            same as core_, <port> = RedMulE master port
    bank_g<g>_t<t>.log          S <bank> <start> <end> <state>

The WP2 wide plane adds its own fabric families, all optional::

    tile_g<g>_t<t>.log          SW/PW/QW lines = the wide side of the boundary
    router_wide_g<g>_{req,resp}.log
                                <chan> S <cls> <dir> <io> <start> <end> <state>
                                cls: req 0=RO 1=RW, resp 0=RD 1=WR

Usage::

    scripts/interco_perf.py build/ --ports-per-core 5 --tiles-per-group 4
    scripts/interco_perf.py build/ --cycle-start 4000 --top 15
"""

import argparse
import collections
import glob
import os
import re
import sys

import memtrace as mt

ROUTER_RE = re.compile(r"router_g(\d+)_(req|resp)\.log$")
TILE_RE = re.compile(r"tile_g(\d+)_t(\d+)\.log$")
CORE_RE = re.compile(r"core_g(\d+)_t(\d+)\.log$")
RM_RE = re.compile(r"rm_g(\d+)_t(\d+)\.log$")
BANK_RE = re.compile(r"bank_g(\d+)_t(\d+)\.log$")
# WP2 wide TCDM plane: it reuses the rm_ family (same grammar, one client per
# tile) and adds its own routers. Which plane carried RedMulE is read from the
# rm_ file's `D <bits>` header, not from its name.
WROUTER_RE = re.compile(r"router_wide_g(\d+)_(req|resp)\.log$")
# Wide router classes per net: req = RO (meta-only read) / RW (payload),
# resp = RD (data) / WR (meta-only write ack).
WCLS = {"req": ("RO", "RW"), "resp": ("RD", "WR")}

DIRS = {0: "N", 1: "E", 2: "S", 3: "W", 4: "local"}
IO = {0: "in", 1: "out"}
NET = {0: "req", 1: "resp"}
# tile port sides, keyed by (net, io) -- see tb_noc_profiling.svh
TILE_SIDE = {(0, 1): "master req out", (0, 0): "slave req in",
             (1, 0): "master resp in", (1, 1): "slave resp out"}
IDLE, STALL, READ, WRITE = range(4)

SCALAR, VECTOR, REDMULE = "snitch scalar", "spatz vector", "redmule"
# The wide-plane engine port is its OWN class: one of its requests moves a
# whole WideTcdmDataWidth line, so its counts and latencies are not
# comparable with a 32-bit port's and must never be summed with them.


class Port:
    """Cycle budget of one port side."""

    def __init__(self):
        self.cyc = [0, 0, 0, 0]     # per state
        self.pkts = 0

    def add(self, s, e, st, lo, hi):
        s, e = max(s, lo), min(e, hi)
        if e > s and 0 <= st < 4:
            self.cyc[st] += e - s

    @property
    def busy(self):
        return self.cyc[READ] + self.cyc[WRITE]

    @property
    def stall(self):
        return self.cyc[STALL]


class ReqClass:
    """Latency accumulator for one requester class."""

    def __init__(self):
        self.all = mt.LatStats()
        self.cls = [mt.LatStats() for _ in range(len(mt.CLASSES) + 1)]
        self.rw = [mt.LatStats(), mt.LatStats()]
        self.n_req = self.n_inflight = self.n_orphan = 0
        self.n_rw = [0, 0]
        self.ports = set()

    def add(self, r):
        self.n_rw[r.wr] += 1
        if r.lat is None:
            return
        self.all.add(r.lat)
        self.cls[r.cls].add(r.lat)
        self.rw[r.wr].add(r.lat)


def _add_run(ports, key, x, lo, hi, last):
    try:
        s, e, st = int(x[0]), int(x[1]), int(x[2])
    except ValueError:
        return last
    ports[key].add(s, e, st, lo, hi)
    return max(last, e)


def load_fabric(noc_dir, spm_dir, lo, hi):
    """Parse the S state-runs of every port -> ({class: {key: Port}}, last)."""
    cls = collections.defaultdict(lambda: collections.defaultdict(Port))
    last = 0
    for path in sorted(glob.glob(os.path.join(noc_dir or "", "*.log"))):
        base = os.path.basename(path)
        m_wrouter = WROUTER_RE.search(base)
        m_router = None if m_wrouter else ROUTER_RE.search(base)
        m_tile = TILE_RE.search(base)
        m_core, m_rm = CORE_RE.search(base), RM_RE.search(base)
        if not (m_router or m_wrouter or m_tile or m_core or m_rm):
            continue
        with open(path, "r", errors="replace") as f:
            for line in f:
                x = line.split()
                if len(x) < 4:
                    continue
                if m_router:
                    g, net = int(m_router.group(1)), m_router.group(2)
                    if x[1] != "S" or len(x) != 7:
                        if x[1] == "P":
                            cls[f"router {net}"][
                                (g, int(x[0]), int(x[2]), int(x[3]))].pkts += 1
                        continue
                    last = _add_run(cls[f"router {net}"],
                                    (g, int(x[0]), int(x[2]), int(x[3])),
                                    x[4:7], lo, hi, last)
                elif m_wrouter:
                    # <chan> S <cls> <dir> <io> <start> <end> <state>
                    if len(x) < 5:      # one column more than the narrow net
                        continue
                    g, net = int(m_wrouter.group(1)), m_wrouter.group(2)
                    key = (g, int(x[0]), int(x[2]), int(x[3]), int(x[4]))
                    if x[1] != "S" or len(x) != 8:
                        if x[1] == "P":
                            cls[f"wide router {net}"][key].pkts += 1
                        continue
                    last = _add_run(cls[f"wide router {net}"], key,
                                    x[5:8], lo, hi, last)
                elif m_tile:
                    # the wide plane shares the file under SW/PW/QW
                    if x[0] not in ("S", "SW") or len(x) != 7:
                        continue
                    g, t = int(m_tile.group(1)), int(m_tile.group(2))
                    name = "tile wide" if x[0] == "SW" else "tile"
                    last = _add_run(cls[name],
                                    (g, t, int(x[1]), int(x[2]), int(x[3])),
                                    x[4:7], lo, hi, last)
                else:
                    m = m_core or m_rm
                    g, t = int(m.group(1)), int(m.group(2))
                    if x[1] != "S" or len(x) != 8:
                        continue
                    name = "core port" if m_core else "redmule port"
                    last = _add_run(cls[name],
                                    (g, t, int(x[0]), int(x[2]), int(x[3])),
                                    x[5:8], lo, hi, last)
    for path in sorted(glob.glob(os.path.join(spm_dir or "", "bank_g*.log"))):
        m = BANK_RE.search(os.path.basename(path))
        if not m:
            continue
        g, t = int(m.group(1)), int(m.group(2))
        with open(path, "r", errors="replace") as f:
            for line in f:
                x = line.split()
                if len(x) != 5 or x[0] != "S":
                    continue
                last = _add_run(cls["spm bank"],
                                (g, t, int(x[1])), x[2:5], lo, hi, last)
    return cls, last


def load_requesters(noc_dir, ppc, tpg, lo, hi):
    """Pair every port's requests with its responses -> {class: ReqClass}."""
    out = {k: ReqClass() for k in (SCALAR, VECTOR, REDMULE)}
    for kind in ("core", "rm"):
        for g, t, ports in mt.iter_port_tiles(noc_dir, kind, tpg):
            for idx, pt in ports.items():
                if kind == "rm":
                    k = REDMULE
                else:
                    k = SCALAR if (ppc <= 1 or idx % ppc == 0) else VECTOR
                rc = out[k]
                rc.ports.add((g, t, idx))
                rc.n_orphan += pt.n_orphan
                for r in pt.recs:
                    if not (lo <= r.cyc < hi):
                        continue
                    rc.n_req += 1
                    if r.lat is None:
                        rc.n_inflight += 1
                    rc.add(r)
    return {k: v for k, v in out.items() if v.n_req}


def _pct(n, d):
    return 100.0 * n / d if d else 0.0


def _lat_row(name, s, total_n, out):
    if not s.n:
        return
    print(f"      {name:<10} {s.n:>10d} {_pct(s.n, total_n):>7.1f}%"
          f" {s.mean:>9.2f} {s.quantile(50):>7d} {s.quantile(90):>7d}"
          f" {s.quantile(99):>7d} {s.hi:>7d}", file=out)


def _req_section(reqs, fabric, span, args, out):
    print("", file=out)
    print("-- requesters: memory round trip (accepted request -> response) "
          + "-" * 14, file=out)
    if not reqs:
        print("    (no paired request/response packets -- the core_/rm_ "
              "P lines need noc_profiling=1)", file=out)
        return
    # port occupancy per class, from the same logs' S runs
    occ = collections.defaultdict(lambda: [0, 0, 0])   # busy, stall, nports
    ppc = max(1, args.ports_per_core)
    for name, ports in (("core port", fabric.get("core port", {})),
                        ("redmule port", fabric.get("redmule port", {}))):
        for (g, t, idx, net, io), p in ports.items():
            if net != 0:          # request side only
                continue
            if name == "redmule port":
                k = REDMULE
            else:
                k = SCALAR if (ppc <= 1 or idx % ppc == 0) else VECTOR
            occ[k][0] += p.busy
            occ[k][1] += p.stall
            occ[k][2] += 1
    for k in (SCALAR, VECTOR, REDMULE):
        rc = reqs.get(k)
        if not rc:
            continue
        b, s, n = occ.get(k, (0, 0, 0))
        tot = n * span
        print("", file=out)
        print(f"  {k}   ports={len(rc.ports)}  requests={rc.n_req}  "
              f"reads={rc.n_rw[0]}  writes={rc.n_rw[1]}", file=out)
        if n:
            print(f"      port occupancy: busy {_pct(b, tot):.2f}%  "
                  f"stall {_pct(s, tot):.2f}%  "
                  f"util {_pct(b + s, tot):.2f}%", file=out)
        print(f"      {'':<10} {'count':>10} {'share':>8} {'mean':>9} "
              f"{'p50':>7} {'p90':>7} {'p99':>7} {'max':>7}", file=out)
        _lat_row("ALL", rc.all, rc.all.n, out)
        for i, nm in enumerate(mt.CLASSES + ["unknown"]):
            _lat_row(nm, rc.cls[i], rc.all.n, out)
        for i, nm in enumerate(["read", "write"]):
            _lat_row(nm, rc.rw[i], rc.all.n, out)
        if rc.n_inflight or rc.n_orphan:
            print(f"      unpaired: {rc.n_inflight} in flight at end, "
                  f"{rc.n_orphan} orphan responses", file=out)


def _row(label, ports, span, out, extra=""):
    n = len(ports)
    if not n:
        return
    tot = n * span
    busy = sum(p.busy for p in ports)
    stall = sum(p.stall for p in ports)
    rd = sum(p.cyc[READ] for p in ports)
    wr = sum(p.cyc[WRITE] for p in ports)
    print(f"    {label:<22}{n:>6}{_pct(busy, tot):>9.2f}"
          f"{_pct(stall, tot):>9.2f}{_pct(busy + stall, tot):>9.2f}"
          f"{_pct(rd, tot):>9.2f}{_pct(wr, tot):>9.2f}  {extra}", file=out)


def _hdr(title, out):
    print("", file=out)
    print(f"-- {title} " + "-" * max(4, 62 - len(title)), file=out)
    print(f"    {'':<22}{'ports':>6}{'busy%':>9}{'stall%':>9}{'util%':>9}"
          f"{'read%':>9}{'write%':>9}", file=out)


def _keyname(name, key, ppc):
    """Human-readable port id (the tuple layout differs per log family)."""
    if name == "core port":
        g, t, idx, net, io = key
        who = (f"C{idx}" if ppc <= 1 else
               (f"C{idx // ppc}" if idx % ppc == 0
                else f"C{idx // ppc}.v{idx % ppc - 1}"))
        return f"G{g} T{t} {who} {NET[net]} {IO[io]}"
    if name == "redmule port":
        g, t, idx, net, io = key
        return f"G{g} T{t} RM{idx} {NET[net]} {IO[io]}"
    if name in ("tile", "tile wide"):
        g, t, net, io, port = key
        w = " wide" if name == "tile wide" else ""
        return f"G{g} T{t} p{port}{w} {TILE_SIDE.get((net, io), '')}"
    if name.startswith("wide router"):
        g, chan, wc, d, io = key
        net = name.split()[-1]
        nm = WCLS[net][wc] if wc < len(WCLS[net]) else str(wc)
        return f"G{g} W{nm}{chan} {DIRS.get(d, d)} {IO[io]}"
    if name.startswith("router"):
        g, rid, d, io = key
        return f"G{g} R{rid} {DIRS.get(d, d)} {IO[io]}"
    if name == "spm bank":
        g, t, b = key
        return f"G{g} T{t} bank {b}"
    return str(key)


def _fabric_section(cls, span, args, out):
    ppc = max(1, args.ports_per_core)
    for name in ("core port", "redmule port"):
        if name not in cls:
            continue
        _hdr(f"{name}s (per requester port side)", out)
        groups = collections.defaultdict(list)
        for (g, t, idx, net, io), p in cls[name].items():
            if name == "core port":
                sub = "scalar" if (ppc <= 1 or idx % ppc == 0) else "vector"
            else:
                sub = "port"
            groups[f"{sub} {NET[net]}"].append(p)
        for k in sorted(groups):
            _row(k, groups[k], span, out)

    for name, title in (("tile", "tile ports (tile <-> NoC)"),
                        ("tile wide",
                         "tile ports, WIDE plane (tile <-> NoC)")):
        if name not in cls:
            continue
        _hdr(title, out)
        groups = collections.defaultdict(list)
        for (g, t, net, io, port), p in cls[name].items():
            groups[TILE_SIDE.get((net, io), f"{NET[net]} {IO[io]}")].append(p)
        for k in sorted(groups):
            _row(k, groups[k], span, out)

    for net in ("req", "resp"):
        key = f"router {net}"
        if key not in cls:
            continue
        _hdr(f"{net} routers (by mesh direction)", out)
        groups = collections.defaultdict(list)
        for (g, rid, d, io), p in cls[key].items():
            groups[f"{DIRS.get(d, d)} {IO[io]}"].append(p)
        for k in sorted(groups, key=lambda s: (s.split()[1], s)):
            _row(k, groups[k], span, out)

    # The wide plane's own routers, split by channel class so an RO/WO
    # (meta-only) channel is never averaged with a payload-bearing one.
    for net in ("req", "resp"):
        key = f"wide router {net}"
        if key not in cls:
            continue
        _hdr(f"WIDE {net} routers (by class and mesh direction)", out)
        groups = collections.defaultdict(list)
        for (g, chan, wc, d, io), p in cls[key].items():
            nm = WCLS[net][wc] if wc < len(WCLS[net]) else str(wc)
            groups[f"{nm} {DIRS.get(d, d)} {IO[io]}"].append(p)
        for k in sorted(groups, key=lambda s: (s.split()[0], s.split()[2], s)):
            _row(k, groups[k], span, out)

    if "spm bank" in cls:
        _hdr("SPM banks", out)
        _row("bank", list(cls["spm bank"].values()), span, out,
             extra="(stall% = arbitration conflict)")

    if args.top:
        print("", file=out)
        print(f"-- top {args.top} ports by stall% " + "-" * 40, file=out)
        allp = []
        for name, ports in cls.items():
            for key, p in ports.items():
                allp.append((p.stall, name, key, p))
        allp.sort(reverse=True, key=lambda z: z[0])
        for stall, name, key, p in allp[:args.top]:
            print(f"    {name:<13} {_keyname(name, key, ppc):<28} "
                  f"stall {_pct(stall, span):>6.2f}%  "
                  f"busy {_pct(p.busy, span):>6.2f}%", file=out)


def main():
    ap = argparse.ArgumentParser("interco_perf")
    ap.add_argument("dir", help="build dir (finds noc_profiling/ and "
                                "spm_profiling/) or one of those dirs")
    ap.add_argument("--ports-per-core", type=int, default=1,
                    help="TCDM data ports per core (1 scalar + Spatz vector "
                         "ports); splits the core ports into scalar/vector")
    ap.add_argument("--tiles-per-group", type=int, default=4,
                    help="tiles per group, to turn a destination tile into an "
                         "address class (local / intra / inter)")
    ap.add_argument("--cycle-start", type=int, default=0)
    ap.add_argument("--cycle-end", type=int, default=None)
    ap.add_argument("--top", type=int, default=10,
                    help="list the N most stalled ports (0 = off)")
    ap.add_argument("--no-requesters", action="store_true",
                    help="fabric occupancy only (skip the request pairing)")
    ap.add_argument("-o", "--output", help="write here (default: stdout)")
    args = ap.parse_args()

    noc_dir = spm_dir = None
    if glob.glob(os.path.join(args.dir, "router_g*.log")) or \
            glob.glob(os.path.join(args.dir, "tile_g*.log")):
        noc_dir = args.dir
    elif glob.glob(os.path.join(args.dir, "bank_g*.log")):
        spm_dir = args.dir
    else:
        cand_n = os.path.join(args.dir, "noc_profiling")
        cand_s = os.path.join(args.dir, "spm_profiling")
        noc_dir = cand_n if os.path.isdir(cand_n) else None
        spm_dir = cand_s if os.path.isdir(cand_s) else None
    if not noc_dir and not spm_dir:
        print(f"no noc_profiling/ or spm_profiling/ logs under {args.dir} "
              "(re-run the sim with noc_profiling=1 spm_profiling=1)",
              file=sys.stderr)
        return 1

    lo = max(0, args.cycle_start)
    hi = args.cycle_end if args.cycle_end is not None else (1 << 62)
    fabric, last = load_fabric(noc_dir, spm_dir, lo, hi)
    span = max(1, min(hi, last) - lo)
    reqs = ({} if (args.no_requesters or not noc_dir)
            else load_requesters(noc_dir, args.ports_per_core,
                                 args.tiles_per_group, lo, hi))

    out = open(args.output, "w") if args.output else sys.stdout
    try:
        print("=" * 78, file=out)
        print("Interconnect performance summary", file=out)
        print("=" * 78, file=out)
        print(f"  logs   : {noc_dir or ''} {spm_dir or ''}", file=out)
        print(f"  window : {lo} .. "
              f"{args.cycle_end if args.cycle_end is not None else last} "
              f"({span} cycles)", file=out)
        print(f"  ports/core: {args.ports_per_core}, "
              f"tiles/group: {args.tiles_per_group}", file=out)
        print("  busy = accepted handshakes, stall = valid without ready, "
              "util = busy+stall", file=out)
        if not args.no_requesters:
            _req_section(reqs, fabric, span, args, out)
        _fabric_section(fabric, span, args, out)
    finally:
        if args.output:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
