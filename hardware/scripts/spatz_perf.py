#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Author: Yinrong Li, ETH Zurich
"""Spatz vector-engine performance summary in PLAIN TEXT.

The Perfetto exporter (``scripts/perfetto_gen.py``) is for browsing; this is
the same data reduced to numbers you can diff, paste into a report, or feed a
model calibration. It is the Spatz analogue of ``scripts/redmule_perf.py``.

Two inputs, split the way RedMulE splits its logs -- engine internals in their
own directory (always on), memory-port traffic captured by the NoC testbench
beside the other port traces (needs ``noc_profiling=1``):

* ``spatz_profiling/spatz_g<g>_t<t>_c<c>.log`` -- the per-cycle ENGINE trace:
  scoreboard occupancy, per-unit busy/sub-state, retire handshakes, active
  memory ports. Gives the occupancy/CPI stack and *VLSU cycles per memory
  instruction*.
* ``noc_profiling/core_g<g>_t<t>.log`` -- the per-REQUEST memory trace of every
  core port; the Spatz vector ports are the flat indices with
  ``idx % ppc != 0`` (``ppc = NrMemPorts + 1``). One record per accepted word
  request, paired with its response to give a round-trip latency and a
  destination
  class. Gives what the engine trace cannot: where the VLSU occupancy goes.

Reported per-memory-instruction decomposition (all per core, per kernel
window):

    VLSU busy / mem insn   how long a vle/vse occupies the VLSU end to end
      requests / mem insn  word requests it fans out into (vl * SEW / 32 / 4)
      grant wait / request cycles the request sat with valid high before the
                           TCDM accepted it (issue-side back-pressure; from
                           the engine trace's memact)
      round trip / request accepted-request -> response, split by address
                           class (local tile / intra-group / inter-group /
                           SoC), read vs write

Usage::

    scripts/spatz_perf.py build/                     # or .../spatz_profiling
    scripts/spatz_perf.py build/ --cycle-start 4000  # kernel window only
    scripts/spatz_perf.py build/ --per-core --csv spatz.csv
"""

import argparse
import collections
import glob
import os
import re
import sys

import memtrace as mt

SPATZ_LOG_RE = re.compile(r"spatz_g(\d+)_t(\d+)_c(\d+)\.log$")

# Occupancy/CPI buckets, same attribution as the Perfetto exporter
# (perfetto_gen.py:_spatz_bucket) so the two views cannot disagree.
CPI = ["compute", "mem", "stall", "drain", "idle"]


# Issue-stall causes, in `istall` bit order (tb_spatz_profiling.svh).
ISTALL = ["VFU not ready", "VLSU not ready", "VSLDU not ready",
          "scoreboard full"]
# VFU stall causes, in `vstall` bit order.
VSTALL = ["operand vs1 wait", "operand vs2 wait", "operand vd wait",
          "FU inputs busy", "IPU<->FPU switch", "scalar wb wait"]
# Retire units, and the ex_unit -> retire-unit map (ex_unit: 0=CON 1=LSU
# 2=SLD 3=VFU; CON instructions retire in the controller, with no R line).
UNITS = ["VFU", "VLSU", "VSLDU", "CSR"]
EXUNIT2UNIT = {3: 0, 1: 1, 2: 2, 0: 3}


class CoreEngine:
    """Engine-trace reduction of one Spatz core."""

    def __init__(self):
        # VLEN, N_FU, NrMemPorts, VRFWordBytes, ELENB, RobDepth
        self.dims = (0, 0, 0, 0, 0, 0)
        # L records: elements fed vs the peak AT THE CURRENT SEW, and the
        # per-lane result-valid bytes. Both are change-detected state runs, so
        # each is weighted by the cycles its run covers.
        self.vfu_elem = 0.0      # sum of elems over cycles
        self.vfu_peak = 0.0      # sum of nr_elem_word over the same cycles
        self.vfu_fed = 0         # cycles with a word fed
        self.lane_cyc = collections.Counter()   # lane -> busy cycles
        self.rob_sum = collections.Counter()    # port -> summed level
        self.rob_cyc = 0                        # cycles covered by B runs
        self.end_cyc = 0
        self.active = 0         # cycles with a C row (engine not idle)
        self.bucket = collections.Counter()
        self.issue = 0
        self.vfu_busy = self.vlsu_busy = self.vsldu_busy = 0
        self.vfu_rsp = self.vlsu_rsp = self.vsldu_rsp = 0
        self.memact = 0         # port-cycles with a request valid
        self.first = None
        self.last = 0
        # -- stall causes (cycles) and VFU lane occupancy (bytes) --
        self.istall = collections.Counter()   # cause -> cycles
        self.iblocked = 0       # cycles blocked by ANY cause (no double count)
        self.iwait = 0          # cycles with an instruction buffered
        self.vstall = collections.Counter()
        self.vfu_req_cyc = 0    # VFU-busy cycles
        self.vfu_clear = 0      # ... of those, with no stall cause asserted
        self.lin = self.lout = 0
        self.feed = 0           # cycles a word was fed to the FUs
        self.extended = False   # log carries the 2026-08 fields
        # -- VLSU reorder buffers (engine's own view of its memory work) --
        self.vlsu_out_sum = 0   # sum of outstanding requests over cycles
        self.vlsu_out_cyc = 0   # cycles with anything outstanding
        self.rob_full_cyc = 0   # cycles with at least one ROB saturated
        self.vlsu_ext = False   # log carries the ROB fields
        # -- per vector instruction (I/R lines) --
        self.n_insn = [0, 0, 0, 0]          # per UNITS
        self.vl_sum = [0, 0, 0, 0]
        self.sew = collections.Counter()    # (unit, vsew) -> count
        self.lmul = collections.Counter()   # (unit, vlmul) -> count
        self.lat = [mt.LatStats() for _ in UNITS]
        self.n_unretired = 0


def _bucket(stall, vfu_busy, vlsu_busy, vsldu_busy):
    if vfu_busy:
        return "compute"
    if vlsu_busy or vsldu_busy:
        return "mem"
    if stall:
        return "stall"
    return "drain"


def load_engine(sp_dir, lo, hi):
    """Parse every spatz_g*_t*_c*.log -> {(g,t,c): CoreEngine}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(sp_dir, "spatz_g*_t*_c*.log"))):
        m = SPATZ_LOG_RE.search(os.path.basename(path))
        if not m:
            continue
        g, t, c = (int(v) for v in m.groups())
        e = CoreEngine()
        # (unit, scoreboard id) -> issue cycles awaiting their retire
        pend = collections.defaultdict(collections.deque)
        # L/B are change-detected state RUNS: a row holds until the next one,
        # so each is charged for the cycles its run actually covers, clipped
        # to the window. Counting rows instead would weight a 1-cycle blip the
        # same as a 1000-cycle plateau.
        l_state = b_state = None

        def flush_l(upto):
            nonlocal l_state
            if l_state is not None:
                n = min(upto, hi) - max(l_state[0], lo)
                if n > 0 and l_state[3]:
                    e.vfu_elem += l_state[3] * n
                    e.vfu_peak += l_state[2] * n
                    e.vfu_fed += n
                    for i, by in enumerate(l_state[4:]):
                        if by:
                            e.lane_cyc[i] += n
            l_state = None

        def flush_b(upto):
            nonlocal b_state
            if b_state is not None:
                n = min(upto, hi) - max(b_state[0], lo)
                if n > 0:
                    for i, lvl in enumerate(b_state[1:]):
                        e.rob_sum[i] += lvl * n
                    e.rob_cyc += n
            b_state = None

        with open(path, "r", errors="replace") as f:
            for line in f:
                x = line.split()
                if not x:
                    continue
                # 4..7 fields: the header grew twice (VRFWordBytes, then
                # ELENB + RobDepth). Pad rather than pin the length -- an
                # exact-length check here silently blanked every dim once.
                if x[0] == "D" and 4 <= len(x) <= 7:
                    e.dims = tuple(
                        [int(v) for v in x[1:]] + [0, 0, 0])[:6]
                    continue
                if x[0] == "E" and len(x) == 2:
                    e.end_cyc = int(x[1])
                    continue
                # I <cyc> <id> <ex_unit> <vl> <vsew> <vlmul>
                if x[0] == "I" and len(x) == 7:
                    try:
                        cyc, iid, exu = int(x[1]), int(x[2]), int(x[3])
                        vl, sew, lmul = int(x[4]), int(x[5]), int(x[6])
                    except ValueError:
                        continue
                    if not (lo <= cyc < hi):
                        continue
                    u = EXUNIT2UNIT.get(exu, 3)
                    e.n_insn[u] += 1
                    e.vl_sum[u] += vl
                    e.sew[(u, sew)] += 1
                    e.lmul[(u, lmul)] += 1
                    if u != 3:                    # CSR ops have no R line
                        pend[(u, iid)].append(cyc)
                    continue
                # R <cyc> <unit> <id>
                if x[0] == "R" and len(x) == 4:
                    try:
                        cyc, u, iid = int(x[1]), int(x[2]), int(x[3])
                    except ValueError:
                        continue
                    if not (lo <= cyc < hi):
                        continue
                    q = pend.get((u, iid))
                    if q:
                        e.lat[u].add(cyc - q.popleft())
                    continue
                # L <cyc> <vsew> <nr_elem_word> <elems> <lane0..laneN-1>
                if x[0] == "L" and len(x) >= 5:
                    try:
                        v = [int(z) for z in x[1:]]
                    except ValueError:
                        continue
                    flush_l(v[0])
                    l_state = v
                    continue
                # B <cyc> <pend0..pendN-1>
                if x[0] == "B" and len(x) >= 3:
                    try:
                        v = [int(z) for z in x[1:]]
                    except ValueError:
                        continue
                    flush_b(v[0])
                    b_state = v
                    continue
                if x[0] != "C" or len(x) not in (14, 18, 21):
                    continue
                v = [int(z) for z in x[1:]]
                cyc = v[0]
                if not (lo <= cyc < hi):
                    continue
                (_cyc, _occ, stall, issue, vfu_b, _vfu_st, vlsu_b, _vlsu_st,
                 vsldu_b, vfu_r, vlsu_r, vsldu_r, memact) = v[:13]
                if len(v) >= 17:                  # 2026-08 fields
                    istall, vstall, lin, lout = v[13:17]
                    e.extended = True
                    # A cause only means "blocked" while something is
                    # actually waiting to issue (bit 4). The RTL's own `stall`
                    # ANDs req_buffer_valid for the same reason; without it a
                    # stale ex_unit or a full scoreboard would look like a
                    # stall on an idle engine.
                    if istall & (1 << 4):
                        e.iwait += 1
                        for i, nm in enumerate(ISTALL):
                            if istall & (1 << i):
                                e.istall[nm] += 1
                        if istall & 0xf:
                            e.iblocked += 1
                    if vstall or True:
                        # vstall is only meaningful while the VFU holds a
                        # request; the RTL already zeroes it otherwise, and a
                        # busy VFU with no cause is a cycle of real work.
                        if vfu_b:
                            e.vfu_req_cyc += 1
                            if not vstall:
                                e.vfu_clear += 1
                        for i, nm in enumerate(VSTALL):
                            if vstall & (1 << i):
                                e.vstall[nm] += 1
                    e.lin += lin
                    e.lout += lout
                    if lin:
                        e.feed += 1
                if len(v) >= 20:                  # VLSU ROB state
                    vlsu_out, rob_full, _rob_empty = v[17:20]
                    e.vlsu_ext = True
                    e.vlsu_out_sum += vlsu_out
                    if vlsu_out:
                        e.vlsu_out_cyc += 1
                    if rob_full:
                        e.rob_full_cyc += 1
                e.active += 1
                e.bucket[_bucket(stall, vfu_b, vlsu_b, vsldu_b)] += 1
                e.issue += issue
                e.vfu_busy += vfu_b
                e.vlsu_busy += vlsu_b
                e.vsldu_busy += vsldu_b
                e.vfu_rsp += vfu_r
                e.vlsu_rsp += vlsu_r
                e.vsldu_rsp += vsldu_r
                e.memact += memact
                if e.first is None:
                    e.first = cyc
                e.last = cyc
        # the last run holds to the E footer
        flush_l(e.end_cyc + 1)
        flush_b(e.end_cyc + 1)
        e.n_unretired = sum(len(q) for q in pend.values())
        out[(g, t, c)] = e
    return out


class CoreMem:
    """Memory-trace reduction of one Spatz core (all latency stats streamed,
    so a 256-core run never holds every record)."""

    def __init__(self, nr_ports):
        self.all = mt.LatStats()
        self.cls = [mt.LatStats() for _ in mt.CLASSES]
        self.rw = [mt.LatStats(), mt.LatStats()]     # [read, write]
        self.port = [mt.LatStats() for _ in range(max(1, nr_ports))]
        self.n_req = self.n_rsp = self.n_inflight = self.n_orphan = 0
        self.n_cls = [0] * len(mt.CLASSES)
        self.n_rw = [0, 0]
        self.occ = 0            # sum of latencies = request-cycles in flight

    def add(self, r):
        self.n_cls[r.cls] += 1
        self.n_rw[r.wr] += 1
        if r.lat is None:
            return
        self.all.add(r.lat)
        self.cls[r.cls].add(r.lat)
        self.rw[r.wr].add(r.lat)
        if r.port < len(self.port):
            self.port[r.port].add(r.lat)
        self.occ += r.lat


def load_mem(noc_dir, lo, hi, nr_mem_ports, tiles_per_group):
    """Parse the VECTOR ports out of noc_profiling/core_g*_t*.log ->
    {(g,t,c): CoreMem}.

    core_ carries every core memory port of a tile on the flat local-port axis
    (idx = ports_per_core*c + p, p=0 the scalar Snitch port). The Spatz vector
    ports are p >= 1, so vector port = p - 1. There is no separate
    engine-boundary trace: the path from the VLSU to this tap is combinational,
    so the cycles are the same ones, and noc_profiling is the single place a
    memory request's life is recorded."""
    out = {}
    ppc = nr_mem_ports + 1
    for g, t, ports in mt.iter_port_tiles(noc_dir, "core", tiles_per_group):
        for idx, ct in ports.items():
            c, p = divmod(idx, ppc)
            if p == 0:                       # the scalar Snitch data port
                continue
            cm = out.get((g, t, c))
            if cm is None:
                cm = out[(g, t, c)] = CoreMem(nr_mem_ports)
            for r in ct.recs:
                if not (lo <= r.cyc < hi):
                    continue
                cm.n_req += 1
                if r.lat is None:
                    cm.n_inflight += 1
                cm.add(r._replace(port=p - 1))
            cm.n_rsp += ct.n_rsp
            cm.n_orphan += ct.n_orphan
    return out


def _fmt(v, w=10, p=2):
    return f"{v:>{w}.{p}f}"


def _pct(n, d):
    return 100.0 * n / d if d else 0.0


def _merged(stats):
    """Merge a list of LatStats into one."""
    m = mt.LatStats()
    for s in stats:
        m.merge(s)
    return m


def _sew_str(c):
    """SEW histogram as e8/e16/e32/e64 shares."""
    tot = sum(c.values())
    return " ".join(f"e{8 << k}:{100.0 * v / tot:.0f}%"
                    for k, v in sorted(c.items())) if tot else "-"


# vlmul_e: 0..3 = 1,2,4,8 ; 5..7 = 1/8,1/4,1/2 (rvv_pkg)
LMUL_NAME = {0: "1", 1: "2", 2: "4", 3: "8",
             4: "res", 5: "1/8", 6: "1/4", 7: "1/2"}


def _lmul_str(c):
    tot = sum(c.values())
    return " ".join(f"m{LMUL_NAME.get(k, k)}:{100.0 * v / tot:.0f}%"
                    for k, v in sorted(c.items())) if tot else "-"


def _lat_line(name, s, total_n):
    if not s.n:
        return f"    {name:<10} {'-':>10}"
    return (f"    {name:<10} {s.n:>10d} {_pct(s.n, total_n):>7.1f}%"
            f" {s.mean:>9.2f} {s.quantile(50):>7d} {s.quantile(90):>7d}"
            f" {s.quantile(99):>7d} {s.hi:>7d}")


def report(eng, mem, args, out):
    cores = sorted(set(eng) | set(mem))
    if not cores:
        print("no Spatz logs found", file=sys.stderr)
        return 1
    ngrp = len({g for g, _t, _c in cores})
    ntile = len({(g, t) for g, t, _c in cores})
    any_e = next(iter(eng.values())) if eng else None
    end_cyc = max((e.end_cyc for e in eng.values()), default=0)
    win_lo = max(0, args.cycle_start)
    win_hi = args.cycle_end if args.cycle_end is not None else end_cyc + 1
    span = max(1, min(win_hi, end_cyc + 1) - win_lo)

    print("=" * 78, file=out)
    print("Spatz performance summary", file=out)
    print("=" * 78, file=out)
    print(f"  logs      : {args.dir}", file=out)
    print(f"  geometry  : {len(cores)} Spatz cores "
          f"({ngrp} groups x {ntile // max(1, ngrp)} tiles x "
          f"{len(cores) // max(1, ntile)} cores/tile)", file=out)
    if any_e:
        print(f"  engine    : VLEN={any_e.dims[0]}, N_FU={any_e.dims[1]}, "
              f"mem ports/core={any_e.dims[2]}", file=out)
    print(f"  run       : {end_cyc} cycles; window "
          f"[{win_lo}, {win_hi if args.cycle_end is not None else 'end'}) "
          f"= {span} cycles", file=out)
    if not mem:
        print("  NOTE: no spmem_g*_t*.log -- engine section only. The "
              "per-request memory trace needs noc_profiling=1 (it is captured "
              "by tb_noc_profiling.svh, with the pe_/rm_ port traces).",
              file=out)

    # ---------------- engine ----------------
    if eng:
        tot = collections.Counter()
        for e in eng.values():
            for b in CPI[:-1]:
                tot[b] += e.bucket[b]
            tot["idle"] += span - e.active
        allc = float(span * len(eng))
        print("", file=out)
        print("-- engine occupancy (all cores, % of window cycles) "
              + "-" * 24, file=out)
        for b in CPI:
            print(f"    {b:<10}{_fmt(_pct(tot[b], allc), 9, 2)}%", file=out)

        s_issue = sum(e.issue for e in eng.values())
        s_vfu_r = sum(e.vfu_rsp for e in eng.values())
        s_vlsu_r = sum(e.vlsu_rsp for e in eng.values())
        s_vsldu_r = sum(e.vsldu_rsp for e in eng.values())
        s_vfu_b = sum(e.vfu_busy for e in eng.values())
        s_vlsu_b = sum(e.vlsu_busy for e in eng.values())
        s_memact = sum(e.memact for e in eng.values())
        n = len(eng)
        print("", file=out)
        print("-- vector instructions (per core, mean) " + "-" * 36, file=out)
        print(f"    issued                    {s_issue / n:>12.1f}", file=out)
        print(f"    VFU retired               {s_vfu_r / n:>12.1f}", file=out)
        print(f"    VLSU retired (mem insns)  {s_vlsu_r / n:>12.1f}", file=out)
        print(f"    VSLDU retired             {s_vsldu_r / n:>12.1f}",
              file=out)
        print(f"    VFU busy cycles           {s_vfu_b / n:>12.1f}", file=out)
        print(f"    VLSU busy cycles          {s_vlsu_b / n:>12.1f}", file=out)
        print("", file=out)
        print("    VFU  cycles / VFU insn    "
              f"{(s_vfu_b / s_vfu_r if s_vfu_r else 0):>12.2f}", file=out)
        print("    VLSU cycles / mem insn    "
              f"{(s_vlsu_b / s_vlsu_r if s_vlsu_r else 0):>12.2f}"
              "   <- model target", file=out)
        print("    mem-port valid cycles     "
              f"{s_memact / n:>12.1f}", file=out)
        # VLSU reorder buffers: the engine's own view of its memory work.
        # Cross-check against the round-trip table below via Little's law --
        # outstanding = request rate * latency, measured independently.
        if any(e.vlsu_ext for e in eng.values()):
            s_out = sum(e.vlsu_out_sum for e in eng.values())
            s_ocyc = sum(e.vlsu_out_cyc for e in eng.values())
            s_full = sum(e.rob_full_cyc for e in eng.values())
            print("", file=out)
            print("    VLSU outstanding / cyc    "
                  f"{s_out / (span * n):>12.2f}", file=out)
            print("    ... while non-zero        "
                  f"{(s_out / s_ocyc if s_ocyc else 0):>12.2f}", file=out)
            print("    ROB-full cycles           "
                  f"{_pct(s_full, span * n):>11.1f}%"
                  "   (VLSU throttled by its own depth)", file=out)

    # ---------------- stall causes + VFU datapath ----------------
    if eng and any(e.extended for e in eng.values()):
        n = len(eng)
        allc = float(span * n)
        print("", file=out)
        print("-- why an instruction did not issue (all cores, % of window) "
              + "-" * 15, file=out)
        print("   causes are a per-cycle bitmap and can overlap, so they do "
              "not partition", file=out)
        tot_i = collections.Counter()
        iwait = iblocked = 0
        for e in eng.values():
            tot_i.update(e.istall)
            iwait += e.iwait
            iblocked += e.iblocked
        for nm in ISTALL:
            print(f"    {nm:<22}{_fmt(_pct(tot_i[nm], allc), 9, 2)}%",
                  file=out)
        print(f"    {'ANY of the above':<22}"
              f"{_fmt(_pct(iblocked, allc), 9, 2)}%   <- true blocked time",
              file=out)
        print(f"    {'insn buffered':<22}"
              f"{_fmt(_pct(iwait, allc), 9, 2)}%   "
              "<- something to issue (waiting or issuing)", file=out)

        peak = next((e.dims[3] for e in eng.values() if e.dims[3]), 0)
        lin = sum(e.lin for e in eng.values())
        lout = sum(e.lout for e in eng.values())
        feed = sum(e.feed for e in eng.values())
        vfu_b = sum(e.vfu_busy for e in eng.values())
        print("", file=out)
        print("-- VFU datapath (peak %d B/cycle) " % peak + "-" * 40, file=out)
        print(f"    cycles feeding the FUs    {feed / n:>12.1f}"
              f"   ({_pct(feed, vfu_b):.1f}% of VFU-busy cycles)", file=out)
        print(f"    bytes fed / core          {lin / n:>12.1f}", file=out)
        print("    byte-mask fill            "
              f"{_pct(lin, feed * peak) if (feed and peak) else 0:>12.2f}%"
              "   <- of the cycles it feeds", file=out)
        print("    byte-mask fill (window)   "
              f"{_pct(lin, allc * peak) if peak else 0:>12.2f}%", file=out)
        print("      (valid_operations is '1 for ordinary vector ops, so the "
              "two above track", file=out)
        print("       word_issued, NOT the vl tail -- use the SEW-relative "
              "figures below)", file=out)
        print(f"    result-valid bytes / core {lout / n:>12.1f}"
              "   (held across writeback back-pressure)", file=out)

        # The honest throughput measure: elements actually fed against the
        # peak AT THE CURRENT ELEMENT WIDTH (nr_elem_word). Filling the lanes
        # with fp8 scores the same 100% as filling them with fp32 -- a narrow
        # SEW is not charged for leaving the byte width unused.
        s_elem = sum(e.vfu_elem for e in eng.values())
        s_peak = sum(e.vfu_peak for e in eng.values())
        s_fed = sum(e.vfu_fed for e in eng.values())
        if s_peak:
            print("", file=out)
            print("    elements / feeding cyc    "
                  f"{s_elem / s_fed if s_fed else 0:>12.2f}", file=out)
            print("    vs SEW peak (feeding)     "
                  f"{_pct(s_elem, s_peak):>12.2f}%"
                  "   <- tail words only", file=out)
            win_peak = (s_peak * allc / s_fed) if s_fed else 0
            print("    vs SEW peak (window)      "
                  f"{_pct(s_elem, win_peak):>12.2f}%"
                  "   <- model target", file=out)
            lanes = sorted({i for e in eng.values() for i in e.lane_cyc})
            if lanes:
                print("    per-lane busy (result-valid, % of window)",
                      file=out)
                for i in lanes:
                    v = sum(e.lane_cyc[i] for e in eng.values())
                    print(f"      lane {i:<17}{_pct(v, allc):>12.2f}%",
                          file=out)

        # Per-port ROB level: which port saturated, and how deep it ran.
        depth = next((e.dims[5] for e in eng.values() if e.dims[5]), 0)
        rcyc = sum(e.rob_cyc for e in eng.values())
        if depth and rcyc:
            print("", file=out)
            print(f"    VLSU ROB level per port (depth {depth})", file=out)
            for p in sorted({i for e in eng.values() for i in e.rob_sum}):
                v = sum(e.rob_sum[p] for e in eng.values())
                print(f"      port {p:<17}{v / rcyc:>12.2f}"
                      f"   ({_pct(v, rcyc * depth):.1f}% full)", file=out)
        vreq = sum(e.vfu_req_cyc for e in eng.values())
        if vreq:
            tot_v = collections.Counter()
            for e in eng.values():
                tot_v.update(e.vstall)
            clear = sum(e.vfu_clear for e in eng.values())
            print(f"    no stall cause asserted   "
                  f"{_pct(clear, vreq):>12.2f}%   <- of VFU-busy cycles",
                  file=out)
            print("    stall causes (% of VFU-busy cycles; overlapping "
                  "bitmap, they do not sum):", file=out)
            for nm in VSTALL:
                if tot_v[nm]:
                    print(f"      {nm:<22}"
                          f"{_fmt(_pct(tot_v[nm], vreq), 9, 2)}%", file=out)

    # ---------------- vector instruction mix ----------------
    if eng and any(sum(e.n_insn) for e in eng.values()):
        n = len(eng)
        print("", file=out)
        print("-- vector instructions issued (per core, mean) " + "-" * 29,
              file=out)
        print(f"    {'unit':<8}{'count':>9}{'mean vl':>10}{'SEW':>22}"
              f"{'LMUL':>16}{'cycles I->R':>13}", file=out)
        for u, nm in enumerate(UNITS):
            cnt = sum(e.n_insn[u] for e in eng.values())
            if not cnt:
                continue
            vl = sum(e.vl_sum[u] for e in eng.values())
            sew = collections.Counter()
            lmul = collections.Counter()
            lat = _merged([e.lat[u] for e in eng.values()])
            for e in eng.values():
                for (uu_, k), v in e.sew.items():
                    if uu_ == u:
                        sew[k] += v
                for (uu_, k), v in e.lmul.items():
                    if uu_ == u:
                        lmul[k] += v
            print(f"    {nm:<8}{cnt / n:>9.1f}{vl / cnt:>10.1f}"
                  f"{_sew_str(sew):>22}{_lmul_str(lmul):>16}"
                  f"{(lat.mean if lat.n else 0):>13.1f}", file=out)
        unret = sum(e.n_unretired for e in eng.values())
        if unret:
            print(f"    ({unret} issued but not retired inside the window)",
                  file=out)

    # ---------------- memory ----------------
    if mem:
        allm = _merged([m.all for m in mem.values()])
        n_req = sum(m.n_req for m in mem.values())
        n_infl = sum(m.n_inflight for m in mem.values())
        n_orph = sum(m.n_orphan for m in mem.values())
        occ = sum(m.occ for m in mem.values())
        n = len(mem)
        print("", file=out)
        print("-- memory requests (per core, mean) " + "-" * 40, file=out)
        print(f"    accepted word requests    {n_req / n:>12.1f}", file=out)
        print(f"    reads / writes            "
              f"{sum(m.n_rw[0] for m in mem.values()) / n:>12.1f} /"
              f"{sum(m.n_rw[1] for m in mem.values()) / n:>10.1f}", file=out)
        if eng:
            s_vlsu_r = sum(e.vlsu_rsp for e in eng.values())
            s_memact = sum(e.memact for e in eng.values())
            if s_vlsu_r:
                print("    requests / mem insn       "
                      f"{n_req / s_vlsu_r:>12.2f}", file=out)
            if n_req:
                # memact counts every cycle a port holds valid, the granted
                # cycle included -> subtract one cycle per accepted request to
                # get the pure wait-for-grant time.
                print("    grant wait / request      "
                      f"{(s_memact - n_req) / n_req:>12.2f}", file=out)
        print("    mean outstanding          "
              f"{occ / span / n:>12.2f}", file=out)

        print("", file=out)
        print("-- round-trip latency (accepted request -> response, cycles) "
              + "-" * 15, file=out)
        print(f"    {'':<10} {'count':>10} {'share':>8} {'mean':>9} "
              f"{'p50':>7} {'p90':>7} {'p99':>7} {'max':>7}", file=out)
        print(_lat_line("ALL", allm, allm.n), file=out)
        for k, name in enumerate(mt.CLASSES):
            print(_lat_line(name, _merged([m.cls[k] for m in mem.values()]),
                            allm.n), file=out)
        for k, name in enumerate(["read", "write"]):
            print(_lat_line(name, _merged([m.rw[k] for m in mem.values()]),
                            allm.n), file=out)
        nports = len(next(iter(mem.values())).port)
        for p in range(nports):
            print(_lat_line(f"port {p}",
                            _merged([m.port[p] for m in mem.values()]),
                            allm.n), file=out)

        if args.hist:
            print("", file=out)
            print("-- latency histogram (all cores) " + "-" * 43, file=out)
            hist = collections.Counter()
            for m in mem.values():
                for lat, cnt in m.all.hist.items():
                    hist[lat] += cnt
            top = sorted(hist.items())
            wide = max((c for _l, c in top), default=1)
            for lat, cnt in top[:args.hist]:
                bar = "#" * max(1, int(40 * cnt / wide))
                print(f"    {lat:>6} cyc {cnt:>10d} "
                      f"{_pct(cnt, allm.n):>6.2f}%  {bar}", file=out)
            if len(top) > args.hist:
                rest = sum(c for _l, c in top[args.hist:])
                print(f"    (+{len(top) - args.hist} longer buckets, "
                      f"{rest} requests)", file=out)

        print("", file=out)
        print("-- acceptance check " + "-" * 56, file=out)
        print(f"    requests (Q)              {n_req:>12d}", file=out)
        print(f"    paired with a response    {allm.n:>12d}", file=out)
        print(f"    in flight at end of trace {n_infl:>12d}", file=out)
        print(f"    orphan responses          {n_orph:>12d}", file=out)
        if n_orph:
            print("    WARNING: responses with no outstanding request -- the "
                  "window may cut a\n             transaction in half "
                  "(use the full run) or the pairing key is wrong.", file=out)

    # ---------------- per-core / per-group ----------------
    if args.per_core or args.per_group:
        _detail(eng, mem, span, args, out)
    return 0


def _core_row(e, m, span):
    """One (engine, memory) row of derived per-core numbers."""
    vlsu_pi = (e.vlsu_busy / e.vlsu_rsp) if (e and e.vlsu_rsp) else 0.0
    req = m.n_req if m else 0
    lat = m.all.mean if m else 0.0
    grant = ((e.memact - req) / req) if (e and req) else 0.0
    return {
        "active_pct": _pct(e.active, span) if e else 0.0,
        "mem_insn": e.vlsu_rsp if e else 0,
        "vlsu_cyc_per_insn": vlsu_pi,
        "requests": req,
        "grant_wait": grant,
        "lat_mean": lat,
        "lat_p99": m.all.quantile(99) if m and m.all.n else 0,
        "local_pct": _pct(m.n_cls[mt.CLS_LOCAL], req) if m else 0.0,
        "intra_pct": _pct(m.n_cls[mt.CLS_INTRA], req) if m else 0.0,
        "inter_pct": _pct(m.n_cls[mt.CLS_INTER], req) if m else 0.0,
    }


COLS = [("act%", "active_pct", 6, 1), ("memins", "mem_insn", 8, 0),
        ("vlsu/ins", "vlsu_cyc_per_insn", 9, 2), ("reqs", "requests", 9, 0),
        ("grant", "grant_wait", 7, 2), ("lat", "lat_mean", 7, 2),
        ("p99", "lat_p99", 6, 0), ("loc%", "local_pct", 6, 1),
        ("intra%", "intra_pct", 7, 1), ("inter%", "inter_pct", 7, 1)]


def _detail(eng, mem, span, args, out):
    keys = sorted(set(eng) | set(mem))
    hdr = "  " + " ".join(f"{h:>{w}}" for h, _k, w, _p in COLS)
    if args.per_group:
        print("", file=out)
        print("-- per group " + "-" * 63, file=out)
        print(f"  {'group':<8}{hdr}", file=out)
        bygrp = collections.defaultdict(list)
        for g, t, c in keys:
            bygrp[g].append((g, t, c))
        for g in sorted(bygrp):
            rows = [_core_row(eng.get(k), mem.get(k), span)
                    for k in bygrp[g]]
            avg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
            print(f"  G{g:<7d}" + " ".join(
                f"{avg[k]:>{w}.{p}f}" for _h, k, w, p in COLS), file=out)
    if args.per_core:
        print("", file=out)
        print("-- per core " + "-" * 64, file=out)
        print(f"  {'core':<12}{hdr}", file=out)
        for k in keys:
            r = _core_row(eng.get(k), mem.get(k), span)
            print(f"  G{k[0]}T{k[1]}C{k[2]:<7}" + " ".join(
                f"{r[c]:>{w}.{p}f}" for _h, c, w, p in COLS), file=out)


def write_csv(path, eng, mem, span):
    keys = sorted(set(eng) | set(mem))
    with open(path, "w") as f:
        f.write("group,tile,core," + ",".join(k for _h, k, _w, _p in COLS)
                + "\n")
        for k in keys:
            r = _core_row(eng.get(k), mem.get(k), span)
            f.write(f"{k[0]},{k[1]},{k[2]}," + ",".join(
                f"{r[c]:.4f}" for _h, c, _w, _p in COLS) + "\n")


def write_dump(path, mem_dir, lo, hi, core, ppc=1, tpg=1):
    """One row per paired request -- the raw events, decoded (class resolved,
    request paired with its response). Streams tile by tile so a whole-machine
    dump never has to fit in memory."""
    want = tuple(int(v) for v in core.split(",")) if core else None
    n = 0
    with open(path, "w") as f:
        f.write("cycle,group,tile,core,port,rw,addr,dst_tile,class,latency\n")
        if not mem_dir:
            return 0
        for g, t, ports in mt.iter_port_tiles(mem_dir, "core", tpg):
            for idx, ct in sorted(ports.items()):
                c, p = divmod(idx, ppc)
                if p == 0 or (want and (g, t, c) != want):
                    continue
                for r in ct.recs:
                    if not (lo <= r.cyc < hi) or r.lat is None:
                        continue
                    f.write(f"{r.cyc},{g},{t},{c},{r.port},"
                            f"{'w' if r.wr else 'r'},0x{r.addr:x},{r.dst},"
                            f"{mt.CLASSES[r.cls]},{r.lat}\n")
                    n += 1
    return n


def _find(root, pattern, subdirs):
    """Directory holding `pattern`: `root` itself, else the first of `subdirs`
    under it that has the files. None if nowhere."""
    for d in (root,) + tuple(os.path.join(root, s) for s in subdirs):
        if glob.glob(os.path.join(d, pattern)):
            return d
    return None


def main():
    ap = argparse.ArgumentParser("spatz_perf")
    ap.add_argument("dir", help="build dir (finds spatz_profiling/ for the "
                                "engine trace and noc_profiling/ for the "
                                "memory trace), or either of those dirs")
    ap.add_argument("--tiles-per-group", type=int, default=1,
                    help="tiles per group -- needed to turn a destination "
                         "tile into a local/intra/inter address class")
    ap.add_argument("--cycle-start", type=int, default=0,
                    help="ignore samples before this cycle (kernel window)")
    ap.add_argument("--cycle-end", type=int, default=None,
                    help="ignore samples at/after this cycle")
    ap.add_argument("--per-core", action="store_true",
                    help="add the per-core table")
    ap.add_argument("--per-group", action="store_true",
                    help="add the per-group table (NUMA spread)")
    ap.add_argument("--hist", type=int, nargs="?", const=20, default=0,
                    metavar="N", help="add a latency histogram (N shortest "
                                      "buckets, default 20)")
    ap.add_argument("--csv", help="also write the per-core table as CSV")
    ap.add_argument("--dump", metavar="CSV",
                    help="also write ONE ROW PER PAIRED REQUEST to this CSV "
                         "(cycle, core, port, rw, addr, dst, class, latency) "
                         "for drilling into a specific discrepancy. Combine "
                         "with --core and --cycle-start/--cycle-end: "
                         "the whole machine is millions of rows.")
    ap.add_argument("--core", metavar="G,T,C",
                    help="restrict --dump to one core, e.g. --core 0,1,2")
    ap.add_argument("-o", "--output", help="write the report here "
                                           "(default: stdout)")
    args = ap.parse_args()

    # The two traces live in different directories (engine internals vs
    # memory-port traffic); accept the build dir, or either dir directly.
    eng_dir = _find(args.dir, "spatz_g*_t*_c*.log", ("spatz_profiling",))
    mem_dir = _find(args.dir, "core_g*_t*.log", ("noc_profiling",))
    args.dir = f"{eng_dir or '-'} + {mem_dir or '-'}"

    lo = max(0, args.cycle_start)
    hi = args.cycle_end if args.cycle_end is not None else (1 << 62)
    eng = load_engine(eng_dir, lo, hi) if eng_dir else {}
    # The engine header gives NrMemPorts, which is what turns a core_ flat port
    # index back into (core, vector port): ports_per_core = NrMemPorts + 1.
    nr_mem_ports = next((e.dims[2] for e in eng.values() if e.dims[2]), 0)
    mem = (load_mem(mem_dir, lo, hi, nr_mem_ports, args.tiles_per_group)
           if mem_dir and nr_mem_ports else {})
    if not eng and not mem:
        print(f"no spatz_g*_t*_c*.log / core_g*_t*.log under {args.dir} "
              "(or its spatz_profiling/ , noc_profiling/)", file=sys.stderr)
        return 1
    out = open(args.output, "w") if args.output else sys.stdout
    try:
        rc = report(eng, mem, args, out)
    finally:
        if args.output:
            out.close()
    if args.csv:
        end_cyc = max((e.end_cyc for e in eng.values()), default=0)
        span = max(1, min(hi, end_cyc + 1) - lo)
        write_csv(args.csv, eng, mem, span)
        print(f"wrote {args.csv}", file=sys.stderr)
    if args.dump:
        n = write_dump(args.dump, mem_dir, lo, hi, args.core,
                       nr_mem_ports + 1, args.tiles_per_group)
        print(f"wrote {args.dump} ({n} requests)", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
