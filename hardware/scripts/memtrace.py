#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Author: Yinrong Li, ETH Zurich
"""Readers for the per-request MEMORY-PORT traces of every requester.

``noc_profiling/`` records the whole LIFE OF EACH MEMORY REQUEST and nothing
else -- requester-internal state lives in the per-engine traces
(``tb/tb_spatz_profiling.svh``, ``tb/tb_redmule_profiling.svh``). Two families
cover every requester, both emitted by ``tb/tb_noc_profiling.svh`` under the
``noc_profiling=1`` gate and both parsed here with the SAME pairing, so all
requester classes are measured with one metric -- see ``iter_port_tiles``:

    core_g<g>_t<t>.log   the Snitch scalar data port AND every Spatz vector
                         port, on the flat local-port axis
                         (idx = ports_per_core*core + port, p=0 scalar)
    rm_g<g>_t<t>.log     the RedMulE master ports, on whichever plane carries
                         them (narrow: several ports; WP2 wide: one)

Pure stdlib, shared by ``interco_perf.py`` / ``spatz_perf.py`` (text reports)
and ``perfetto_gen.py`` (timeline counters), which must not depend on each
other.

Log format (one file per tile, every record prefixed with the port index)::

    D <data_width_bits>
    <idx> S <net> <io> 0 <start> <end> <state>
    <idx> P 1 0 <cyc> <wen> <addr_hex> <meta_id> <dst_tile>   request accepted
    <idx> P 0 0 <cyc> <meta_id>                               response arrived

``D`` gives the bits moved by one record: 32 on the narrow plane, and
WIDE_TCDM_DATA_WIDTH for ``rm_`` when the WP2 wide plane carries the engine.
Request COUNTS are therefore not comparable across a plane A/B but count*width
is; see ``read_width_bits``.

Pairing key: ``(port idx, meta_id)``. meta_id is unique among that port's
outstanding requests and round-trips unchanged, so within one key the k-th
response belongs to the k-th request; across keys responses do come back out of
order, which is exactly why the pairing is per key. The one exception is the
RedMulE Z writeback, which reuses ONE meta_id for many outstanding stores --
that path is an in-order FIFO, so FIFO pairing stays exact even though no
individual request can be followed across a NoC hop.

Latency convention: ``lat = rsp_cycle - req_cycle``, from the accepted request
handshake to the response landing back at the port. A local-tile hit is the
floor of the distribution; issue-side back-pressure (valid waiting for ready)
is NOT part of it -- that is the ``S`` state runs, and the engine trace's
``memact`` / ``vlsu_out``.
"""

import collections
import glob
import os
import re

# Address classes, by destination relative to the requesting core's own tile.
# CLS_UNKNOWN (= len(CLASSES), defined below) is only for logs predating the
# dst_tile field -- never a guess.
CLASSES = ["local", "intra", "inter", "soc"]
CLS_LOCAL, CLS_INTRA, CLS_INTER, CLS_SOC = range(4)

# One paired request. `lat` is None for a request still outstanding at the end
# of the trace (counted separately; never folded into a latency statistic).
Rec = collections.namedtuple("Rec", "cyc lat port wr cls dst addr")


class CoreTrace:
    """Every memory request of one Spatz core, plus the pairing bookkeeping
    that the acceptance check needs."""

    def __init__(self):
        self.recs = []          # [Rec] in request order
        self.n_req = 0          # accepted requests (Q lines)
        self.n_rsp = 0          # responses seen (P lines)
        self.n_inflight = 0     # requests with no response by end of trace
        self.n_orphan = 0       # responses matching no outstanding request

    def latencies(self, cls=None, wr=None):
        """Round-trip latencies, optionally filtered by address class and/or
        write flag. In-flight requests are excluded (no latency)."""
        return [r.lat for r in self.recs
                if r.lat is not None
                and (cls is None or r.cls == cls)
                and (wr is None or r.wr == wr)]


def classify(dst, g, t, tiles_per_group):
    """Address class of a request from tile (g,t) to global tile id `dst`
    (dst < 0 = a non-TCDM SoC/L2 access)."""
    if dst < 0:
        return CLS_SOC
    if dst == g * tiles_per_group + t:
        return CLS_LOCAL
    if dst // tiles_per_group == g:
        return CLS_INTRA
    return CLS_INTER


# ---------------------------------------------------------------------------
# Tile-side port logs. The P lines carry both directions, so pairing them gives
# the same round-trip metric for every requester class:
#   req  <idx> P 1 0 <cyc> <wen> <addr_hex> <meta_id> <dst_tile>
#   resp <idx> P 0 0 <cyc> <meta_id>
# `dst_tile` is absent in logs from before 2026-08-04 -> class falls back to
# "unknown" rather than being guessed.
# ---------------------------------------------------------------------------
PORT_LOG_RE = re.compile(r"(?:core|rm)_g(\d+)_t(\d+)\.log$")
CLS_UNKNOWN = len(CLASSES)
# Default record width when a log predates the `D <bits>` header.
DEFAULT_WIDTH_BITS = 32


def read_width_bits(path):
    """Bits moved by ONE record of this port log, from its `D <bits>` header.

    RedMulE emits rm_ on either plane, and one wide record is
    WIDE_TCDM_DATA_WIDTH bits where a narrow one is 32 -- so request COUNTS are
    not comparable across a plane A/B, but count*width is. Reading the width
    from the file (rather than from its name) is what lets a consumer normalise
    without knowing which plane ran."""
    with open(path, "r", errors="replace") as f:
        for line in f:
            x = line.split()
            if x and x[0] == "D" and len(x) >= 2:
                try:
                    return int(x[1])
                except ValueError:
                    break
            if x:
                break
    return DEFAULT_WIDTH_BITS


def parse_port_log(path, g, t, tiles_per_group):
    """Parse one core_/rm_ log -> {port idx: CoreTrace} (P lines only; the S
    state runs are the fabric view and are read separately)."""
    ports = {}
    pending = collections.defaultdict(collections.deque)
    with open(path, "r", errors="replace") as f:
        for line in f:
            x = line.split()
            if len(x) < 5 or x[1] != "P":
                continue
            try:
                idx, io, cyc = int(x[0]), int(x[2]), int(x[4])
            except ValueError:
                continue
            pt = ports.get(idx)
            if pt is None:
                pt = ports[idx] = CoreTrace()
            if io == 1 and len(x) >= 8:            # request
                try:
                    wr, addr, mid = int(x[5]), int(x[6], 16), int(x[7])
                    dst = int(x[8]) if len(x) > 8 else None
                except ValueError:
                    continue
                pending[(idx, mid)].append((cyc, addr, dst, wr))
                pt.n_req += 1
            elif io == 0 and len(x) >= 6:          # response
                try:
                    mid = int(x[5])
                except ValueError:
                    continue
                pt.n_rsp += 1
                q = pending.get((idx, mid))
                if q:
                    qcyc, addr, dst, wr = q.popleft()
                    pt.recs.append(Rec(
                        qcyc, cyc - qcyc, idx, wr,
                        CLS_UNKNOWN if dst is None
                        else classify(dst, g, t, tiles_per_group),
                        -1 if dst is None else dst, addr))
                else:
                    pt.n_orphan += 1
    for (idx, _mid), q in pending.items():
        pt = ports.setdefault(idx, CoreTrace())
        for qcyc, addr, dst, wr in q:
            pt.n_inflight += 1
            pt.recs.append(Rec(qcyc, None, idx, wr,
                               CLS_UNKNOWN if dst is None
                               else classify(dst, g, t, tiles_per_group),
                               -1 if dst is None else dst, addr))
    for pt in ports.values():
        pt.recs.sort(key=lambda r: r.cyc)
    return ports


def iter_port_tiles(noc_dir, kind, tiles_per_group):
    """Yield (g, t, {port idx: CoreTrace}) for every core_ (kind='core', the
    Snitch scalar port plus every Spatz vector port) or rm_ (kind='rm', the
    RedMulE ports on whichever plane carries them) log, in (g,t) order.
    `tiles_per_group` is needed to turn a destination tile into an address
    class."""
    for g, t, path in sorted(port_log_paths(noc_dir, kind)):
        yield g, t, parse_port_log(path, g, t, tiles_per_group)


def port_log_paths(noc_dir, kind):
    """[(g, t, path)] for one port-log family -- use when the width header or
    the file name itself is what you need, not the parsed records."""
    paths = []
    for path in glob.glob(os.path.join(noc_dir, f"{kind}_g*_t*.log")):
        m = PORT_LOG_RE.search(os.path.basename(path))
        if m:
            paths.append((int(m.group(1)), int(m.group(2)), path))
    return paths


def percentile(sorted_vals, q):
    """Nearest-rank percentile of an already-sorted list (q in [0,100])."""
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1,
                   int(round(q / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


class LatStats:
    """Streaming latency accumulator: count/sum/min/max plus an exact integer
    histogram (latencies are small cycle counts, so a dict histogram is exact
    and far cheaper than keeping every sample)."""

    def __init__(self):
        self.n = 0
        self.total = 0
        self.lo = None
        self.hi = None
        self.hist = collections.defaultdict(int)

    def add(self, lat):
        self.n += 1
        self.total += lat
        self.hist[lat] += 1
        if self.lo is None or lat < self.lo:
            self.lo = lat
        if self.hi is None or lat > self.hi:
            self.hi = lat

    def merge(self, other):
        for lat, n in other.hist.items():
            self.hist[lat] += n
        self.n += other.n
        self.total += other.total
        if other.lo is not None and (self.lo is None or other.lo < self.lo):
            self.lo = other.lo
        if other.hi is not None and (self.hi is None or other.hi > self.hi):
            self.hi = other.hi

    @property
    def mean(self):
        return self.total / self.n if self.n else 0.0

    def quantile(self, q):
        """Exact q-th percentile (0..100) from the histogram."""
        if not self.n:
            return 0
        target = max(1, int(round(q / 100.0 * self.n)))
        seen = 0
        for lat in sorted(self.hist):
            seen += self.hist[lat]
            if seen >= target:
                return lat
        return self.hi
