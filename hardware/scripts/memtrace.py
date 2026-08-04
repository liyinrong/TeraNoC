#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Author: Yinrong Li, ETH Zurich
"""Readers for the per-request MEMORY-PORT traces of every requester.

Parses ``noc_profiling/spmem_g<g>_t<t>.log`` (emitted by
``tb/tb_noc_profiling.svh`` beside the other memory-port traces
``pe_``/``rm_``, under the same ``noc_profiling=1`` gate;
``tb/tb_spatz_profiling.svh`` keeps the engine-internal trace) and pairs every
request with its response, so
consumers get one record per memory word request with its round-trip latency
and its address class. The same pairing runs on the tile-side port logs
``pe_g<g>_t<t>.log`` (Snitch scalar + Spatz vector ports) and
``rm_g<g>_t<t>.log`` (RedMulE master ports), so all three requester classes are
measured with ONE metric -- see ``iter_port_tiles``. Pure stdlib, shared by
``interco_perf.py`` / ``spatz_perf.py`` (text reports) and ``perfetto_gen.py``
(timeline counters), which must not depend on each other.

Log format (one file per tile, every line prefixed with the core index)::

    D <NrMemPorts> <NumCoresPerTile> <NumTilesPerGroup> <NumTiles> <IdBits>
    <c> Q <cyc> <port> <id> <addr_hex> <wr> <dst_tile>
    <c> P <cyc> <port> <id> <wr>
    E <last_cyc>

Pairing key: ``(port, id & (2**IdBits - 1), wr)``. The VLSU puts its per-port
ROB id on the request and gets it back on the response, and only the low
IdBits round-trip through the TCDM (mempool_cc.sv packs {write, id[IdBits-1:0]}
into meta_id) -- so the id is masked to IdBits before matching. `wr`
distinguishes a store-ack from a load reply that reuses the same id (the store
path frees its ROB entry at send, not at response). Within one key the memory
system cannot reorder (an id is not reused until it is freed), so the k-th
response pairs with the k-th request; ACROSS keys responses do come back out of
order, which is exactly why the pairing is per key.

Latency convention: ``lat = rsp_cycle - req_cycle``, i.e. the number of cycles
from the accepted request handshake to the response landing at the Spatz
boundary. A local-tile hit is the floor of this distribution; the issue-side
back-pressure (valid waiting for ready) is NOT part of it -- that is the engine
trace's ``memact``.
"""

import collections
import glob
import os
import re

SPMEM_LOG_RE = re.compile(r"spmem_g(\d+)_t(\d+)\.log$")

# Address classes, by destination relative to the requesting core's own tile.
# CLS_UNKNOWN (= len(CLASSES), defined below) is only for logs predating the
# dst_tile field -- never a guess.
CLASSES = ["local", "intra", "inter", "soc"]
CLS_LOCAL, CLS_INTRA, CLS_INTER, CLS_SOC = range(4)

# One paired request. `lat` is None for a request still outstanding at the end
# of the trace (counted separately; never folded into a latency statistic).
Rec = collections.namedtuple("Rec", "cyc lat port wr cls dst addr")

Dims = collections.namedtuple(
    "Dims", "nr_ports cores_per_tile tiles_per_group num_tiles id_bits")


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


def parse_tile_log(path, g, t):
    """Parse one spmem_g<g>_t<t>.log -> (dims, {core: CoreTrace}, end_cyc).

    Returns dims=None if the file carries no D header (not a trace)."""
    dims = None
    end_cyc = 0
    cores = {}
    # (core, port, id, wr) -> deque of (req_cyc, addr, dst)
    pending = collections.defaultdict(collections.deque)
    idmask = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            x = line.split()
            if not x:
                continue
            if x[0] == "D" and len(x) == 6:
                dims = Dims(*(int(v) for v in x[1:]))
                idmask = (1 << dims.id_bits) - 1
                continue
            if x[0] == "E" and len(x) == 2:
                end_cyc = int(x[1])
                continue
            if len(x) < 5 or dims is None:  # D header comes first
                continue
            try:
                c = int(x[0])
            except ValueError:
                continue
            ct = cores.get(c)
            if ct is None:
                ct = cores[c] = CoreTrace()
            if x[1] == "Q" and len(x) == 8:
                cyc, port = int(x[2]), int(x[3])
                rid = int(x[4]) & (idmask if idmask is not None else ~0)
                addr, wr, dst = int(x[5], 16), int(x[6]), int(x[7])
                pending[(c, port, rid, wr)].append((cyc, addr, dst))
                ct.n_req += 1
            elif x[1] == "P" and len(x) == 6:
                cyc, port = int(x[2]), int(x[3])
                rid = int(x[4]) & (idmask if idmask is not None else ~0)
                wr = int(x[5])
                ct.n_rsp += 1
                q = pending.get((c, port, rid, wr))
                if q:
                    qcyc, addr, dst = q.popleft()
                    ct.recs.append(Rec(qcyc, cyc - qcyc, port, wr,
                                       classify(dst, g, t,
                                                dims.tiles_per_group),
                                       dst, addr))
                else:
                    ct.n_orphan += 1
    # whatever never came back
    for (c, port, _rid, wr), q in pending.items():
        ct = cores.setdefault(c, CoreTrace())
        for qcyc, addr, dst in q:
            ct.n_inflight += 1
            ct.recs.append(Rec(qcyc, None, port, wr,
                               classify(dst, g, t, dims.tiles_per_group),
                               dst, addr))
    for ct in cores.values():
        ct.recs.sort(key=lambda r: r.cyc)
    return dims, cores, end_cyc


def iter_spmem_tiles(sp_dir):
    """Yield (g, t, dims, {core: CoreTrace}, end_cyc) per tile log, in
    (g,t) order. Streams one tile at a time -- a whole 256-core run is a few
    million records, so consumers should aggregate as they go."""
    paths = []
    for path in glob.glob(os.path.join(sp_dir, "spmem_g*_t*.log")):
        m = SPMEM_LOG_RE.search(os.path.basename(path))
        if m:
            paths.append((int(m.group(1)), int(m.group(2)), path))
    for g, t, path in sorted(paths):
        dims, cores, end_cyc = parse_tile_log(path, g, t)
        if dims is None:
            continue
        yield g, t, dims, cores, end_cyc


def have_trace(sp_dir):
    """True if `sp_dir` holds at least one per-request memory trace."""
    return bool(glob.glob(os.path.join(sp_dir, "spmem_g*_t*.log")))


# ---------------------------------------------------------------------------
# Tile-side port logs (pe_ = Snitch scalar + Spatz vector ports, rm_ = RedMulE
# master ports). Same idea as the spmem trace, one level further out: the P
# lines already carry both directions, so pairing them gives the SAME
# round-trip metric for every requester class.
#   req  <idx> P 1 0 <cyc> <wen> <addr_hex> <meta_id> <dst_tile>
#   resp <idx> P 0 0 <cyc> <meta_id>
# `dst_tile` is absent in logs from before 2026-08-04 -> class falls back to
# "unknown" rather than being guessed.
#
# Pairing key is (port idx, meta_id). For a reorder-buffered requester (Snitch
# loads, Spatz VLSU, RedMulE X/W/Y streams) meta_id is unique among that port's
# outstanding requests, so FIFO order within the key is exact. The RedMulE Z
# writeback instead reuses ONE meta_id for many concurrently-outstanding stores
# -- but that path is an in-order FIFO, so the k-th response still belongs to
# the k-th request and FIFO pairing stays exact. (The Perfetto exporter refuses
# to draw flow arrows for that case only because a NoC hop cannot be tied to a
# specific request there; a port-local round trip can.)
# ---------------------------------------------------------------------------
PORT_LOG_RE = re.compile(r"(?:pe|rm)_g(\d+)_t(\d+)\.log$")
CLS_UNKNOWN = len(CLASSES)


def parse_port_log(path, g, t, tiles_per_group):
    """Parse one pe_/rm_ log -> {port idx: CoreTrace} (P lines only; the S
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
    """Yield (g, t, {port idx: CoreTrace}) for every pe_ (kind='pe') or rm_
    (kind='rm') log, in (g,t) order. `tiles_per_group` is needed to turn a
    destination tile into an address class."""
    paths = []
    for path in glob.glob(os.path.join(noc_dir, f"{kind}_g*_t*.log")):
        m = PORT_LOG_RE.search(os.path.basename(path))
        if m:
            paths.append((int(m.group(1)), int(m.group(2)), path))
    for g, t, path in sorted(paths):
        yield g, t, parse_port_log(path, g, t, tiles_per_group)


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
