#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Engine utilization from a GVSoC run, and dashboard telemetry for it.

Two questions, for each instrumented engine class:

  utilization while busy   non-stalled cycles / busy cycles
  machine busy share       busy cycles / (installed engines * layer cycles)

The summary also retains `busy_share` over engines that emitted a record, so
an unassigned engine cannot silently disappear from the machine denominator.

The model emits the raw material under three environment variables, all
printf-based and inert when unset, because a per-cycle VCD of 128 cores and 64
engines is not a usable instrument:

  RM_STALL_REPORT=1         one line per GEMM per RedMulE, carrying the busy
                            cycles and the cycles held by each operand stream
  SPATZ_ACTIVITY_REPORT=1   cumulative busy/exec per vector unit, emitted every
                            SPATZ_ACTIVITY_PERIOD cycles, so the last line per
                            core gives totals and the series gives a timeline
  IDMA_ACTIVITY_REPORT=1    one interval per group stream while at least one
                            backend slice is outstanding, with bytes counted
                            on backend acknowledgement

All records carry absolute cycle bounds, so they land on a time axis rather
than collapsing to a single bar.

A run stopped early still yields valid numbers for the cycles it covered; it
simply covers a prefix of the layer. The report says which, and never scales a
partial capture up to a whole layer.

New guest logs include a timer start/end pair for aligning engine records.
Spatz samples are cumulative and periodic, so the aggregate still includes
the sample edges around that interval; do not call it an exact clipped counter.

With ``--telemetry`` the same records are written as schema-v1 JSON Lines
(``sim_dashboard/SCHEMA.md``) so the dashboard can render them:

    python engine_activity.py run/model.log --mesh-x 4 --mesh-y 4 \\
        --telemetry engines.jsonl
    python sim_dashboard/generate.py --telemetry engines.jsonl \\
        --mesh 4x4 --window 50000 --out engines.html
"""

import argparse
import json
import re

SPATZ = re.compile(r"^SPATZ_ACT path=(\S+) cyc=(\d+) busy=(\d+) exec=(\d+)$")
REDMULE = re.compile(
    r"^RM_STALL path=(\S+) eng=(\d+) end=(\d+) cyc=(\d+) ticks=(\d+) "
    r"x=(\d+) w=(\d+) y=(\d+) z=(\d+) m=(\d+) n=(\d+) k=(\d+)")
# Older RedMulE builds print no path; every group's tile N then aliases onto
# one engine id, which is still a valid whole-chip total.
REDMULE_FLAT = re.compile(
    r"^RM_STALL eng=(\d+) end=(\d+) cyc=(\d+) ticks=(\d+) "
    r"x=(\d+) w=(\d+) y=(\d+) z=(\d+)")
IDMA = re.compile(
    r"^IDMA_ACTIVITY path=(\S+) stream=(\d+) start=(\d+) end=(\d+) "
    r"bytes=(\d+) slices=(\d+)$")
GROUP = re.compile(r"group_(\d+)_(\d+)")
TILE = re.compile(r"tile_(\d+)")
CORE = re.compile(r"/pe(\d+)")
LAYER = re.compile(r"\[qwen38_layer\][^\n]*cycles=(\d+)")
LAYER_WINDOW = re.compile(
    r"\[qwen38_layer\][^\n]*start=(\d+) end=(\d+) cycles=(\d+)")
PROBE = re.compile(r"\[qwen38_probe\][^\n]*numerical_checked=0")


def is_schedule_probe(path):
    with open(path, errors="replace") as stream:
        return any(PROBE.search(line) for line in stream)


def place(path, mesh_y):
    """(group, tile, core) from a model path; group is x * NumY + y."""
    group = GROUP.search(path)
    tile = TILE.search(path)
    core = CORE.search(path)
    return (int(group[1]) * mesh_y + int(group[2]) if group else 0,
            int(tile[1]) if tile else 0,
            int(core[1]) if core else 0)


def read(path, mesh_y):
    """Engine records, in file order, plus whatever the guest reported."""
    spatz, redmule, idma, layer_cycles, last_cycle = [], [], [], None, 0
    layer_window = None
    with open(path, errors="replace") as stream:
        for line in stream:
            found = SPATZ.match(line)
            if found:
                where = place(found[1], mesh_y)
                cycle, busy, ran = (int(found[i]) for i in (2, 3, 4))
                spatz.append((where, cycle, busy, ran))
                last_cycle = max(last_cycle, cycle)
                continue
            found = REDMULE.match(line)
            if found:
                where = place(found[1], mesh_y)
                end, span = int(found[3]), int(found[4])
                stalls = sum(int(found[i]) for i in (6, 7, 8, 9))
                redmule.append((where, end, span, stalls))
                last_cycle = max(last_cycle, end)
                continue
            found = REDMULE_FLAT.match(line)
            if found:
                end, span = int(found[2]), int(found[3])
                stalls = sum(int(found[i]) for i in (5, 6, 7, 8))
                redmule.append(((0, int(found[1]), 0), end, span, stalls))
                last_cycle = max(last_cycle, end)
                continue
            found = IDMA.match(line)
            if found:
                group = place(found[1], mesh_y)[0]
                stream = int(found[2])
                start, end, count, slices = (int(found[i])
                                             for i in (3, 4, 5, 6))
                if end <= start:
                    raise ValueError("Nonpositive iDMA activity interval")
                if count == 0 or slices == 0:
                    raise ValueError("An acknowledged iDMA slice needs bytes")
                idma.append(((group, stream), start, end, count, slices))
                last_cycle = max(last_cycle, end)
                continue
            if (line.startswith(("SPATZ_ACT", "IDMA_ACTIVITY")) or
                    "IDMA_ACTIVITY path=" in line):
                raise ValueError("Malformed or interleaved activity record")
            found = LAYER.search(line)
            if found:
                layer_cycles = int(found[1])
                interval = LAYER_WINDOW.search(line)
                if interval:
                    start = int(interval[1])
                    end = int(interval[2])
                    cycles = int(interval[3])
                    if (end - start) % (1 << 32) != cycles:
                        raise ValueError("Guest timer interval disagrees "
                                         "with layer cycles")
                    layer_window = (start, end)
    return spatz, redmule, idma, layer_cycles, last_cycle, layer_window


def spatz_windows(records):
    """Cumulative samples to per-window deltas, per vector unit."""
    previous = {}
    windows = []
    for where, cycle, busy, ran in records:
        before = previous.get(where, (0, 0, 0))
        start, busy_before, exec_before = before
        if cycle > start:
            windows.append((where, start, cycle,
                            busy - busy_before, ran - exec_before))
        previous[where] = (cycle, busy, ran)
    return windows


def totals(spatz, redmule, span, installed_spatz=None,
           installed_redmule=None):
    """The two headline numbers per engine class."""
    out = {}
    busy = sum(b for _, _, _, b, _ in
               [(w, s, e, b, x) for w, s, e, b, x in spatz_windows(spatz)])
    ran = sum(x for _, _, _, _, x in spatz_windows(spatz))
    units = len({where for where, _, _, _ in spatz})
    out["spatz"] = {
        "units": units, "busy_cycles": busy, "unstalled_cycles": ran,
        "utilization_while_busy": ran / busy if busy else None,
        "busy_share": busy / (span * units) if span and units else None}
    if installed_spatz is not None:
        out["spatz"]["installed_units"] = installed_spatz
        out["spatz"]["machine_busy_share"] = (
            busy / (span * installed_spatz) if span and installed_spatz
            else None)
    rbusy = sum(s for _, _, s, _ in redmule)
    rstall = sum(t for _, _, _, t in redmule)
    engines = len({where for where, _, _, _ in redmule})
    out["redmule"] = {
        "units": engines, "gemms": len(redmule), "busy_cycles": rbusy,
        "unstalled_cycles": rbusy - rstall,
        "utilization_while_busy": (rbusy - rstall) / rbusy if rbusy else None,
        "busy_share": rbusy / (span * engines) if span and engines else None}
    if installed_redmule is not None:
        out["redmule"]["installed_units"] = installed_redmule
        out["redmule"]["machine_busy_share"] = (
            rbusy / (span * installed_redmule) if span and installed_redmule
            else None)
    return out


def telemetry(spatz, redmule, span, mesh, tiles_per_group, cores_per_tile,
              complete, layer_window=None, idma=(), dma_streams_per_group=1,
              correctness=None):
    """Schema-v1 records: one `stage` per engine per measured interval."""
    rows = [{"kind": "meta", "schema_version": 1, "mesh": list(mesh),
             "backend": "gvsoc-teranoc", "name": "engine activity",
             "tiles_per_group": tiles_per_group,
             "cores_per_tile": cores_per_tile,
             # Views TeraNoC cannot fill, removed rather than shown empty.
             # An MSHR tab on a system with no MSHR is not "no data", it is a
             # claim about the hardware that is not true. `noc` and `memory`
             # need the RTL trace families, which GVSoC does not emit; `hash`
             # models an MSHR bank hash that does not exist here; `roof` needs
             # a peak table that is not part of this repository.
             "unavailable_views": ["memory", "noc", "hash", "roof",
                                   "diagnostics"],
             "teranoc": {
                 "source": "RM_STALL, SPATZ_ACT and optional IDMA_ACTIVITY "
                           "model reports",
                 "cycles_covered": span,
                 "layer_complete": complete,
                 "guest_timer_window": list(layer_window)
                 if layer_window else None,
                 "dma_boundary": "backend slice enqueue to final slice "
                                 "acknowledgement; bytes are original slice "
                                 "payloads published after acknowledgement",
                 "definitions": {
                     "stage hsk": "cycles busy and not stalled",
                     "stage stall": "cycles busy but unable to proceed",
                     "utilization while busy": "hsk / (hsk + stall)",
                     "busy share": "(hsk + stall) / cycles covered"},
                 "unavailable": {
                     "bank/link": "GVSoC emits no RTL trace family; use an "
                                  "RTL run for the fabric views"}}}]
    if correctness is not None:
        rows[0]["correctness"] = correctness
    if idma:
        channels = {}
        for (group, stream), start, end, count, slices in idma:
            name = str(group * dma_streams_per_group + stream)
            channel = channels.setdefault(name, {
                "programmed_bytes": 0, "completed_bytes": 0,
                "transactions": 0, "windows": 0, "active_windows": 0,
                "active_cycles": 0})
            channel["programmed_bytes"] += count
            channel["completed_bytes"] += count
            channel["transactions"] += slices
            channel["windows"] += 1
            channel["active_windows"] += 1
            channel["active_cycles"] += end - start
        for channel in channels.values():
            channel["bytes_per_cycle"] = (
                channel["completed_bytes"] / channel["active_cycles"])
        rows[0]["dma"] = {
            "measurement": "model",
            "note": "iDMA backend slice payloads published after "
                    "acknowledgement; "
                    "active means at least one backend slice outstanding. "
                    "These are not AXI bus beats or NoC/bank handshakes.",
            "channels": channels}
    for (group, tile, core), start, end, busy, ran in spatz_windows(spatz):
        if end <= start or busy > end - start:
            continue
        rows.append({"kind": "stage", "start": start, "end": end,
                     "phase": "unknown", "g": group,
                     "label": "spatz vector",
                     "t": tile * cores_per_tile + core,
                     "hsk": ran, "stall": busy - ran,
                     "idle": end - start - busy})
    # No `fpu` series: each vector unit samples on its own activity, so its
    # window boundaries differ from its neighbours' and a per-group roll-up
    # would emit overlapping intervals -- which the generator rightly flags,
    # and which binning onto a common grid could only fix by inventing a
    # distribution inside each window. The `stage` records above already carry
    # busy, stalled and idle per unit, which is what both metrics need.
    seen = set()
    for (group, tile, _), end, busy, stall in redmule:
        start = end - busy
        key = (group, tile, start, end)
        if start < 0 or busy <= 0 or key in seen:
            continue
        seen.add(key)
        rows.append({"kind": "stage", "start": start, "end": end,
                     "phase": "unknown", "g": group,
                     "label": "redmule array", "t": tile,
                     "hsk": busy - stall, "stall": stall, "idle": 0})
    for (group, stream), start, end, count, slices in idma:
        rows.append({"kind": "dma", "start": start, "end": end,
                     "phase": "bench", "g": group, "scope": "channel",
                     "channel": group * dma_streams_per_group + stream,
                     "measurement": "model", "completed_bytes": count,
                     "transactions": slices, "active_cycles": end - start})
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="model.log from the GVSoC run")
    ap.add_argument("--mesh-x", type=int, default=4)
    ap.add_argument("--mesh-y", type=int, default=4)
    ap.add_argument("--tiles-per-group", type=int, default=4)
    ap.add_argument("--cores-per-tile", type=int, default=2)
    ap.add_argument("--dma-streams-per-group", type=int, default=1)
    ap.add_argument("--telemetry", help="write schema-v1 JSONL here")
    ap.add_argument("--json", help="write the summary here")
    ap.add_argument("--result", help="run_layers result.json for a verified "
                    "dashboard correctness label")
    args = ap.parse_args()
    if args.dma_streams_per_group <= 0:
        ap.error("DMA streams per group must be positive")

    spatz, redmule, idma, layer_cycles, last, layer_window = read(
        args.log, args.mesh_y)
    schedule_probe = is_schedule_probe(args.log)
    if any(stream >= args.dma_streams_per_group
           for (_, stream), _, _, _, _ in idma):
        ap.error("DMA stream index exceeds --dma-streams-per-group")
    if not spatz and not redmule and not idma:
        raise SystemExit(
            "No activity records. Enable RM_STALL_REPORT=1, "
            "SPATZ_ACTIVITY_REPORT=1, or IDMA_ACTIVITY_REPORT=1 before "
            "model construction.")
    span = layer_cycles or last
    complete = layer_cycles is not None and not schedule_probe
    summary = {"cycles_covered": span,
               "layer_complete": complete,
               "schedule_probe_complete": bool(layer_cycles and
                                               schedule_probe),
               "guest_timer_window": list(layer_window)
               if layer_window else None,
               "engines": totals(
                   spatz, redmule, span,
                   args.mesh_x * args.mesh_y * args.tiles_per_group *
                   args.cores_per_tile,
                   args.mesh_x * args.mesh_y * args.tiles_per_group)}
    if idma:
        busy = sum(end - start for _, start, end, _, _ in idma)
        installed = args.mesh_x * args.mesh_y * args.dma_streams_per_group
        summary["dma"] = {
            "channels": len({where for where, _, _, _, _ in idma}),
            "installed_channels": installed,
            "active_cycles": busy,
            "completed_bytes": sum(count for _, _, _, count, _ in idma),
            "completed_slices": sum(slices for _, _, _, _, slices in idma),
            "machine_busy_share": busy / (span * installed)
            if span and installed else None,
            "measurement": "backend slices outstanding"}
    partial = ("  (SCHEDULE PROBE: partial layer, no numerical check)"
               if schedule_probe else "" if layer_cycles else
               "  (PARTIAL: no layer total was reported, so this is a prefix)")
    print("cycles covered: %d%s" % (span, partial))
    for name in ("redmule", "spatz"):
        e = summary["engines"][name]
        if not e["busy_cycles"]:
            print("%-8s no activity recorded" % name)
            continue
        print("%-8s active=%-4d installed=%-4d machine busy share %6.2f%% "
              "active-engine share %6.2f%% utilization while busy %6.2f%%" %
              (name, e["units"], e["installed_units"],
               100 * e["machine_busy_share"], 100 * e["busy_share"],
               100 * e["utilization_while_busy"]))
    if idma:
        dma = summary["dma"]
        print("idma     active=%-4d installed=%-4d machine busy share %6.2f%% "
              "completed %d bytes in %d slices" % (
                  dma["channels"], dma["installed_channels"],
                  100 * dma["machine_busy_share"], dma["completed_bytes"],
                  dma["completed_slices"]))
    if args.json:
        with open(args.json, "w") as stream:
            json.dump(summary, stream, indent=2)
    if args.telemetry:
        correctness = None
        if args.result:
            with open(args.result) as stream:
                result = json.load(stream)
            valid = (result.get("passed") is True and
                     result.get("checked_elements", 0) > 0 and
                     result.get("configuration", {}).get("agree") is True and
                     result.get("configuration", {}).get("checked") is True)
            if result.get("numerical_checked") is False:
                correctness = "not checked: schedule probe"
            else:
                correctness = ("passed" if valid else "failed" if
                               result.get("passed") is False else "unknown")
        rows = telemetry(spatz, redmule, span, (args.mesh_x, args.mesh_y),
                         args.tiles_per_group, args.cores_per_tile,
                         complete, layer_window, idma,
                         args.dma_streams_per_group, correctness)
        with open(args.telemetry, "w") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        print("%s: %d records" % (args.telemetry, len(rows) - 1))


if __name__ == "__main__":
    main()
