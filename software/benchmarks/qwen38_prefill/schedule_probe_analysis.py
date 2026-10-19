# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Exact engine/DMA occupancy and sampled Spatz work in a scheduling probe."""

from engine_activity import read, spatz_windows


def interval_stats(records, start, end, installed, spans=None):
    """Count simultaneous exact intervals within a guest timer window.

    `spans`, when given, collects the (first, last) cycle ranges with at
    least 80% of the installed units busy."""
    events = {}
    units = set()
    busy = 0
    for unit, first, last in records:
        first, last = max(start, first), min(end, last)
        if first >= last:
            continue
        events[first] = events.get(first, 0) + 1
        events[last] = events.get(last, 0) - 1
        units.add(unit)
        busy += last - first
    active = peak = high = 0
    previous = start
    threshold = (4 * installed + 4) // 5
    for cycle in sorted(events):
        if active >= threshold:
            high += cycle - previous
            if spans is not None and cycle > previous:
                if spans and spans[-1][1] == previous:
                    spans[-1] = (spans[-1][0], cycle)
                else:
                    spans.append((previous, cycle))
        active += events[cycle]
        peak = max(peak, active)
        previous = cycle
    return {"active_units": len(units), "peak_concurrent": peak,
            "busy_engine_cycles": busy,
            "machine_busy_share": busy / ((end - start) * installed),
            "cycles_at_80_percent": high}


def overlap_cycles(first, second, start, end):
    """Exact cycles with at least one interval from each source."""
    events = {}
    for kind, records in enumerate((first, second)):
        for _, begin, finish in records:
            begin, finish = max(start, begin), min(end, finish)
            if begin >= finish:
                continue
            events.setdefault(begin, [0, 0])[kind] += 1
            events.setdefault(finish, [0, 0])[kind] -= 1
    active = [0, 0]
    previous = start
    overlap = 0
    for cycle in sorted(events):
        if active[0] and active[1]:
            overlap += cycle - previous
        active[0] += events[cycle][0]
        active[1] += events[cycle][1]
        previous = cycle
    return overlap


def analyze(path, profile, coverage, installed_spatz, burst_only=False,
            installed_redmule=None, probe_phase="all"):
    """Summarize a completed probe without claiming exact Spatz intervals."""
    mesh_y = 8 if "8x8" in profile else 4
    spatz, redmule, idma, _, _, window = read(path, mesh_y)
    if window is None:
        raise ValueError("Probe needs an absolute guest timer window")
    start, end = window
    burst_start = coverage["burst_start"]
    burst_end = coverage["burst_end"]
    mixer_end = coverage["mixer_end"]
    ffn_end = coverage["ffn_end"]
    if installed_redmule is None:
        installed_redmule = installed_spatz // 2
    rm_intervals = [(where, finish - span, finish)
                    for where, finish, span, _ in redmule]
    dma_intervals = [(where, begin, finish)
                     for where, begin, finish, _, _ in idma]
    has_burst = coverage.get("burst_jobs", installed_redmule) > 0
    phases = [("burst", burst_start, burst_end)] if has_burst else []
    if not burst_only:
        # A phase-only probe leaves the other stage empty.
        if probe_phase != "ffn":
            phases.append(("mixer", burst_end if has_burst else start,
                           mixer_end))
        if probe_phase != "mixer":
            phases.append(("ffn", mixer_end, ffn_end))
    # A GDN ready queue brackets the recurrence it overlapped. Its serial
    # prologue and output projection also fill RedMulE, so the mixer's
    # integrated gate is judged on this window alone.
    overlap = (coverage.get("overlap_start", 0),
               coverage.get("overlap_end", 0))
    if not burst_only and overlap[0] and overlap[1] > overlap[0]:
        phases.append(("mixer_overlap",) + overlap)
    windows = list(spatz_windows(spatz))

    def sampled_spatz(ranges):
        busy = 0.0
        for _, begin, finish, work, _ in windows:
            for first, last in ranges:
                inside = max(0, min(finish, last) - max(begin, first))
                busy += work * inside / (finish - begin)
        return busy

    stages = {}
    for name, first, last in phases:
        vector_busy = sampled_spatz([(first, last)])
        spans = []
        stages[name] = {
            "cycles": last - first,
            "redmule": interval_stats(rm_intervals, first, last,
                                      installed_redmule, spans),
            "idma": interval_stats(dma_intervals, first, last,
                                   mesh_y * mesh_y),
            "redmule_idma_overlap_cycles": overlap_cycles(
                rm_intervals, dma_intervals, first, last),
            "spatz_busy_engine_cycles_sampled": round(vector_busy),
            "spatz_busy_share_sampled": (
                vector_busy / ((last - first) * installed_spatz))}
        high = sum(finish - begin for begin, finish in spans)
        # Spatz samples are periodic, so this is a prorated share, not an
        # exact same-cycle measurement.
        stages[name]["spatz_busy_share_sampled_during_redmule_80"] = (
            sampled_spatz(spans) / (high * installed_spatz) if high else None)
        complete_jobs = [(span, stalls) for _, finish, span, stalls in redmule
                         if first <= finish - span and finish <= last]
        total_ticks = sum(span for span, _ in complete_jobs)
        stages[name]["redmule"]["unstalled_while_busy"] = (
            1 - sum(stalls for _, stalls in complete_jobs) / total_ticks
            if total_ticks else None)
    burst = stages.get("burst")
    burst_gate = bool(
        burst and burst["redmule"]["active_units"] == installed_redmule and
        burst["redmule"]["cycles_at_80_percent"] >= 1000 and
        burst["redmule"]["unstalled_while_busy"] is not None and
        burst["redmule"]["unstalled_while_busy"] >= 0.9 and
        burst["redmule_idma_overlap_cycles"] > 0 and
        coverage["burst_spatz_ops"] == installed_spatz)
    gated = tuple(stage for stage in (
        stages.get("mixer_overlap", stages.get("mixer")), stages.get("ffn"))
        if stage is not None)
    integrated_gate = (None if burst_only else all(
        stage["redmule"]["cycles_at_80_percent"] >= 1000 and
        stage["redmule_idma_overlap_cycles"] > 0 and
        stage["spatz_busy_engine_cycles_sampled"] > 0
        for stage in gated))
    if burst_only:
        design_gate = burst_gate
    elif has_burst:
        design_gate = burst_gate and integrated_gate
    else:
        design_gate = integrated_gate
    return {"stages": stages,
            "setup_cycles": (burst_start if has_burst else mixer_end) - start,
            "installed_redmule": installed_redmule,
            "installed_spatz": installed_spatz,
            "burst_gate_passed": burst_gate if has_burst else None,
            "integrated_gate_passed": integrated_gate,
            "design_gate_passed": design_gate,
            "design_gate": "The capacity burst must use every RedMulE, "
                           "sustain 80% occupancy for 1000 cycles, have at "
                           "least 90% unstalled time while busy, overlap "
                           "iDMA and schedule a vector task on every core. "
                           "The mixer and FFN separately need 80% RedMulE "
                           "occupancy, iDMA overlap and sampled Spatz work; "
                           "with a GDN ready queue the mixer is judged on "
                           "the queued-recurrence overlap window only. "
                           "Periodic Spatz samples do not prove exact "
                           "three-way hardware overlap."}
