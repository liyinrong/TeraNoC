# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Complete resident layers: independent oracle and paired schedule checks."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np

from backing import memory_layout
from attn_layout import max_projection_geometry, projection_layout_check
from fixture_cache import build as build_fixture
from reference import Geometry, TOY
from run_softmax_probe import (RTL, LANE, NATIVE, PROFILES, SPATZ_PER_TILE,
                               BANKS_PER_TILE, REDMULE_ROB, WIDE_RO, execute)
from schedule_probe_analysis import analyze as analyze_probe
from snapshot import freeze_build_inputs, record_model_install
from ctrlcheck import check as check_ctrl
from stackdepth import check as check_stack


# --probe-phase -> Q38_PROBE_PHASE.
PROBE_PHASES = {"all": 0, "mixer": 1, "ffn": 2}


def redmules(profile):
    """One RedMulE per tile, whatever the Spatz count per tile."""
    return PROFILES[profile][1] // SPATZ_PER_TILE.get(profile, 2)


STACK = re.compile(r"-DSTACK_SIZE=(\d+)")


def attn_vc_l2_check(directory, install, l2_bytes):
    """The phased attention keeps its L2 buffers at the top of L2, outside
    the ELF images (q38_attn_vc.c): they must clear the data sidecar."""
    defines = dict(re.findall(r"#define (Q38_\w+) (\d+)u",
                              (directory / "layer_data.h").read_text()))
    qh, rows, hd, qdim = (int(defines[key]) for key in
                          ("Q38_QH", "Q38_R", "Q38_HD", "Q38_QDIM"))
    reserved = (2 * qh * rows * hd + rows * qdim) * 2
    readelf = install / "riscv-gcc/bin/riscv32-unknown-elf-readelf"
    text = subprocess.check_output(
        [str(readelf), "-lW", str(directory / "layer_data.elf")], text=True)
    end = max((int(f[3], 16) + int(f[5], 16) for f in
               (line.split() for line in text.splitlines())
               if f and f[0] == "LOAD"), default=0)
    if end > 0x80000000 + l2_bytes - reserved:
        raise RuntimeError(f"phased-attention L2 buffers overlap the data "
                           f"sidecar (ends {end:#x}, buffers from "
                           f"{0x80000000 + l2_bytes - reserved:#x})")


def ffn_l2_check(directory, install, l2_bytes, rows):
    """Bounded G scratch must clear the captured data-only ELF."""
    defines = dict(re.findall(r"#define (Q38_\w+) (\d+)u",
                              (directory / "layer_data.h").read_text()))
    reserved = int(defines["Q38_R"]) * int(defines["Q38_I"]) * 2
    readelf = install / "riscv-gcc/bin/riscv32-unknown-elf-readelf"
    text = subprocess.check_output(
        [str(readelf), "-lW", str(directory / "layer_data.elf")], text=True)
    end = max((int(f[3], 16) + int(f[5], 16) for f in
               (line.split() for line in text.splitlines())
               if f and f[0] == "LOAD"), default=0)
    if end > 0x80000000 + l2_bytes - reserved:
        raise RuntimeError("FFN G scratch overlaps the data sidecar")


def l1_footprint(binary, install):
    """Bytes of interleaved L1 the guest reserves, excluding the per-core
    sequential region. With bounded windows this must not grow with S."""
    size = install / "riscv-gcc/bin/riscv32-unknown-elf-size"
    text = subprocess.check_output([str(size), "-A", str(binary)], text=True)
    sections = {}
    for line in text.splitlines():
        field = line.split()
        if len(field) >= 2 and field[0].startswith("."):
            try:
                sections[field[0]] = int(field[1])
            except ValueError:
                continue
    return sections.get(".l1", 0)


def stack_gate(build_log, binary, install, reserved=0):
    """The per-core stack is a fixed slice of the sequential L1 region and
    overflow is silent, so a binary is rejected before it is ever simulated."""
    limit = STACK.search(build_log)
    if not limit:
        raise ValueError("Build log does not record STACK_SIZE")
    objdump = install / "riscv-gcc/bin/riscv32-unknown-elf-objdump"
    result = check_stack(str(objdump), binary, int(limit[1]) - reserved)
    result["path"] = [{"function": name, "bytes": size}
                      for name, size in result["path"]]
    return result


def ctrl_gate(binary, install, entry="avs_sched_stream"):
    """A scheduler core is rv32ima: an F/V/Xpulp opcode reachable from its
    entry parks it in the boot ROM (a hang), so such a binary is rejected.
    Binaries without the entry (no scheduler core) pass unchecked."""
    objdump = install / "riscv-gcc/bin/riscv32-unknown-elf-objdump"
    symbols = subprocess.check_output([str(objdump), "-t", str(binary)],
                                      text=True)
    if not re.search(r"\s" + entry + r"$", symbols, re.M):
        return {"checked": False, "clean": True}
    result = check_ctrl(str(objdump), binary, (entry,))
    result["checked"] = True
    result["violations"] = [
        {"function": n, "address": a, "insn": f"{m} {o}".strip(), "why": w}
        for n, a, m, o, w in result["violations"]]
    return result


def phase_checks(fixture, phase):
    """Select numerical descriptors in the guest's phase-specific order."""
    if phase == "all":
        return fixture
    if fixture["kind"] != "attention" or fixture["check_mode"] != "full":
        raise ValueError("numerical phase selection needs full-tensor "
                         "attention fixtures")
    names = ("output",) if phase == "ffn" else ("context", "cache", "mixer")
    checks = {name: fixture["checks"][name] for name in names}
    return dict(fixture, checks=checks, checked_elements=sum(
        math.prod(check["shape"]) for check in checks.values()))


def attn_sweep_probe_counts(base, rows, window, batches, block, tokens):
    """Count head batches and the first-two/last causal key-block probe."""
    windows = len(range(base, base + rows, window)) * batches
    blocks = batches * sum(
        min(3, math.ceil(min(first + window, tokens) / block))
        for first in range(base, base + rows, window))
    return windows, blocks


def evaluate(log, code, expected, configuration=None, expected_checks=4):
    checks = re.findall(
        r"\[DPI_CHECK\] Check (\d+): (\d+) ERRORS out of (\d+) CHECKS", log)
    total = sum(int(count) for _, _, count in checks)
    marker = re.search(r"\[qwen38_layer\].*cycles=(\d+)", log)
    bounds = re.search(
        r"\[qwen38_bounds\] mixer_rows=(\d+) ffn_rows=(\d+) "
        r"weight_panels=(\d+) context_export_cycles=(\d+)", log)
    passed = bool((configuration is None or configuration["agree"])
                  and code == 0 and marker and bounds
                  and len(checks) == expected_checks
                  and {int(index) for index, _, _ in checks} ==
                  set(range(expected_checks))
                  and not any(int(errors) for _, errors, _ in checks)
                  and total == expected
                  and re.search(r"EOC register return value: 0x0\b", log)
                  and not re.search(
                      r"Fatal|Traceback|WATCHDOG|qwen38_dma_error", log))
    result = {"passed": passed, "checked_elements": total,
              "cycles": int(marker[1]) if marker else None}
    if bounds:
        # The context export keeps a diagnostic tensor out of the window
        # buffer; it is inside the measured interval and reported separately.
        result["resolved_mixer_rows"] = int(bounds[1])
        result["context_export_cycles"] = int(bounds[4])
    return result


def gdn_queue_shape(geometry, supertile_rows, panels, columns,
                    projection_rows):
    """Groups, jobs and L2 spill bytes of one GDN ready-queue supertile."""
    g = geometry or TOY
    qkv = 2 * g.key_heads * g.key_dim + g.value_heads * g.value_dim
    gated = g.value_heads * g.value_dim
    control = -(-2 * g.value_heads // columns) * columns
    counts = [qkv // columns, gated // columns, control // columns]
    groups = sum(-(-count // panels) for count in counts)
    jobs = supertile_rows // projection_rows * sum(counts)
    spill = 2 * groups * supertile_rows * panels * columns * 2
    return {"groups": groups, "jobs": jobs, "spill_bytes": spill}


def evaluate_schedule_probe(log, code, configuration, expected, queue=None):
    """Require every selected stage and all three activity sources."""
    marker = re.search(
        r"\[qwen38_layer\][^\n]*start=(\d+) end=(\d+) cycles=(\d+)",
        log)
    probe = re.search(r"\[qwen38_probe\][^\n]*", log)
    observed = ({key: int(value) for key, value in
                 re.findall(r"(\w+)=(\d+)", probe[0])} if probe else {})
    counts_match = all(observed.get(key) == value for key, value in
                       expected.items())
    counts_match = (counts_match and
                    (expected["burst_only"] or
                     observed.get("projection_groups", 0) > 0) and
                    observed.get("gateup_groups", 0) >=
                    3 * expected["ffn_blocks"])
    phases_ordered = bool(marker and probe and
                          int(marker[1]) <= observed.get("burst_start", -1) <
                          observed.get("burst_end", -1) <=
                          observed.get("mixer_end", -1) <=
                          observed.get("ffn_end", -1) <= int(marker[2]))
    if expected["burst_jobs"] == 0:
        phases_ordered = bool(
            marker and probe and observed.get("burst_start") == 0 and
            observed.get("burst_end") == 0 and
            int(marker[1]) <= observed.get("mixer_end", -1) <=
            observed.get("ffn_end", -1) <= int(marker[2]))
    queue_line = re.search(r"\[qwen38_queue\][^\n]*", log)
    queue_observed = ({key: int(value) for key, value in
                       re.findall(r"(\w+)=(\d+)", queue_line[0])}
                      if queue_line else {})
    if queue is not None:
        # Every queued rectangle retired, and the queue overlapped a
        # recurrence window strictly inside the mixer phase.
        counts_match = counts_match and all(
            queue_observed.get(key) == value for key, value in queue.items())
        phases_ordered = phases_ordered and bool(
            observed.get("burst_end", -1) <=
            queue_observed.get("overlap_start", -1) <
            queue_observed.get("overlap_end", -1) <=
            observed.get("mixer_end", -1))
        observed["overlap_start"] = queue_observed.get("overlap_start", 0)
        observed["overlap_end"] = queue_observed.get("overlap_end", 0)
    activity = {"redmule": "RM_STALL path=" in log,
                "spatz": "SPATZ_ACT path=" in log,
                "idma": "IDMA_ACTIVITY path=" in log}
    passed = bool(configuration["agree"] and code == 0 and marker and probe
                  and counts_match and phases_ordered
                  and all(activity.values())
                  and observed.get("numerical_checked") == 0
                  and re.search(r"EOC register return value: 0x0\b", log)
                  and not re.search(
                      r"Fatal|Traceback|WATCHDOG|qwen38_dma_error", log))
    return {"passed": passed, "schedule_passed": passed,
            "numerical_checked": False, "checked_elements": 0,
            "cycles": int(marker[3]) if marker else None,
            "probe_coverage": observed, "activity_sources": activity,
            "queue_coverage": queue_observed or None}


def extract_actual(log, fixture, directory):
    """DPI verbose output is host-side and outside the measured interval."""
    names = list(fixture["checks"])
    pending = {}
    hashes = {}
    for line in log.splitlines():
        element = re.fullmatch(
            r"CHECK\((\d+)\): EXP = [0-9A-Fa-f]+ - RESP = ([0-9A-Fa-f]+)",
            line)
        if element:
            index = int(element[1])
            if index in pending:
                raise ValueError("Duplicate DPI element index")
            pending[index] = int(element[2], 16)
        end = re.fullmatch(
            r"\[DPI_CHECK\] Check (\d+): \d+ ERRORS out of (\d+) CHECKS", line)
        if end:
            number, count = int(end[1]), int(end[2])
            if number >= len(names) or set(pending) != set(range(count)):
                raise ValueError("Incomplete DPI tensor dump")
            name = names[number]
            single = fixture["checks"][name]["dtype"] == "float32"
            dtype = "<u4" if single else "<u2"
            data = np.asarray([pending[i]
                              for i in range(count)], dtype=dtype).tobytes()
            (directory / ("actual_" + name + ".bin")).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
            pending.clear()
    if pending or set(hashes) != set(names):
        raise ValueError("Missing DPI tensor dump")
    return hashes


def check_contract(directory, required=False, timing_only=None):
    """Does the hardware the model built match the one the guest was told?

    `qwen38_model` publishes its resolved lane geometry as `qwen38_contract`;
    the guest's is in its compile line. Comparing them is the plan's
    "assert the resolved model and ELF configurations match", and it is not
    optional: three S=64 sweep points once returned byte-identical cycle
    counts because the model had quietly kept the base vector unit.
    """
    found = {}
    config = directory / "gvsoc_config.json"
    if config.exists():
        def walk(node):
            if isinstance(node, dict):
                if "qwen38_contract" in node:
                    yield node["qwen38_contract"]
                for value in node.values():
                    yield from walk(value)
            elif isinstance(node, list):
                for value in node:
                    yield from walk(value)
        for contract in walk(json.loads(config.read_text())):
            found = contract
            break
    build = directory / "build.log"
    guest = {}
    if build.exists():
        for name, key in (("N_FPU", "n_fpu"), ("VLEN", "vlen"),
                          ("NUM_CORES", "cores")):
            match = re.search(rf"-D{name}=(\d+)", build.read_text())
            if match:
                guest[key] = int(match[1])
        if "spatz_block_alloc" in found:
            for name, key in (("Q38_SPATZ_BLOCK_ALLOC", "spatz_block_alloc"),
                              ("Q38_SPATZ_MAX_BURST_WORDS",
                               "spatz_max_burst_words"),
                              ("Q38_SPATZ_ROB_DEPTH", "spatz_rob_depth")):
                match = re.search(rf"-D{name}=(\d+)", build.read_text())
                if match:
                    guest[key] = int(match[1])
        # Bank geometry: only contracts that report it (older installs don't).
        banks = re.search(r"-DNUM_BANKS=(\d+)", build.read_text())
        size = re.search(r"-DL1_BANK_SIZE=(\d+)", build.read_text())
        if found and banks and size and "l1_bank_bytes" in found:
            guest["l1_bank_bytes"] = int(size[1])
            guest["l1_bytes"] = int(banks[1]) * int(size[1])
    if not found or not guest:
        return {"agree": not required, "checked": False,
                "reason": "no qwen38_contract or no compile line to compare"}
    differ = {k: {"model": found.get(k), "guest": v}
              for k, v in guest.items() if found.get(k) != v}
    if (timing_only is not None and
            found.get("redmule_timing_only") != timing_only):
        differ["redmule_timing_only"] = {
            "model": found.get("redmule_timing_only"),
            "expected": timing_only}
    return {"agree": not differ, "checked": True, "model": found,
            "guest": guest, "differ": differ}


def attention_model_target(profile, l2_mib, timing_only):
    """Stable target names for qualified full-burst attention model builds."""
    fpus, vlen = LANE.get(profile, (4, 512))
    target = ("qwen38_model:system_config=" + PROFILES[profile][0] +
              ":virtual_cluster=true" + f":qwen_l2_mib={l2_mib}" +
              f":redmule_timing_only={str(timing_only).lower()}" +
              ":qwen_spatz_burst=1")
    # S1 four-core filenames otherwise exceed CMake's filesystem limit.
    if profile != "dse_s1_winner_b32_c4_f4_v512":
        target += f":qwen_fpus={fpus}:qwen_vlen={vlen}"
    target += (f":spatz_max_burst_words={vlen // 32}" +
               f":spatz_rob_depth={32 if vlen == 1024 else 16}")
    if profile in BANKS_PER_TILE:
        target += f":qwen_banks_per_tile={BANKS_PER_TILE[profile]}"
    if profile in SPATZ_PER_TILE:
        target += f":qwen_spatz_per_tile={SPATZ_PER_TILE[profile]}"
    return target


def run_case(root, build_source, model_install, profile, kind, schedule,
             tokens, timeout, dump, delay_seed, placement, kv_level,
             attention_kv_heads, weight_panels, external_data, l2_mib,
             ffn_rows, mixer_rows, check_samples, check_run_length,
             geometry=None, fixture_cache=None, adaptive_panels=False,
             projection_rows=None, gateup_pipeline=False,
             projection_columns=64, coordinator_dma=False,
             kv_prefetch=False, interleave_kv_query=False,
             gdn_chunk_rows=64, separate_stages=False,
             ffn_norm_stream=False, background_ffn_norm=False,
             gdn_double_chunk=False, gdn_recur_projection_overlap=False,
             gdn_triple_pipeline=False, probe_region=None, probe_rows=0,
             probe_recurrent_k=32, build_only=False,
             probe_burst_only=False, gdn_ready_queue=False,
             gdn_project_rows=256, gdn_queue_panels=4, ffn_wide=False,
             ffn_down_panels=4, gdn_queue_depth=1, gdn_spill_serial=False,
             gdn_group_sync=True, gdn_retire_copy=1, gdn_drain_sleep=False,
             gdn_weight_prefetch=True, gdn_z_fixed=False,
             probe_burst_rounds=1, probe_burst_slot=0, ffn_down_slices=1,
             attn_spread=False, attn_reuse_norm=False, redmule_matmul=False,
             ffn_dataflow=False, probe_lead_groups=2, attn_st_rows=0,
             proj_dataflow=False, attn_local=False, attn_rm_accum=False,
             attn_fused_softmax=False, attn_local_pipe=False,
             attn_rm_fold=False, attn_tile_local=False, attn_isolate=0,
             attn_drain=False, spatz_burst=False, key_block=64,
             gu_slots=2, gu_panels=0, dp_slots=2, ffn_gu_slices=1,
             ffn_pipe=False, probe_phase="all", ffn_vc=0, ffn_vc_poll=False,
             ffn_vc_dma_stress=False, attn_vc=False, attn_vc_rows=256,
             attn_vc_orows=0, attn_vc_sweep=False, attn_sweep_rows=0,
             attn_sweep_tile=0, ffn_l2=False, ffn_gu_rows=0, ffn_down_rows=0,
             attn_vc_panel=False, attn_vc_dual=False, attn_vc_max_rows=False,
             attn_sweep_mode="group", attn_sweep_reuse_query=False,
             attn_sweep_register=False, attn_sweep_interleave=False,
             attn_sweep_tree_max=False, attn_sweep_row_dma=False,
             attn_sweep_lean=False, attn_sweep_qscale=False,
             attn_sweep_exp2=False, attn_sweep_fast=False,
             attn_sweep_tile_win=False, attn_sweep_async=False,
             attn_sweep_late_drain=0, attn_sweep_finish=False,
             attn_sweep_overlap=False,
             attn_vc_direct_dma=False, attn_vc_cores=None,
             attn_vc_row_unroll=None,
             attn_vc_tile_stage=False, attn_vc_cache_gamma=None):
    if attn_vc_cores is None:
        attn_vc_cores = SPATZ_PER_TILE.get(profile, 2) if attn_vc_dual else 0
    if attn_vc_row_unroll is None:
        attn_vc_row_unroll = ((4 if LANE.get(profile, (4, 512))[1] == 1024
                               else 2) if attn_vc_cores else 1)
    if attn_vc_cache_gamma is None:
        attn_vc_cache_gamma = bool(
            attn_vc_cores and LANE.get(profile, (4, 512))[1] == 1024)
    if attn_vc_tile_stage:
        raise ValueError("Tile-private indexed shipping is unqualified")
    if (attn_vc_direct_dma or attn_vc_cores or
            attn_vc_row_unroll != 1) and not (
            attn_vc_panel and attn_vc_dual and probe_phase == "mixer"):
        raise ValueError("Projection alternatives need mixer-only dual panel "
                         "attention")
    if attn_vc_cores not in (0, 2, 4) or attn_vc_cores > SPATZ_PER_TILE.get(
            profile, 2):
        raise ValueError("Static projection core count exceeds tile cores")
    if attn_vc_row_unroll not in (1, 2, 4):
        raise ValueError("Projection row unroll must be one, two or four")
    if attn_vc_cache_gamma and not (
            attn_vc_cores and LANE.get(profile, (4, 512))[1] == 1024):
        raise ValueError("Gamma retention needs static VLEN1024 row blocks")
    if attn_sweep_mode not in ("group", "tile", "early"):
        raise ValueError("Unknown attention sweep scheduling mode")
    if (attn_sweep_mode != "group" or attn_sweep_reuse_query or
            attn_sweep_register or attn_sweep_interleave or
            attn_sweep_tree_max or attn_sweep_row_dma or
            attn_sweep_lean or attn_sweep_fast or
            attn_sweep_tile_win) and not (
                attn_vc_sweep and probe_phase == "mixer"):
        raise ValueError("Sweep alternatives require mixer-only VC attention")
    if attn_sweep_exp2 and not attn_sweep_lean:
        raise ValueError("exp2 needs the lean softmax row")
    if attn_sweep_qscale and not (attn_sweep_lean or attn_sweep_fast):
        raise ValueError("Folded scale needs the lean or FP16 softmax row")
    if attn_sweep_fast and (
            not attn_sweep_qscale or attn_sweep_lean or attn_sweep_register or
            key_block * 2 > LANE.get(profile, (4, 512))[1] or
            key_block % (LANE.get(profile, (4, 512))[1] // 8)):
        raise ValueError("FP16 softmax needs QSCALE and B <= VLEN/2")
    if attn_sweep_tile_win and attn_sweep_mode == "group":
        raise ValueError("Tile-window intermediates need tile-owned rows")
    if attn_sweep_async and attn_sweep_mode != "early":
        raise ValueError("Dependency-only pipeline needs early mode")
    if attn_sweep_late_drain not in (0, 1, 2) or (
            attn_sweep_late_drain and not attn_sweep_async):
        raise ValueError("Late drains (1/2) need the dependency-only pipeline")
    if attn_sweep_finish and not attn_sweep_fast:
        raise ValueError("Fused finish needs the FP16 softmax row")
    if attn_sweep_overlap and not (attn_sweep_async and attn_sweep_finish and
                                   attn_sweep_tile_win):
        raise ValueError("Overlap needs async, fused finish and tile window")
    if attn_sweep_lean and (attn_sweep_register or key_block <= 128):
        raise ValueError("Lean softmax replaces the legacy B>128 row only")
    if attn_sweep_qscale and not attn_vc_panel:
        raise ValueError("Folded QK scale needs the panel-aware Q epilogue")
    if attn_sweep_tree_max and key_block <= 128:
        raise ValueError("Tree-max softmax is qualified only above B128")
    if attn_sweep_register and LANE.get(profile, (4, 512))[1] != 1024:
        raise ValueError("Register softmax is currently qualified at VLEN1024")
    if attn_vc_max_rows and not attn_vc_panel:
        raise ValueError("Maximum projection rows need panel-aware attention")
    if attn_vc_panel and not (attn_vc and attn_vc_sweep and
                              probe_phase == "mixer"):
        raise ValueError("Panel-aware projection needs mixer-only VC sweep")
    if attn_vc_dual and not attn_vc_panel:
        raise ValueError("Two-core projection needs panel-aware epilogues")
    if attn_vc_max_rows:
        selected = geometry or TOY
        projection_rows, attn_vc_rows = max_projection_geometry(
            profile, {"d": selected.d}, tokens,
            probe_rows if probe_region else 0)
        attn_vc_orows = attn_vc_rows
        projection_columns = (32 if PROFILES[profile][0] == "dse_s1_winner"
                              else 64)
    if attn_vc_sweep and (key_block % 128 or
                          attn_sweep_rows < 0 or attn_sweep_tile < 0):
        raise ValueError("KV-affine geometry needs nonnegative row overrides "
                         "and B=128/256/512")
    if key_block > 128 and not attn_vc_sweep:
        raise ValueError("Key blocks above 128 need --attn-vc-sweep")
    if attn_vc_sweep and not (attn_vc and attn_fused_softmax and
                              probe_phase == "mixer"):
        raise ValueError("The KV-affine sweep needs mixer-only phased, "
                         "fused attention")
    bound = "resident" if mixer_rows == 0 else f"w{mixer_rows}"
    suffix = f"_probe_{probe_region}" if probe_region else ""
    name = f"{profile}_{kind}_{placement}_{bound}_{schedule}{suffix}"
    directory = root / name
    directory.mkdir()
    probe_base = tokens - probe_rows if probe_region == "tail" else 0
    mixer_windows = 1 if kind == "gdn" else 2
    if attn_st_rows and kind == "attention":
        mixer_windows = max(2, attn_st_rows // mixer_rows)
    if attn_vc and kind == "attention":
        # Whole blocks of both projections, and at least two of the smaller.
        rows = max(attn_vc_rows, 2 * min(attn_vc_rows,
                                         attn_vc_orows or attn_vc_rows))
        mixer_windows = math.lcm(rows, attn_vc_orows or attn_vc_rows) // \
            mixer_rows
    if gdn_ready_queue:
        # Two supertiles: the second's projections overlap the first's
        # recurrence, and the second then runs without a queued successor.
        mixer_windows = 2 * gdn_project_rows // mixer_rows
    queue_shape = (gdn_queue_shape(geometry, gdn_project_rows,
                                   gdn_queue_panels, projection_columns,
                                   projection_rows or 64)
                   if gdn_ready_queue else None)
    mixer_probe_rows = mixer_windows * mixer_rows if probe_region else 0
    mixer_base = (probe_base + probe_rows - mixer_probe_rows
                  if probe_region == "tail" else probe_base)
    parameters = {"kind": kind, "tokens": tokens,
                  "attention_kv_heads": attention_kv_heads,
                  "external_data": external_data, "l2_mib": l2_mib,
                  "check_samples": check_samples,
                  "check_run_length": check_run_length, "geometry": geometry}
    if projection_columns != 64:
        parameters["panel_columns"] = projection_columns
    if key_block != 64:
        parameters["key_block"] = key_block
    if gdn_ready_queue:
        # The spill ring is guest BSS; the sidecar must start above it.
        parameters["l2_scratch_bytes"] = queue_shape["spill_bytes"]
    fixture = build_fixture(directory, parameters, fixture_cache)
    if not probe_region:
        fixture = phase_checks(fixture, probe_phase)
    sweep_defines = (f"-DQ38_AV_SWEEP_ROWS={attn_sweep_rows} "
                     if attn_sweep_rows else "")
    sweep_defines += (f"-DQ38_AV_SWEEP_TILE={attn_sweep_tile} "
                      if attn_sweep_tile else "")
    defines = (sweep_defines + f"-DQ38_ASYNC={int(schedule == 'async')} "
               f"-DQ38_VERBOSE_CHECKS={int(dump)} "
               f"-DQ38_DELAY_SEED={delay_seed} "
               f"-DQ38_KV_REPLICATED={int(placement == 'replicated')} "
               f"-DQ38_KV_LEVEL={kv_level} "
               f"-DQ38_WEIGHT_PANELS={weight_panels} "
               f"-DQ38_ADAPTIVE_PANELS={int(adaptive_panels)} "
               f"-DQ38_GATEUP_PIPELINE={int(gateup_pipeline)} "
               f"-DQ38_COORDINATOR_DMA={int(coordinator_dma)} "
               f"-DQ38_KV_PREFETCH={int(kv_prefetch)} "
               f"-DQ38_INTERLEAVE_KV_QUERY={int(interleave_kv_query)} "
               f"-DQ38_GDN_CHUNK_ROWS={gdn_chunk_rows} "
               f"-DQ38_STAGE_SEPARATE={int(separate_stages)} "
               f"-DQ38_FFN_NORM_STREAM={int(ffn_norm_stream)} "
               f"-DQ38_BACKGROUND_FFN_NORM={int(background_ffn_norm)} "
               f"-DQ38_GDN_DOUBLE_CHUNK={int(gdn_double_chunk)} "
               f"-DQ38_GDN_RECUR_PROJECTION_OVERLAP="
               f"{int(gdn_recur_projection_overlap)} "
               f"-DQ38_GDN_TRIPLE_PIPELINE={int(gdn_triple_pipeline)} "
               f"-DQ38_GDN_READY_QUEUE={int(gdn_ready_queue)} "
               f"-DQ38_GDN_PROJECT_ROWS={gdn_project_rows} "
               f"-DQ38_GDN_QUEUE_PANELS={gdn_queue_panels} "
               f"-DQ38_GDN_QUEUE_DEPTH={gdn_queue_depth} "
               f"-DQ38_GDN_SPILL_SERIAL={int(gdn_spill_serial)} "
               f"-DQ38_GDN_GROUP_SYNC={int(gdn_group_sync)} "
               f"-DQ38_GDN_RETIRE_COPY={gdn_retire_copy} "
               f"-DQ38_GDN_DRAIN_SLEEP={int(gdn_drain_sleep)} "
               f"-DQ38_GDN_WEIGHT_PREFETCH={int(gdn_weight_prefetch)} "
               f"-DQ38_GDN_Z_FIXED={int(gdn_z_fixed)} "
               f"-DQ38_PROBE_BURST_ROUNDS={probe_burst_rounds} "
               f"-DQ38_PROBE_BURST_SLOT={probe_burst_slot} "
               f"-DQ38_FFN_WIDE={int(ffn_wide)} "
               f"-DQ38_FFN_DOWN_PANELS={ffn_down_panels} "
               f"-DQ38_FFN_DOWN_SLICES={ffn_down_slices} "
               f"-DQ38_GU_SLOTS={gu_slots} "
               f"-DQ38_GU_PANELS={gu_panels or weight_panels} "
               f"-DQ38_DP_SLOTS={dp_slots} "
               f"-DQ38_FFN_GU_SLICES={ffn_gu_slices} "
               f"-DQ38_FFN_PIPE={int(bool(ffn_pipe))} "
               f"-DQ38_FP_SLOTS={max(2, int(ffn_pipe))} "
               f"-DQ38_FFN_VC={int(ffn_vc)} "
               f"-DQ38_FFN_L2={int(ffn_l2)} "
               + (f"-DQ38_FFN_GU_ROWS={ffn_gu_rows} " if ffn_gu_rows else "")
               + (f"-DQ38_FFN_DN_ROWS={ffn_down_rows} "
                  if ffn_down_rows else "")
               +
               f"-DQ38_VC_POLL={int(ffn_vc_poll)} "
               f"-DQ38_VC_DMA_STRESS={int(ffn_vc_dma_stress)} "
               f"-DQ38_ATTN_VC={int(attn_vc)} "
               f"-DQ38_ATTN_VC_ROWS={attn_vc_rows} "
               f"-DQ38_ATTN_VC_OROWS={attn_vc_orows or attn_vc_rows} "
               f"-DQ38_ATTN_VC_SWEEP={int(attn_vc_sweep)} "
               f"-DQ38_AV_SWEEP_MODE="
               f"{('group', 'tile', 'early').index(attn_sweep_mode)} "
               f"-DQ38_AV_SWEEP_REUSE_QUERY={int(attn_sweep_reuse_query)} "
               f"-DQ38_AV_SWEEP_REGISTER={int(attn_sweep_register)} "
               f"-DQ38_AV_SWEEP_INTERLEAVE={int(attn_sweep_interleave)} "
               f"-DQ38_AV_SWEEP_TREE_MAX={int(attn_sweep_tree_max)} "
               f"-DQ38_AV_SWEEP_ROW_DMA={int(attn_sweep_row_dma)} "
               f"-DQ38_AV_SWEEP_LEAN={int(attn_sweep_lean)} "
               f"-DQ38_AV_SWEEP_QSCALE={int(attn_sweep_qscale)} "
               f"-DQ38_AV_SWEEP_EXP2={int(attn_sweep_exp2)} "
               f"-DQ38_AV_SWEEP_FAST={int(attn_sweep_fast)} "
               f"-DQ38_AV_SWEEP_TILE_WIN={int(attn_sweep_tile_win)} "
               f"-DQ38_AV_SWEEP_ASYNC={int(attn_sweep_async)} "
               f"-DQ38_AV_SWEEP_LATE_DRAIN={attn_sweep_late_drain} "
               f"-DQ38_AV_SWEEP_FINISH={int(attn_sweep_finish)} "
               f"-DQ38_AV_SWEEP_OVERLAP={int(attn_sweep_overlap)} "
               f"-DQ38_AV_PANEL_AWARE={int(attn_vc_panel)} "
               f"-DQ38_AV_DUAL={int(attn_vc_dual)} "
               f"-DQ38_AV_DIRECT_DMA={int(attn_vc_direct_dma)} "
               f"-DQ38_AV_STATIC_CORES={attn_vc_cores} "
               f"-DQ38_AV_ROW_UNROLL={attn_vc_row_unroll} "
               f"-DQ38_AV_CACHE_GAMMA={int(attn_vc_cache_gamma)} "
               # The pipeline's deep queue must stay below the iDMA FIFO.
               f"-DQ38_DMA_QUEUE_MAX_INFLIGHT="
               f"{5 if ffn_pipe or ffn_vc else 64} "
               f"-DQ38_FP_CHECK={int(bool(ffn_pipe))} "
               # Two epochs per gate/up slot plus the first down slice's
               # pieces; one per projection slot.
               f"-DQ38_DMA_QUEUE_SLOTS="
               f"{max(8, 2 * gu_slots + ffn_down_panels, dp_slots,
                      64 if ffn_pipe or ffn_vc else 0)} "
               f"-DQ38_ATTN_SPREAD={int(attn_spread)} "
               f"-DQ38_REDMULE_MATMUL={int(redmule_matmul)} "
               f"-DQ38_FFN_DATAFLOW={int(ffn_dataflow)} "
               f"-DQ38_PROJ_DATAFLOW={int(proj_dataflow)} "
               f"-DQ38_ATTN_LOCAL={int(attn_local)} "
               f"-DQ38_ATTN_RM_ACCUM={int(attn_rm_accum)} "
               f"-DQ38_ATTN_FUSED_SOFTMAX={int(attn_fused_softmax)} "
               f"-DQ38_ATTN_LOCAL_PIPE={int(attn_local_pipe)} "
               f"-DQ38_ATTN_RM_FOLD={int(attn_rm_fold)} "
               f"-DQ38_ATTN_TILE_LOCAL={int(attn_tile_local)} "
               f"-DQ38_ATTN_ISOLATE={attn_isolate} "
               f"-DQ38_ATTN_DRAIN={int(attn_drain)} "
               f"-DQ38_BK={key_block} "
               f"-DQ38_PROBE_LEAD_GROUPS={probe_lead_groups} "
               f"-DQ38_ATTN_SUPERTILE={int(bool(attn_st_rows))} "
               f"-DQ38_ATTN_ST_ROWS={attn_st_rows or 256} "
               f"-DQ38_ATTN_REUSE_NORM={int(attn_reuse_norm)} "
               f"-DQ38_PROJECTION_ROWS={projection_rows or 64} "
               f"-DQ38_PROJECTION_COLUMNS={projection_columns} "
               f"-DQ38_FFN_ROWS={ffn_rows} "
               f"-DQ38_MIXER_ROWS={mixer_rows} "
               f"-DQ38_SCHEDULE_PROBE={int(bool(probe_region))} "
               f"-DQ38_PROBE_BASE={probe_base} "
               f"-DQ38_PROBE_ROWS={probe_rows} "
               f"-DQ38_PROBE_TAIL={int(probe_region == 'tail')} "
               f"-DQ38_PROBE_BURST_ONLY={int(probe_burst_only)} "
               f"-DQ38_PROBE_PHASE={PROBE_PHASES[probe_phase]} "
               f"-DQ38_PROBE_MIXER_WINDOWS={mixer_windows} "
               f"-DQ38_PROBE_RECURRENT_K={probe_recurrent_k}")
    sweep_burst = attn_vc_sweep and spatz_burst
    spatz_vlen = LANE.get(profile, (4, 512))[1]
    spatz_rob = 32 if profile.startswith("dse_s2_") else 16
    if sweep_burst:
        defines += (f" -DQ38_SPATZ_BLOCK_ALLOC=1 "
                    f"-DQ38_SPATZ_MAX_BURST_WORDS={spatz_vlen // 32} "
                    f"-DQ38_SPATZ_ROB_DEPTH={spatz_rob}")
    build_make = build_source / "software/benchmarks/qwen38_prefill/Makefile"
    l2_bytes, rom_base = memory_layout(l2_mib)
    build = ["make", "-f", str(build_make), "layer_resident.elf",
             f"MEMPOOL_DIR={build_source}", f"INSTALL_DIR={RTL / 'install'}",
             f"config={profile}", "COMPILER=llvm", "virtual_cluster=1",
             "APP_DEFINES=" + defines, f"l2_size={l2_bytes}"]
    # A native profile keeps its ROM (boot_addr follows bootrom_base): linked
    # for the moved one, a core leaving main parks at a ROM that is not there.
    if profile not in NATIVE:
        build.append(f"boot_addr={rom_base}")
    if sweep_burst:
        build += ["spatz_block_alloc=1",
                  f"spatz_max_burst_words={spatz_vlen // 32}",
                  f"spatz_rob_depth={spatz_rob}", f"vlen={spatz_vlen}"]
    code, _ = execute(build, directory, "build.log", timeout)
    if attn_vc and code == 0 and kind == "attention":
        attn_vc_l2_check(directory, RTL / "install", l2_bytes)
    if ffn_l2 and code == 0 and external_data:
        ffn_l2_check(directory, RTL / "install", l2_bytes, ffn_rows)
    activations = ("resident whole-sequence control" if mixer_rows == 0
                   else f"bounded {mixer_rows}-row mixer windows")
    dimensions = "full-geometry" if geometry else "reduced-geometry"
    scope = (f"{probe_region} scheduling probe at {dimensions}; " +
             activations + "; first two and final column-panel groups"
             if probe_region else
             f"complete {dimensions} layer; " + activations)
    if ffn_l2:
        scope += "; VC FFN with independent phase batches and G staged in L2"
    elif ffn_wide:
        scope += (f"; wide FFN: joint gate/up job lists and single-buffered "
                  f"{ffn_down_panels}-panel down groups")
    if not probe_region and probe_phase != "all":
        scope = f"{probe_phase}-only numerical run; " + scope
    if gdn_ready_queue:
        scope += (f"; GDN ready queue of {gdn_project_rows}-row supertiles "
                  f"with {gdn_queue_panels}-panel groups, every queued "
                  "group executed")
    if build_only:
        scope = "build-only capacity gate for " + scope
    elif probe_burst_only:
        scope = "RedMulE/Spatz/iDMA capacity burst only at real geometry"
    result = {"profile": profile, "kind": kind, "tokens": tokens,
              "schedule": schedule, "delay_seed": delay_seed,
              "weight_panels": weight_panels, "external_data": external_data,
              "adaptive_panels": adaptive_panels,
              "gateup_pipeline": gateup_pipeline,
              "coordinator_dma": coordinator_dma,
              "kv_prefetch": kv_prefetch,
              "interleave_kv_query": interleave_kv_query,
              "gdn_chunk_rows": gdn_chunk_rows,
              "separate_stages": separate_stages,
              "ffn_norm_stream": ffn_norm_stream,
              "background_ffn_norm": background_ffn_norm,
              "gdn_double_chunk": gdn_double_chunk,
              "gdn_recur_projection_overlap": gdn_recur_projection_overlap,
              "gdn_triple_pipeline": gdn_triple_pipeline,
              "gdn_ready_queue": gdn_ready_queue,
              "gdn_project_rows": (gdn_project_rows if gdn_ready_queue
                                   else None),
              "gdn_queue_panels": (gdn_queue_panels if gdn_ready_queue
                                   else None),
              "gdn_queue_shape": queue_shape,
              "gdn_queue_depth": (gdn_queue_depth if gdn_ready_queue
                                  else None),
              "gdn_spill_serial": gdn_spill_serial,
              "gdn_group_sync": gdn_group_sync,
              "gdn_retire_copy": gdn_retire_copy,
              "gdn_drain_sleep": gdn_drain_sleep,
              "gdn_weight_prefetch": gdn_weight_prefetch,
              "gdn_z_fixed": gdn_z_fixed,
              "probe_burst_rounds": probe_burst_rounds,
              "probe_burst_slot": probe_burst_slot,
              "ffn_wide": ffn_wide,
              "ffn_down_panels": ffn_down_panels if ffn_wide else None,
              "ffn_down_slices": ffn_down_slices if ffn_wide else None,
              "gu_slots": gu_slots,
              "gu_panels": gu_panels or weight_panels,
              "dp_slots": dp_slots,
              "ffn_gu_slices": ffn_gu_slices,
              "ffn_pipe": ffn_pipe,
              "ffn_vc": ffn_vc,
              "ffn_l2": ffn_l2,
              "ffn_gu_rows": ffn_gu_rows,
              "ffn_down_rows": ffn_down_rows,
              "attn_vc": attn_vc,
              "attn_vc_rows": attn_vc_rows if attn_vc else None,
              "attn_vc_orows": (attn_vc_orows or attn_vc_rows) if attn_vc
              else None,
              "attn_spread": attn_spread,
              "redmule_matmul": redmule_matmul,
              "ffn_dataflow": ffn_dataflow,
              "proj_dataflow": proj_dataflow,
              "attn_local": attn_local,
              "attn_rm_accum": attn_rm_accum,
              "attn_fused_softmax": attn_fused_softmax,
              "attn_local_pipe": attn_local_pipe,
              "attn_rm_fold": attn_rm_fold,
              "attn_tile_local": attn_tile_local,
              "attn_isolate": attn_isolate,
              "attn_drain": attn_drain,
              "spatz_burst": spatz_burst,
              "spatz_block_alloc": 1 if sweep_burst else None,
              "spatz_max_burst_words": (spatz_vlen // 32 if sweep_burst
                                        else None),
              "spatz_rob_depth": spatz_rob if sweep_burst else None,
              "key_block": key_block,
              "attn_vc_sweep": attn_vc_sweep,
              "attn_vc_panel": attn_vc_panel,
              "attn_vc_dual": attn_vc_dual,
              "attn_vc_direct_dma": attn_vc_direct_dma,
              "attn_vc_cores": attn_vc_cores,
              "attn_vc_row_unroll": attn_vc_row_unroll,
              "attn_vc_tile_stage": attn_vc_tile_stage,
              "attn_vc_cache_gamma": attn_vc_cache_gamma,
              "attn_vc_max_rows": attn_vc_max_rows,
              "attn_sweep_rows": attn_sweep_rows,
              "attn_sweep_tile": attn_sweep_tile,
              "attn_sweep_mode": attn_sweep_mode,
              "attn_sweep_reuse_query": attn_sweep_reuse_query,
              "attn_sweep_register": attn_sweep_register,
              "attn_sweep_interleave": attn_sweep_interleave,
              "attn_sweep_tree_max": attn_sweep_tree_max,
              "attn_sweep_row_dma": attn_sweep_row_dma,
              "attn_sweep_lean": attn_sweep_lean,
              "attn_sweep_qscale": attn_sweep_qscale,
              "attn_sweep_exp2": attn_sweep_exp2,
              "attn_sweep_fast": attn_sweep_fast,
              "attn_sweep_tile_win": attn_sweep_tile_win,
              "attn_sweep_async": attn_sweep_async,
              "attn_sweep_late_drain": attn_sweep_late_drain,
              "attn_sweep_finish": attn_sweep_finish,
              "attn_sweep_overlap": attn_sweep_overlap,
              "probe_lead_groups": probe_lead_groups,
              "attn_supertile_rows": attn_st_rows or None,
              "attn_reuse_norm": attn_reuse_norm,
              "projection_rows": projection_rows or 64,
              "projection_columns": projection_columns,
              "l2_mib": l2_mib, "ffn_rows": ffn_rows,
              "mixer_rows": mixer_rows,
              "check_mode": ("disabled" if probe_region else
                             fixture["check_mode"]),
              "check_samples": 0 if probe_region else check_samples,
              "check_run_length": fixture["check_run_length"],
              "check_coverage": {} if probe_region else fixture["samples"],
              "probe_region": probe_region, "probe_base": probe_base,
              "probe_rows": probe_rows, "probe_mixer_base": mixer_base,
              "probe_mixer_rows": mixer_probe_rows,
              "probe_burst_only": probe_burst_only,
              "probe_phase": (probe_phase if probe_region or
                              probe_phase != "all" else None),
              "checked_tensors": ([] if probe_region else
                                  list(fixture["checks"])),
              "ffn_input_source": ("gold_mixer" if not probe_region and
                                   probe_phase == "ffn" else None),
              "probe_recurrent_k": (
                  probe_recurrent_k if probe_region and kind == "gdn"
                  else None),
              "redmule_timing_only": bool(probe_region),
              "kv_placement": placement, "kv_level": kv_level,
              "dimensions": fixture["dimensions"],
              "model_install": str(model_install),
              "geometry": fixture["geometry"],
              "reference": ("not_checked" if probe_region else
                            fixture["reference"]),
              "scope": scope,
              "build_returncode": code, "build_passed": False,
              "build_only": build_only, "passed": False,
              "header_sha256": hashlib.sha256(
                  (directory / "layer_data.h").read_bytes()).hexdigest()}
    if code == 0:
        binary = directory / "layer_resident.elf"
        result["elf_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        result["l1_bytes"] = l1_footprint(binary, RTL / "install")
        if attn_vc_panel:
            result["attention_projection_layout"] = projection_layout_check(
                directory, RTL / "install", fixture["dimensions"],
                attn_vc_rows, attn_vc_orows, projection_rows,
                projection_columns, key_block=key_block,
                sweep_tile=attn_sweep_tile)
        result["stack_depth"] = stack_gate(
            (directory / "build.log").read_text(), binary, RTL / "install",
            # crt0.S starts SP one word below the end of its512-byte slice.
            # Apply this attention guard without changing FFN qualification.
            reserved=4 if attn_vc else 0)
        if not result["stack_depth"]["fits"]:
            return directory, result
        result["ctrl_isa"] = ctrl_gate(binary, RTL / "install")
        if not result["ctrl_isa"]["clean"]:
            return directory, result
        result["build_passed"] = True
        if build_only:
            result["numerical_checked"] = False
            return directory, result
        target = ("teranoc_v2:system_config=" + PROFILES[profile][0] +
                  ":virtual_cluster=true")
        target_directory = []
        if external_data:
            target = ("qwen38_model:system_config=" + PROFILES[profile][0] +
                      ":virtual_cluster=true" + f":qwen_l2_mib={l2_mib}")
            if probe_region:
                target += ":redmule_timing_only=true"
            if profile in LANE:
                fpus, vlen = LANE[profile]
                target += f":qwen_fpus={fpus}:qwen_vlen={vlen}"
            if profile in SPATZ_PER_TILE:
                target += f":qwen_spatz_per_tile={SPATZ_PER_TILE[profile]}"
            if spatz_burst:
                target += ":qwen_spatz_burst=1"
                if sweep_burst:
                    target += (f":spatz_max_burst_words={spatz_vlen // 32}"
                               f":spatz_rob_depth={spatz_rob}")
            if profile in BANKS_PER_TILE:
                target += f":qwen_banks_per_tile={BANKS_PER_TILE[profile]}"
            if profile in REDMULE_ROB:
                target += f":qwen_redmule_rob={REDMULE_ROB[profile]}"
            if profile in WIDE_RO:
                target += f":qwen_wide_ro={WIDE_RO[profile]}"
            if sweep_burst and profile in (
                    "dse_s1_winner_b32", "dse_s2_winner_f8_v1024",
                    "dse_s1_winner_b32_c4_f4_v512",
                    "dse_s2_winner_c4_f8_v1024"):
                target = attention_model_target(profile, l2_mib,
                                                bool(probe_region))
            target_directory = ["--target-dir=" + str(build_make.parent),
                                "--parameter=qwen_data_elf=" +
                                str(directory / "layer_data.elf")]
        command = ["conda", "run", "--no-capture-output", "-n", "gvsoc",
                   str(model_install / "bin/gvrun"),
                   "--target=" + target,
                   "--parameter=binary=" + str(binary),
                   "--work-dir=" + str(directory)] + target_directory + ["run"]
        if probe_region or attn_vc_sweep:
            command = ["env", "RM_STALL_REPORT=1", "SPATZ_ACTIVITY_REPORT=1",
                       ("SPATZ_ACTIVITY_PERIOD=10000" if probe_burst_only
                        else "SPATZ_ACTIVITY_PERIOD=100000"),
                       "IDMA_ACTIVITY_REPORT=1"] + command
        code, elapsed = execute(command, directory, "model.log", timeout)
        log = (directory / "model.log").read_text()
        result.update(returncode=code, elapsed_seconds=elapsed)
        # Externally backed Qwen runs can otherwise pass numerically against
        # the wrong prebuilt platform tree without ever proving model/ELF
        # geometry agreement.
        agreement = check_contract(directory, required=external_data,
                                   timing_only=bool(probe_region))
        result["configuration"] = agreement
        # Numerical success must never overwrite a configuration failure.
        if probe_region:
            stage_rows = (gdn_chunk_rows
                          if gdn_double_chunk or gdn_ready_queue
                          else mixer_rows)
            key_blocks = 0
            expected_windows = mixer_windows
            mixer_run = not probe_burst_only and probe_phase != "ffn"
            ffn_run = not probe_burst_only and probe_phase != "mixer"
            if kind == "attention" and mixer_run:
                key_blocks = sum(
                    min(3, -(-(base + mixer_rows) // key_block))
                    for base in range(mixer_base,
                                      mixer_base + mixer_probe_rows,
                                      mixer_rows))
                if attn_vc_sweep:
                    build_text = (directory / "build.log").read_text()
                    height = int(re.search(
                        r"-DREDMULE_H=(\d+)", build_text)[1])
                    tile = attn_sweep_tile or min(height * 4, key_block // 8)
                    window = attn_sweep_rows or 8 * tile
                    dims = fixture["dimensions"]
                    kv_heads = dims["kv_heads"]
                    replicas = redmules(profile) // 16 // kv_heads
                    batches = math.ceil(
                        dims["q_heads"] / kv_heads / (2 * replicas))
                    expected_windows, key_blocks = attn_sweep_probe_counts(
                        mixer_base, mixer_probe_rows, window, batches,
                        key_block, tokens)
            expected = {
                "mixer_start_row": mixer_base,
                "mixer_rows": mixer_probe_rows,
                "ffn_start_row": probe_base, "ffn_rows": probe_rows,
                "mixer_windows": expected_windows if mixer_run else 0,
                "ffn_blocks": (probe_rows //
                               (ffn_down_rows or
                                (512 if profile.startswith("dse_s2") else 256)
                                if ffn_l2 else ffn_rows)
                               if ffn_run else 0),
                "gdn_chunks": (mixer_probe_rows // stage_rows
                               if kind == "gdn" and mixer_run else 0),
                "recurrent_k": (probe_recurrent_k
                                if kind == "gdn" and not probe_burst_only
                                else 0),
                "burst_only": int(probe_burst_only),
                "burst_jobs": (0 if ffn_l2 or attn_vc_panel
                               else redmules(profile)),
                "burst_spatz_ops": (0 if ffn_l2 or attn_vc_panel
                                    else PROFILES[profile][1]),
                "key_blocks": key_blocks}
            queue = None
            if gdn_ready_queue and not probe_burst_only:
                supertiles = mixer_probe_rows // gdn_project_rows
                queue = {"supertile_rows": gdn_project_rows,
                         "panels": gdn_queue_panels,
                         "groups": queue_shape["groups"],
                         "queue_groups": supertiles * queue_shape["groups"],
                         "queue_jobs": supertiles * queue_shape["jobs"]}
            result.update(evaluate_schedule_probe(
                log, code, agreement, expected, queue))
            if result["schedule_passed"]:
                try:
                    activity = analyze_probe(
                        directory / "model.log", profile,
                        result["probe_coverage"], PROFILES[profile][1],
                        probe_burst_only, redmules(profile),
                        probe_phase=probe_phase)
                    result["schedule_activity"] = activity
                    result["design_gate_passed"] = activity[
                        "design_gate_passed"]
                except ValueError as error:
                    result.update(passed=False, schedule_passed=False,
                                  activity_error=str(error))
        else:
            result.update(evaluate(log, code, fixture["checked_elements"],
                                   agreement, len(fixture["checks"])))
        kv_metrics = re.search(
            r"\[qwen38_kv\] replicas=(\d+) level=(\d+) "
            r"requested_read_bytes=(\d+) load_join_cycles=(\d+)", log)
        if kv_metrics:
            result["kv_replicas"] = int(kv_metrics[1])
            result["kv_requested_read_bytes"] = int(kv_metrics[3])
            result["kv_load_join_cycles"] = int(kv_metrics[4])
        if dump:
            try:
                result["actual_hashes"] = extract_actual(
                    log, fixture, directory)
            except ValueError as error:
                result.update(passed=False, dump_error=str(error))
    return directory, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-install", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES),
                        default=list(PROFILES))
    parser.add_argument("--kinds", nargs="+", choices=("attention", "gdn"),
                        default=["attention", "gdn"])
    parser.add_argument("--schedules", nargs="+",
                        choices=("serialized", "async"),
                        default=["serialized", "async"])
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--schedule-probe", action="store_true",
                        help="run start/tail schedule windows with full-K "
                             "RedMulE timing and traffic, no numerical checks")
    parser.add_argument("--build-only", action="store_true",
                        help="compile and check L1/stack without simulation; "
                             "only valid with --schedule-probe")
    parser.add_argument("--probe-burst-only", action="store_true",
                        help="run only the full-K engine-capacity burst, "
                             "without mixer or FFN loops")
    parser.add_argument("--probe-regions", nargs="+",
                        choices=("start", "tail"),
                        default=["start", "tail"])
    parser.add_argument("--probe-recurrent-k", type=int,
                        choices=(16, 32, 64, 128), default=32,
                        help="GDN state rows updated per selected token in "
                             "the scheduling probe")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--dump-results", action="store_true")
    parser.add_argument("--delay-seed", type=int, default=0)
    parser.add_argument("--kv-placements", nargs="+",
                        choices=("global", "replicated"))
    parser.add_argument("--kv-level", type=int, default=0)
    parser.add_argument("--attention-kv-heads", type=int,
                        choices=(1, 2), default=1)
    parser.add_argument("--weight-panels", type=int, default=2,
                        help="L1 panels per buffer; 0 is resident control")
    parser.add_argument("--adaptive-panels", action="store_true",
                        help="pack more output-column panels into each L1 "
                             "weight slot when the reduction K is shorter")
    parser.add_argument("--gateup-pipeline", action="store_true",
                        help="overlap activation of completed gate/up columns "
                             "with RedMulE and next-panel DMA")
    parser.add_argument("--coordinator-dma", action="store_true",
                        help="issue gate/up prefetch through the bounded "
                             "coordinator-only DMA queue")
    parser.add_argument("--kv-prefetch", action="store_true",
                        help="overlap the next global attention KV block DMA "
                             "with QK, softmax and PV on the current block")
    parser.add_argument("--interleave-kv-query", action="store_true",
                        help="run a query window after its causal KV prefix "
                             "is built, before later KV windows")
    parser.add_argument("--gdn-chunk-rows", type=int, choices=(16, 32, 64),
                        default=64,
                        help="ordered recurrent rows per completion chunk")
    parser.add_argument("--separate-stages", action="store_true",
                        help="reserve independent mixer and FFN stage arenas")
    parser.add_argument("--ffn-norm-stream", action="store_true",
                        help="temporarily hold normalized FFN input in the "
                             "final-output L2 stream")
    parser.add_argument("--background-ffn-norm", action="store_true",
                        help="normalize the preceding attention window on "
                             "sibling Spatz cores while query engines run")
    parser.add_argument("--gdn-double-chunk", action="store_true",
                        help="reuse two bounded GDN mixer microchunk slots")
    parser.add_argument("--gdn-recur-projection-overlap", action="store_true",
                        help="start the next GDN QKV panel while the current "
                             "chunk advances its recurrent state")
    parser.add_argument("--gdn-triple-pipeline", action="store_true",
                        help="DMA-stage GDN chunk n+2 while RedMulE projects "
                             "n+1 and Spatz advances recurrence n")
    parser.add_argument("--gdn-ready-queue", action="store_true",
                        help="queue the next GDN supertile's full-K "
                             "projection rectangles during the ordered "
                             "recurrence, spilling them to an L2 ring")
    parser.add_argument("--gdn-project-rows", type=int, default=256,
                        help="rows per GDN ready-queue supertile")
    parser.add_argument("--gdn-queue-panels", type=int, default=4,
                        help="column panels per GDN ready-queue group")
    parser.add_argument("--gdn-queue-depth", type=int, choices=(1, 2),
                        default=1,
                        help="queued RedMulE jobs in flight per engine; two "
                             "aborts the timing-only model at real geometry")
    parser.add_argument("--gdn-spill-serial", action="store_true",
                        help="diagnostic: start a queued group only after "
                             "the previous group's L2 spill completes")
    parser.add_argument("--gdn-no-group-sync", action="store_true",
                        help="let engines start the next queued group before "
                             "every engine retired the current one")
    parser.add_argument("--gdn-retire-copy", type=int, choices=(0, 1, 2),
                        default=1,
                        help="Z-tile retire copy: one vector copy, one per "
                             "row, or none (diagnostic, stale projections)")
    parser.add_argument("--gdn-drain-sleep", action="store_true",
                        help="diagnostic: draining agents sleep on RedMulE "
                             "EOC instead of polling its status register")
    parser.add_argument("--gdn-no-weight-prefetch", action="store_true",
                        help="fetch a queued group's weights only after the "
                             "previous group retires")
    parser.add_argument("--gdn-z-fixed", action="store_true",
                        help="diagnostic: every queued job writes Z slot 0")
    parser.add_argument("--probe-burst-rounds", type=int, default=1,
                        help="back-to-back rounds of the capacity burst")
    parser.add_argument("--probe-burst-slot", type=int, choices=(0, 1),
                        default=0,
                        help="weight slot the capacity burst streams")
    parser.add_argument("--ffn-wide", action="store_true",
                        help="one gate/up job list per panel group and "
                             "single-buffered multi-panel down groups")
    parser.add_argument("--ffn-down-panels", type=int, default=4,
                        help="full-K down panels staged per wide FFN group")
    parser.add_argument("--attn-supertile-rows", type=int, default=0,
                        help="run attention projections over supertiles of "
                             "this many rows, one full-K rectangle per engine "
                             "per group; 0 keeps per-window projections")
    parser.add_argument("--probe-lead-groups", type=int, default=2,
                        help="leading panel groups a probe executes before "
                             "the final one")
    parser.add_argument("--redmule-matmul", action="store_true",
                        help="start zero-addend jobs as MATMUL, without "
                             "zeroing or reading their Z tile")
    parser.add_argument("--ffn-dataflow", action="store_true",
                        help="synchronize wide gate/up groups by dependency "
                             "counters instead of cluster barriers")
    parser.add_argument("--attn-local", action="store_true",
                        help="group-local key sweep: owner state in level-0 "
                             "VC arenas, group-only barriers per key block")
    parser.add_argument("--attn-rm-accum", action="store_true",
                        help="RedMulE accumulates the context in FP16 "
                             "(Z = PV + Z); Spatz rescales moved rows only")
    parser.add_argument("--key-block", type=int, default=64,
                        choices=(64, 128, 256, 512),
                        help="attention key block: softmax row length per "
                             "pass and K/V tile size")
    parser.add_argument("--attn-vc-sweep", action="store_true",
                        help="KV-affine two-head VC batches with private "
                             "group operands and next-query prefetch")
    parser.add_argument("--attn-sweep-rows", type=int, default=0,
                        help="KV-affine query rows "
                             "(default: eight owner tiles)")
    parser.add_argument("--attn-sweep-tile", type=int, default=0,
                        help="KV-affine owner rows (default: min(MT, B/8))")
    parser.add_argument("--attn-sweep-mode",
                        choices=("group", "tile", "early"),
                        default="group",
                        help="compute joins: group, tile-local, or tile-local "
                             "with rescale overlapping next QK")
    parser.add_argument("--attn-sweep-reuse-query", action="store_true",
                        help="reuse exhausted Q bank for gated finish output")
    parser.add_argument("--attn-sweep-row-dma", action="store_true",
                        help="batch gated-result exports with strided DMA")
    parser.add_argument("--attn-sweep-tree-max", action="store_true",
                        help="use an attention-only vector max tree")
    parser.add_argument("--attn-sweep-interleave", action="store_true",
                        help="interleave owner rows within aligned key blocks")
    parser.add_argument("--attn-sweep-register", action="store_true",
                        help="retain softmax scores in vector registers")
    parser.add_argument("--attn-sweep-lean", action="store_true",
                        help="legacy softmax row with fewer vector ops")
    parser.add_argument("--attn-sweep-qscale", action="store_true",
                        help="apply 1/sqrt(HD) in the Q epilogue (lean)")
    parser.add_argument("--attn-sweep-exp2", action="store_true",
                        help="degree-4 exp2 in the lean softmax row")
    parser.add_argument("--attn-sweep-fast", action="store_true",
                        help="FP16 base-2 softmax row (needs qscale)")
    parser.add_argument("--attn-sweep-tile-win", action="store_true",
                        help="scores/Z in the tile window (model extension; "
                             "needs tile or early mode)")
    parser.add_argument("--attn-sweep-async", action="store_true",
                        help="dependency-only sweep pipeline (early mode)")
    parser.add_argument("--attn-sweep-late-drain", type=int, default=0,
                        choices=(0, 1, 2),
                        help="drain stores after rescale (1) or after "
                             "softmax and rescale (2), not per row")
    parser.add_argument("--attn-sweep-finish", action="store_true",
                        help="fused output gate and batched gated export "
                             "(no full context export)")
    parser.add_argument("--attn-sweep-overlap", action="store_true",
                        help="one key-block stream over all windows")
    parser.add_argument("--spatz-burst", action="store_true",
                        help="model Spatz VLSU bursts on the wide TCDM plane "
                             "(RTL spatz_burst=1); needs --external-data")
    parser.add_argument("--attn-drain", action="store_true",
                        help="read back Spatz rows RedMulE consumes before "
                             "the barrier (fence misses vector stores)")
    parser.add_argument("--attn-tile-local", action="store_true",
                        help="softmax rows and their state on the core's "
                             "own tile")
    parser.add_argument("--attn-isolate", type=int, default=0,
                        choices=(0, 1, 2, 3, 4),
                        help="probe only: drop the pipelined sweep's RedMulE "
                             "jobs (1), K/V DMA (2) or both (3)")
    parser.add_argument("--attn-rm-fold", action="store_true",
                        help="fresh FP16 PV per key block, folded into the "
                             "context by a RedMulE identity GEMM")
    parser.add_argument("--attn-local-pipe", action="store_true",
                        help="pipelined local sweep: softmax(n) runs while "
                             "RedMulE runs PV(n-1) and QK(n+1)")
    parser.add_argument("--attn-fused-softmax", action="store_true",
                        help="one-pass softmax rows (same vector arithmetic)")
    parser.add_argument("--proj-dataflow", action="store_true",
                        help="synchronize attention supertile projections by "
                             "dependency counters instead of cluster barriers")
    parser.add_argument("--attn-spread", action="store_true",
                        help="spread attention softmax and accumulation rows "
                             "over every core between the QK and PV jobs")
    parser.add_argument("--attn-reuse-norm", action="store_true",
                        help="reuse the K/V pass's normalized rows for the "
                             "interleaved query pass")
    parser.add_argument("--gu-slots", type=int, default=2,
                        help="dataflow gate/up weight ring depth; above two "
                             "the slots are carved from the weight area")
    parser.add_argument("--ffn-vc", action="store_const", const=2,
                        default=0,
                        help="VC-scoped FFN: each engine computes gate/up "
                             "pairs; each 2x2 VC double-buffers its weights "
                             "and syncs its 16 tiles on fractal_sync at every "
                             "buffer switch")
    parser.add_argument("--ffn-l2", action="store_true",
                        help="independent FFN phases with G staged through L2;"
                             " "
                             "requires --ffn-vc and --probe-phase ffn")
    parser.add_argument("--ffn-gu-rows", type=int, default=0,
                        help="Gate/Up output rows per job (default: H*(P+1))")
    parser.add_argument("--ffn-down-rows", type=int, default=0,
                        help="Down batch rows "
                             "(default: s1 256, s2 512)")
    parser.add_argument("--attn-vc", action="store_true",
                        help="phased attention: Q/gate/K/V and output "
                             "projections of all mixer rows on the VC FFN "
                             "scheme, with the key sweep between them")
    parser.add_argument("--attn-vc-rows", type=int, default=256,
                        help="rows per Q/gate/K/V projection block of "
                             "--attn-vc")
    parser.add_argument("--attn-vc-panel", action="store_true",
                        help="consume projection panels in vector registers; "
                             "requires mixer-only --attn-vc-sweep")
    parser.add_argument("--attn-vc-dual", action="store_true",
                        help="tile cores process projection epilogues; "
                             "requires --attn-vc-panel")
    parser.add_argument("--attn-vc-direct-dma", action="store_true",
                        help="export V/gate panels without a Spatz copy")
    parser.add_argument("--attn-vc-cache-gamma",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="retain head-normalization weights in vregs "
                             "(default: static VLEN1024 projection)")
    parser.add_argument("--attn-vc-cores", type=int, choices=(0, 2, 4),
                        default=None,
                        help="fixed projection row blocks on this many tile "
                             "cores (default: available tile cores with "
                             "--attn-vc-dual); zero selects the baseline "
                             "queue")
    parser.add_argument("--attn-vc-row-unroll", type=int, choices=(1, 2, 4),
                        default=None,
                        help="worker row unroll (default: S1 two, S2 four); "
                             "the controller services one row at a time")
    parser.add_argument("--attn-vc-max-rows", action="store_true",
                        help="select audited maximum projection tiles/batches "
                             "within the execution extent; requires "
                             "--attn-vc-panel, overrides projection row sizes")
    parser.add_argument("--attn-vc-orows", type=int, default=0,
                        help="rows per output projection block of --attn-vc "
                             "(default: --attn-vc-rows)")
    parser.add_argument("--ffn-vc-poll", action="store_true",
                        help="debug: the VC FFN polls instead of sleeping "
                             "on events")
    parser.add_argument("--ffn-vc-dma-stress", action="store_true",
                        help="debug: keep an L1 -> L2 DMA running under every "
                             "VC FFN job")
    parser.add_argument("--probe-phase", choices=tuple(PROBE_PHASES),
                        default="all",
                        help="select the mixer or FFN for a schedule probe "
                             "or full-tensor toy numerical check; numerical "
                             "FFN checks read the golden mixer residual")
    parser.add_argument("--ffn-pipe", type=int, nargs="?", const=2,
                        default=0, metavar="SLOTS",
                        help="whole FFN as one barrier-free dataflow "
                             "pipeline over SLOTS weight slots (default 2)")
    parser.add_argument("--ffn-gu-slices", type=int, default=1,
                        help="ordered K slices of gate/up; above one, a "
                             "double-buffered step loop with one job per "
                             "engine (needs --ffn-dataflow and down slices)")
    parser.add_argument("--dp-slots", type=int, default=2,
                        help="projection dataflow weight ring depth")
    parser.add_argument("--gu-panels", type=int, default=0,
                        help="panels per gate/up ring slot (0: weight panels)")
    parser.add_argument("--ffn-down-slices", type=int, default=1,
                        help="ordered K slices of each wide down rectangle; "
                             "above one, slice weights are double-buffered")
    parser.add_argument("--projection-rows", type=int, choices=(8, 16, 32, 64),
                        default=64,
                        help="RedMulE output rows per projection task")
    parser.add_argument("--projection-columns", type=int, choices=(32, 64),
                        default=64,
                        help="output columns per packed RedMulE weight panel")
    parser.add_argument("--external-data", action="store_true")
    parser.add_argument("--l2-mib", type=int, choices=(512, 1024),
                        default=512)
    parser.add_argument("--ffn-rows", type=int, default=64)
    parser.add_argument("--check-samples", type=int, default=0,
                        help="elements checked per tensor; 0 checks whole "
                             "tensors, which whole-geometry runs cannot do")
    parser.add_argument("--check-run-length", type=int, default=32,
                        help="elements per contiguous run; sensitivity "
                             "scales with samples/run-length, gather cost "
                             "with the number of runs")
    parser.add_argument("--geometry", choices=("toy", "real"), default="toy",
                        help="reduced dimensions, or the real Qwen3.8 model. "
                             "Real geometry checks bounded samples against "
                             "bounded_reference and needs the data sidecar.")
    parser.add_argument("--fixture-cache", type=Path,
                        help="reuse generated fixtures across cases; a "
                             "full-geometry fixture is tens of minutes and "
                             "three quarters of a gigabyte")
    parser.add_argument("--mixer-rows", nargs="+", type=int, default=[64],
                        help="mixer activation window rows; 0 is the resident "
                             "whole-sequence control. The first value is the "
                             "bitwise reference for the others.")
    args = parser.parse_args()
    if args.ffn_rows <= 0 or args.ffn_rows % 64:
        parser.error("FFN rows must be a positive multiple of 64")
    if args.check_samples < 0:
        parser.error("check samples must be nonnegative")
    if args.schedule_probe and (args.geometry != "real" or args.tokens % 64 or
                                args.tokens < 256 or 0 in args.mixer_rows):
        parser.error("schedule probe requires real geometry, S >= 256 in "
                     "multiples of 64, and bounded mixer windows")
    if args.schedule_probe and (args.schedules != ["async"] or
                                not args.gateup_pipeline or
                                args.weight_panels < 2):
        parser.error("schedule probe needs async gate/up and at least two "
                     "weight panels")
    if (args.schedule_probe and
            len(set(args.probe_regions)) != len(args.probe_regions)):
        parser.error("Do not repeat probe regions")
    if args.build_only and not args.schedule_probe:
        parser.error("--build-only needs --schedule-probe")
    if args.probe_burst_only and not args.schedule_probe:
        parser.error("--probe-burst-only needs --schedule-probe")
    if args.ffn_vc and args.ffn_pipe:
        parser.error("--ffn-vc and --ffn-pipe are alternative FFNs")
    if args.ffn_l2 and (not args.ffn_vc or args.ffn_pipe or
                        args.probe_phase != "ffn" or
                        args.ffn_gu_slices != 1 or args.ffn_down_slices < 1 or
                        args.attn_vc or not args.external_data):
        parser.error("--ffn-l2 needs FFN-only VC, external data, no attention "
                     "VC, full-K Gate/Up and positive Down K slices")
    if args.ffn_gu_rows < 0 or args.ffn_down_rows < 0:
        parser.error("FFN row overrides must be nonnegative")
    if args.probe_phase != "all" and not args.schedule_probe and (
            args.kinds != ["attention"] or args.check_samples or
            args.geometry != "toy"):
        parser.error("numerical --probe-phase needs full-tensor toy "
                     "attention fixtures")
    if args.check_run_length <= 0:
        parser.error("check run length must be positive")
    if any(rows < 0 or rows % 64 for rows in args.mixer_rows):
        parser.error("Mixer rows must be zero or a multiple of 64")
    if len(set(args.mixer_rows)) != len(args.mixer_rows):
        parser.error("Do not repeat mixer row bounds")
    if args.l2_mib != 512 and not args.external_data:
        parser.error("extended backing requires --external-data")
    geometry = Geometry() if args.geometry == "real" else None
    if geometry and not (args.check_samples and args.external_data and
                         args.l2_mib == 1024):
        parser.error("real geometry needs --check-samples, --external-data "
                     "and --l2-mib 1024")
    if args.weight_panels < 0:
        parser.error("weight panels must be nonnegative")
    if args.adaptive_panels and args.weight_panels == 0:
        parser.error("adaptive panels need staged weights")
    if args.gateup_pipeline and args.weight_panels == 0:
        parser.error("gate/up overlap needs staged weights")
    if args.coordinator_dma and not args.gateup_pipeline:
        parser.error("coordinator DMA needs gate/up overlap")
    if args.separate_stages and not (args.gateup_pipeline and
                                     args.schedules == ["async"]):
        parser.error("separate stage arenas need async gate/up execution")
    if args.ffn_norm_stream and not args.separate_stages:
        parser.error("FFN norm stream needs separate stage arenas")
    if args.background_ffn_norm and not (
            args.ffn_norm_stream and args.interleave_kv_query and
            args.kv_prefetch and
            args.kinds == ["attention"] and args.kv_placements == ["global"]):
        parser.error("background FFN norm needs interleaved global attention "
                     "and the FFN norm stream")
    if args.gdn_double_chunk and (
            args.projection_rows > args.gdn_chunk_rows or
            args.gdn_chunk_rows % args.projection_rows):
        parser.error("GDN double chunks must hold whole projection row tiles")
    if args.gdn_recur_projection_overlap and not (
            args.gdn_double_chunk and args.weight_panels > 0 and
            args.schedules == ["async"] and args.kinds == ["gdn"]):
        parser.error("GDN projection/recurrence overlap needs async GDN, "
                     "staged weights, and double chunks")
    if args.gdn_triple_pipeline and not (
            args.gdn_recur_projection_overlap and args.coordinator_dma):
        parser.error("GDN triple pipeline needs QKV overlap and "
                     "coordinator DMA")
    if args.gdn_ready_queue and (
            args.kinds != ["gdn"] or args.schedules != ["async"] or
            args.separate_stages or args.gdn_double_chunk or
            args.weight_panels == 0 or args.gdn_queue_panels <= 0 or
            args.gdn_project_rows % args.gdn_chunk_rows or
            args.gdn_chunk_rows % args.projection_rows or
            any(rows == 0 or args.gdn_project_rows % rows
                for rows in args.mixer_rows)):
        parser.error("GDN ready queue needs async GDN with shared arenas, "
                     "no double chunks, and supertiles holding whole mixer "
                     "windows, chunks and projection row tiles")
    if args.attn_local and not (args.attn_supertile_rows or args.attn_vc):
        parser.error("--attn-local needs --attn-supertile-rows or --attn-vc")
    if args.attn_vc_sweep and not (
            args.attn_vc and args.attn_fused_softmax and
            args.probe_phase == "mixer"):
        parser.error("--attn-vc-sweep needs --attn-vc and "
                     "--attn-fused-softmax and --probe-phase mixer")
    if args.attn_vc_panel and not (
            args.attn_vc_sweep and args.probe_phase == "mixer"):
        parser.error("--attn-vc-panel needs mixer-only --attn-vc-sweep")
    if args.attn_vc_dual and not args.attn_vc_panel:
        parser.error("--attn-vc-dual needs --attn-vc-panel")
    if (args.attn_vc_direct_dma or args.attn_vc_cores or
            args.attn_vc_row_unroll not in (None, 1)) and not (
                args.attn_vc_dual):
        parser.error("Projection alternatives need --attn-vc-dual")
    if args.attn_vc_max_rows and not args.attn_vc_panel:
        parser.error("--attn-vc-max-rows needs --attn-vc-panel")
    if args.key_block > 128 and not args.attn_vc_sweep:
        parser.error("--key-block above 128 needs --attn-vc-sweep")
    if args.attn_vc and (args.attn_supertile_rows or not (
            args.attn_local_pipe and (args.ffn_vc or args.ffn_pipe)) or
            any(rows == 0 or args.attn_vc_rows % rows
                for rows in args.mixer_rows)):
        parser.error("--attn-vc replaces supertiles, needs --attn-local-pipe "
                     "and --ffn-vc or --ffn-pipe, and blocks of whole mixer "
                     "windows")
    if args.attn_local_pipe and not (args.attn_rm_accum and args.kv_prefetch):
        parser.error("--attn-local-pipe needs --attn-rm-accum and "
                     "--kv-prefetch")
    if args.attn_tile_local and not args.attn_local:
        parser.error("--attn-tile-local needs --attn-local")
    if args.attn_isolate and not (args.schedule_probe and
                                  args.attn_local_pipe):
        parser.error("--attn-isolate is a scheduling probe of "
                     "--attn-local-pipe")
    if args.attn_rm_fold and not args.attn_local_pipe:
        parser.error("--attn-rm-fold needs --attn-local-pipe")
    if args.attn_rm_accum and not (args.attn_local and args.redmule_matmul):
        parser.error("--attn-rm-accum needs --attn-local and --redmule-matmul")
    if args.proj_dataflow and not (args.coordinator_dma and
                                   args.gateup_pipeline):
        parser.error("--proj-dataflow needs --coordinator-dma and "
                     "--gateup-pipeline")
    if args.ffn_dataflow and not (args.ffn_wide and args.coordinator_dma):
        parser.error("--ffn-dataflow needs --ffn-wide and --coordinator-dma")
    if args.attn_supertile_rows and (
            not args.interleave_kv_query or
            any(rows == 0 or args.attn_supertile_rows % rows
                for rows in args.mixer_rows) or
            args.attn_supertile_rows > max(max(args.mixer_rows),
                                           args.ffn_rows)):
        parser.error("attention supertiles need interleaved K/V and query "
                     "windows, whole mixer windows, and at most FFN-block "
                     "rows")
    if args.attn_reuse_norm and not args.interleave_kv_query:
        parser.error("--attn-reuse-norm needs --interleave-kv-query")
    if args.attn_spread and not (args.gateup_pipeline and
                                 args.schedules == ["async"]):
        parser.error("--attn-spread needs async gate/up stage buffers")
    if args.gu_slots < 2 or args.gu_panels < 0 or (
            args.gu_slots > 2 and not args.ffn_dataflow):
        parser.error("a gate/up ring deeper than two needs --ffn-dataflow")
    if args.dp_slots < 2 or (args.dp_slots > 2 and not args.proj_dataflow):
        parser.error("a projection ring deeper than two needs "
                     "--proj-dataflow")
    if args.ffn_down_slices < 1 or (args.ffn_down_slices > 1 and
                                    not args.ffn_wide):
        parser.error("down K slices need --ffn-wide")
    if args.ffn_wide and not (args.gateup_pipeline and
                              args.schedules == ["async"] and
                              args.ffn_down_panels > 0):
        parser.error("wide FFN needs the async gate/up pipeline")
    if args.kv_prefetch and args.kv_placements and \
            "replicated" in args.kv_placements:
        parser.error("KV prefetch currently supports global placement only")
    if args.kv_prefetch and args.kinds != ["gdn"] and \
            any("8x8" in profile for profile in args.profiles) and \
            not args.kv_placements:
        parser.error("8x8 KV prefetch needs --kv-placements global")
    if args.timeout <= 0 or args.delay_seed < 0:
        parser.error("timeout must be positive and delay seed nonnegative")
    if len(set(args.schedules)) != len(args.schedules):
        parser.error("Do not repeat schedules")
    if args.kv_level < 0 or args.kv_level > 1:
        parser.error("KV placement supports one-group or 2x2-group VCs")
    schedules = [s for s in ("serialized", "async") if s in args.schedules]
    dump = not args.schedule_probe and (
        args.dump_results or len(schedules) == 2 or
        len(args.kv_placements or []) == 2 or len(args.mixer_rows) > 1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    build_source = freeze_build_inputs(RTL, root)
    record_model_install(args.model_install, root)
    results = []
    for profile in args.profiles:
        for kind in args.kinds:
            default = "replicated" if "8x8" in profile else "global"
            selected = args.kv_placements or [default]
            choices = ("global", "replicated")
            placements = ([p for p in choices if p in selected]
                          if kind == "attention" else ["none"])
            global_hashes = {}
            # The first mixer bound is the reference every other bound must
            # reproduce bit for bit: shrinking the activation window must not
            # change a single output, state or cache element.
            reference = {}
            for placement in placements:
                for mixer_rows in args.mixer_rows:
                    probe_rows = 0
                    if args.schedule_probe:
                        unit = math.lcm(mixer_rows, args.ffn_rows,
                                        args.attn_supertile_rows or 1,
                                        args.attn_vc_rows if args.attn_vc
                                        else 1,
                                        args.attn_vc_orows or 1)
                        minimum = max(2 * mixer_rows,
                                      args.ffn_rows *
                                      (1 if args.ffn_l2 else 2),
                                      2 * args.attn_vc_rows if args.attn_vc
                                      else 0)
                        if args.gdn_ready_queue:
                            unit = math.lcm(unit, args.gdn_project_rows)
                            minimum = max(minimum,
                                          2 * args.gdn_project_rows)
                        probe_rows = ((minimum + unit - 1) // unit) * unit
                        if probe_rows > args.tokens:
                            parser.error("probe needs at least two FFN blocks "
                                         "and room for the mixer windows")
                    baseline = None
                    regions = (args.probe_regions if args.schedule_probe
                               else [None])
                    for schedule, probe_region in (
                            (schedule, region) for schedule in schedules
                            for region in regions):
                        directory, result = run_case(
                            root, build_source, args.model_install.resolve(),
                            profile, kind, schedule, args.tokens, args.timeout,
                            dump, args.delay_seed, placement, args.kv_level,
                            args.attention_kv_heads, args.weight_panels,
                            args.external_data, args.l2_mib, args.ffn_rows,
                            mixer_rows, args.check_samples,
                            args.check_run_length, geometry,
                            args.fixture_cache, args.adaptive_panels,
                            args.projection_rows, args.gateup_pipeline,
                            args.projection_columns, args.coordinator_dma,
                            args.kv_prefetch, args.interleave_kv_query,
                            args.gdn_chunk_rows, args.separate_stages,
                            args.ffn_norm_stream, args.background_ffn_norm,
                            args.gdn_double_chunk,
                            args.gdn_recur_projection_overlap,
                            args.gdn_triple_pipeline, probe_region, probe_rows,
                            args.probe_recurrent_k, args.build_only,
                            args.probe_burst_only, args.gdn_ready_queue,
                            args.gdn_project_rows, args.gdn_queue_panels,
                            args.ffn_wide, args.ffn_down_panels,
                            args.gdn_queue_depth, args.gdn_spill_serial,
                            not args.gdn_no_group_sync,
                            args.gdn_retire_copy, args.gdn_drain_sleep,
                            not args.gdn_no_weight_prefetch,
                            args.gdn_z_fixed, args.probe_burst_rounds,
                            args.probe_burst_slot, args.ffn_down_slices,
                            args.attn_spread, args.attn_reuse_norm,
                            args.redmule_matmul, args.ffn_dataflow,
                            args.probe_lead_groups, args.attn_supertile_rows,
                            args.proj_dataflow, args.attn_local,
                            args.attn_rm_accum, args.attn_fused_softmax,
                            args.attn_local_pipe, args.attn_rm_fold,
                            args.attn_tile_local, args.attn_isolate,
                            args.attn_drain, args.spatz_burst,
                            args.key_block, args.gu_slots, args.gu_panels,
                            args.dp_slots, args.ffn_gu_slices,
                            args.ffn_pipe, probe_phase=args.probe_phase,
                            ffn_vc=args.ffn_vc, ffn_vc_poll=args.ffn_vc_poll,
                            ffn_l2=args.ffn_l2, ffn_gu_rows=args.ffn_gu_rows,
                            ffn_down_rows=args.ffn_down_rows,
                            ffn_vc_dma_stress=args.ffn_vc_dma_stress,
                            attn_vc=args.attn_vc,
                            attn_vc_rows=args.attn_vc_rows,
                            attn_vc_orows=args.attn_vc_orows,
                            attn_vc_sweep=args.attn_vc_sweep,
                            attn_vc_panel=args.attn_vc_panel,
                            attn_vc_dual=args.attn_vc_dual,
                            attn_vc_direct_dma=args.attn_vc_direct_dma,
                            attn_vc_cores=args.attn_vc_cores,
                            attn_vc_row_unroll=args.attn_vc_row_unroll,
                            attn_vc_cache_gamma=args.attn_vc_cache_gamma,
                            attn_vc_max_rows=args.attn_vc_max_rows,
                            attn_sweep_rows=args.attn_sweep_rows,
                            attn_sweep_tile=args.attn_sweep_tile,
                            attn_sweep_mode=args.attn_sweep_mode,
                            attn_sweep_reuse_query=args.attn_sweep_reuse_query,
                            attn_sweep_register=args.attn_sweep_register,
                            attn_sweep_interleave=args.attn_sweep_interleave,
                            attn_sweep_tree_max=args.attn_sweep_tree_max,
                            attn_sweep_row_dma=args.attn_sweep_row_dma,
                            attn_sweep_lean=args.attn_sweep_lean,
                            attn_sweep_qscale=args.attn_sweep_qscale,
                            attn_sweep_exp2=args.attn_sweep_exp2,
                            attn_sweep_fast=args.attn_sweep_fast,
                            attn_sweep_tile_win=args.attn_sweep_tile_win,
                            attn_sweep_async=args.attn_sweep_async,
                            attn_sweep_late_drain=args.attn_sweep_late_drain,
                            attn_sweep_finish=args.attn_sweep_finish,
                            attn_sweep_overlap=args.attn_sweep_overlap)
                        key = (placement, mixer_rows, schedule)
                        if schedule == "serialized":
                            baseline = result.get("actual_hashes")
                        elif baseline is not None:
                            equal = baseline == result.get("actual_hashes")
                            result["schedule_bitwise_equal"] = equal
                            result["passed"] = result["passed"] and equal
                        if placement == "global":
                            global_hashes[key[1:]] = result.get(
                                "actual_hashes")
                        elif (placement == "replicated" and
                              global_hashes.get(key[1:])):
                            equal = (global_hashes[key[1:]] ==
                                     result.get("actual_hashes"))
                            result["placement_bitwise_equal"] = equal
                            result["passed"] = result["passed"] and equal
                        if mixer_rows == args.mixer_rows[0]:
                            reference[(placement, schedule)] = result.get(
                                "actual_hashes")
                        elif reference.get((placement, schedule)):
                            equal = (reference[(placement, schedule)] ==
                                     result.get("actual_hashes"))
                            result["mixer_bound_bitwise_equal"] = equal
                            result["mixer_bound_reference"] = (
                                args.mixer_rows[0])
                            result["passed"] = result["passed"] and equal
                        (directory / "result.json").write_text(
                            json.dumps(result, indent=2) + "\n")
                        results.append(result)
                        (root / "summary.json").write_text(
                            json.dumps(results, indent=2) + "\n")
                        print(json.dumps(result), flush=True)
                        if not (result["build_passed"] if args.build_only
                                else result["passed"]):
                            raise SystemExit(1)


if __name__ == "__main__":
    main()
