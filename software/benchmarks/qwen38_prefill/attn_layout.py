# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Phase-specific attention allocations, matching av_setup's bank-row map."""

import re
import subprocess


def max_projection_geometry(profile, dimensions, tokens, probe_rows=0):
    """Largest target tile/batch within the requested execution extent.

    Full-geometry linked audits rule out the next power-of-two tile. Smaller
    numerical fixtures may need a smaller tile to supply all output engines.
    """
    if profile not in ("dse_s1_winner_b32", "dse_s2_winner",
                       "dse_s2_winner_f8_v1024",
                       "dse_s1_winner_b32_c4_f4_v512",
                       "dse_s1_winner_b32_f8_v512",
                       "dse_s1_winner_b32_f8_v1024",
                       "dse_s2_winner_c4_f8_v1024"):
        raise ValueError("Maximum rows need a qualified S1/S2 profile")
    height = 8 if profile.startswith("dse_s1_") else 16
    tile = height * 4
    extent = (tokens + 63) // 64 * 64
    if probe_rows:
        extent = min(extent, probe_rows // 2)
    columns = 64 if height == 16 else 32
    output_panels = dimensions["d"] // columns
    while True:
        batch = 16 * tile
        while batch > extent or extent % batch:
            batch //= 2
            if batch < tile or batch < 64:
                raise ValueError("Execution extent cannot hold a VC batch")
        if batch // tile * output_panels >= 64 or tile == height:
            return tile, batch
        tile //= 2


def projection_layout(floor, capacity, banks, groups, engines, dimensions,
                      rows, tile, columns, output=False, panel_aware=True):
    """Projection owns no sweep KV ring; its reusable arena starts at floor."""
    group_row = banks // groups * 4
    global_row = banks * 4
    vc_row = 4 * group_row
    row_tiles = rows // tile
    engines_per_vc = engines // (groups // 4)
    if rows % tile or not row_tiles or engines_per_vc % row_tiles:
        raise ValueError("Projection batch cannot cover whole VC row tiles")
    pairs = engines_per_vc // row_tiles
    d, hd = dimensions["d"], dimensions["head_dim"]
    contraction = dimensions["q_heads"] * hd if output else d
    panels = 1 if output else hd // columns
    if hd % columns:
        raise ValueError("Head width must contain whole output panels")

    def align(value, quantum):
        return (value + quantum - 1) // quantum * quantum

    cursor = align(floor, global_row)
    weights = 2 * align(pairs * contraction * columns * 2, vc_row)
    weights = weights // vc_row * global_row
    cursor += weights
    x = rows * contraction * 2
    residual = rows * d * 2 if output else 0
    cursor = align(cursor + x + residual, global_row)
    local_engines = engines // groups
    z = align(local_engines * 2 * panels * tile * columns * 2,
              group_row) // group_row * global_row
    cursor += z
    scratch, rotary, stage = 0, 0, 0
    if not output:
        if not panel_aware:
            scratch = align(local_engines * hd * 2, group_row)
            scratch = scratch // group_row * global_row
            cursor += scratch
        rotary = 2 * align(rows * dimensions["rotary_dim"], 256)
        cursor = align(cursor + rotary, global_row)
        stage = local_engines * 2 * tile * hd * 2
        if stage % group_row:
            raise ValueError("Shipping slots must occupy whole group rows")
        stage = stage // group_row * global_row
        cursor += stage
    return {"phase": "output" if output else "qkv", "rows": rows,
            "tile": tile, "columns": columns, "heads_per_vc": pairs,
            "weight_bytes": weights, "x_bytes": x,
            "residual_bytes": residual, "z_bytes": z,
            "scratch_bytes": scratch, "rotary_bytes": rotary,
            "stage_bytes": stage, "kv_bytes": 0,
            "end": cursor, "free": capacity - cursor,
            "fits": cursor <= capacity}


def sweep_layout(floor, capacity, banks, groups, engines, hd, tile, block,
                 reuse_query=False):
    """Only the sweep owns its Q/gate banks, score slots and four KV slots."""
    group_row, global_row = banks // groups * 4, banks * 4
    local_tiles = engines // groups
    rows = local_tiles * tile
    stripe = banks // engines * 4

    def align(value, quantum):
        return (value + quantum - 1) // quantum * quantum

    state_slots, alpha_slots = stripe // 8, stripe // 4
    state = align(align(tile, state_slots) * local_tiles * 8, group_row)
    alpha = align(align(tile, alpha_slots) * local_tiles * 4, group_row)
    vectors = 5 if reuse_query else 6
    local = (vectors * rows * hd * 2 + 2 * rows * block * 2 + state + alpha)
    local = align(local, group_row)
    kv = 4 * 2 * hd * block * 2 * (groups // 4)
    end = align(floor, global_row) + local * groups + kv
    return {"tile": tile, "query_rows": 8 * tile, "key_block": block,
            "group_bytes": local, "kv_bytes": kv, "end": end,
            "free": capacity - end, "fits": end <= capacity}


def projection_layout_check(directory, install, dimensions, rows, output_rows,
                            tile, columns, panel_aware=True, key_block=0,
                            sweep_tile=0):
    """Check linked arena bounds before launching even a build-only case."""
    readelf = install / "riscv-gcc/bin/riscv32-unknown-elf-readelf"
    text = subprocess.check_output(
        [str(readelf), "-sW", str(directory / "layer_resident.elf")],
        text=True)
    symbols = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0].rstrip(":").isdigit():
            symbols[fields[-1]] = int(fields[1], 16)
    floor = min(symbols["stage"], symbols["weights"])
    capacity = symbols["__heap_end"]
    build = (directory / "build.log").read_text()
    defines = dict(re.findall(r"-D(\w+)=(\d+)", build))
    banks, groups, engines = (int(defines[key]) for key in
                              ("NUM_BANKS", "NUM_GROUPS",
                               "NUM_REDMULE_TILES"))
    nominal = int(defines["REDMULE_H"]) * (int(defines["REDMULE_P"]) + 1)
    if tile is None:
        tile = nominal
    result = {"floor": floor, "capacity": capacity, "phases": [],
              "next_tile_candidates": []}
    for output, batch in [(False, rows), (True, output_rows or rows)]:
        phase = projection_layout(floor, capacity, banks, groups, engines,
                                  dimensions, batch, tile, columns, output,
                                  panel_aware)
        if not phase["fits"]:
            raise ValueError(f"Attention {phase['phase']} exceeds L1 by "
                             f"{-phase['free']} bytes")
        result["phases"].append(phase)
        for tiles in [1, 2, 4, 8, 16]:
            larger_rows = 2 * tile * tiles
            if larger_rows > dimensions.get("rows", 2**32):
                continue
            candidate = projection_layout(
                floor, capacity, banks, groups, engines, dimensions,
                larger_rows, 2 * tile, columns, output, panel_aware)
            result["next_tile_candidates"].append(candidate)
    if key_block:
        reuse_query = bool(int(defines.get("Q38_AV_SWEEP_REUSE_QUERY", "0")))
        owner = sweep_tile or min(nominal, key_block // 8)
        hd = dimensions["head_dim"]
        current = sweep_layout(floor, capacity, banks, groups, engines,
                               hd, owner, key_block, reuse_query)
        if not current["fits"]:
            raise ValueError("Attention sweep exceeds L1 capacity")
        result["sweep"] = current
        result["larger_sweep_owner"] = sweep_layout(
            floor, capacity, banks, groups, engines, hd, owner * 2, key_block,
            reuse_query)
        result["larger_sweep_key_block"] = sweep_layout(
            floor, capacity, banks, groups, engines, hd, owner, key_block * 2,
            reuse_query)
        result["peak"] = max([phase["end"] for phase in result["phases"]] +
                             [current["end"]])
    return result
