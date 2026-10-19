# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Small complete-layer guest fixtures, with explicit physical padding.

The resident baseline intentionally rejects full model dimensions until the
streaming/backing gate is implemented. Weight matrices are column-panel major.
"""

import io
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from backing import DataElf
from bounded_reference import evaluate, pack_kv_cache, pad_rows
from reference import (TOY, attention_layer, gdn_layer, linear,
                       linear_f16_fma, make_fixture, weight_dims)
from sampling import describe_runs, expand_runs, sample_runs
from run_softmax_probe import write_array


def check_shapes(kind, g, rows, tokens, block=64):
    """Logical shape and boundary hints per checked tensor.

    The shapes are what the tensor means, not how it is stored, so the
    sampler can straddle the real/padded token split, the last head and the
    tail of each block rather than taking an arbitrary prefix.
    """
    shapes = {"output": ((rows, g.d), {0: [tokens]})}
    if kind == "attention":
        shapes["context"] = ((g.q_heads, rows, g.head_dim), {1: [tokens]})
        # Block-major K-transpose/V: one row per key block.
        width = 2 * g.kv_heads * g.head_dim * block
        shapes["cache"] = ((rows // block, width), {0: [tokens // block]})
        shapes["mixer"] = ((rows, g.d), {0: [tokens]})
    else:
        shapes["context"] = ((g.value_heads, rows, g.value_dim), {1: [tokens]})
        shapes["conv"] = ((g.conv_width, g.qkv_dim), {})
        shapes["recurrent"] = ((g.value_heads, g.key_dim, g.value_dim), {})
    return shapes


def pad_columns(n, panel=64):
    return ((n + panel - 1) // panel) * panel


def panel_of(name, panel_columns=64, panel_overrides=None):
    """Panel width of weight `name`: its override, else the layer default."""
    return (panel_overrides or {}).get(name, panel_columns)


def layer_constants(kind, g, tokens, rows, panel_columns=64,
                    panel_overrides=None):
    """Guest geometry defines, derived from declared shapes so that they can
    be produced for geometries too large to materialize."""
    dims = weight_dims(kind, g)
    return {
        "KIND": int(kind == "gdn"), "S": tokens, "R": rows,
        "D": g.d, "I": g.intermediate, "QH": g.q_heads,
        "KVH": g.kv_heads, "HD": g.head_dim, "ROT": g.rotary_dim,
        "NK": g.key_heads, "NV": g.value_heads, "KD": g.key_dim,
        "VD": g.value_dim, "QDIM": g.q_dim, "KVDIM": g.kv_dim,
        "GDIM": g.gdn_value_dim, "QKVDIM": g.qkv_dim,
        "MAX_WIDTH": max(g.intermediate, 2 * g.q_dim, g.qkv_dim),
        "MAX_WEIGHT": max(
            k * pad_columns(n, panel_of(name, panel_columns, panel_overrides))
            for name, (k, n) in dims.items()),
        "MAX_K": max(k for k, _ in dims.values()),
    }


def pack_weight(weight, panel=64):
    k, n = weight.shape
    padded = np.zeros((k, ((n + panel - 1) // panel) * panel), np.float16)
    padded[:, :n] = weight
    return padded.reshape(k, -1, panel).transpose(1, 0, 2).copy()


# Beyond this the weights stop being a C initializer and have to be placed
# in the data sidecar instead.
INLINE_LIMIT = 64 * 1024 * 1024

# Written out for checking only: the layer consumes it in L1 and throws it
# away, so in sampled mode it never needs to exist in L2 at all.
DIAGNOSTIC = ("context",)


def golden_tensors(kind, rows, expected, state, intermediate,
                   packed_cache=True, key_block=64):
    """Whole-tensor goldens in the guest's physical layout.

    Shared with the test that qualifies `bounded_reference`, so a sampled
    reference and a whole-tensor one cannot drift apart in layout.
    """
    gold = {"output": pad_rows(expected, rows),
            "context": pad_rows(intermediate["context"],
                                rows).transpose(1, 0, 2).copy()}
    if kind == "attention":
        gold["cache"] = np.stack(
            [pad_rows(state[name], rows).transpose(1, 0, 2)
             for name in ("key", "value")])
        if packed_cache:
            gold["cache"] = pack_kv_cache(gold["cache"], key_block)
        gold["mixer"] = pad_rows(intermediate["mixer_residual"], rows)
    else:
        gold["conv"] = state["conv"][0]
        gold["recurrent"] = state["recurrent"][0]
    return gold


def backing_base(l2_mib, result_bytes, sidecar_bytes,
                 headroom=16 * 1024 * 1024, strict=True):
    """Where the data sidecar starts.

    It has to clear the executable and the result BSS, which are linked up
    from the L2 base, and it has to leave room for its own tensors below the
    aperture end. The historical constants are kept while they satisfy both,
    so reduced-geometry fixtures keep the addresses their archives recorded;
    full geometry, whose weights alone are 710 MiB, moves the base down.
    `strict=False` reports the floor instead of refusing, which is what the
    capacity probe wants: its job is to say how far over the map a geometry
    would be.
    """
    end = 0x80000000 + l2_mib * 1024 * 1024
    floor = (0x80000000 + headroom + result_bytes + 65535) & -65536
    legacy = 0x90000000 if l2_mib == 512 else 0xa0010000
    if floor <= legacy and legacy + sidecar_bytes <= end:
        return legacy
    if strict and floor + sidecar_bytes > end:
        raise ValueError("Layer does not fit the configured L2 aperture")
    return floor


def generate_layer_header(directory, kind, tokens=17, seed=20260921,
                          packed_cache=True, attention_kv_heads=1,
                          external_data=False, l2_mib=512,
                          check_samples=0, check_run_length=32,
                          geometry=None, bounded=None, chunk=256,
                          panel_columns=64, l2_scratch_bytes=0,
                          key_block=64, panel_overrides=None):
    """Emit the guest header, the data sidecar and the reference values.

    `geometry` defaults to the reduced `TOY` dimensions, where the whole layer
    is materialized and every checked tensor has a whole-tensor golden. Pass a
    full `Geometry` for the real model: there the whole layer costs hours and
    would not fit beside the weights, so the reference comes from
    `bounded_reference`, which evaluates only the sampled coordinates and is
    bit-identical to the whole-tensor one (`test_bounded_reference`).

    `panel_overrides` maps weight names to their own panel width (e.g. the
    f2 matrices of a full attention layer at 64 while its VC projections keep
    32); every other weight uses `panel_columns`.
    """
    if kind not in ("attention", "gdn"):
        raise ValueError("Unknown layer kind")
    reduced = geometry is None
    if bounded is None:
        bounded = not reduced
    limit = 768 if reduced else 4096
    if not 1 <= tokens <= limit:
        raise ValueError(f"S must be 1..{limit} for this geometry")
    if reduced and attention_kv_heads not in (1, 2):
        raise ValueError("Reduced attention supports one or two KV heads")
    if bounded and not check_samples:
        raise ValueError("A bounded reference only produces sampled values")
    if panel_columns not in (32, 64) or any(
            p not in (32, 64) for p in (panel_overrides or {}).values()):
        raise ValueError("Weight panels must have 32 or 64 columns")
    directory = Path(directory)
    g = geometry or (replace(TOY, kv_heads=attention_kv_heads,
                             q_heads=6 * attention_kv_heads)
                     if kind == "attention" else TOY)
    dims = weight_dims(kind, g)
    if set(panel_overrides or {}) - set(dims):
        raise ValueError("Panel overrides must name 2-D weights")
    inline = 2 * sum(
        k * pad_columns(n, panel_of(name, panel_columns, panel_overrides))
        for name, (k, n) in dims.items())
    if not external_data and inline > INLINE_LIMIT:
        # A 178 MiB C initializer is not a compilable thing; at real geometry
        # the weights only reach the guest through the data sidecar.
        raise ValueError("Weights this large must go in the data sidecar")
    rows = ((tokens + 63) // 64) * 64
    x, weights = make_fixture(kind, g, tokens=tokens, seed=seed)
    shapes = check_shapes(kind, g, rows, tokens, key_block)
    extent = {name: int(np.prod(shape)) for name, (shape, _) in shapes.items()}
    width = {name: (4 if name == "recurrent" else 2) for name in shapes}
    guest_type = {name: ("float" if name == "recurrent" else "__fp16")
                  for name in shapes}

    # The sampled coordinates are chosen before any arithmetic: a bounded
    # reference has to know which rows it owes before it can skip the rest.
    selection, samples = {}, {}
    for name, (shape, edges) in shapes.items():
        if not check_samples:
            continue
        starts, length = sample_runs(shape, min(check_samples, extent[name]),
                                     seed + len(name), edges, check_run_length)
        selection[name] = (shape, starts, length)
        samples[name] = describe_runs(shape, starts, length)
        np.save(directory / ("sample_runs_" + name + ".npy"), starts,
                allow_pickle=False)
    wanted = {name: (shape, expand_runs(starts, length))
              for name, (shape, starts, length) in selection.items()}

    gold = None
    if bounded:
        # Retain the independent mathematical result alongside the arithmetic
        # companion, at the sampled coordinates rather than everywhere.
        sample = evaluate(kind, g, tokens, rows, x, weights, wanted,
                          key_block=key_block, linear_op=linear_f16_fma,
                          chunk=chunk)
        ideal = evaluate(kind, g, tokens, rows, x, weights, wanted,
                         key_block=key_block, linear_op=linear, chunk=chunk)
        arithmetic_delta = {}
        for name, value in ideal.items():
            np.save(directory / ("ideal_sample_" + name + ".npy"), value,
                    allow_pickle=False)
            arithmetic_delta[name + "_max_abs"] = float(np.max(np.abs(
                sample[name].astype(np.float32) - value.astype(np.float32))))
    else:
        layer = attention_layer if kind == "attention" else gdn_layer
        kwargs = ({"online": True, "key_block": key_block}
                  if kind == "attention" else {})
        ideal, ideal_state, _ = layer(x, weights, g, **kwargs)
        expected, state, intermediate = layer(
            x, weights, g, linear_op=linear_f16_fma, **kwargs)
        np.save(directory / "ideal_output.npy", ideal, allow_pickle=False)
        arithmetic_delta = {"output_max_abs": float(np.max(np.abs(
            expected.astype(np.float32) - ideal.astype(np.float32))))}
        for name in state:
            np.save(directory / ("ideal_state_" + name + ".npy"),
                    ideal_state[name], allow_pickle=False)
            difference = (state[name].astype(np.float32) -
                          ideal_state[name].astype(np.float32))
            arithmetic_delta[name + "_max_abs"] = float(
                np.max(np.abs(difference)))
        gold = golden_tensors(kind, rows, expected, state, intermediate,
                              packed_cache, key_block)
        if set(gold) != set(shapes):
            raise ValueError("Every checked tensor needs a sampling shape")
        sample = {name: gold[name].ravel()[index]
                  for name, (_, index) in wanted.items()}

    matrices = {name: pack_weight(value, panel_of(name, panel_columns,
                                                  panel_overrides))
                for name, value in weights.items()
                if value.ndim == 2 and name != "conv"}
    constants = layer_constants(kind, g, tokens, rows, panel_columns,
                                panel_overrides)
    rope = {}
    if kind == "attention":
        angle = np.arange(rows)[:, None] * 10000000.0 ** (
            -np.arange(0, g.rotary_dim, 2) / g.rotary_dim)
        rope = {"rope_cos": np.cos(angle).astype(np.float16),
                "rope_sin": np.sin(angle).astype(np.float16)}

    def emit_header(stream, backing):
        def array(name, value):
            if backing:
                backing.add(name, value, stream)
            else:
                write_array(stream, name, value)

        stream.write("#define Q38_SAMPLED_CHECKS %d\n"
                     % int(bool(check_samples)))
        for name, value in constants.items():
            stream.write(f"#define Q38_{name} {value}u\n")
        array("input_init", pad_rows(x, rows))
        for name, weight in weights.items():
            array("w_" + name, matrices.get(name, weight))
        for name, value in rope.items():
            array(name, value)
        for name in shapes:
            dtype = guest_type[name]
            if check_samples:
                # Bounded checking: the guest gathers contiguous runs out of
                # its own result -- from L1 for the diagnostic tensor, from L2
                # for the rest -- so only the sampled reference has to exist.
                _, starts, length = selection[name]
                array("run_start_" + name, starts)
                array("gold_sample_" + name, sample[name])
                stream.write(f"#define Q38_RUNS_{name.upper()} "
                             f"{len(starts)}u\n")
                stream.write(f"#define Q38_RUNLEN_{name.upper()} {length}u\n")
                stream.write(f"static {dtype} result_sample_{name}"
                             f"[{len(starts) * length}] "
                             '__attribute__((section(".l2_bss"), '
                             'aligned(4 * NUM_BANKS)));\n')
                if name in DIAGNOSTIC:
                    continue
            else:
                array("gold_" + name, gold[name])
            stream.write(f"static {dtype} result_{name}[{extent[name]}] "
                         '__attribute__((section(".l2_bss"), '
                         'aligned(4 * NUM_BANKS)));\n')
        if "mixer" not in shapes:
            # The post-mixer residual stream is staged through L2 so its L1
            # footprint stays window bounded. Attention checks it against a
            # golden tensor; GDN only needs the buffer.
            stream.write(f"static __fp16 result_mixer[{rows * g.d}] "
                         '__attribute__((section(".l2_bss"), '
                         'aligned(4 * NUM_BANKS)));\n')

    result_bytes = sum(extent[name] * width[name] for name in shapes
                       if not (check_samples and name in DIAGNOSTIC))
    if "mixer" not in shapes:
        result_bytes += rows * g.d * 2
    result_bytes += sum(len(starts) * length * width[name]
                        for name, (_, starts, length) in selection.items())
    # Guest-side L2 BSS beyond the results, e.g. the GDN ready-queue spill.
    result_bytes += l2_scratch_bytes
    backing = None
    if external_data:
        # Place the tensors once against a throwaway base to learn how much
        # room they need, then choose a base that clears the result BSS.
        probe = DataElf(0x80010000, 2 ** 32)
        emit_header(io.StringIO(), probe)
        backing = DataElf(
            backing_base(l2_mib, result_bytes, probe.cursor - probe.base),
            0x80000000 + l2_mib * 1024 * 1024)
    with (directory / "layer_data.h").open("w") as stream:
        emit_header(stream, backing)
    external = (backing.write(directory / "layer_data.elf")
                if backing else None)
    manifest = {"external_data": external, "kind": kind,
                "tokens": tokens, "padded_rows": rows,
                "geometry": "toy" if reduced else "real", "seed": seed,
                "dimensions": asdict(g),
                "kv_layout": ("block-major K-transpose/V" if packed_cache
                              else "head-major"),
                "projection_arithmetic": "ascending-K FP16 FMA; X shift 0",
                "reference": "bounded" if bounded else "whole-tensor",
                "reference_chunk": chunk if bounded else 0,
                "arithmetic_delta_from_fp32_projection": arithmetic_delta,
                "layout": (f"{panel_columns}-column weight panels, "
                           "64-row physical padding"
                           + "".join(f", {name} {width}-column"
                                     for name, width in
                                     sorted((panel_overrides or {}).items()))),
                "result_bss_bytes": result_bytes,
                "check_mode": "sampled" if check_samples else "full",
                "check_run_length": check_run_length if check_samples else 0,
                "checked_elements": (
                    sum(entry["gathered"] for entry in samples.values())
                    if check_samples else sum(extent.values())),
                "samples": samples,
                "checks": {name: {"shape": list(shapes[name][0]),
                                  "dtype": ("float32" if name == "recurrent"
                                            else "float16")}
                           for name in shapes}}
    (directory / "fixture.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest
