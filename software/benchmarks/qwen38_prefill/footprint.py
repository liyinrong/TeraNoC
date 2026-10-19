# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Full model geometry capacity probe: emit a data-free guest header.

Golden tensors for the real geometry cannot be produced by the host reference
at a useful cost — `linear_f16_fma` is sequential in K, so one full FFN down
projection is 17408 passes over a tokens-by-5120 array. This probe therefore
answers only the capacity questions, which need no data: how much interleaved
L1 a given profile and window bound reserves, and how much L2 the layer's
tensors occupy. It compiles; it does not run and it checks nothing numerical.
"""

import argparse
import io
import json
from pathlib import Path

from backing import DataElf, memory_layout
from layer_fixture import backing_base, layer_constants, pad_columns
from reference import Geometry, TOY, vector_dims, weight_dims


def tensor_sizes(kind, g, tokens, rows, panel_columns=64):
    """Every L2 tensor the guest header declares, in emission order."""
    sizes = {"input_init": (rows * g.d, "float16")}
    for name, (k, n) in sorted(weight_dims(kind, g).items()):
        sizes["w_" + name] = (k * pad_columns(n, panel_columns), "float16")
    for name, shape in vector_dims(kind, g).items():
        count = 1
        for extent in shape:
            count *= extent
        single = name in ("a_log", "dt_bias")
        sizes["w_" + name] = (count, "float32" if single else "float16")
    if kind == "attention":
        sizes["rope_cos"] = (rows * g.rotary_dim // 2, "float16")
        sizes["rope_sin"] = (rows * g.rotary_dim // 2, "float16")
    golds = {"output": (rows * g.d, "float16")}
    if kind == "attention":
        golds["context"] = (g.q_heads * rows * g.head_dim, "float16")
        golds["cache"] = (2 * rows * g.kv_dim, "float16")
        golds["mixer"] = (rows * g.d, "float16")
    else:
        golds["context"] = (g.value_heads * rows * g.value_dim, "float16")
        golds["conv"] = (g.conv_width * g.qkv_dim, "float16")
        golds["recurrent"] = (g.value_heads * g.key_dim * g.value_dim,
                              "float32")
    return sizes, golds


def generate_header(directory, kind, tokens, geometry=None, l2_mib=1024,
                    base=None, goldens=True, panel_columns=64):
    """Place every tensor and report the capacity verdict.

    The aperture is not enforced here: the point of the probe is to report how
    much L2 the geometry needs, including when that exceeds the configured
    map."""
    g = geometry or Geometry()
    rows = ((tokens + 63) // 64) * 64
    constants = layer_constants(kind, g, tokens, rows, panel_columns)
    sizes, golds = tensor_sizes(kind, g, tokens, rows, panel_columns)
    width = {"float16": 2, "float32": 4}
    results = sum(count * width[dtype] for count, dtype in golds.values())
    if "mixer" not in golds:
        results += rows * g.d * 2

    def place(stream, backing):
        lines = []
        for name, value in constants.items():
            stream.write(f"#define Q38_{name} {value}u\n")
        for name, (count, dtype) in sizes.items():
            backing.reserve(name, dtype, count * width[dtype], stream)
        for name, (count, dtype) in golds.items():
            # The guest always references its reference tensors, so they are
            # always declared. Whether their bytes belong in L2 is a separate
            # question: at full size they would be checked from host files.
            backing.reserve("gold_" + name, dtype, count * width[dtype],
                            stream)
            guest = "float" if dtype == "float32" else "__fp16"
            lines.append(f"static {guest} result_{name}[{count}] "
                         '__attribute__((section(".l2_bss"), '
                         'aligned(4 * NUM_BANKS)));')
        if "mixer" not in golds:
            lines.append(f"static __fp16 result_mixer[{rows * g.d}] "
                         '__attribute__((section(".l2_bss"), '
                         'aligned(4 * NUM_BANKS)));')
        stream.write("\n".join(lines) + "\n")

    probe = DataElf(0x80010000, 2 ** 32)
    place(io.StringIO(), probe)
    sidecar = probe.cursor - probe.base
    # The aperture is deliberately not enforced: the probe reports what the
    # geometry needs even when that exceeds the configured map.
    recommended = backing_base(l2_mib, results, sidecar, strict=False)
    if base is None:
        base = recommended
    backing = DataElf(base, 2 ** 32)
    with (Path(directory) / "layer_data.h").open("w") as stream:
        place(stream, backing)
    l2_size, rom = memory_layout(l2_mib)
    weights = sum(count * width[dtype]
                  for name, (count, dtype) in sizes.items()
                  if name.startswith("w_"))
    # The executable's text, rodata and result BSS are linked from the L2
    # base upwards and the sidecar is placed above them, so the requirement is
    # the sum, not the distance from L2 base to whatever base was passed.
    executable = 16 * 1024 * 1024
    golden = sum(count * width[dtype] for count, dtype in golds.values())
    needed = recommended - 0x80000000 + sidecar
    if not goldens:
        needed -= golden
    return {"kind": kind, "tokens": tokens, "padded_rows": rows,
            "geometry": g.__dict__, "constants": constants,
            "l2_mib": l2_mib, "l2_bytes": l2_size, "bootrom_base": rom,
            "goldens_in_l2": goldens, "golden_bytes": golden,
            "backing_base": base, "backing_end": backing.cursor,
            "weight_bytes": weights,
            "backing_bytes": backing.cursor - base,
            "input_and_reference_bytes": backing.cursor - base - weights,
            "result_bss_bytes": results,
            "executable_headroom_bytes": executable,
            "recommended_backing_base": recommended,
            "base_is_high_enough": base >= recommended,
            "l2_bytes_needed": needed, "fits_configured_l2": needed <= l2_size,
            "panel_columns": panel_columns,
            "scope": "capacity probe only; no data and no numerical check"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("attention", "gdn"),
                        default="attention")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--geometry", choices=("real", "toy"), default="real")
    parser.add_argument("--l2-mib", type=int, choices=(512, 1024),
                        default=1024)
    parser.add_argument("--backing-base", type=lambda v: int(v, 0),
                        help="sidecar base; defaults to the fixture's choice")
    parser.add_argument("--no-goldens", action="store_true",
                        help="exclude reference tensors from the L2 total, "
                             "as a full-size run checking bounded samples "
                             "from host files would")
    parser.add_argument("--panel-columns", type=int, choices=(32, 64),
                        default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = generate_header(
        args.output, args.kind, args.tokens,
        None if args.geometry == "real" else TOY, args.l2_mib,
        args.backing_base, not args.no_goldens, args.panel_columns)
    (args.output / "footprint.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
