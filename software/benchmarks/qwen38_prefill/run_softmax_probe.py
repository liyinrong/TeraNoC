# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Build isolated Spatz qualification ELFs and require numerical DPI success.

Run under conda teranoc. GVSoC subprocesses use conda gvsoc. No RTL jobs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import numpy as np


HERE = Path(__file__).resolve().parent
RTL = HERE.parents[2]
PROFILES = {
    # The 1 + 3 S1 tile (config/dse_s1_winner.mk): core 0 is a Snitch-only
    # scheduler, cores 1-3 Snitch-Spatz; the model profile is plan B as is.
    "dse_s1_winner": ("dse_s1_winner", 256),
    "dse_s2_winner": ("dse_s2_winner", 128),
    "dse_s1_8x8_winner": ("dse_s1_winner_8x8", 512),
    "dse_s2_8x8_winner": ("dse_s2_winner_8x8", 512),
    # Lane-first DSE points (plan section 3). The profile name is the RTL
    # flavor (config/dse_s2_winner_f*_v*.mk), which is what compiles the guest;
    # the MODEL side is the winner profile plus target parameters, see LANE.
    # `dse_s2_winner` itself is the f4/v512 corner.
    "dse_s2_winner_f8_v512": ("dse_s2_winner", 128),
    "dse_s2_winner_f4_v1024": ("dse_s2_winner", 128),
    "dse_s2_winner_f8_v1024": ("dse_s2_winner", 128),
    # Four Spatz per tile (256 cores), still one RedMulE per tile.
    "dse_s2_winner_c4_f8_v512": ("dse_s2_winner", 256),
    "dse_s2_winner_c4_f8_v1024": ("dse_s2_winner", 256),
    "dse_s1_winner_b32_c4_f4_v512": ("dse_s1_winner", 256),
    # Two Spatz x 8 FPUs per tile (lane trial against c4_f4).
    "dse_s1_winner_b32_f8_v512": ("dse_s1_winner", 128),
    "dse_s1_winner_b32_f8_v1024": ("dse_s1_winner", 128),
    # Reduced S1: 32 banks per tile at 8 KiB (config/dse_s1_winner_b32.mk).
    "dse_s1_winner_b32": ("dse_s1_winner", 128),
    # Reduced S1 with RedMulE ROB 16, then also 4-2-4-2 wide channels.
    "dse_s1_winner_b32_rob16": ("dse_s1_winner", 128),
    "dse_s1_winner_b32_rob16_w4": ("dse_s1_winner", 128),
}

# (FPUs per Spatz, VLEN bits) for the lane sweep, as TARGET PARAMETERS.
#
# Not as bespoke system configs: `qwen38_model.prepare_arch` replaces the
# vector unit from `qwen_fpus`/`qwen_vlen` (defaulting to 4 and 512) after the
# system config resolves, so a config that sets them is silently overridden and
# the run measures the winner's hardware while looking like it measured a
# variant. Three S=64 runs measured exactly 55,560,662 cycles that way.
LANE = {
    "dse_s2_winner_f8_v512": (8, 512),
    "dse_s2_winner_f4_v1024": (4, 1024),
    "dse_s2_winner_f8_v1024": (8, 1024),
    "dse_s2_winner_c4_f8_v512": (8, 512),
    "dse_s2_winner_c4_f8_v1024": (8, 1024),
    "dse_s1_winner_b32_c4_f4_v512": (4, 512),
    "dse_s1_winner_b32_f8_v512": (8, 512),
    "dse_s1_winner_b32_f8_v1024": (8, 1024),
}

# Profiles the model runs as is (qwen38_model.native_arch): the boot ROM stays
# at the config's bootrom_base, so the guest must not be linked for the moved
# lane-sweep ROM.
NATIVE = {"dse_s1_winner"}

# Spatz cores per tile as a TARGET PARAMETER, for the same reason as LANE.
SPATZ_PER_TILE = {"dse_s1_winner": 4,  # cores per tile; core 0 is scalar
                  "dse_s2_winner_c4_f8_v512": 4,
                  "dse_s2_winner_c4_f8_v1024": 4,
                  "dse_s1_winner_b32_c4_f4_v512": 4}

# Banks per tile at constant L1 bytes, as a TARGET PARAMETER (same reason).
BANKS_PER_TILE = {"dse_s1_winner_b32": 32,
                  "dse_s1_winner_b32_c4_f4_v512": 32,
                  "dse_s1_winner_b32_f8_v512": 32,
                  "dse_s1_winner_b32_f8_v1024": 32,
                  "dse_s1_winner_b32_rob16": 32,
                  "dse_s1_winner_b32_rob16_w4": 32}
# RedMulE ROB depth and wide RO request channels, as TARGET PARAMETERS.
REDMULE_ROB = {"dse_s1_winner_b32_rob16": 16, "dse_s1_winner_b32_rob16_w4": 16}
WIDE_RO = {"dse_s1_winner_b32_rob16_w4": 4}


GUEST_TYPE = {"float16": "__fp16", "float32": "float", "uint32": "uint32_t"}


def write_array(stream, name, value):
    dtype = GUEST_TYPE[value.dtype.name]
    stream.write(f"static {dtype} {name}[{value.size}] "
                 '__attribute__((section(".l2"), '
                 'aligned(4 * NUM_BANKS))) = {\n')
    flattened = value.ravel()
    # Floats go out as hex literals so no decimal round trip can move a bit;
    # indices are plain unsigned.
    if value.dtype.name == "uint32":
        def literal(v):
            return f"{int(v)}u"
    else:
        def literal(v):
            return float(v).hex() + "f"
    for start in range(0, value.size, 8):
        stream.write(",".join(literal(v) for v in flattened[start:start + 8]))
        stream.write(",\n")
    stream.write("};\n")


def generate_header(path, cores, columns, block):
    rng = np.random.default_rng(20260921)
    scores = rng.normal(0, 2, (cores, columns)).astype(np.float16)
    scores[:, -1] = scores.max(axis=1) + 2
    scores[::4] = np.float16(0.25)
    scores[1::4] -= np.float16(32)
    maximum = scores.astype(np.float64).max(axis=1)
    exponentials = np.exp(scores.astype(np.float64) - maximum[:, None])
    denominator = exponentials.sum(axis=1)
    expected = (exponentials / denominator[:, None]).astype(np.float16)
    statistics = np.stack((maximum, denominator), axis=1).astype(np.float32)
    stride = ((block + 31) // 32) * 32
    blocks = (columns + block - 1) // block
    storage = stride * blocks
    packed_scores = np.full((cores, storage), -65504, np.float16)
    packed_expected = np.zeros((cores, storage), np.float16)
    for tile in range(blocks):
        start = tile * block
        count = min(block, columns - start)
        packed_scores[:, tile * stride:tile * stride + count] = (
            scores[:, start:start + count])
        packed_expected[:, tile * stride:tile * stride + count] = (
            expected[:, start:start + count])
    with path.open("w") as stream:
        stream.write(
            f"#define Q38_COLUMNS {columns}u\n#define Q38_BLOCK {block}u\n")
        stream.write(f"#define Q38_STRIDE {stride}u\n")
        stream.write(f"#define Q38_STORAGE {storage}u\n")
        write_array(stream, "scores_init", packed_scores)
        write_array(stream, "probabilities_expected", packed_expected)
        write_array(stream, "statistics_expected", statistics)
        stream.write(f"static __fp16 probabilities_result[{cores * storage}] "
                     '__attribute__((section(".l2_bss"), '
                     'aligned(4 * NUM_BANKS)));\n')
        stream.write(f"static __fp16 scores_result[{cores * storage}] "
                     '__attribute__((section(".l2_bss"), '
                     'aligned(4 * NUM_BANKS)));\n')
        stream.write(f"static float statistics_result[{cores * 2}] "
                     '__attribute__((section(".l2_bss"), aligned(64)));\n')


def execute(command, directory, logfile, timeout, environment=None):
    (directory / (logfile + ".command.json")).write_text(
        json.dumps(command, indent=2) + "\n")
    started = time.monotonic()
    with (directory / logfile).open("w") as stream:
        process = subprocess.Popen(command, cwd=directory, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    return code, time.monotonic() - started


def evaluate_log(log, returncode, cores, columns, block=67):
    checks = re.findall(r"\[DPI_CHECK\] (\d+) ERRORS out of (\d+) CHECKS", log)
    storage = ((block + 31) // 32) * 32 * ((columns + block - 1) // block)
    expected = cores * (2 * storage + 2)
    checked = any(int(errors) == 0 and int(count) == expected
                  for errors, count in checks)
    eoc = bool(re.search(
        r"EOC register return value: 0x0\b|\[EOC\].*retval = 0\)", log))
    summary = re.search(r"\[qwen38_softmax\] cores=(\d+) columns=(\d+) "
                        r"block=(\d+) cycles=(\d+) errors=(\d+)", log)
    clean = not re.search(r"[1-9]\d* ERRORS|\[WATCHDOG\]|Fatal|Traceback", log)
    passed = bool(returncode == 0 and checked and eoc and clean and summary
                  and int(summary[1]) == cores and int(summary[2]) == columns
                  and int(summary[5]) == 0)
    return {"passed": passed, "checked_elements": expected if checked else 0,
            "eoc": eoc, "cycles": int(summary[4]) if summary else None}


def run(output, model_install, profile, columns, block, timeout):
    model, cores = PROFILES[profile]
    directory = output / profile
    directory.mkdir(parents=True, exist_ok=False)
    header = directory / "softmax_data.h"
    generate_header(header, cores, columns, block)
    command = ["make", "-f", str(HERE / "Makefile"), "softmax_probe.elf",
               f"MEMPOOL_DIR={RTL}", f"config={profile}", "COMPILER=llvm"]
    build_code, _ = execute(command, directory, "build.log", timeout)
    result = {"profile": profile, "model_config": model, "cores": cores,
              "columns": columns, "block": block,
              "build_returncode": build_code,
              "passed": False, "scope": "online-softmax primitive only",
              "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest()}
    if build_code == 0:
        binary = directory / "softmax_probe.elf"
        result["elf_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        command = ["conda", "run", "--no-capture-output", "-n", "gvsoc",
                   str(model_install / "bin/gvrun"),
                   "--target=teranoc_v2:system_config=" + model,
                   "--parameter=binary=" + str(binary),
                   "--work-dir=" + str(directory), "run"]
        code, elapsed = execute(command, directory, "model.log", timeout)
        result.update(returncode=code, elapsed_seconds=elapsed)
        result.update(evaluate_log((directory / "model.log").read_text(),
                                   code, cores, columns, block))
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-install", type=Path,
                        help="Matching install; default: model-root/install")
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES),
                        default=list(PROFILES))
    parser.add_argument("--columns", type=int, default=197)
    parser.add_argument("--block", type=int, default=67)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if min(args.columns, args.block, args.timeout) <= 0:
        parser.error("columns, block and timeout must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = list(HERE.glob("*.py")) + \
        list(HERE.glob("*.c")) + [HERE / "Makefile"]
    kernels = RTL / "software/kernels/baremetal"
    sources += [kernels / "mempool_qwen38_online_softmax.h",
                kernels / "mempool_llm_ffn_f16.h"]
    provenance = {
        str(path.relative_to(RTL)):
        hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (output / "sources.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    revisions = {}
    for name, root, environment in (
            ("rtl", RTL, "teranoc"), ("model_meta", args.model_root, "gvsoc"),
            ("model_core", args.model_root / "core", "gvsoc"),
            ("model_pulp", args.model_root / "pulp", "gvsoc"),
            ("model_engine", args.model_root / "engine", "gvsoc")):
        prefix = ["conda", "run", "-n", environment, "git", "-C", str(root)]
        sha = subprocess.check_output(
            prefix + ["rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(
            prefix + ["status", "--short"], text=True)
        diff = subprocess.check_output(prefix + ["diff", "HEAD"])
        revisions[name] = {"root": str(root.resolve()), "sha": sha,
                           "status": status,
                           "diff_sha256": hashlib.sha256(diff).hexdigest()}
    (output / "revisions.json").write_text(
        json.dumps(revisions, indent=2) + "\n")
    model_install = (args.model_install or args.model_root /
                     "install").resolve()
    results = []
    for profile in args.profiles:
        result = run(output, model_install, profile,
                     args.columns, args.block, args.timeout)
        results.append(result)
        print(json.dumps(result), flush=True)
        (output / "summary.json").write_text(
            json.dumps(results, indent=2) + "\n")
        if not result["passed"]:
            break
    raise SystemExit(0 if len(results) == len(args.profiles)
                     and all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
