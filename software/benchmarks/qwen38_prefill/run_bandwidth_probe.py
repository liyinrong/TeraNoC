# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Short isolated VLSU measurements with fixed caches and ROB controls."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from snapshot import freeze_build_inputs


HERE = Path(__file__).resolve().parent
POINTS = (("s2", 4, 1024, 32), ("s2", 8, 1024, 32),
          ("s2", 8, 1024, 64), ("s1", 4, 512, 16), ("s1", 4, 512, 32))


def execute(command, directory, name, environment):
    (directory / (name + ".command.json")).write_text(
        json.dumps(command, indent=2) + "\n")
    start = time.monotonic()
    with (directory / name).open("w") as log:
        process = subprocess.Popen(command, cwd=directory, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            # gvrun owns a launcher child. Only signal this invocation's
            # descendants, identified by parent relationships and start time.
            pairs = [tuple(map(int, line.split())) for line in
                     subprocess.check_output(["ps", "-eo", "pid,ppid"],
                                             text=True).splitlines()[1:]]
            owned, pending = [], [process.pid]
            while pending:
                parent = pending.pop()
                for pid, ppid in pairs:
                    if ppid == parent:
                        owned.append(pid)
                        pending.append(pid)
            for pid in owned[::-1] + [process.pid]:
                path = Path("/proc") / str(pid)
                try:
                    born = path.joinpath("stat").read_text().split()[21]
                    if (path.stat().st_uid == os.getuid() and
                            path.joinpath("stat").read_text().split()[21] ==
                            born):
                        os.kill(pid, signal.SIGTERM)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            process.wait(timeout=10)
            code = 124
    return code, time.monotonic() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--tile-local", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = freeze_build_inputs(args.repo, args.output)
    environment = os.environ.copy()
    results = []
    for arch, fpus, vlen, rob in POINTS:
        directory = args.output / f"{arch}-f{fpus}-v{vlen}-rob{rob}"
        directory.mkdir()
        profile = ("dse_s2_winner_c4_f8_v1024" if arch == "s2" else
                   "dse_s1_winner_b32_c4_f4_v512")
        banks = 64 if arch == "s2" else 32
        makefile = source / "software/benchmarks/qwen38_prefill/Makefile"
        command = ["make", "-f", str(makefile), "bandwidth_probe.elf",
                   f"MEMPOOL_DIR={source}",
                   f"INSTALL_DIR={args.repo / 'install'}",
                   f"config={profile}", f"n_fpu={fpus}", f"vlen={vlen}",
                   f"banking_factor={banks // (4 * fpus)}",
                   f"APP_DEFINES=-DBW_TILE_LOCAL={int(args.tile_local)}",
                   f"spatz_rob_depth={rob}", "spatz_block_alloc=1",
                   f"spatz_max_burst_words={vlen // 32}"]
        code, _ = execute(command, directory, "build.log", environment)
        result = {"arch": arch, "fpus": fpus, "vlen": vlen, "rob": rob,
                  "build_returncode": code, "passed": False}
        result["placement"] = "tile" if args.tile_local else "group"
        if not code:
            binary = directory / "bandwidth_probe.elf"
            result["elf_sha256"] = hashlib.sha256(
                binary.read_bytes()).hexdigest()
            target = ("q38_bandwidth_model:system_config=dse_" + arch +
                      f"_winner:virtual_cluster=true:f={fpus}:v={vlen}:"
                      f"r={rob}")
            command = [str(args.install / "bin/gvrun"), "--target=" + target,
                       "--target-dir=" + str(args.target_dir),
                       "--parameter=binary=" + str(binary),
                       "--work-dir=" + str(directory), "run"]
            code, elapsed = execute(command, directory, "model.log",
                                    environment)
            log = (directory / "model.log").read_text()

            def contracts(node):
                if isinstance(node, dict):
                    if "bandwidth_contract" in node:
                        yield node["bandwidth_contract"]
                    for value in node.values():
                        yield from contracts(value)
                elif isinstance(node, list):
                    for value in node:
                        yield from contracts(value)

            contract = next(contracts(json.loads(
                (directory / "gvsoc_config.json").read_text())), {})
            result["contract"] = contract
            rows = [dict(zip(("mode", "bytes", "repeats", "cycles",
                              "errors", "ports", "vlen"), map(int, row)))
                    for row in re.findall(
                        r"\[bandwidth_probe\] mode=(\d+) bytes=(\d+) "
                        r"repeats=(\d+) cycles=(\d+) errors=(\d+) "
                        r"ports=(\d+) vlen=(\d+)", log)]
            for row in rows:
                directions = 2 if row["mode"] < 2 else 1
                row["payload_bytes_per_cycle"] = (
                    directions * row["bytes"] * row["repeats"] /
                    row["cycles"] if row["cycles"] > 0 else None)
            result.update(returncode=code, elapsed_seconds=elapsed, rows=rows)
            result["passed"] = (code == 0 and len(rows) == 4 and all(
                row["errors"] == 0 and row["cycles"] > 0 and
                row["bytes"] == 2048 and row["mode"] in range(4) and
                row["ports"] == fpus and
                row["vlen"] == vlen for row in rows))
            expected = {"fpus": fpus, "ports": fpus, "vlen": vlen,
                        "rob": rob, "block_alloc": 1,
                        "burst_words": vlen // 32, "cores": 256,
                        "banks_per_tile": banks, "icache_bytes": 16384,
                        "icache_line_bytes": 256}
            result["contract_agrees"] = all(
                contract.get(k) == v for k, v in expected.items())
            result["passed"] &= result["contract_agrees"]
        (directory / "result.json").write_text(
            json.dumps(result, indent=2) + "\n")
        results.append(result)
        (args.output / "summary.json").write_text(
            json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
