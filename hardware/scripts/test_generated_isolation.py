#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

"""Generate two topologies concurrently and check their Bender source paths.

Run from the teranoc environment. Requires the installed Bender and FlooGen
dependencies, but no simulator license. Temporary outputs are removed on exit.
"""

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor


ROOT = Path(__file__).resolve().parents[2]
HARDWARE = ROOT / "hardware"
PACKAGES = ("floo_terapool_noc_pkg.sv", "l2_bank_map_pkg.sv")


def shared_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (HARDWARE / "generated").glob("*.sv")}


def run(*args, env=None):
    result = subprocess.run(args, cwd=HARDWARE, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout


def generate(build, config):
    args = ("make", "--no-print-directory", "-j4", "update-floogen",
            f"config={config}", f"buildpath={build}")
    run(*args)
    generated = build / "generated" / config
    for name in PACKAGES:
        assert (generated / name).is_file(), name

    # Exercise the environment contract used by every Bender script backend.
    env = dict(os.environ,
               MEMPOOL_GENERATED_DIR=os.path.relpath(generated, ROOT))
    bender = str(ROOT / "install" / "bender" / "bender")
    files = run(bender, "script", "flist", "-t", "rtl", env=env)
    paths = {Path(line).resolve() for line in files.splitlines()
             if line.endswith(".sv")}
    for name in PACKAGES:
        assert generated / name in paths, files
        assert HARDWARE / "generated" / name not in paths, files

    for backend in ("vsim", "vcs", "verilator"):
        script = run(bender, "script", backend, "-t", "rtl", env=env)
        for name in PACKAGES:
            assert f'{env["MEMPOOL_GENERATED_DIR"]}/{name}' in script
            assert f"/hardware/generated/{name}" not in script

    before = {name: (generated / name).stat().st_mtime_ns
              for name in PACKAGES}
    run(*args)
    run(*args, "-n")
    after = {name: (generated / name).stat().st_mtime_ns
             for name in PACKAGES}
    assert before == after, "unchanged generation must be incremental"
    return (generated / PACKAGES[0]).read_bytes()


def main():
    before = shared_hashes()
    with tempfile.TemporaryDirectory(prefix="build_isolation_",
                                     dir=HARDWARE) as directory:
        build = Path(directory)
        with ThreadPoolExecutor(max_workers=2) as executor:
            jobs = [executor.submit(generate, build / config, config)
                    for config in ("minpool", "tensorpool256_spatz4")]
            packages = [job.result() for job in jobs]
        assert packages[0] != packages[1], "test needs different topologies"
        # An endpoint mismatch must fail again on retry, rather than reusing
        # the output left behind by the failed generator recipe.
        invalid = build / "invalid"
        for _ in range(2):
            try:
                run("make", "update-floogen", "config=minpool",
                    "num_groups=1", f"buildpath={invalid}")
            except RuntimeError as error:
                assert "ERROR: floo_terapool_noc_pkg" in str(error)
            else:
                raise AssertionError("invalid topology was accepted")
            assert not (invalid / "generated" / "minpool" /
                        PACKAGES[0]).exists()
    assert shared_hashes() == before, "shared generated sources changed"
    print("PASS: isolated generation, source paths, incremental and dry runs")


if __name__ == "__main__":
    main()
