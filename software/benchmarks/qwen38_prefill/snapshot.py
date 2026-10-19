# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Freeze build inputs against edits made during a run."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import re


def freeze_build_inputs(repository, output):
    repository = Path(repository).resolve()
    if Path(output).resolve().is_relative_to(repository):
        raise ValueError(
            "Place frozen build/result directories outside the repository")
    snapshot = Path(output) / "source"
    snapshot.mkdir()
    selected = []
    for folder in ("config", "software/runtime",
                   "software/benchmarks/qwen38_prefill",
                   "hardware/deps/redmule/sw"):
        for source in (repository / folder).rglob("*"):
            if source.is_file() and "__pycache__" not in source.parts:
                if (source.name == "arch.ld" or
                        source.suffix in (".o", ".elf", ".dump", ".pyc")):
                    continue
                selected.append(source)
    kernels = repository / "software/kernels/baremetal"
    selected.extend(kernels.glob("mempool_qwen38*.h"))
    selected.append(kernels / "mempool_llm_ffn_f16.h")
    hashes = {}
    for source in selected:
        relative = source.relative_to(repository)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = {"repository": str(repository),
                "git_sha": subprocess.check_output(
                    ["git", "-C", str(repository), "rev-parse", "HEAD"],
                    text=True).strip(),
                "files": hashes}
    (Path(output) / "sources.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    return snapshot


def record_model_install(install, output):
    """Identify installed DMA/engine binaries and available source state."""
    install, output = Path(install).resolve(), Path(output)
    manifest = {"install": str(install), "binaries": {}, "sources": {}}
    binaries = [install / "lib/libpulpvp.so", install / "bin/gvsoc_launcher"]
    binaries += [p for p in (install / "models").glob("*idma_group_cpp_*.so")
                 if not p.stem.endswith(("_debug", "_profile", "_asserts"))]
    for path in binaries:
        if path.is_file():
            manifest["binaries"][str(path.relative_to(install))] = (
                hashlib.sha256(path.read_bytes()).hexdigest())
    cache = install.parent / "build/CMakeCache.txt"
    home = (re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$",
                      cache.read_text(), re.MULTILINE)
            if cache.exists() else None)
    if home:
        source = Path(home[1])
        for name in ("meta", "core", "pulp", "engine"):
            repo = source if name == "meta" else source / name
            prefix = ["git", "-C", str(repo)]
            sha = subprocess.check_output(
                prefix + ["rev-parse", "HEAD"], text=True).strip()
            diff = subprocess.check_output(
                prefix + ["diff", "--binary", "HEAD"])
            (output / ("model-" + name + ".patch")).write_bytes(diff)
            manifest["sources"][name] = {
                "root": str(repo), "sha": sha,
                "diff_sha256": hashlib.sha256(diff).hexdigest()}
    (output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
