# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Generate a fixture once and reuse it across the cases that share it.

At full model geometry one fixture is tens of minutes of host arithmetic and
three quarters of a gigabyte of data sidecar, while a sweep runs that same
fixture across profiles, schedules and window bounds -- none of which change
a single emitted byte. The cache key is every parameter that does change
them; `chunk` is excluded because it only moves the cost around.
"""

import hashlib
import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path

from layer_fixture import generate_layer_header

# Parameters that change how long generation takes but not what it produces.
COST_ONLY = ("chunk",)


def key(parameters):
    """Stable digest of the parameters that determine the emitted bytes."""
    material = {}
    for name, value in parameters.items():
        if name in COST_ONLY:
            continue
        material[name] = asdict(value) if is_dataclass(value) else value
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, default=str).encode()
    ).hexdigest()[:32]


def _place(source, target):
    """Link the cached file into the case, copying only if linking fails."""
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def build(directory, parameters, cache=None):
    """The fixture for `parameters`, materialized in `directory`.

    Without a cache this is `generate_layer_header`. With one, the fixture is
    generated into a staging slot and moved into place afterwards, so an
    interrupted generation is never mistaken for a hit.
    """
    directory = Path(directory)
    if not cache:
        return generate_layer_header(directory, **parameters)
    slot = Path(cache) / key(parameters)
    if not (slot / "fixture.json").exists():
        staging = slot.with_name(slot.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        generate_layer_header(staging, **parameters)
        (staging / "parameters.json").write_text(json.dumps(
            {name: (asdict(value) if is_dataclass(value) else value)
             for name, value in parameters.items()},
            indent=2, sort_keys=True, default=str) + "\n")
        try:
            staging.rename(slot)
        except OSError:
            # Someone else finished the same fixture first; theirs is the
            # same bytes, so take it and drop ours.
            shutil.rmtree(staging, ignore_errors=True)
    for item in sorted(slot.iterdir()):
        if item.name != "parameters.json":
            _place(item, directory / item.name)
    return json.loads((directory / "fixture.json").read_text())
