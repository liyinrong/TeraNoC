# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Lane-first sweep manifest, with fixed banks and two cores per tile.

These are requested configurations, not a claim that the model supports them
or that experiments ran. The runner must validate resolved configuration.
"""

import json


def configurations():
    for family in ("s1", "s2"):
        for mesh in (4, 8):
            tiles = mesh * mesh * 4
            base = f"dse_{family}_winner"
            rtl = base if mesh == 4 else f"dse_{family}_8x8_winner"
            model = base if mesh == 4 else base + "_8x8"
            for fpus in (4, 8):
                for vlen in (512, 1024):
                    yield {"name": f"{base}_{mesh}x{mesh}_f{fpus}_v{vlen}",
                           "rtl_config": rtl, "model_config": model,
                           "spatz_per_tile": 2, "n_fpu": fpus, "n_ipu": 4,
                           "vlen": vlen, "banking_factor": 32 // fpus,
                           "expected_banks_per_tile": 64,
                           "expected_cores": tiles * 2,
                           "expected_redmules": tiles,
                           "expected_l1_bytes": tiles * 64 * (
                               4096 if family == "s1" else 8192),
                           "status": "unqualified"}


if __name__ == "__main__":
    print(json.dumps({"version": 1, "tokens": [2048, 4096],
                      "layers": ["attention", "gdn"],
                      "configurations": list(configurations())}, indent=2))
