# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""External, hashed input/reference packs; no generated megabyte C headers."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from reference import (TOY, Geometry, attention_layer, gdn_layer, make_fixture)


def save_tensor(directory, name, value):
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite tensor: {name}")
    dtype = array.dtype.newbyteorder("<")
    data = np.ascontiguousarray(array, dtype=dtype).tobytes()
    path = directory / f"{name}.bin"
    path.write_bytes(data)
    return {"file": path.name, "shape": list(array.shape), "dtype": dtype.str,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def load_tensor(directory, description):
    path = Path(directory) / description["file"]
    data = path.read_bytes()
    if len(data) != description["bytes"]:
        raise ValueError(f"Incorrect byte count: {path}")
    if hashlib.sha256(data).hexdigest() != description["sha256"]:
        raise ValueError(f"Hash mismatch: {path}")
    array = np.frombuffer(data, dtype=np.dtype(description["dtype"]))
    return array.reshape(description["shape"])


def generate(output, kind, geometry=TOY, tokens=17, batch=1, seed=20260921):
    output = Path(output)
    # Existing packs are immutable, including failed/incomplete generations.
    output.mkdir(parents=True, exist_ok=False)
    x, weights = make_fixture(kind, geometry, tokens, batch, seed)
    layer = attention_layer if kind == "attention" else gdn_layer
    kwargs = {"online": True} if kind == "attention" else {}
    expected, state, intermediates = layer(x, weights, geometry, **kwargs)
    tensors = {"input": x, "output": expected}
    tensors.update({f"weight_{name}": value for name,
                   value in weights.items()})
    tensors.update({f"state_{name}": value for name, value in state.items()})
    tensors.update({f"intermediate_{name}": value
                    for name, value in intermediates.items()})
    manifest = {"version": 1, "kind": kind, "geometry": asdict(geometry),
                "batch": batch, "tokens": tokens, "seed": seed,
                "weights": "synthetic", "positions": "text-only, start=0",
                "reference": "FP16 interfaces; "
                             "FP32 linear/recurrent arithmetic; "
                             "FP64 attention core; not bit-exact RedMulE",
                "tensors": {name: save_tensor(output, name, value)
                            for name, value in tensors.items()}}
    manifest["total_bytes"] = sum(t["bytes"]
                                  for t in manifest["tensors"].values())
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--kind", choices=("attention", "gdn"), required=True)
    parser.add_argument("--geometry", choices=("toy", "real"), default="toy")
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    manifest = generate(args.output, args.kind,
                        TOY if args.geometry == "toy" else Geometry(),
                        args.tokens, args.batch, args.seed)
    print(json.dumps({"manifest": str(args.output / "manifest.json"),
                      "bytes": manifest["total_bytes"]}))


if __name__ == "__main__":
    main()
