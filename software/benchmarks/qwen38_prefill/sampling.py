# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, stratified element selection for bounded checking.

At full model geometry a golden tensor can neither be produced at a useful
cost nor held in L2 beside the weights, so the guest checks a bounded set of
elements instead of whole tensors (plan §7). The set has to be chosen, not
taken from the front: an arbitrary prefix would miss the padding edge, the
last head and every tail column, which is where the tiling bugs live.

Every axis therefore contributes its first and last coordinates plus any
boundary the caller names -- the real/padded token split, a head start, a
block edge -- and the product of those seeds the set. A seeded draw fills the
remainder so that interior elements are covered too. The result is sorted,
unique, reproducible from `seed`, and independent of how the tensor is stored.
"""

import numpy as np


def axis_coordinates(extent, boundaries=()):
    """First, last and named coordinates of one axis, in range and unique."""
    if extent <= 0:
        raise ValueError("Axis extent must be positive")
    wanted = {0, extent - 1}
    if extent > 2:
        wanted.add(extent - 2)
    for edge in boundaries:
        # A boundary names the first element on its far side; the element just
        # before it is the interesting one, so both are taken.
        for coordinate in (edge - 1, edge):
            if 0 <= coordinate < extent:
                wanted.add(int(coordinate))
    return sorted(wanted)


def sample_elements(shape, count, seed, boundaries=None, edge_limit=4096):
    """Flat indices into a tensor of `shape`, stratified then filled.

    `boundaries` maps an axis index to extra coordinates on that axis.
    `edge_limit` caps the structured product so a high-rank tensor cannot
    make the set explode; the cap is applied deterministically.
    """
    shape = tuple(int(extent) for extent in shape)
    size = int(np.prod(shape))
    if count <= 0 or count > size:
        raise ValueError("Sample count must be within the tensor")
    boundaries = boundaries or {}
    if any(axis >= len(shape) for axis in boundaries):
        raise ValueError("Boundary names an axis the tensor does not have")

    grids = [axis_coordinates(extent, boundaries.get(axis, ()))
             for axis, extent in enumerate(shape)]
    edges = np.ravel_multi_index(
        tuple(grid.ravel() for grid in np.meshgrid(*grids, indexing="ij")),
        shape)
    edges = np.unique(edges)
    if edges.size > min(edge_limit, count):
        keep = min(edge_limit, count)
        # Even stride keeps the corners rather than one face of the product.
        edges = edges[np.linspace(0, edges.size - 1, keep).astype(np.int64)]
        edges = np.unique(edges)

    if edges.size >= count:
        return edges[:count].astype(np.uint32)

    # Fill the remainder without ever dropping an edge: draw into a pool,
    # remove anything already structured, shuffle so the slice is not biased
    # towards low indices, and only then take what is needed.
    rng = np.random.default_rng(seed)
    need = count - edges.size
    pool = np.zeros(0, np.int64)
    while pool.size < need:
        draw = rng.integers(0, size, size=need * 2 + 16, dtype=np.int64)
        pool = np.unique(np.concatenate((pool, draw)))
        pool = pool[~np.isin(pool, edges)]
    pool = rng.permutation(pool)[:need]
    return np.unique(np.concatenate((edges, pool))).astype(np.uint32)


def sample_runs(shape, count, seed, boundaries=None, run_length=64):
    """Anchored contiguous runs instead of scattered single elements.

    A run never crosses a row of the last axis, so the guest can copy it with
    one DMA rather than reading elements one at a time -- which is what makes
    a large budget affordable, and what keeps the gather off the many-core
    scalar L2 path the model cannot survive.

    Runs are anchored the same way single elements are: the product of each
    axis's ends and named boundaries picks the rows, and both the head and the
    tail of every chosen row are taken. A seeded draw fills the rest.

    Returns `(starts, length)`. Runs may overlap where a row is not a multiple
    of `length`; that costs a little redundancy and nothing else, since both
    the reference and the gather address elements the same way.
    """
    shape = tuple(int(extent) for extent in shape)
    if len(shape) < 2:
        raise ValueError("Runs need a last axis to live inside")
    width = shape[-1]
    length = min(int(run_length), width)
    if length <= 0 or count <= 0:
        raise ValueError("Run length and count must be positive")
    boundaries = boundaries or {}
    if any(axis >= len(shape) for axis in boundaries):
        raise ValueError("Boundary names an axis the tensor does not have")

    rows = tuple(shape[:-1])
    row_count = int(np.prod(rows))
    wanted = max(1, count // length)

    grids = [axis_coordinates(extent, boundaries.get(axis, ()))
             for axis, extent in enumerate(rows)]
    chosen_rows = np.unique(np.ravel_multi_index(
        tuple(g.ravel() for g in np.meshgrid(*grids, indexing="ij")), rows))
    # Head and tail of every structured row, plus any named column.
    columns = sorted({0, width - length} |
                     {min(max(int(edge) - length // 2, 0), width - length)
                      for edge in boundaries.get(len(shape) - 1, ())})
    starts = np.unique(np.array(
        [row * width + column for row in chosen_rows for column in columns],
        dtype=np.int64)) if chosen_rows.size else np.zeros(0, np.int64)

    if starts.size > wanted:
        keep = np.linspace(0, starts.size - 1, wanted).astype(np.int64)
        starts = np.unique(starts[keep])
    if starts.size < wanted:
        rng = np.random.default_rng(seed)
        need = wanted - starts.size
        pool = np.zeros(0, np.int64)
        while pool.size < need:
            size = need * 2 + 16
            row = rng.integers(0, row_count, size=size, dtype=np.int64)
            column = rng.integers(0, width - length + 1, size=row.size,
                                  dtype=np.int64)
            pool = np.unique(np.concatenate((pool, row * width + column)))
            pool = pool[~np.isin(pool, starts)]
        pool = rng.permutation(pool)[:need]
        starts = np.unique(np.concatenate((starts, pool)))
    return starts.astype(np.uint32), length


def expand_runs(starts, length):
    """Flat indices of every element the runs cover, in gather order."""
    return (np.asarray(starts, np.int64)[:, None] +
            np.arange(int(length), dtype=np.int64)[None, :]).ravel()


def describe_runs(shape, starts, length):
    """Coverage facts for a run set, in the same spirit as `describe`."""
    shape = tuple(int(extent) for extent in shape)
    facts = describe(shape, np.sort(np.unique(expand_runs(starts, length))))
    facts.update(runs=int(len(starts)), run_length=int(length),
                 gathered=int(len(starts) * length))
    return facts


def describe(shape, indices):
    """Coverage facts a manifest can record and a reviewer can check."""
    shape = tuple(int(extent) for extent in shape)
    coordinates = np.unravel_index(np.asarray(indices, np.int64), shape)
    return {"shape": list(shape), "count": int(len(indices)),
            "tensor_size": int(np.prod(shape)),
            "distinct_per_axis": [int(np.unique(axis).size)
                                  for axis in coordinates],
            "first": int(indices[0]), "last": int(indices[-1])}
