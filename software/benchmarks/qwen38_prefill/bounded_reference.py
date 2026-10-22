# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Reference values at sampled coordinates only, for full model geometry.

Evaluating a whole layer under the accelerator's FP16 FMA arithmetic is
sequential in K, so it costs hours at real dimensions and cannot be
parallelised away (plan §7). What makes a reference affordable is that most of
the layer is row-wise: only a few tensors are needed for every token, and the
rest can be evaluated for the token rows the sample actually touches.

Computed in full, because every row depends on them:

  attention  the key/value cache -- any query row attends to all earlier keys
  GDN        the projection, causal convolution and recurrence -- the state at
             token t carries all of 0..t

Everything else -- the query/gate projection, the attention core for the
chosen rows, the gating, the output projection and the whole FFN -- runs only
for the sampled rows. The arithmetic is the same functions the whole-layer
reference uses, so the result is bit-identical to evaluating everything and
then indexing; `test_bounded_reference` asserts exactly that.
"""

import numpy as np

from reference import (attention_cache, attention_online_at, ffn, fp16,
                       gdn_recurrence, gdn_tail, linear_f16_fma, rmsnorm,
                       rope_at, sigmoid)

# Which axis of each checked tensor is the token row, where there is one.
ROW_AXIS = {"output": 0, "mixer": 0, "context": 1}
# What each layer kind can be asked for; `layer_fixture.check_shapes` names
# exactly these, and `test_bounded_reference` holds the two together.
CHECKED = {"attention": ("output", "context", "cache", "mixer"),
           "gdn": ("output", "context", "conv", "recurrent")}


def pad_rows(value, rows):
    """Pad a [1, tokens, ...] tensor out to the guest's physical row count."""
    out = np.zeros((rows,) + value.shape[2:], value.dtype)
    out[:value.shape[1]] = value[0]
    return out


def pack_kv_cache(head_major, block=64):
    """[2,H,R,D] to block-major K^T/V; queries are not replicated."""
    _, heads, rows, dimension = head_major.shape
    if rows % block:
        raise ValueError("KV storage rows must include complete key blocks")
    chunks = []
    for start in range(0, rows, block):
        key = head_major[0, :, start:start + block].transpose(0, 2, 1)
        value = head_major[1, :, start:start + block]
        chunks.extend((key.ravel(), value.ravel()))
    return np.concatenate(chunks).astype(np.float16)


def required_rows(wanted, tokens, ignore=()):
    """Real token rows the sampled coordinates need, sorted and unique.

    `ignore` names tensors that come out of the part computed in full, so
    their rows cost nothing extra and must not enlarge the restricted set.
    """
    needed = set()
    for name, (shape, index) in wanted.items():
        axis = ROW_AXIS.get(name)
        if axis is None or name in ignore:
            continue
        coordinate = np.unravel_index(np.asarray(index, np.int64), shape)[axis]
        needed.update(int(row) for row in np.unique(coordinate))
    return np.array(sorted(row for row in needed if row < tokens), np.int64)


def chunked(linear_op, chunk):
    """`linear_op` applied to blocks of token rows.

    Output rows depend only on their own input row, so this is bit-identical
    to one call. The point is the working set: at real geometry one reduction
    step of a 2048-row projection touches hundreds of megabytes, and chunking
    keeps it in cache.
    """
    if not chunk:
        return linear_op

    def apply(x, weight):
        x = np.asarray(x)
        if x.shape[-2] <= chunk:
            return linear_op(x, weight)
        return np.concatenate([linear_op(x[..., i:i + chunk, :], weight)
                               for i in range(0, x.shape[-2], chunk)],
                              axis=-2)
    return apply


def _gather(shape, index, axis, rows, dense):
    """Read sampled coordinates out of a row-restricted result.

    `dense` holds only `rows`; coordinates on any other row are padding and
    read as zero, which is what the guest writes there.
    """
    index = np.asarray(index, np.int64)
    coordinate = list(np.unravel_index(index, shape))
    # The restricted rows come from every checked tensor, so one of them can
    # sit beyond this tensor's own sampled coordinates.
    extent = max(int(coordinate[axis].max()) + 1,
                 int(rows[-1]) + 1 if len(rows) else 0, 1)
    position = np.full(extent, -1, np.int64)
    position[rows] = np.arange(len(rows))
    taken = position[coordinate[axis]]
    values = np.zeros(len(index), dense.dtype)
    live = taken >= 0
    coordinate[axis] = np.where(live, taken, 0)
    values[live] = dense[tuple(c[live] for c in coordinate)]
    return values


def evaluate(kind, geometry, tokens, rows, x, weights, wanted, key_block=64,
             linear_op=linear_f16_fma, chunk=None):
    """Sampled values of every checked tensor.

    `wanted` maps a tensor name to `(padded shape, flat indices)`; the return
    maps the same names to the values at those indices, in that order.
    `linear_op` selects the arithmetic -- the guest's FP16 FMA companion by
    default, `reference.linear` for the independent FP32 oracle.
    """
    if kind not in CHECKED:
        raise ValueError("Unknown layer kind")
    if set(wanted) - set(CHECKED[kind]):
        raise ValueError("This layer kind has no such checked tensor")
    # GDN's context falls out of the recurrence, which runs in full, so its
    # rows are free; every other checked tensor has to be evaluated.
    needed = required_rows(wanted, tokens,
                           ("context",) if kind == "gdn" else ())
    linear = chunked(linear_op, chunk)
    g = geometry
    out = {}

    if kind == "attention":
        # Every query row reads the whole cache, so it is built in full and
        # doubles as the `cache` golden.
        normalized = rmsnorm(x, weights["input_norm"])
        key, value = attention_cache(normalized, weights, g, 0, linear)
        if "cache" in wanted:
            _, index = wanted["cache"]
            packed = pack_kv_cache(np.stack([
                pad_rows(part, rows).transpose(1, 0, 2)
                for part in (key, value)]))
            out["cache"] = packed[np.asarray(index, np.int64)]
        span = len(needed)
        q_gate = linear(normalized[:, needed], weights["q_gate"]).reshape(
            1, span, g.q_heads, 2 * g.head_dim)
        query, gate = np.split(q_gate, 2, axis=-1)
        query = rope_at(rmsnorm(query, weights["q_norm"]), needed,
                        g.rotary_dim)
        context = fp16(attention_online_at(query, needed, key, value,
                                           key_block))
        gated = fp16(context.astype(np.float32) *
                     fp16(sigmoid(gate)).astype(np.float32))
        mixer = linear(gated.reshape(1, span, g.q_dim), weights["out"])
        residual = fp16(x[:, needed].astype(np.float32) +
                        mixer.astype(np.float32))
        final, _ = ffn(residual, weights, linear)
        dense = {"context": context[0].transpose(1, 0, 2),
                 "mixer": residual[0], "output": final[0]}
    else:
        # The recurrence carries every token, so it runs in full; only the
        # gating, output projection and FFN restrict.
        normalized, contexts, history, recurrent = gdn_recurrence(
            x, weights, g, None, linear)
        if "conv" in wanted:
            _, index = wanted["conv"]
            out["conv"] = history[0].ravel()[np.asarray(index, np.int64)]
        if "recurrent" in wanted:
            _, index = wanted["recurrent"]
            out["recurrent"] = recurrent[0].ravel()[
                np.asarray(index, np.int64)]
        final, _ = gdn_tail(x, weights, g, normalized, contexts, linear,
                            needed)
        if "context" in wanted:
            # The recurrence already produced every row, so the context is
            # padded and indexed directly rather than gathered.
            _, index = wanted["context"]
            head_major = pad_rows(contexts, rows).transpose(1, 0, 2)
            out["context"] = head_major.reshape(-1)[
                np.asarray(index, np.int64)]
        dense = {"output": final[0]}

    for name, (shape, index) in wanted.items():
        if name in out:
            continue
        # `context` is head-major; `_gather` maps its row axis either way.
        out[name] = _gather(shape, index, ROW_AXIS[name], needed, dense[name])
    return out
