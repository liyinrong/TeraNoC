# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Independent full-layer equations; no guest tiling or engine emulation.

Synthetic FP16 storage, FP32 linear accumulation and explicit FP16 interfaces.
This reference is not bit-exact RedMulE accumulation or pretrained-model
validation. Text positions only; multimodal RoPE is outside the benchmark.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Geometry:
    d: int = 5120
    intermediate: int = 17408
    q_heads: int = 24
    kv_heads: int = 4
    head_dim: int = 256
    rotary_dim: int = 64
    key_heads: int = 16
    value_heads: int = 48
    key_dim: int = 128
    value_dim: int = 128
    conv_width: int = 4

    def __post_init__(self):
        if any(value <= 0 for value in self.__dict__.values()):
            raise ValueError("Geometry dimensions must be positive")
        if self.q_heads != 6 * self.kv_heads:
            raise ValueError("Qwen attention requires six-way GQA")
        if self.value_heads != 3 * self.key_heads:
            raise ValueError(
                "Qwen GDN requires three value heads per key head")
        if self.rotary_dim % 2 or self.rotary_dim > self.head_dim:
            raise ValueError("Invalid partial rotary dimension")

    @property
    def q_dim(self):
        return self.q_heads * self.head_dim

    @property
    def kv_dim(self):
        return self.kv_heads * self.head_dim

    @property
    def gdn_key_dim(self):
        return self.key_heads * self.key_dim

    @property
    def gdn_value_dim(self):
        return self.value_heads * self.value_dim

    @property
    def qkv_dim(self):
        return 2 * self.gdn_key_dim + self.gdn_value_dim


TOY = Geometry(d=256, intermediate=896, q_heads=6, kv_heads=1,
               head_dim=64, rotary_dim=16, key_heads=2, value_heads=6,
               key_dim=32, value_dim=32)


def fp16(x):
    return np.asarray(x, dtype=np.float16)


def linear(x, weight):
    return fp16(np.asarray(x, np.float32) @ np.asarray(weight, np.float32))


def linear_f16_fma(x, weight):
    """FP16 feedback after each FMA, ascending K, zero addend.

    The independent FP32-accumulation reference remains `linear`. This
    companion follows GVSoC LightRedmuleV3's FP16 feedback order with X shift
    disabled. FP64 preserves exact products/sums for our bounded synthetic
    fixtures before a single half-precision rounding per FMA.
    """
    x = np.asarray(x, np.float16)
    weight = np.asarray(weight, np.float16)
    out = np.zeros(x.shape[:-1] + (weight.shape[1],), np.float16)
    for k in range(weight.shape[0]):
        out = fp16(x[..., k, None].astype(np.float64) *
                   weight[k].astype(np.float64) + out.astype(np.float64))
    return out


def linear_f16_fma_at(x, weight, rows, columns, chunk=None):
    """`linear_f16_fma` evaluated only at the given (row, column) pairs.

    Identical arithmetic -- ascending K, FP16 feedback after each FMA, zero
    addend -- but the cost is O(K * pairs) instead of O(K * tokens * N). That
    is what makes a reference affordable at real model geometry, where one
    full FFN down projection would otherwise be 17408 passes over a
    tokens-by-5120 array.

    `rows` and `columns` are parallel index arrays: pair i is
    (rows[i], columns[i]). Pass `chunk` to bound peak memory on very large
    K by splitting the pair set.
    """
    x = np.asarray(x, np.float16)
    weight = np.asarray(weight, np.float16)
    rows = np.asarray(rows, np.intp)
    columns = np.asarray(columns, np.intp)
    if rows.shape != columns.shape or rows.ndim != 1:
        raise ValueError("rows and columns must be parallel 1-D index arrays")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError("Inner dimensions disagree")
    if chunk and len(rows) > chunk:
        return np.concatenate([
            linear_f16_fma_at(x, weight, rows[i:i + chunk],
                              columns[i:i + chunk])
            for i in range(0, len(rows), chunk)])

    flat = x.reshape(-1, x.shape[-1])
    # K-major copies so each step of the reduction reads contiguously.
    left = np.ascontiguousarray(flat[rows].T)
    right = np.ascontiguousarray(weight[:, columns])
    out = np.zeros(len(rows), np.float16)
    for k in range(weight.shape[0]):
        out = fp16(left[k].astype(np.float64) * right[k].astype(np.float64) +
                   out.astype(np.float64))
    return out


def sigmoid(x):
    x = np.asarray(x, np.float32)
    return np.exp(-np.logaddexp(np.float32(0), -x))


def silu(x):
    return np.asarray(x, np.float32) * sigmoid(x)


def rmsnorm(x, scale, eps=1e-6):
    """Scale is the effective multiplier, not Qwen's stored 1+weight offset."""
    x = np.asarray(x, np.float32)
    norm = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return fp16(norm * np.asarray(scale, np.float32))


def rope(x, start, rotary_dim, theta=10000000.0):
    """Qwen rotate-half convention on the rotary prefix; [B,T,H,D]."""
    return rope_at(x, np.arange(start, start + x.shape[1]), rotary_dim, theta)


def rope_at(x, positions, rotary_dim, theta=10000000.0):
    """`rope` for rows that carry their own absolute positions.

    Needed because a bounded reference evaluates scattered query rows: the
    rotation depends on the token index, not on where the row sits in the
    array it was gathered into.
    """
    positions = np.asarray(positions)
    if positions.shape != (x.shape[1],):
        raise ValueError("One position per row is required")
    result = x.copy()
    half = rotary_dim // 2
    frequencies = theta ** (-np.arange(0, rotary_dim, 2) / rotary_dim)
    angles = positions[:, None] * frequencies
    # The model casts sine/cosine to the activation dtype before rotation.
    cosine = fp16(np.cos(angles))[None, :, None, :].astype(np.float32)
    sine = fp16(np.sin(angles))[None, :, None, :].astype(np.float32)
    first = x[..., :half].astype(np.float32)
    second = x[..., half:rotary_dim].astype(np.float32)
    result[..., :half] = fp16(first * cosine - second * sine)
    result[..., half:rotary_dim] = fp16(second * cosine + first * sine)
    return result


def attention_dense(q, k, v, start=0):
    """Independent materialized softmax reference; small correctness cases."""
    ratio = q.shape[2] // k.shape[2]
    keys = np.repeat(k.astype(np.float64), ratio, axis=2)
    values = np.repeat(v.astype(np.float64), ratio, axis=2)
    scores = np.einsum("bthd,bshd->bhts", q.astype(np.float64), keys)
    scores /= np.sqrt(q.shape[-1])
    allowed = np.arange(k.shape[1])[None, :] <= (
        start + np.arange(q.shape[1]))[:, None]
    scores = np.where(allowed[None, None, :, :], scores, -np.inf)
    probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return np.einsum("bhts,bshd->bthd", probabilities, values)


def attention_online(q, k, v, start=0, key_block=64):
    """Bounded score storage, complete-row ownership, FP64 algorithm oracle.

    No early FP16 probability rounding: this isolates the online algorithm
    from the later accelerator's numerical contract.
    """
    if key_block <= 0:
        raise ValueError("key_block must be positive")
    batch, tokens, heads, dim = q.shape
    result = np.zeros((batch, tokens, heads, dim), dtype=np.float64)
    ratio = heads // k.shape[2]
    for b in range(batch):
        for h in range(heads):
            kh = h // ratio
            for t in range(tokens):
                maximum, denominator = -np.inf, 0.0
                accumulator = np.zeros(dim, dtype=np.float64)
                limit = min(start + t + 1, k.shape[1])
                for offset in range(0, limit, key_block):
                    end = min(offset + key_block, limit)
                    scores = k[b, offset:end, kh].astype(np.float64) @ (
                        q[b, t, h].astype(np.float64) / np.sqrt(dim))
                    new_maximum = max(maximum, float(scores.max()))
                    correction = np.exp(maximum - new_maximum)
                    probabilities = np.exp(scores - new_maximum)
                    accumulator = correction * accumulator + probabilities @ (
                        v[b, offset:end, kh].astype(np.float64))
                    denominator = (correction * denominator +
                                   probabilities.sum())
                    maximum = new_maximum
                if denominator == 0:
                    raise ValueError("Attention row has no visible key")
                result[b, t, h] = accumulator / denominator
    return result


def attention_online_at(q, positions, k, v, key_block=64):
    """`attention_online` for query rows carrying their own positions.

    Same loop and same FP64 accumulation; only the set of rows differs, and
    each row's causal limit comes from its own token index. Every key is still
    read, which is why the key/value cache is the one thing a bounded
    reference must still build in full.
    """
    if key_block <= 0:
        raise ValueError("key_block must be positive")
    positions = np.asarray(positions)
    batch, count, heads, dim = q.shape
    if positions.shape != (count,):
        raise ValueError("One position per query row is required")
    result = np.zeros((batch, count, heads, dim), dtype=np.float64)
    ratio = heads // k.shape[2]
    for b in range(batch):
        for h in range(heads):
            kh = h // ratio
            for index in range(count):
                maximum, denominator = -np.inf, 0.0
                accumulator = np.zeros(dim, dtype=np.float64)
                limit = min(int(positions[index]) + 1, k.shape[1])
                for offset in range(0, limit, key_block):
                    end = min(offset + key_block, limit)
                    scores = k[b, offset:end, kh].astype(np.float64) @ (
                        q[b, index, h].astype(np.float64) / np.sqrt(dim))
                    new_maximum = max(maximum, float(scores.max()))
                    correction = np.exp(maximum - new_maximum)
                    probabilities = np.exp(scores - new_maximum)
                    accumulator = correction * accumulator + probabilities @ (
                        v[b, offset:end, kh].astype(np.float64))
                    denominator = (correction * denominator +
                                   probabilities.sum())
                    maximum = new_maximum
                if denominator == 0:
                    raise ValueError("Attention row has no visible key")
                result[b, index, h] = accumulator / denominator
    return result


def attention_cache(normalized, weights, geometry, start=0, linear_op=linear):
    """The key/value cache: what every query row depends on."""
    g = geometry
    batch, tokens, _ = normalized.shape
    k = linear_op(normalized, weights["key"]).reshape(
        batch, tokens, g.kv_heads, g.head_dim)
    v = linear_op(normalized, weights["value"]).reshape(k.shape)
    return rope(rmsnorm(k, weights["k_norm"]), start, g.rotary_dim), v


def ffn(x, weights, linear_op=linear):
    normalized = rmsnorm(x, weights["ffn_norm"])
    gate = linear_op(normalized, weights["gate"])
    up = linear_op(normalized, weights["up"])
    activated = fp16(fp16(silu(gate)).astype(
        np.float32) * up.astype(np.float32))
    down = linear_op(activated, weights["down"])
    output = fp16(x.astype(np.float32) + down.astype(np.float32))
    return output, {"ffn_norm": normalized, "ffn_activation": activated}


def attention_layer(x, weights, geometry, state=None, online=False,
                    key_block=64, linear_op=linear):
    g = geometry
    batch, tokens, _ = x.shape
    normalized = rmsnorm(x, weights["input_norm"])
    # The checkpoint's Q/gate projection is interleaved per head.
    q_gate = linear_op(normalized, weights["q_gate"]).reshape(
        batch, tokens, g.q_heads, 2 * g.head_dim)
    q, gate = np.split(q_gate, 2, axis=-1)
    start = 0 if state is None else state["key"].shape[1]
    k, v = attention_cache(normalized, weights, g, start, linear_op)
    q = rope(rmsnorm(q, weights["q_norm"]), start, g.rotary_dim)
    if state is not None:
        k = np.concatenate((state["key"], k), axis=1)
        v = np.concatenate((state["value"], v), axis=1)
    context = (attention_online(q, k, v, start, key_block) if online
               else attention_dense(q, k, v, start))
    context = fp16(context)
    gated = fp16(context.astype(np.float32) *
                 fp16(sigmoid(gate)).astype(np.float32))
    mixer = linear_op(gated.reshape(batch, tokens, g.q_dim), weights["out"])
    residual = fp16(x.astype(np.float32) + mixer.astype(np.float32))
    output, intermediates = ffn(residual, weights, linear_op)
    intermediates.update(input_norm=normalized, context=context,
                         mixer_residual=residual)
    return output, {"key": k, "value": v}, intermediates


def gdn_recurrence(x, weights, geometry, state=None, linear_op=linear):
    """The part of GDN every token depends on.

    Projection, causal convolution and the recurrence itself: the state at
    token t carries all of 0..t, so none of this can be restricted to sampled
    rows. A bounded reference runs this in full and restricts only the tail.
    Returns the normalized input, the per-token contexts and both states.
    """
    g = geometry
    batch, tokens, _ = x.shape
    normalized = rmsnorm(x, weights["input_norm"])
    projected = linear_op(normalized, weights["qkv"])
    controls = linear_op(normalized, weights["ab"])
    a, b = np.split(controls.astype(np.float32), 2, axis=-1)
    beta = sigmoid(b)
    decay = np.exp(-np.exp(weights["a_log"].astype(np.float32)) *
                   np.logaddexp(
                       0.0, a + weights["dt_bias"].astype(np.float32)))
    history = (np.zeros((batch, g.conv_width, g.qkv_dim), np.float16)
               if state is None else state["conv"].copy())
    recurrent = (np.zeros((batch, g.value_heads, g.key_dim, g.value_dim),
                          np.float32)
                 if state is None else state["recurrent"].copy())
    contexts = np.empty(
        (batch, tokens, g.value_heads, g.value_dim), np.float16)
    for t in range(tokens):
        history = np.concatenate(
            (history[:, 1:], projected[:, t:t + 1]), axis=1)
        convolved = fp16(np.sum(history.astype(np.float32) *
                                weights["conv"].astype(np.float32), axis=1))
        convolved = fp16(silu(convolved))
        q, k, v = np.split(
            convolved, [g.gdn_key_dim, 2 * g.gdn_key_dim], axis=-1)
        q = q.reshape(batch, g.key_heads, g.key_dim).astype(np.float32)
        k = k.reshape(q.shape).astype(np.float32)
        v = v.reshape(batch, g.value_heads, g.value_dim).astype(np.float32)
        q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        q /= np.sqrt(g.key_dim)
        for h in range(g.value_heads):
            kh = h // 3
            recurrent[:, h] *= decay[:, t, h, None, None]
            predicted = np.einsum("bk,bkv->bv", k[:, kh], recurrent[:, h])
            delta = beta[:, t, h, None] * (v[:, h] - predicted)
            recurrent[:, h] += k[:, kh, :, None] * delta[:, None, :]
            contexts[:, t, h] = fp16(np.einsum(
                "bk,bkv->bv", q[:, kh], recurrent[:, h]))
    return normalized, contexts, history, recurrent


def gdn_tail(x, weights, geometry, normalized, contexts, linear_op=linear,
             rows=None):
    """Everything after the recurrence. Row-wise, so it restricts.

    `rows` evaluates only those token rows; the result is what evaluating all
    of them and then indexing would give.
    """
    g = geometry
    if rows is not None:
        x = x[:, rows]
        normalized = normalized[:, rows]
        contexts = contexts[:, rows]
    batch, tokens, _ = x.shape
    z = linear_op(normalized, weights["z"]).reshape(
        batch, tokens, g.value_heads, g.value_dim)
    # Gated RMSNorm narrows the normalized value before applying its learned
    # scale; unlike the zero-centered decoder RMSNorm, that boundary matters.
    normalized_context = contexts.astype(np.float32)
    normalized_context /= np.sqrt(np.mean(normalized_context ** 2,
                                          axis=-1, keepdims=True) + 1e-6)
    scaled_context = fp16(fp16(normalized_context).astype(np.float32) *
                          weights["gated_norm"].astype(np.float32))
    gated = fp16(scaled_context.astype(np.float32) * silu(z))
    mixer = linear_op(gated.reshape(
        batch, tokens, g.gdn_value_dim), weights["out"])
    residual = fp16(x.astype(np.float32) + mixer.astype(np.float32))
    output, intermediates = ffn(residual, weights, linear_op)
    intermediates.update(input_norm=normalized, context=contexts,
                         mixer_residual=residual)
    return output, intermediates


def gdn_layer(x, weights, geometry, state=None, linear_op=linear):
    normalized, contexts, history, recurrent = gdn_recurrence(
        x, weights, geometry, state, linear_op)
    output, intermediates = gdn_tail(x, weights, geometry, normalized,
                                     contexts, linear_op)
    return output, {"conv": history, "recurrent": recurrent}, intermediates


def weight_dims(kind, geometry=TOY):
    """(rows, columns) of every 2-D projection `make_fixture` produces.

    Declared separately from `make_fixture` so that shapes can be derived for
    geometries too large to materialize. `test_reference` asserts they agree.
    """
    if kind not in ("attention", "gdn"):
        raise ValueError("Unknown layer kind")
    g = geometry
    dims = {"gate": (g.d, g.intermediate), "up": (g.d, g.intermediate),
            "down": (g.intermediate, g.d)}
    if kind == "attention":
        dims.update(q_gate=(g.d, 2 * g.q_dim), key=(g.d, g.kv_dim),
                    value=(g.d, g.kv_dim), out=(g.q_dim, g.d))
    else:
        dims.update(qkv=(g.d, g.qkv_dim), z=(g.d, g.gdn_value_dim),
                    ab=(g.d, 2 * g.value_heads),
                    out=(g.gdn_value_dim, g.d))
    return dims


def vector_dims(kind, geometry=TOY):
    """Shapes of the remaining fixture tensors, in the same emission order."""
    g = geometry
    dims = {"input_norm": (g.d,), "ffn_norm": (g.d,)}
    if kind == "attention":
        dims.update(q_norm=(g.head_dim,), k_norm=(g.head_dim,))
    else:
        dims.update(conv=(g.conv_width, g.qkv_dim),
                    a_log=(g.value_heads,), dt_bias=(g.value_heads,),
                    gated_norm=(g.value_dim,))
    return dims


def make_fixture(kind, geometry=TOY, tokens=17, batch=1, seed=20260921):
    if kind not in ("attention", "gdn"):
        raise ValueError("Unknown layer kind")
    if tokens <= 0 or batch <= 0:
        raise ValueError("tokens and batch must be positive")
    g = geometry
    rng = np.random.default_rng(seed)

    def matrix(rows, columns):
        return fp16(rng.normal(0, 0.25 / np.sqrt(rows), (rows, columns)))

    def scale(size):
        return fp16(rng.uniform(0.8, 1.2, size))

    weights = {"input_norm": scale(g.d), "ffn_norm": scale(g.d),
               "gate": matrix(g.d, g.intermediate),
               "up": matrix(g.d, g.intermediate),
               "down": matrix(g.intermediate, g.d)}
    if kind == "attention":
        weights.update(q_gate=matrix(g.d, 2 * g.q_dim),
                       key=matrix(g.d, g.kv_dim), value=matrix(g.d, g.kv_dim),
                       out=matrix(g.q_dim, g.d), q_norm=scale(g.head_dim),
                       k_norm=scale(g.head_dim))
    else:
        weights.update(qkv=matrix(g.d, g.qkv_dim),
                       z=matrix(g.d, g.gdn_value_dim),
                       ab=matrix(g.d, 2 * g.value_heads),
                       out=matrix(g.gdn_value_dim, g.d),
                       conv=fp16(rng.normal(
                           0, 0.25, (g.conv_width, g.qkv_dim))),
                       a_log=rng.uniform(-1, 1,
                                         g.value_heads).astype(np.float32),
                       dt_bias=rng.uniform(-2, -1,
                                           g.value_heads).astype(np.float32),
                       gated_norm=scale(g.value_dim))
    x = fp16(rng.normal(0, 0.5, (batch, tokens, g.d)))
    return x, weights
