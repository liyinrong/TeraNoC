# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""A sampled reference must be the whole one, indexed -- bit for bit."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

import bounded_reference
from bounded_reference import (CHECKED, chunked, evaluate,
                               required_rows)
from layer_fixture import check_shapes, generate_layer_header, golden_tensors
from reference import (TOY, attention_layer, gdn_layer, linear_f16_fma,
                       make_fixture)
from sampling import sample_runs

KEY_BLOCK = 64


def fixture(kind, tokens, kv_heads=1, seed=20260921):
    g = (replace(TOY, kv_heads=kv_heads, q_heads=6 * kv_heads)
         if kind == "attention" else TOY)
    rows = ((tokens + 63) // 64) * 64
    x, weights = make_fixture(kind, g, tokens=tokens, seed=seed)
    return g, rows, x, weights


def whole(kind, g, rows, x, weights):
    layer = attention_layer if kind == "attention" else gdn_layer
    kwargs = {"online": True, "key_block": KEY_BLOCK} \
        if kind == "attention" else {}
    expected, state, intermediate = layer(
        x, weights, g, linear_op=linear_f16_fma, **kwargs)
    return golden_tensors(kind, rows, expected, state, intermediate)


def wanted_runs(kind, g, rows, tokens, count=512, length=16, seed=5):
    """The same selection `generate_layer_header` makes, expanded flat."""
    out = {}
    for name, (shape, edges) in check_shapes(kind, g, rows, tokens).items():
        size = int(np.prod(shape))
        starts, taken = sample_runs(shape, min(count, size), seed + len(name),
                                    edges, length)
        index = (starts.astype(np.int64)[:, None] +
                 np.arange(taken)[None, :]).ravel()
        out[name] = (shape, index)
    return out


class BoundedReferenceTests(unittest.TestCase):
    def test_sampled_values_match_the_whole_layer_bit_for_bit(self):
        for kind, tokens, kv_heads in (("attention", 37, 1),
                                       ("attention", 70, 2),
                                       ("gdn", 37, 1), ("gdn", 70, 1)):
            with self.subTest(kind=kind, tokens=tokens, kv_heads=kv_heads):
                g, rows, x, weights = fixture(kind, tokens, kv_heads)
                gold = whole(kind, g, rows, x, weights)
                wanted = wanted_runs(kind, g, rows, tokens)
                got = evaluate(kind, g, tokens, rows, x, weights, wanted,
                               KEY_BLOCK)
                self.assertEqual(set(got), set(gold))
                for name, (shape, index) in wanted.items():
                    expected = gold[name].ravel()[index]
                    self.assertEqual(got[name].dtype, expected.dtype, name)
                    np.testing.assert_array_equal(
                        got[name].view(np.uint16 if expected.dtype
                                       == np.float16 else np.uint32),
                        expected.view(np.uint16 if expected.dtype
                                      == np.float16 else np.uint32),
                        err_msg=f"{kind} {name}")

    def test_the_cache_is_packed_at_the_key_block(self):
        # A key block other than 64: the bounded cache must follow it, as the
        # whole-tensor golden does (the default block once hid that).
        block, tokens = 128, 200
        g, rows, x, weights = fixture("attention", tokens, 2)
        expected, state, intermediate = attention_layer(
            x, weights, g, online=True, key_block=block,
            linear_op=linear_f16_fma)
        gold = golden_tensors("attention", rows, expected, state,
                              intermediate, key_block=block)
        shape, edges = check_shapes("attention", g, rows, tokens,
                                    block)["cache"]
        starts, taken = sample_runs(shape, 256, 7, edges, 16)
        index = (starts.astype(np.int64)[:, None] +
                 np.arange(taken)[None, :]).ravel()
        got = evaluate("attention", g, tokens, rows, x, weights,
                       {"cache": (shape, index)}, block)
        np.testing.assert_array_equal(
            got["cache"].view(np.uint16),
            gold["cache"].ravel()[index].view(np.uint16))

    def test_a_single_tensor_costs_only_the_rows_it_names(self):
        kind, tokens = "attention", 70
        g, rows, x, weights = fixture(kind, tokens)
        gold = whole(kind, g, rows, x, weights)
        every = wanted_runs(kind, g, rows, tokens)
        for name in ("output", "context", "mixer"):
            with self.subTest(name=name):
                wanted = {name: every[name]}
                needed = required_rows(wanted, tokens)
                self.assertLess(len(needed), tokens)
                got = evaluate(kind, g, tokens, rows, x, weights, wanted,
                               KEY_BLOCK)
                self.assertEqual(list(got), [name])
                np.testing.assert_array_equal(
                    got[name], gold[name].ravel()[every[name][1]])

    def test_padded_rows_read_as_the_zeros_the_guest_leaves(self):
        kind, tokens = "gdn", 70
        g, rows, x, weights = fixture(kind, tokens)
        shape = (rows, g.d)
        index = np.arange(tokens * g.d, rows * g.d, 37, dtype=np.int64)
        got = evaluate(kind, g, tokens, rows, x, weights,
                       {"output": (shape, index)})
        np.testing.assert_array_equal(got["output"],
                                      np.zeros(len(index), np.float16))

    def test_gdn_context_rows_do_not_enlarge_the_restricted_set(self):
        kind, tokens = "gdn", 70
        g, rows, x, weights = fixture(kind, tokens)
        wanted = wanted_runs(kind, g, rows, tokens)
        both = required_rows(wanted, tokens)
        tail = required_rows(wanted, tokens, ("context",))
        self.assertLess(len(tail), len(both))
        # And the values are still right, which the bit-exact test covers for
        # the whole set; here only that the saving is real.
        got = evaluate(kind, g, tokens, rows, x, weights, wanted, KEY_BLOCK)
        gold = whole(kind, g, rows, x, weights)
        np.testing.assert_array_equal(
            got["context"], gold["context"].ravel()[wanted["context"][1]])

    def test_row_requirements_ignore_padding_and_rowless_tensors(self):
        rows = required_rows({"output": ((8, 4), np.array([0, 9, 30])),
                              "conv": ((4, 4), np.array([15]))}, 5)
        np.testing.assert_array_equal(rows, np.array([0, 2]))

    def test_unknown_kinds_and_tensors_are_refused(self):
        g, rows, x, weights = fixture("gdn", 5)
        for kind, wanted in (("hybrid", {}), ("gdn", {"cache": ((4,), [0])})):
            with self.assertRaises(ValueError):
                evaluate(kind, g, 5, rows, x, weights, wanted)

    def test_the_checked_set_matches_the_fixture(self):
        for kind, kv_heads in (("attention", 2), ("gdn", 1)):
            g, rows, _, _ = fixture(kind, 70, kv_heads)
            self.assertEqual(set(CHECKED[kind]),
                             set(check_shapes(kind, g, rows, 70)))

    def test_chunking_a_projection_changes_nothing(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(1, 13, 8)).astype(np.float16)
        weight = rng.normal(size=(8, 5)).astype(np.float16)
        whole = linear_f16_fma(x, weight)
        for chunk in (1, 2, 5, 13, 64):
            with self.subTest(chunk=chunk):
                np.testing.assert_array_equal(
                    chunked(linear_f16_fma, chunk)(x, weight).view(np.uint16),
                    whole.view(np.uint16))
        self.assertIs(chunked(linear_f16_fma, 0), linear_f16_fma)

    def test_the_chunk_size_never_changes_a_sampled_value(self):
        for kind in ("attention", "gdn"):
            g, rows, x, weights = fixture(kind, 70)
            wanted = wanted_runs(kind, g, rows, 70)
            base = evaluate(kind, g, 70, rows, x, weights, wanted, KEY_BLOCK)
            for chunk in (1, 9, 256):
                with self.subTest(kind=kind, chunk=chunk):
                    got = evaluate(kind, g, 70, rows, x, weights, wanted,
                                   KEY_BLOCK, chunk=chunk)
                    for name in base:
                        np.testing.assert_array_equal(
                            got[name].view(np.uint16 if base[name].dtype
                                           == np.float16 else np.uint32),
                            base[name].view(np.uint16 if base[name].dtype
                                            == np.float16 else np.uint32),
                            err_msg=f"{kind} {name} chunk={chunk}")

    def test_the_generator_emits_the_same_header_either_way(self):
        """The wiring, not just the arithmetic: a bounded fixture and a
        whole-tensor one must produce the identical guest header."""
        for kind in ("attention", "gdn"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                made = []
                for index, geometry in enumerate((None, TOY)):
                    out = Path(tmp) / str(index)
                    out.mkdir()
                    made.append(generate_layer_header(
                        out, kind, tokens=70, check_samples=256,
                        check_run_length=16, geometry=geometry,
                        bounded=geometry is not None))
                whole, sampled = made
                self.assertEqual(whole["reference"], "whole-tensor")
                self.assertEqual(sampled["reference"], "bounded")
                self.assertEqual(
                    (Path(tmp) / "0" / "layer_data.h").read_bytes(),
                    (Path(tmp) / "1" / "layer_data.h").read_bytes())
                self.assertEqual(whole["samples"], sampled["samples"])
                self.assertEqual(json.dumps(whole["checks"], sort_keys=True),
                                 json.dumps(sampled["checks"],
                                            sort_keys=True))

    def test_the_helpers_have_one_home(self):
        import layer_fixture
        self.assertIs(layer_fixture.pack_kv_cache,
                      bounded_reference.pack_kv_cache)
        self.assertIs(layer_fixture.pad_rows, bounded_reference.pad_rows)


if __name__ == "__main__":
    unittest.main()
