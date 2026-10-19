# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Metamorphic checks independent of the future asynchronous guest schedule."""

import unittest

import numpy as np

from reference import (TOY, attention_dense, attention_layer, attention_online,
                       gdn_layer, linear_f16_fma, linear_f16_fma_at,
                       make_fixture, rope)


class ReferenceTests(unittest.TestCase):
    def test_online_attention_tails_and_late_maximum(self):
        rng = np.random.default_rng(9)
        q = rng.normal(size=(2, 11, 6, 16)).astype(np.float16)
        k = rng.normal(size=(2, 11, 1, 16)).astype(np.float16)
        v = rng.normal(size=k.shape).astype(np.float16)
        k[:, -1] *= 12  # Changes the running maximum late in visible rows.
        expected = attention_dense(q, k, v)
        for block in (1, 3, 7, 16):
            np.testing.assert_allclose(
                attention_online(q, k, v, key_block=block), expected,
                rtol=1e-12, atol=1e-12)

    def test_equal_logits_average_visible_values(self):
        q = np.zeros((1, 7, 6, 8), np.float16)
        k = np.zeros((1, 7, 1, 8), np.float16)
        v = np.arange(56).reshape(1, 7, 1, 8).astype(np.float16)
        got = attention_online(q, k, v, key_block=3)
        for t in range(7):
            expected = np.broadcast_to(v[0, :t + 1, 0].mean(axis=0), (6, 8))
            np.testing.assert_array_equal(got[0, t], expected)

    def test_rope_only_rotates_prefix(self):
        x = np.arange(64).reshape(1, 2, 1, 32).astype(np.float16)
        rotated = rope(x, 0, 8)
        np.testing.assert_array_equal(rotated[:, 0], x[:, 0])
        np.testing.assert_array_equal(rotated[..., 8:], x[..., 8:])
        self.assertFalse(np.array_equal(rotated[:, 1, :, :8], x[:, 1, :, :8]))

    def test_complete_attention_dense_online(self):
        x, w = make_fixture("attention", tokens=13)
        expected, state, _ = attention_layer(x, w, TOY)
        actual, other_state, _ = attention_layer(
            x, w, TOY, online=True, key_block=3)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0.0005)
        for name in state:
            np.testing.assert_array_equal(state[name], other_state[name])

    def test_causal_prefix_and_state_continuation(self):
        for kind, layer in (("attention", attention_layer),
                            ("gdn", gdn_layer)):
            with self.subTest(kind=kind):
                x, w = make_fixture(kind, tokens=13, batch=2)
                whole, final, _ = layer(x, w, TOY)
                prefix, initial, _ = layer(x[:, :5], w, TOY)
                saved = {name: array.copy() for name, array in initial.items()}
                suffix, split_final, _ = layer(x[:, 5:], w, TOY, state=initial)
                np.testing.assert_allclose(
                    prefix, whole[:, :5], rtol=0, atol=0.0005)
                np.testing.assert_allclose(
                    suffix, whole[:, 5:], rtol=0, atol=0.0005)
                for name in final:
                    np.testing.assert_allclose(final[name], split_final[name],
                                               rtol=1e-5, atol=1e-6)
                    np.testing.assert_array_equal(initial[name], saved[name])

    def test_ffn_and_mixer_are_both_live(self):
        for kind, layer in (("attention", attention_layer),
                            ("gdn", gdn_layer)):
            x, w = make_fixture(kind, tokens=5)
            out, state, intermediate = layer(x, w, TOY)
            self.assertFalse(np.array_equal(
                out, intermediate["mixer_residual"]))
            self.assertFalse(np.array_equal(x, intermediate["mixer_residual"]))
            for value in (out, *state.values(), *intermediate.values()):
                self.assertTrue(np.isfinite(value).all())


class PointwiseReferenceTests(unittest.TestCase):
    def test_pointwise_matches_the_whole_tensor_reference(self):
        rng = np.random.default_rng(4)
        x = np.asarray(rng.normal(0, 0.3, (1, 7, 40)), np.float16)
        w = np.asarray(rng.normal(0, 0.2, (40, 23)), np.float16)
        full = linear_f16_fma(x, w).reshape(-1, 23)
        rows, columns = np.meshgrid(np.arange(7), np.arange(23),
                                    indexing="ij")
        got = linear_f16_fma_at(x, w, rows.ravel(), columns.ravel())
        # Bitwise: the point evaluation must not be a different arithmetic.
        np.testing.assert_array_equal(got.view(np.uint16),
                                      full.ravel().view(np.uint16))

    def test_chunking_does_not_change_a_single_bit(self):
        rng = np.random.default_rng(9)
        x = np.asarray(rng.normal(0, 0.3, (11, 33)), np.float16)
        w = np.asarray(rng.normal(0, 0.2, (33, 17)), np.float16)
        rows = rng.integers(0, 11, 40)
        columns = rng.integers(0, 17, 40)
        whole = linear_f16_fma_at(x, w, rows, columns)
        split = linear_f16_fma_at(x, w, rows, columns, chunk=7)
        np.testing.assert_array_equal(whole.view(np.uint16),
                                      split.view(np.uint16))

    def test_mismatched_index_arrays_are_refused(self):
        x = np.ones((3, 4), np.float16)
        w = np.ones((4, 5), np.float16)
        with self.assertRaises(ValueError):
            linear_f16_fma_at(x, w, [0, 1], [0])
        with self.assertRaises(ValueError):
            linear_f16_fma_at(x, np.ones((6, 5), np.float16), [0], [0])


if __name__ == "__main__":
    unittest.main()
