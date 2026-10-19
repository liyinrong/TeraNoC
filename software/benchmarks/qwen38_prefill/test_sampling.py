# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from sampling import (axis_coordinates, describe, describe_runs,
                      sample_elements, sample_runs)


class SamplingTests(unittest.TestCase):
    def test_axis_keeps_ends_and_straddles_a_boundary(self):
        self.assertEqual(axis_coordinates(1), [0])
        self.assertEqual(axis_coordinates(2), [0, 1])
        self.assertEqual(axis_coordinates(8), [0, 6, 7])
        # A padding edge at 5 must bring both the last real and first padded
        # element into the set.
        self.assertEqual(axis_coordinates(8, [5]), [0, 4, 5, 6, 7])
        # Out-of-range boundaries are ignored rather than clipped onto an end.
        self.assertEqual(axis_coordinates(4, [0, 99]), [0, 2, 3])

    def test_selection_is_deterministic_sorted_and_unique(self):
        a = sample_elements((64, 32), 200, seed=7)
        b = sample_elements((64, 32), 200, seed=7)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.dtype, np.uint32)
        self.assertEqual(len(a), 200)
        self.assertEqual(len(np.unique(a)), 200)
        self.assertTrue(np.all(np.diff(a.astype(np.int64)) > 0))
        self.assertTrue(a[-1] < 64 * 32)
        self.assertNotEqual(list(a), list(sample_elements((64, 32), 200, 8)))

    def test_padding_edge_and_last_head_are_always_covered(self):
        heads, rows, width, tokens = 6, 320, 32, 257
        indices = sample_elements((heads, rows, width), 512, seed=3,
                                  boundaries={1: [tokens]})
        h, r, w = np.unravel_index(indices.astype(np.int64),
                                   (heads, rows, width))
        for axis, value in ((h, heads - 1), (r, tokens - 1), (r, tokens),
                            (w, width - 1), (h, 0), (r, 0), (w, 0)):
            self.assertIn(value, axis, f"missing coordinate {value}")

    def test_a_small_budget_still_spans_the_tensor(self):
        # Fewer samples than the structured product: the cap must keep both
        # ends rather than one face of it.
        indices = sample_elements((24, 2048, 256), 64, seed=5,
                                  boundaries={1: [2000]})
        self.assertEqual(len(indices), 64)
        h, r, _ = np.unravel_index(indices.astype(np.int64),
                                   (24, 2048, 256))
        self.assertEqual(h.min(), 0)
        self.assertEqual(h.max(), 23)
        self.assertGreater(r.max(), 2000)

    def test_gathering_by_index_reproduces_the_tensor_values(self):
        tensor = np.arange(8 * 9 * 10).reshape(8, 9, 10).astype(np.float16)
        indices = sample_elements(tensor.shape, 120, seed=11)
        np.testing.assert_array_equal(tensor.ravel()[indices],
                                      tensor.reshape(-1)[indices])
        facts = describe(tensor.shape, indices)
        self.assertEqual(facts["count"], 120)
        self.assertEqual(facts["tensor_size"], 720)
        self.assertEqual(len(facts["distinct_per_axis"]), 3)

    def test_impossible_budgets_are_refused(self):
        with self.assertRaises(ValueError):
            sample_elements((4, 4), 17, seed=1)
        with self.assertRaises(ValueError):
            sample_elements((4, 4), 0, seed=1)
        with self.assertRaises(ValueError):
            sample_elements((4, 4), 4, seed=1, boundaries={2: [1]})
        with self.assertRaises(ValueError):
            axis_coordinates(0)


class RunSamplingTests(unittest.TestCase):
    def test_runs_stay_inside_one_row_of_the_last_axis(self):
        shapes = ((6, 128, 64), (24, 2048, 256), (2, 8192), (48, 128, 128))
        for shape in shapes:
            starts, length = sample_runs(shape, 512, seed=3)
            width = shape[-1]
            self.assertLessEqual(length, width)
            self.assertTrue(np.all((starts % width) + length <= width),
                            f"run crosses a row in {shape}")
            self.assertTrue(np.all(np.diff(starts.astype(np.int64)) > 0))
            self.assertEqual(starts.dtype, np.uint32)

    def test_runs_take_both_ends_of_a_row_and_the_padding_edge(self):
        shape, tokens = (6, 320, 64), 257
        starts, length = sample_runs(shape, 1024, seed=5,
                                     boundaries={1: [tokens]})
        head, row, column = np.unravel_index(starts.astype(np.int64), shape)
        self.assertIn(0, column)
        self.assertIn(shape[-1] - length, column)
        self.assertIn(tokens - 1, row)
        self.assertIn(tokens, row)
        self.assertIn(shape[0] - 1, head)

    def test_run_length_is_clamped_to_a_narrow_last_axis(self):
        starts, length = sample_runs((48, 128, 16), 256, seed=1,
                                     run_length=64)
        self.assertEqual(length, 16)
        self.assertTrue(np.all(starts % 16 == 0))

    def test_runs_are_deterministic_and_described(self):
        a, la = sample_runs((8, 64), 128, seed=2)
        b, lb = sample_runs((8, 64), 128, seed=2)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(la, lb)
        facts = describe_runs((8, 64), a, la)
        self.assertEqual(facts["gathered"], len(a) * la)
        self.assertEqual(facts["runs"], len(a))

    def test_degenerate_run_requests_are_refused(self):
        with self.assertRaises(ValueError):
            sample_runs((64,), 16, seed=1)
        with self.assertRaises(ValueError):
            sample_runs((8, 8), 0, seed=1)
        with self.assertRaises(ValueError):
            sample_runs((8, 8), 16, seed=1, boundaries={5: [1]})


if __name__ == "__main__":
    unittest.main()
