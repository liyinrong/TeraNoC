# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import unittest

from design_space import configurations
from run_softmax_probe import evaluate_log


class RunnerTests(unittest.TestCase):
    def test_eoc_without_numerical_coverage_fails(self):
        log = ("[qwen38_softmax] cores=128 columns=197 "
               "block=67 cycles=123 errors=0\n")
        log += "EOC register return value: 0x0\n"
        self.assertFalse(evaluate_log(log, 0, 128, 197)["passed"])
        log += "[DPI_CHECK] 0 ERRORS out of 73984 CHECKS\n"
        self.assertTrue(evaluate_log(log, 0, 128, 197)["passed"])
        self.assertFalse(evaluate_log(log, 124, 128, 197)["passed"])
        self.assertFalse(evaluate_log(log.replace(
            "73984", "1"), 0, 128, 197)["passed"])
        self.assertFalse(evaluate_log(log + "1 ERRORS", 0, 128, 197)["passed"])

    def test_lane_sweep_preserves_banks(self):
        cases = list(configurations())
        self.assertEqual(len(cases), 16)
        self.assertEqual(len({case["name"] for case in cases}), 16)
        for case in cases:
            self.assertEqual(case["spatz_per_tile"], 2)
            self.assertLessEqual(case["n_fpu"], 8)
            self.assertEqual(case["banking_factor"] * case["n_fpu"] * 2, 64)


if __name__ == "__main__":
    unittest.main()
