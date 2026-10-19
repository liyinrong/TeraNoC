# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Check that occupancy bins conserve the recorded engine work."""

import unittest

from engine_timeline import bin_activity


class TimelineTests(unittest.TestCase):
    def test_intervals_and_samples_keep_their_denominators(self):
        meta = {"mesh": [1, 1], "tiles_per_group": 2,
                "cores_per_tile": 1}
        records = [
            {"kind": "stage", "label": "redmule array", "start": 0,
             "end": 50, "hsk": 45, "stall": 5},
            {"kind": "stage", "label": "spatz vector", "start": 0,
             "end": 100, "hsk": 35, "stall": 5, "idle": 60},
            {"kind": "dma", "channel": 0, "start": 25, "end": 75,
             "active_cycles": 50}]
        result = bin_activity(meta, records, 50)
        first, second = result["bins"]
        self.assertEqual(first["redmule"], 0.5)
        self.assertEqual(second["redmule"], 0.0)
        self.assertEqual(first["dma"], second["dma"])
        self.assertEqual(first["dma"], 0.5)
        self.assertEqual(first["spatz"], second["spatz"])
        self.assertEqual(first["spatz"], 0.4)
        self.assertEqual(first["spatz_coverage"], 0.5)
        self.assertEqual(second["spatz_coverage"], 0.5)
        self.assertEqual(sum(b["redmule"] * 2 * 50 for b in result["bins"]),
                         50)
        self.assertEqual(sum(b["dma"] * 50 for b in result["bins"]), 50)
        self.assertEqual(sum(b["spatz"] * b["spatz_coverage"] * 2 * 50
                             for b in result["bins"]), 40)

    def test_missing_dma_is_unmeasured_not_zero(self):
        meta = {"mesh": [1, 1], "tiles_per_group": 1,
                "cores_per_tile": 1}
        records = [{"kind": "stage", "label": "redmule array",
                    "start": 0, "end": 10, "hsk": 10, "stall": 0}]
        result = bin_activity(meta, records, 10)
        self.assertIsNone(result["bins"][0]["dma"])
        self.assertIsNone(result["bins"][0]["spatz"])


if __name__ == "__main__":
    unittest.main()
