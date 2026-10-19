# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Check that idle installed engines remain in the utilization denominator."""

import unittest
from pathlib import Path
import tempfile

from engine_activity import is_schedule_probe, read, telemetry, totals


class ActivityTests(unittest.TestCase):
    def test_idma_backend_interval_becomes_channel_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "model.log"
            log.write_text("IDMA_ACTIVITY path=/system/group_1_2/dma/stream_0 "
                           "stream=0 start=10 end=30 bytes=128 slices=2\n")
            spatz, redmule, idma, _, last, _ = read(log, 4)
            self.assertEqual(last, 30)
            rows = telemetry(spatz, redmule, last, (4, 4), 4, 2, False,
                             idma=idma, correctness="passed")
            self.assertEqual(rows[0]["correctness"], "passed")
            self.assertIn("backend slice", rows[0]["dma"]["note"])
            self.assertEqual(
                rows[0]["dma"]["channels"]["6"]["completed_bytes"], 128)
            dma = [row for row in rows if row["kind"] == "dma"]
            self.assertEqual(dma, [{
                "kind": "dma", "start": 10, "end": 30, "phase": "bench",
                "g": 6, "scope": "channel", "channel": 6,
                "measurement": "model", "completed_bytes": 128,
                "transactions": 2, "active_cycles": 20}])
            log.write_text(log.read_text().replace("bytes=128", "bytes=0"))
            with self.assertRaises(ValueError):
                read(log, 4)
            log.write_text("SPATZ_ACT path=/system/group_1_2/pe0/ara "
                           "cyc=30 busy=10 exec=8IDMA_ACTIVITY path=x\n")
            with self.assertRaises(ValueError):
                read(log, 4)

    def test_layer_timer_window_is_kept_and_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "model.log"
            log.write_text("[qwen38_layer] kind=0 start=100 end=250 "
                           "cycles=150\n")
            self.assertEqual(read(log, 4)[-1], (100, 250))
            log.write_text("[qwen38_layer] kind=0 start=100 end=250 "
                           "cycles=151\n")
            with self.assertRaises(ValueError):
                read(log, 4)

    def test_partial_schedule_marker_is_never_a_full_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "model.log"
            log.write_text("[qwen38_probe] start_row=0 rows=256 "
                           "mixer_windows=4 numerical_checked=0\n"
                           "[qwen38_layer] start=10 end=30 cycles=20\n")
            self.assertTrue(is_schedule_probe(log))
            self.assertEqual(read(log, 4)[-1], (10, 30))
            log.write_text("[qwen38_layer] start=10 end=30 cycles=20\n")
            self.assertFalse(is_schedule_probe(log))

    def test_machine_share_counts_engines_without_jobs(self):
        spatz = [((0, 0, 0), 10, 6, 5)]
        redmule = [((0, 0, 0), 10, 4, 1)]
        out = totals(spatz, redmule, 10, installed_spatz=2,
                     installed_redmule=4)
        self.assertEqual(out["spatz"]["busy_share"], 0.6)
        self.assertEqual(out["spatz"]["machine_busy_share"], 0.3)
        self.assertEqual(out["redmule"]["busy_share"], 0.4)
        self.assertEqual(out["redmule"]["machine_busy_share"], 0.1)
        self.assertEqual(out["redmule"]["utilization_while_busy"], 0.75)


if __name__ == "__main__":
    unittest.main()
