# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path
import tempfile

from schedule_probe_analysis import analyze, interval_stats, overlap_cycles


class ProbeActivityTests(unittest.TestCase):
    def test_peak_and_sustained_occupancy_use_exact_intervals(self):
        jobs = [(0, 10, 30), (1, 20, 40), (2, 20, 35), (3, 25, 30)]
        counts = interval_stats(jobs, 10, 40, 4)
        self.assertEqual(counts["peak_concurrent"], 4)
        self.assertEqual(counts["cycles_at_80_percent"], 5)
        self.assertEqual(counts["busy_engine_cycles"], 60)
        self.assertAlmostEqual(counts["machine_busy_share"], 60 / 120)

    def test_dma_overlap_counts_union_not_channel_multiplicity(self):
        jobs = [(0, 0, 20), (1, 10, 30)]
        dma = [(0, 5, 15), (1, 10, 25)]
        self.assertEqual(overlap_cycles(jobs, dma, 0, 30), 20)
        self.assertEqual(overlap_cycles(jobs, dma, 15, 20), 5)

    def test_capacity_burst_cannot_make_the_mixer_gate_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.log"
            lines = [
                f"RM_STALL path=/system/group_0_0/tile_{i}/redmule "
                "eng=0 end=1400 cyc=1300 ticks=1300 x=0 w=0 y=0 z=0 "
                "m=16 n=5120 k=64\n"
                for i in range(4)]
            lines += [
                "IDMA_ACTIVITY path=/system/group_0_0/idma/stream_0 "
                "stream=0 start=500 end=1000 bytes=128 slices=1\n",
                "SPATZ_ACT path=/system/group_0_0/tile_0/pe0/ara "
                "cyc=3000 busy=100 exec=100\n",
                "[qwen38_layer] start=0 end=3000 cycles=3000\n"]
            path.write_text("".join(lines))
            coverage = {"burst_start": 100, "burst_end": 1500,
                        "mixer_end": 2000, "ffn_end": 2500,
                        "burst_spatz_ops": 8}
            result = analyze(path, "dse_s2_winner", coverage, 8)
            self.assertTrue(result["burst_gate_passed"])
            self.assertFalse(result["integrated_gate_passed"])
            self.assertEqual(result["stages"]["mixer"]["redmule"]
                             ["peak_concurrent"], 0)
            burst = analyze(path, "dse_s2_winner", coverage, 8,
                            burst_only=True)
            self.assertTrue(burst["design_gate_passed"])
            self.assertIsNone(burst["integrated_gate_passed"])
            self.assertEqual(set(burst["stages"]), {"burst"})

    def test_queue_overlap_window_judges_the_mixer(self):
        # A serial prologue fills all four engines; only the overlap window,
        # where the recurrence runs, may decide the mixer gate.
        def run(overlap_jobs):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.log"
                # Prologue [200, 1500), then disjoint [600+1000j, 1500+1000j).
                lines = [
                    f"RM_STALL path=/system/group_0_0/tile_{i}/redmule "
                    f"eng=0 end={1500 + 1000 * j} cyc={900 if j else 1300} "
                    f"ticks={900 if j else 1300} "
                    "x=0 w=0 y=0 z=0 m=16 n=5120 k=64\n"
                    for i in range(4) for j in range(1 + overlap_jobs)]
                lines += [
                    "IDMA_ACTIVITY path=/system/group_0_0/idma/stream_0 "
                    "stream=0 start=400 end=5000 bytes=128 slices=1\n",
                    "SPATZ_ACT path=/system/group_0_0/tile_0/pe0/ara "
                    "cyc=10000 busy=5000 exec=5000\n",
                    "[qwen38_layer] start=0 end=10000 cycles=10000\n"]
                path.write_text("".join(lines))
                coverage = {"burst_start": 100, "burst_end": 200,
                            "mixer_end": 6000, "ffn_end": 8000,
                            "overlap_start": 1600, "overlap_end": 5000,
                            "burst_spatz_ops": 8}
                return analyze(path, "dse_s2_winner", coverage, 8)

        serial = run(0)
        self.assertGreaterEqual(
            serial["stages"]["mixer"]["redmule"]["cycles_at_80_percent"],
            1000)
        self.assertEqual(serial["stages"]["mixer_overlap"]["redmule"]
                         ["peak_concurrent"], 0)
        self.assertIsNone(serial["stages"]["mixer_overlap"]
                          ["spatz_busy_share_sampled_during_redmule_80"])
        overlapped = run(3)
        window = overlapped["stages"]["mixer_overlap"]
        self.assertEqual(window["redmule"]["peak_concurrent"], 4)
        self.assertEqual(window["redmule"]["cycles_at_80_percent"], 2700)
        self.assertAlmostEqual(
            window["spatz_busy_share_sampled_during_redmule_80"],
            5000 / 10000 / 8)


if __name__ == "__main__":
    unittest.main()
