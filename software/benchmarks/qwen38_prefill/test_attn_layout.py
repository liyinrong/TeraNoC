# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import inspect
import unittest

from attn_layout import (max_projection_geometry, projection_layout,
                         sweep_layout)
from run_layers import run_case


class AttentionLayoutTests(unittest.TestCase):
    dimensions = {"d": 5120, "head_dim": 256, "q_heads": 24,
                  "rotary_dim": 64}

    def test_maximum_target_geometry_and_limited_execution_extent(self):
        self.assertEqual(max_projection_geometry(
            "dse_s1_winner_b32", self.dimensions, 2048), (32, 512))
        self.assertEqual(max_projection_geometry(
            "dse_s2_winner", self.dimensions, 2048), (64, 1024))
        self.assertEqual(max_projection_geometry(
            "dse_s2_winner_f8_v1024", self.dimensions, 2048), (64, 1024))
        self.assertEqual(max_projection_geometry(
            "dse_s2_winner", self.dimensions, 2048, 1024), (64, 512))
        for profile, expected in [("dse_s1_winner_b32_c4_f4_v512",
                                   (32, 512)),
                                  ("dse_s2_winner_c4_f8_v1024",
                                   (64, 1024))]:
            self.assertEqual(max_projection_geometry(
                profile, self.dimensions, 2048), expected)
        # Four toy O panels need sixteen row tiles for all64 engines.
        self.assertEqual(max_projection_geometry(
            "dse_s2_winner", {"d": 256}, 512), (32, 512))

    def test_projection_reserves_head_tiles_without_sweep_kv(self):
        phase = projection_layout(352256, 16 * 2**20, 2048, 16, 64,
                                  self.dimensions, 512, 32, 32)
        self.assertTrue(phase["fits"])
        self.assertEqual(phase["kv_bytes"], 0)
        self.assertEqual(phase["scratch_bytes"], 0)
        self.assertEqual(phase["z_bytes"], 2 * 2**20)
        self.assertEqual(phase["stage_bytes"], 2 * 2**20)

    def test_sweep_ring_is_separate_and_larger_pairs_exceed_capacity(self):
        for tile, block, banks, capacity in [(32, 256, 2048, 16 * 2**20),
                                             (64, 512, 4096, 32 * 2**20)]:
            current = sweep_layout(700000, capacity, banks, 16, 64,
                                   256, tile, block)
            self.assertTrue(current["fits"])
            self.assertGreater(current["kv_bytes"], 0)
            for owner, bk in [(tile * 2, block), (tile, block * 2)]:
                larger = sweep_layout(700000, capacity, banks, 16, 64,
                                      256, owner, bk)
                self.assertFalse(larger["fits"])

    def test_finish_reuse_saves_one_owner_vector_without_removing_kv(self):
        for tile, block, banks, capacity in [(32, 256, 2048, 16 * 2**20),
                                             (64, 512, 4096, 32 * 2**20)]:
            old = sweep_layout(524288, capacity, banks, 16, 64,
                               256, tile, block)
            new = sweep_layout(524288, capacity, banks, 16, 64,
                               256, tile, block, reuse_query=True)
            self.assertEqual(old["end"] - new["end"], 64 * tile * 256 * 2)
            self.assertEqual(old["kv_bytes"], new["kv_bytes"])
            self.assertTrue(new["fits"])

    def test_next_power_of_two_tile_exceeds_phase_memory_budgets(self):
        for tile, banks, capacity in [(32, 2048, 16 * 2**20),
                                      (64, 4096, 32 * 2**20)]:
            for output in [False, True]:
                current = projection_layout(
                    700000, capacity, banks, 16, 64, self.dimensions,
                    tile * 16, tile, tile, output)
                self.assertTrue(current["fits"])
                for row_tiles in [1, 2, 4, 8, 16]:
                    larger = projection_layout(
                        700000, capacity, banks, 16, 64, self.dimensions,
                        2 * tile * row_tiles, 2 * tile, tile, output)
                    self.assertFalse(larger["fits"])

    def test_partial_vc_row_tiles_are_rejected(self):
        for rows in [513, 640]:
            with self.assertRaises(ValueError):
                projection_layout(352256, 16 * 2**20, 2048, 16, 64,
                                  self.dimensions, rows, 32, 32)

    def test_rejected_b128_tree_variant_never_reaches_build(self):
        required = {name: None for name, parameter in
                    inspect.signature(run_case).parameters.items()
                    if parameter.default is inspect.Parameter.empty}
        with self.assertRaisesRegex(ValueError, "only above B128"):
            run_case(**required, attn_sweep_tree_max=True,
                     attn_vc_sweep=True, probe_phase="mixer", key_block=128)

    def test_projection_options_reject_incompatible_phases_before_build(self):
        required = {name: None for name, parameter in
                    inspect.signature(run_case).parameters.items()
                    if parameter.default is inspect.Parameter.empty}
        for options in [{"attn_sweep_mode": "early"},
                        {"attn_sweep_reuse_query": True},
                        {"attn_sweep_register": True},
                        {"attn_sweep_tree_max": True},
                        {"attn_sweep_row_dma": True},
                        {"attn_vc_dual": True},
                        {"attn_vc_max_rows": True},
                        {"attn_vc_panel": True},
                        {"attn_vc_panel": True, "attn_vc": True,
                         "attn_vc_sweep": True, "probe_phase": "ffn"}]:
            with self.assertRaises(ValueError):
                run_case(**required, **options)


if __name__ == "__main__":
    unittest.main()
