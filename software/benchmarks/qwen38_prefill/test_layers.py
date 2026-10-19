# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from layer_fixture import pack_weight, pack_kv_cache, generate_layer_header
from run_layers import (check_contract, evaluate, evaluate_schedule_probe,
                        extract_actual, gdn_queue_shape, phase_checks,
                        attn_sweep_probe_counts)
from reference import Geometry, linear, linear_f16_fma


class LayerTests(unittest.TestCase):
    def test_kv_affine_probe_counts_batches_and_partial_key_blocks(self):
        # s1: four 128-row windows, three head batches; B exceeds the window.
        self.assertEqual(attn_sweep_probe_counts(0, 512, 128, 3, 256, 2048),
                         (12, 18))
        # S2: two windows; the distant prefix is sampled at three blocks.
        self.assertEqual(attn_sweep_probe_counts(1536, 512, 256, 3, 512, 2048),
                         (6, 18))
        # A final partial query window cannot extend the causal key limit.
        self.assertEqual(attn_sweep_probe_counts(256, 256, 256, 1, 256, 384),
                         (1, 2))

    def test_phase_checks_select_only_computed_tensors(self):
        fixture = {"kind": "attention", "check_mode": "full",
                   "checks": {name: {"shape": [2, 3]} for name in
                              ("output", "context", "cache", "mixer")},
                   "checked_elements": 24}
        mixer = phase_checks(fixture, "mixer")
        ffn = phase_checks(fixture, "ffn")
        self.assertEqual(list(mixer["checks"]), ["context", "cache", "mixer"])
        self.assertEqual(mixer["checked_elements"], 18)
        self.assertEqual(list(ffn["checks"]), ["output"])
        self.assertEqual(ffn["checked_elements"], 6)
        self.assertEqual(len(fixture["checks"]), 4)
        for invalid in (dict(fixture, kind="gdn"),
                        dict(fixture, check_mode="sampled")):
            with self.assertRaises(ValueError):
                phase_checks(invalid, "ffn")

    def test_phase_evaluation_requires_all_selected_descriptors(self):
        log = ("[qwen38_layer] cycles=123\n"
               "[qwen38_bounds] mixer_rows=64 ffn_rows=64 "
               "weight_panels=2 context_export_cycles=7\n"
               "EOC register return value: 0x0\n")
        checks = [f"[DPI_CHECK] Check {i}: 0 ERRORS out of 6 CHECKS\n"
                  for i in range(3)]
        self.assertTrue(evaluate(log + checks[0], 0, 6,
                                 expected_checks=1)["passed"])
        self.assertTrue(evaluate(log + "".join(checks), 0, 18,
                                 expected_checks=3)["passed"])
        self.assertFalse(evaluate(log + "".join(checks[:2]), 0, 18,
                                  expected_checks=3)["passed"])
        self.assertFalse(evaluate(log + "".join(checks).replace(
            "Check 2:", "Check 3:"), 0, 18, expected_checks=3)["passed"])

    def test_external_run_requires_resolved_model_elf_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertFalse(check_contract(root, required=True)["agree"])
            self.assertTrue(check_contract(root)["agree"])
            (root / "build.log").write_text(
                "-DN_FPU=4 -DVLEN=512 -DNUM_CORES=128\n")
            config = {"qwen38_contract": {
                "n_fpu": 4, "vlen": 512, "cores": 128}}
            (root / "gvsoc_config.json").write_text(json.dumps(config))
            self.assertTrue(check_contract(root, required=True)["agree"])
            self.assertFalse(check_contract(
                root, required=True, timing_only=True)["agree"])
            config["qwen38_contract"]["redmule_timing_only"] = True
            (root / "gvsoc_config.json").write_text(json.dumps(config))
            self.assertTrue(check_contract(
                root, required=True, timing_only=True)["agree"])
            config["qwen38_contract"]["vlen"] = 1024
            (root / "gvsoc_config.json").write_text(json.dumps(config))
            self.assertFalse(check_contract(root, required=True)["agree"])

    def test_kv_blocks_keep_heads_and_key_value_orientations(self):
        data = np.arange(2 * 2 * 8 * 4).reshape(2, 2, 8, 4).astype(np.float16)
        packed = pack_kv_cache(data, 4)
        restored = np.zeros_like(data)
        for block in range(2):
            tile = packed[block * 64:(block + 1) * 64]
            restored[0, :, block * 4:(block + 1) * 4] = (
                tile[:32].reshape(2, 4, 4).transpose(0, 2, 1))
            restored[1, :, block * 4:(block + 1) * 4] = (
                tile[32:].reshape(2, 4, 4))
        np.testing.assert_array_equal(restored, data)
        with self.assertRaises(ValueError):
            pack_kv_cache(data[:, :, :7], 4)

    def test_raw_dump_preserves_bits_and_requires_coverage(self):
        fixture = {"checks": {"half": {"dtype": "float16"},
                              "single": {"dtype": "float32"}}}
        log = ("CHECK(0): EXP = 00000000 - RESP = 00008000\n"
               "[DPI_CHECK] Check 0: 0 ERRORS out of 1 CHECKS\n"
               "CHECK(0): EXP = 3F800000 - RESP = 3F800000\n"
               "[DPI_CHECK] Check 1: 0 ERRORS out of 1 CHECKS\n")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hashes = extract_actual(log, fixture, root)
            self.assertEqual(set(hashes), {"half", "single"})
            self.assertEqual(
                (root / "actual_half.bin").read_bytes(), b"\x00\x80")
            with self.assertRaises(ValueError):
                extract_actual(log.split("CHECK(0): EXP = 3F800000")[
                               0], fixture, root)

    def test_fp16_feedback_and_fused_product_rounding(self):
        x = np.ones((1, 3), np.float16)
        w = np.array([[2048], [1], [-2048]], np.float16)
        self.assertEqual(float(linear(x, w)[0, 0]), 1)
        self.assertEqual(float(linear_f16_fma(x, w)[0, 0]), 0)
        x = np.array([[1, 1 + 2 ** -10]], np.float16)
        w = np.array([[-1 - 2 ** -9], [1 + 2 ** -10]], np.float16)
        self.assertEqual(float(linear_f16_fma(x, w)[0, 0]), 2 ** -20)

    def test_column_panels_preserve_values_and_zero_padding(self):
        source = np.arange(128 * 77).reshape(128, 77).astype(np.float16)
        for panel in (32, 64):
            packed = pack_weight(source, panel)
            width = ((77 + panel - 1) // panel) * panel
            restored = packed.transpose(1, 0, 2).reshape(128, width)
            np.testing.assert_array_equal(restored[:, :77], source)
            self.assertTrue(np.all(restored[:, 77:] == 0))

    def test_layers_have_four_checks_and_reject_oversized_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for kind in ("attention", "gdn"):
                meta = generate_layer_header(root, kind, 3)
                self.assertEqual(len(meta["checks"]), 4)
                self.assertEqual(meta["padded_rows"], 64)
            with self.assertRaises(ValueError):
                generate_layer_header(root, "gdn", 4096)

    def test_missing_or_failed_descriptor_never_passes(self):
        bounds = ("[qwen38_bounds] mixer_rows=64 ffn_rows=64 "
                  "weight_panels=2 context_export_cycles=7\n")
        log = ("[qwen38_layer] cycles=123\n" + bounds +
               "EOC register return value: 0x0\n")
        for i in range(4):
            log += f"[DPI_CHECK] Check {i}: 0 ERRORS out of 10 CHECKS\n"
        verdict = evaluate(log, 0, 40)
        self.assertTrue(verdict["passed"])
        self.assertFalse(evaluate(log, 0, 40, {"agree": False})["passed"])
        self.assertTrue(evaluate(log, 0, 40, {"agree": True})["passed"])
        self.assertEqual(verdict["resolved_mixer_rows"], 64)
        self.assertEqual(verdict["context_export_cycles"], 7)
        self.assertFalse(evaluate(log, 0, 41)["passed"])
        self.assertFalse(evaluate(log.replace(
            "Check 2", "Check 1"), 0, 40)["passed"])
        self.assertFalse(evaluate(log.replace(
            "0 ERRORS", "1 ERRORS"), 0, 40)["passed"])
        # The resolved bound must be in the log: a run that cannot state which
        # window it used is not a qualified bounded-window run.
        self.assertFalse(evaluate(log.replace(bounds, ""), 0, 40)["passed"])

    def test_schedule_probe_requires_its_region_and_eoc(self):
        log = ("[qwen38_probe] mixer_start_row=1984 mixer_rows=64 "
               "ffn_start_row=1792 ffn_rows=256 "
               "mixer_windows=1 ffn_blocks=2 gdn_chunks=4 key_blocks=0 "
               "projection_groups=11 gateup_groups=6 recurrent_k=32 "
               "burst_only=0 burst_start=10 burst_end=40 burst_jobs=64 "
               "burst_spatz_ops=128 mixer_end=80 ffn_end=110 "
               "numerical_checked=0\n"
               "[qwen38_layer] start=0 end=123 cycles=123\n"
               "RM_STALL path=engine\nSPATZ_ACT path=core\n"
               "IDMA_ACTIVITY path=dma\n"
               "EOC register return value: 0x0\n")
        expected = {"mixer_start_row": 1984, "mixer_rows": 64,
                    "ffn_start_row": 1792, "ffn_rows": 256,
                    "mixer_windows": 1, "ffn_blocks": 2,
                    "gdn_chunks": 4, "key_blocks": 0,
                    "recurrent_k": 32, "burst_only": 0, "burst_jobs": 64,
                    "burst_spatz_ops": 128}
        verdict = evaluate_schedule_probe(log, 0, {"agree": True}, expected)
        self.assertTrue(verdict["schedule_passed"])
        self.assertFalse(verdict["numerical_checked"])
        self.assertEqual(verdict["checked_elements"], 0)
        self.assertFalse(evaluate_schedule_probe(
            log, 0, {"agree": True},
            {**expected, "mixer_start_row": 0})["passed"])
        self.assertFalse(evaluate_schedule_probe(
            log, 0, {"agree": False}, expected)["passed"])
        self.assertFalse(evaluate_schedule_probe(
            log.replace("return value: 0x0", "return value: 0x1"),
            0, {"agree": True}, expected)["passed"])
        self.assertFalse(evaluate_schedule_probe(
            log.replace("IDMA_ACTIVITY path=dma\n", ""),
            0, {"agree": True}, expected)["passed"])

        burst_log = (log.replace("burst_only=0", "burst_only=1")
                     .replace("recurrent_k=32", "recurrent_k=0")
                     .replace("mixer_windows=1 ffn_blocks=2 gdn_chunks=4",
                              "mixer_windows=0 ffn_blocks=0 gdn_chunks=0")
                     .replace("projection_groups=11 gateup_groups=6",
                              "projection_groups=0 gateup_groups=0")
                     .replace("mixer_end=80 ffn_end=110",
                              "mixer_end=40 ffn_end=40"))
        burst_expected = {**expected, "burst_only": 1,
                          "mixer_windows": 0, "ffn_blocks": 0,
                          "gdn_chunks": 0, "recurrent_k": 0}
        self.assertTrue(evaluate_schedule_probe(
            burst_log, 0, {"agree": True}, burst_expected)["passed"])

    def test_gdn_queue_shape_counts_every_panel_once(self):
        # 160 QKV, 96 Z and two control panels of 64 columns in groups of 4.
        shape = gdn_queue_shape(Geometry(), 256, 4, 64, 16)
        self.assertEqual(shape["groups"], 40 + 24 + 1)
        self.assertEqual(shape["jobs"], 16 * (160 + 96 + 2))
        self.assertEqual(shape["spill_bytes"], 2 * 65 * 256 * 4 * 64 * 2)
        # S1 physical tiles: 96 control columns are three 32-column panels.
        shape = gdn_queue_shape(Geometry(), 128, 4, 32, 8)
        self.assertEqual(shape["groups"], 80 + 48 + 1)
        self.assertEqual(shape["jobs"], 16 * (320 + 192 + 3))

    def test_gdn_queue_probe_needs_every_group_and_an_overlap(self):
        log = ("[qwen38_probe] mixer_start_row=0 mixer_rows=512 "
               "ffn_start_row=0 ffn_rows=512 "
               "mixer_windows=8 ffn_blocks=4 gdn_chunks=32 key_blocks=0 "
               "projection_groups=6 gateup_groups=12 recurrent_k=32 "
               "burst_only=0 burst_start=10 burst_end=40 burst_jobs=64 "
               "burst_spatz_ops=128 mixer_end=900 ffn_end=950 "
               "numerical_checked=0\n"
               "[qwen38_queue] supertile_rows=256 panels=4 groups=65 "
               "queue_groups=130 queue_jobs=8256 overlap_start=100 "
               "overlap_end=800 prologue_cycles=50 recur_cycles=600 "
               "drain_cycles=5 output_cycles=40\n"
               "[qwen38_layer] start=0 end=999 cycles=999\n"
               "RM_STALL path=engine\nSPATZ_ACT path=core\n"
               "IDMA_ACTIVITY path=dma\n"
               "EOC register return value: 0x0\n")
        expected = {"mixer_start_row": 0, "mixer_rows": 512,
                    "ffn_start_row": 0, "ffn_rows": 512,
                    "mixer_windows": 8, "ffn_blocks": 4,
                    "gdn_chunks": 32, "key_blocks": 0,
                    "recurrent_k": 32, "burst_only": 0, "burst_jobs": 64,
                    "burst_spatz_ops": 128}
        queue = {"supertile_rows": 256, "panels": 4, "groups": 65,
                 "queue_groups": 130, "queue_jobs": 8256}
        verdict = evaluate_schedule_probe(log, 0, {"agree": True}, expected,
                                          queue)
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["probe_coverage"]["overlap_start"], 100)
        self.assertEqual(verdict["queue_coverage"]["recur_cycles"], 600)
        self.assertFalse(evaluate_schedule_probe(
            log.replace("queue_jobs=8256", "queue_jobs=8255"), 0,
            {"agree": True}, expected, queue)["passed"])
        self.assertFalse(evaluate_schedule_probe(
            log.replace("overlap_end=800", "overlap_end=950"), 0,
            {"agree": True}, expected, queue)["passed"])
        self.assertFalse(evaluate_schedule_probe(
            log.replace("[qwen38_queue]", "[other]"), 0,
            {"agree": True}, expected, queue)["passed"])


def header_array(text, name):
    """Values of one emitted array, whether embedded or externally backed."""
    body = re.search(r"\b" + name + r"\[\d+\] [^=]*= \{(.*?)\};",
                     text, re.S)
    if not body:
        raise KeyError(name)
    items = [item.strip() for item in body[1].split(",") if item.strip()]
    if items[0].endswith("u"):
        return np.array([int(i[:-1]) for i in items], np.uint32)
    return np.array([float.fromhex(i[:-1]) for i in items], np.float16)


class SampledCheckTests(unittest.TestCase):
    def test_sampled_goldens_are_the_full_goldens_at_those_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            for kind in ("attention", "gdn"):
                full = Path(temp) / ("full_" + kind)
                part = Path(temp) / ("part_" + kind)
                full.mkdir()
                part.mkdir()
                whole = generate_layer_header(full, kind, 3)
                meta = generate_layer_header(part, kind, 3, check_samples=64)
                self.assertEqual(meta["check_mode"], "sampled")
                self.assertEqual(whole["check_mode"], "full")
                whole_text = (full / "layer_data.h").read_text()
                part_text = (part / "layer_data.h").read_text()
                length = meta["check_run_length"]
                for name in whole["checks"]:
                    starts = np.load(part / f"sample_runs_{name}.npy")
                    facts = meta["samples"][name]
                    span = np.arange(min(length, facts["shape"][-1]))
                    index = (starts.astype(np.int64)[:, None] + span).ravel()
                    want = header_array(whole_text, "gold_" + name)
                    got = header_array(part_text, "gold_sample_" + name)
                    self.assertEqual(len(got), facts["gathered"])
                    np.testing.assert_array_equal(
                        got.view(np.uint16), want[index].view(np.uint16),
                        f"{kind}/{name}")
                    np.testing.assert_array_equal(
                        header_array(part_text, "run_start_" + name), starts)
                    # No run may straddle a row of the last axis.
                    width = facts["shape"][-1]
                    inside = (starts % width) + facts["run_length"] <= width
                    self.assertTrue(inside.all())
                # The whole golden must not reach L2 in sampled mode: that is
                # the point of bounded checking at full geometry.
                self.assertNotIn("gold_" + name + "[", part_text)
                # Nor the diagnostic context result, which is gathered from L1.
                self.assertNotIn("result_context[", part_text)
                self.assertIn("result_context[", whole_text)
                self.assertIn("#define Q38_SAMPLED_CHECKS 1", part_text)
                self.assertIn("#define Q38_SAMPLED_CHECKS 0", whole_text)

    def test_sampled_mode_places_indices_in_the_sidecar(self):
        # The full-size configuration: externally backed tensors and bounded
        # checking. Indices must be placed as unsigned words and no whole
        # golden may reach L2.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meta = generate_layer_header(root, "attention", 65,
                                         external_data=True,
                                         check_samples=128)
            names = [s["name"] for s in meta["external_data"]["segments"]]
            for tensor in meta["checks"]:
                self.assertIn("run_start_" + tensor, names)
                self.assertIn("gold_sample_" + tensor, names)
                self.assertNotIn("gold_" + tensor, names)
            header = (root / "layer_data.h").read_text()
            self.assertIn("uint32_t *)0x", header)


if __name__ == "__main__":
    unittest.main()
