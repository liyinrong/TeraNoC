# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""A cached fixture has to be the same fixture."""

import json
import tempfile
import unittest
from pathlib import Path

import fixture_cache
from fixture_cache import build, key
from reference import TOY

PARAMETERS = {"kind": "attention", "tokens": 70, "check_samples": 128,
              "check_run_length": 16}


class FixtureCacheTests(unittest.TestCase):
    def test_a_hit_reproduces_the_uncached_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain, first, second = (root / n for n in ("plain", "a", "b"))
            for directory in (plain, first, second):
                directory.mkdir()
            cache = root / "cache"
            reference = build(plain, PARAMETERS)
            build(first, PARAMETERS, cache)
            self.assertEqual(len(list(cache.iterdir())), 1)
            manifest = build(second, PARAMETERS, cache)
            self.assertEqual(len(list(cache.iterdir())), 1)
            self.assertEqual(manifest, reference)
            for name in ("layer_data.h", "sample_runs_output.npy"):
                self.assertEqual((plain / name).read_bytes(),
                                 (second / name).read_bytes(), name)

    def test_a_second_run_does_not_regenerate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            calls = []
            real = fixture_cache.generate_layer_header

            def counted(directory, **parameters):
                calls.append(directory)
                return real(directory, **parameters)

            fixture_cache.generate_layer_header = counted
            try:
                for name in ("a", "b", "c"):
                    (root / name).mkdir()
                    build(root / name, PARAMETERS, cache)
            finally:
                fixture_cache.generate_layer_header = real
            self.assertEqual(len(calls), 1)

    def test_every_emitted_parameter_changes_the_key(self):
        base = key(PARAMETERS)
        for name, value in (("kind", "gdn"), ("tokens", 71),
                            ("check_samples", 129), ("check_run_length", 8),
                            ("external_data", True), ("l2_mib", 1024),
                            ("attention_kv_heads", 2), ("seed", 1),
                            ("geometry", TOY), ("packed_cache", False)):
            with self.subTest(name=name):
                self.assertNotEqual(base, key({**PARAMETERS, name: value}))

    def test_cost_only_parameters_share_a_slot(self):
        self.assertEqual(key(PARAMETERS), key({**PARAMETERS, "chunk": 1}))

    def test_an_interrupted_generation_is_not_a_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            slot = cache / key(PARAMETERS)
            (slot.parent / (slot.name + ".partial")).mkdir(parents=True)
            (root / "a").mkdir()
            manifest = build(root / "a", PARAMETERS, cache)
            self.assertTrue((slot / "fixture.json").exists())
            self.assertEqual(
                manifest,
                json.loads((slot / "fixture.json").read_text()))


if __name__ == "__main__":
    unittest.main()
