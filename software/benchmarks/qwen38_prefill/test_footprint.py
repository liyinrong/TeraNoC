# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import re
import tempfile
import unittest

from footprint import generate_header
from layer_fixture import generate_layer_header
from reference import TOY, make_fixture, vector_dims, weight_dims


CONSTANT = re.compile(r"^#define (Q38_\w+) (\d+)u$", re.MULTILINE)
POINTER = re.compile(r"^#define (\w+) \(\(", re.MULTILINE)
RESULT = re.compile(r"^static \w+ (result_\w+)\[", re.MULTILINE)


def symbols(text):
    return (dict(CONSTANT.findall(text)), set(POINTER.findall(text)),
            set(RESULT.findall(text)))


class FootprintTests(unittest.TestCase):
    def test_declared_shapes_match_the_generated_fixture(self):
        for kind in ("attention", "gdn"):
            _, weights = make_fixture(kind, TOY, tokens=3)
            declared = dict(weight_dims(kind, TOY))
            declared.update(vector_dims(kind, TOY))
            self.assertEqual(set(declared), set(weights))
            for name, shape in declared.items():
                self.assertEqual(tuple(weights[name].shape), tuple(shape),
                                 f"{kind}/{name}")

    def test_probe_header_declares_the_same_guest_interface(self):
        with tempfile.TemporaryDirectory() as temp:
            for kind in ("attention", "gdn"):
                real = Path(temp) / ("real_" + kind)
                probe = Path(temp) / ("probe_" + kind)
                real.mkdir()
                probe.mkdir()
                generate_layer_header(real, kind, 65, external_data=True)
                generate_header(probe, kind, 65, TOY, goldens=True)
                want = symbols((real / "layer_data.h").read_text())
                have = symbols((probe / "layer_data.h").read_text())
                self.assertEqual(want[0], have[0], kind)
                self.assertLessEqual(want[1], have[1], kind)
                self.assertLessEqual(want[2], have[2], kind)

    def test_real_geometry_reports_a_capacity_verdict(self):
        with tempfile.TemporaryDirectory() as temp:
            small = generate_header(temp, "attention", 2048, TOY,
                                    goldens=False)
            full = generate_header(temp, "attention", 2048, goldens=False)
            self.assertTrue(small["fits_configured_l2"])
            # The real layer's FP16 weights are hundreds of megabytes; the
            # probe must say so rather than silently truncating.
            self.assertGreater(full["weight_bytes"], 600 * 1024 * 1024)
            self.assertGreater(full["l2_bytes_needed"],
                               full["weight_bytes"])
            self.assertGreaterEqual(full["recommended_backing_base"],
                                    0x80000000 + full["result_bss_bytes"])


if __name__ == "__main__":
    unittest.main()
